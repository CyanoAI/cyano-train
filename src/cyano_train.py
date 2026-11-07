#!/usr/bin/env python3
# v7 ROBUST trainer: dynamic loss scaling, skip-with-descale, full checkpoint
# resume (model + optimizer internals + data cursor + scale), eval holdout,
# optional async flat-buffer sync for world>1.
import os, sys, json, time, mmap, argparse, shutil, datetime, hashlib
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Extra module locations. The optimizer/checkpoint helpers live in
# CYANO_TRAIN_LIB; the model definition ships next to this file.
_LIB = os.environ.get("CYANO_TRAIN_LIB", "")
if _LIB:
    sys.path.insert(0, _LIB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyano_model import Transformer, CONFIGS, Config, get_op_call_counts
from rank_checkpoint import load_rank_checkpoint, save_rank_checkpoint


class ChunkedCE(torch.autograd.Function):
    """Memory-lean cross-entropy (+optional z-loss) over fp16 logits.

    Persistent state: only the fp16 logits. fp32 softmax internals live
    only transiently per chunk, instead of a full fp32 copy of logits."""

    @staticmethod
    def forward(ctx, logits, targets, z_coef, chunks=8):
        N, V = logits.shape
        step = (N + chunks - 1) // chunks
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        zsum = torch.zeros_like(total)
        for lo in range(0, N, step):
            hi = min(N, lo + step)
            ch = logits[lo:hi].float()
            logp = F.log_softmax(ch, dim=-1)
            total = total - logp.gather(1, targets[lo:hi, None]).sum()
            if z_coef:
                z = ch.logsumexp(dim=-1)
                zsum = zsum + z.square().sum()
        ctx.save_for_backward(logits, targets)
        ctx.z_coef = z_coef
        ctx.chunks = chunks
        return total / N + (z_coef * zsum / N if z_coef else 0.0)

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        N, V = logits.shape
        step = (N + ctx.chunks - 1) // ctx.chunks
        grads = torch.empty_like(logits)
        g = grad_out.float() / N
        for lo in range(0, N, step):
            hi = min(N, lo + step)
            chf = logits[lo:hi].float()
            sm = F.softmax(chf, dim=-1)
            if ctx.z_coef:
                z = chf.logsumexp(dim=-1, keepdim=True)
                sm.mul_(1.0 + 2.0 * ctx.z_coef * z)
            rows = torch.arange(hi - lo, device=logits.device)
            sm[rows, targets[lo:hi]] -= 1.0
            grads[lo:hi] = (sm * g).to(logits.dtype)
        return grads, None, None, None

CONFIGS["dense85m"] = Config(dim=512, n_layers=16, n_heads=8, hidden_dim=1376)
CONFIGS["dense05b"] = Config(dim=1408, n_layers=16, n_heads=11, hidden_dim=4048)
CONFIGS["dense200m"] = Config(dim=768, n_layers=12, n_heads=12, hidden_dim=3072)


class MuonAdamW:
    """Muon for 2D hidden matrices (NS5 orthogonalized momentum, FP32),
    AdamW (pure-torch, FP32 master) for the rest. Same _params_info layout
    as FusedAdamW for checkpoint/resume compatibility."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, muon_lr=0.02, muon_momentum=0.95,
                 ns_steps=5, ns_map=None, vocab_size=50257):
        self.lr = lr
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.ns_steps = ns_steps
        self.ns_map = ns_map or {}
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self._params_info = []
        for parameter in params:
            if not parameter.requires_grad:
                continue
            is_matrix = (parameter.ndim == 2
                         and parameter.shape[0] != vocab_size
                         and parameter.shape[1] != vocab_size
                         and min(parameter.shape) > 1)
            self._params_info.append({
                "parameter": parameter,
                "master": parameter.detach().float().clone(),
                "m": torch.zeros_like(parameter, dtype=torch.float32),
                "v": torch.zeros_like(parameter, dtype=torch.float32),
                "output": torch.empty_like(parameter),
                "use_muon": is_matrix,
                "ns_steps": self.ns_map.get(tuple(parameter.shape), ns_steps),
            })

    def param_groups(self):
        return [{"params": [e["parameter"] for e in self._params_info],
                 "lr": self.lr}]

    def set_lr(self, lr):
        self.lr = lr

    def set_muon_lr(self, lr):
        self.muon_lr = lr

    def zero_grad(self, set_to_none=True):
        for entry in self._params_info:
            p = entry["parameter"]
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @staticmethod
    def _ns5(G, steps=5):
        # Newton-Schulz iteration in fp16 (fp16 GEMM is much faster here).
        # Keller Jordan coefficients; iterations bounded in [-1,1] by construction.
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = (G.float() / (G.float().norm() + 1e-7)).to(torch.float16)
        transposed = False
        if X.size(0) > X.size(1):
            X = X.t()
            transposed = True
        for _ in range(steps):
            A = X @ X.t()
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.t()
        return X

    @staticmethod
    def _ns5_batched(D, steps=5):
        # Same math as _ns5 but D is (k, m, n): batched fp16 matmuls.
        # NS5 coefficients are floated in fp16; norm in fp32 for stability.
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = (D.float() / (D.float().norm(dim=(1, 2), keepdim=True) + 1e-7)).to(torch.float16)
        transposed = False
        if X.size(1) > X.size(2):
            X = X.transpose(1, 2)
            transposed = True
        for _ in range(steps):
            A = X @ X.transpose(1, 2)
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.transpose(1, 2)
        return X

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        muon_work = []
        for entry in self._params_info:
            p = entry["parameter"]
            if p.grad is None:
                continue
            g = p.grad.float()
            master = entry["master"]
            master.mul_(1.0 - self.lr * self.weight_decay)
            if entry["use_muon"]:
                buf = entry["m"]
                buf.mul_(self.muon_momentum).add_(g)
                d = g.add(buf, alpha=self.muon_momentum)  # nesterov
                if d.ndim > 2:
                    d = d.reshape(d.size(0), -1)
                muon_work.append((entry, p, d.contiguous()))
            else:
                m, v = entry["m"], entry["v"]
                m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                bc1 = 1 - self.beta1 ** self.step_count
                bc2 = 1 - self.beta2 ** self.step_count
                mhat = m / bc1
                vhat = v / bc2
                master.addcdiv_(mhat, vhat.sqrt().add_(self.eps),
                                value=-self.lr)
                entry["output"].copy_(master)
                p.copy_(entry["output"])
        if muon_work:
            groups = {}
            order = []
            for item in muon_work:
                shape = item[2].shape
                if shape not in groups:
                    groups[shape] = []
                    order.append(shape)
                groups[shape].append(item)
            for shape in order:
                items = groups[shape]
                ns = items[0][0].get("ns_steps", self.ns_steps)
                if len(items) > 1:
                    batched = torch.stack([d for _, _, d in items])
                    outs = self._ns5_batched(batched, steps=ns)
                    for (entry, p, _), u in zip(items, outs.unbind(0)):
                        master = entry["master"]
                        scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                        master.add_(u.reshape_as(master),
                                    alpha=-self.muon_lr * scale)
                        entry["output"].copy_(master)
                        p.copy_(entry["output"])
                else:
                    entry, p, d = items[0]
                    u = self._ns5(d, steps=ns)
                    master = entry["master"]
                    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    master.add_(u.reshape_as(master), alpha=-self.muon_lr * scale)
                    entry["output"].copy_(master)
                    p.copy_(entry["output"])

    def assert_finite(self):
        for entry in self._params_info:
            if not torch.isfinite(entry["parameter"]).all().item():
                raise FloatingPointError("non-finite parameter")

LOGDIR = os.environ.get("CYANO_LOG_DIR", "./logs")


def checkpoint_space_available(free_gb, required_gb, world, device):
    if world > 1:
        minimum_free = torch.tensor(free_gb, dtype=torch.float64, device=device)
        dist.all_reduce(minimum_free, op=dist.ReduceOp.MIN)
        free_gb = float(minimum_free.item())
    return free_gb >= required_gb, free_gb


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--master", default="192.168.0.3")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--max_run_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--decay", default="linear", choices=["none", "cosine", "linear"])
    ap.add_argument("--untied", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--scale_init", type=float, default=1024.0)
    ap.add_argument("--scale_max", type=float, default=65536.0)
    ap.add_argument("--eval_every", type=int, default=128)
    ap.add_argument("--save_every", type=int, default=256)
    ap.add_argument("--ckpt", default=os.environ.get("CYANO_CKPT", "./ckpt.pt"))
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--init", default=None)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval_data", default=None)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--sync_bucket_mb", type=float, default=0)
    ap.add_argument("--heartbeat_every", type=int, default=16)
    ap.add_argument("--config", default="dense30m", choices=CONFIGS.keys())
    ap.add_argument("--attention-chunk", type=int, default=0,
                    help="attention chunk size; 0 = use SDPA")
    ap.add_argument("--exit_file", default=None)
    ap.add_argument("--optimizer", default="muon", choices=["adamw", "muon"])
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--muon_ns_steps", type=int, default=5)
    ap.add_argument("--ns-map", default="",
                    help="per-shape NS steps, e.g. '1408x8096:1,1408x4224:2'")
    ap.add_argument("--z_loss", type=float, default=1e-4)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--ckpt_save_optimizer", type=int, default=1,
                    help="0 = save model/step only (for throughput runs); "
                         "cannot resume optimizer state from such a checkpoint")
    ap.add_argument("--ckpt_min_free_gb", type=float, default=2.0,
                    help="skip checkpoint (and keep training) below this free space")
    a = ap.parse_args()
    if a.max_run_steps < 0 or a.sync_bucket_mb < 0 or a.save_every < 0 or a.muon_ns_steps < 1:
        ap.error("--max_run_steps, --sync_bucket_mb and --save_every must be nonnegative; --muon_ns_steps must be positive")
    if a.sync_bucket_mb > 0:
        from local_sgd_sync import average_masters

    os.environ["UNTIED"] = str(a.untied)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.set_device(0)
    use_dist = a.world > 1
    if use_dist and not use_cuda:
        ap.error("distributed training requires CUDA")
    if use_dist:
        dist.init_process_group(backend="nccl", init_method="tcp://%s:29528" % a.master,
                                rank=a.rank, world_size=a.world,
                                timeout=datetime.timedelta(seconds=120))
        dummy = torch.zeros(1024, device="cuda")
        dist.all_reduce(dummy)

    cfg = CONFIGS[a.config]; cfg.max_seq_len = a.seq_len
    model = Transformer(cfg, chunk_size=a.attention_chunk).to(device)
    if use_cuda:
        model = model.half()
    model.train()
    ns_map = {}
    if a.ns_map:
        for part in a.ns_map.split(","):
            shape, n = part.split(":")
            dims = shape.split("x")
            ns_map[(int(dims[0]), int(dims[1]))] = int(n)
        print("ns_map:", ns_map, flush=True)

    if a.optimizer == "muon":
        opt = MuonAdamW(model.parameters(), lr=a.lr, muon_lr=a.muon_lr,
                        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                        ns_steps=a.muon_ns_steps, ns_map=ns_map,
                        vocab_size=cfg.vocab_size)
    else:
        from train_champion import FusedAdamW

        opt = FusedAdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                         eps=1e-8, weight_decay=0.1)
    params = list(model.parameters())

    if a.init and a.resume:
        ap.error("--init and --resume are mutually exclusive")
    if a.init:
        init_ck = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ck["model"])
        with torch.no_grad():
            for e in opt._params_info:
                e["master"].copy_(e["parameter"])
                e["output"].copy_(e["parameter"])
        del init_ck
        print("INITIALIZED model from", a.init, flush=True)

    fd = open(a.data, "rb")
    mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
    n_tokens = len(mm) // 4
    B, S = a.batch_size, a.seq_len
    required_tokens = B * (S + 1)
    # train region: first 98%; eval region: last 2%
    train_end = int(n_tokens * 0.98)
    lo, hi = 0, train_end
    if train_end - lo < required_tokens:
        ap.error("--data does not contain one complete training batch")
    rng = torch.Generator().manual_seed(a.seed + a.rank)
    cursor = lo + int(torch.randint(0, max(1, hi - lo - B*(S+1) - 1), (1,), generator=rng))

    _pin = None
    def get_train_batch():
        nonlocal cursor, _pin
        need = B * (S + 1)
        if cursor + need > hi:
            cursor = lo
        if _pin is None:
            _pin = torch.empty(need, dtype=torch.int64,
                               pin_memory=use_cuda)
        view = np.frombuffer(mm, dtype=np.uint32, count=need,
                             offset=cursor * 4)
        _pin.copy_(torch.from_numpy(view), non_blocking=False)
        cursor += B * S
        return _pin.view(B, S + 1).to(device, non_blocking=use_cuda)

    eval_src = a.eval_data or a.data
    efd = open(eval_src, "rb")
    emm = mmap.mmap(efd.fileno(), 0, access=mmap.ACCESS_READ)
    en_tokens = len(emm) // 4
    eval_lo = 0 if a.eval_data else int(en_tokens * 0.98)
    eval_hi = en_tokens if a.eval_data else en_tokens - 1
    eval_protocol = ("external-whole-file-fixed4batch" if a.eval_data
                     else "legacy-tail2pct-fixed4batch")
    if en_tokens - eval_lo < required_tokens:
        source = "--eval_data" if a.eval_data else "the default tail evaluation region"
        ap.error(f"{source} must contain at least one complete evaluation batch")

    def get_eval_batch(j):
        need = B * (S + 1)
        span = eval_hi - eval_lo - need + (1 if a.eval_data else 0)
        start = eval_lo + (j * need) % max(1, span)
        arr = torch.frombuffer(bytearray(emm[start*4:(start+need)*4]),
                               dtype=torch.uint32).long()
        return arr.view(B, S+1).to(device)

    def lr_at(step):
        # WSD: warmup -> constant -> final decay_frac linear to ~0
        if a.warmup > 0 and step <= a.warmup:
            return a.lr * step / a.warmup
        if a.decay == "none":
            return a.lr
        decay_start = a.steps - max(1, int(a.steps * a.decay_frac))
        if step <= decay_start:
            return a.lr
        t_ = (step - decay_start) / max(1, a.steps - decay_start)
        if a.decay == "cosine":
            return a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t_)))
        return a.lr * max(0.0, 1.0 - t_)

    import math
    scale = a.scale_init
    clean_streak = 0
    start_step = 1
    tokens_done = 0
    wall_prev = 0.0
    eval_hist = []
    skips = 0
    last_sync_step = 0
    legacy_resume = False

    resume_contract = {
        # batch_size/seq_len affect only the data cursor stride, never the
        # saved state tensor shapes, so they are safe to change on resume.
        "optimizer": a.optimizer, "lr": a.lr, "muon_lr": a.muon_lr,
        "muon_ns_steps": a.muon_ns_steps,
        "z_loss": a.z_loss,
        "k": a.k,
        "config": a.config, "untied": a.untied, "steps": a.steps,
        "warmup": a.warmup, "decay": a.decay, "decay_frac": a.decay_frac,
        "data": {"path": os.path.realpath(a.data), "size": len(mm),
                 "sha256": file_sha256(a.data)},
        "eval_data": {"path": os.path.realpath(eval_src), "size": len(emm),
                       "sha256": file_sha256(eval_src)},
        "eval_protocol": eval_protocol,
        "param_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "param_shapes": [list(p.shape) for _, p in model.named_parameters() if p.requires_grad],
    }
    if a.resume:
        local_error = None
        ck = None
        try:
            ck = load_rank_checkpoint(a.ckpt, a.rank, a.world, map_location="cpu")
            legacy_resume = ck.get("legacy_resume", False)
            if not legacy_resume:
                if "resume_contract" not in ck:
                    raise ValueError("rank checkpoint missing resume_contract; run a new pilot")
                saved_contract = ck["resume_contract"]
                if not isinstance(saved_contract, dict):
                    raise ValueError("resume_contract must be a dict; run a new pilot")
                mismatches = sorted(key for key in saved_contract.keys() | resume_contract.keys()
                                    if key not in saved_contract or key not in resume_contract
                                    or saved_contract[key] != resume_contract[key])
                if mismatches:
                    raise ValueError("resume_contract mismatch: " + ", ".join(mismatches))
            if "opt_extra" in ck:
                if len(ck["opt_extra"]) != len(opt._params_info):
                    raise ValueError("opt_extra length does not match optimizer parameters")
                for index, e in enumerate(opt._params_info):
                    sd = ck["opt_extra"][index]
                    for key in ("master", "m", "v", "output"):
                        if legacy_resume and key == "output" and key not in sd:
                            continue
                        if key not in sd or not torch.is_tensor(sd[key]) or sd[key].shape != e[key].shape:
                            raise ValueError(f"opt_extra[{index}].{key} missing or shape mismatch")
            elif not ck.get("optimizer_state_saved", True):
                print("RESUME without optimizer state: model/step only", flush=True)
            else:
                raise ValueError("checkpoint has no opt_extra but claims optimizer state")
            model.load_state_dict(ck["model"])
            if "opt_extra" in ck:
                with torch.no_grad():
                    for e, sd in zip(opt._params_info, ck["opt_extra"]):
                        e["master"].copy_(sd["master"])
                        e["m"].copy_(sd["m"])
                        e["v"].copy_(sd["v"])
                        if "output" in sd:
                            e["output"].copy_(sd["output"])
                        else:
                            e["output"].copy_(e["master"])
                    del sd
            else:
                with torch.no_grad():
                    for e in opt._params_info:
                        e["master"].copy_(e["parameter"])
                        e["m"].zero_()
                        e["v"].zero_()
                        e["output"].copy_(e["parameter"])
            scale = ck["scale"]; clean_streak = ck["clean_streak"]
            if legacy_resume:
                counts = ck.get("optimizer_step_counts")
                opt.step_count = counts[a.rank] if counts is not None else ck["optimizer_step_count"]
                cursors = ck.get("cursors")
                cursor = cursors[a.rank] if cursors is not None else ck["cursor"]
            else:
                opt.step_count = ck["optimizer_step_count"]
                cursor = ck["cursor"]
            start_step = ck["step"] + 1
            tokens_done = ck["tokens_done"]; wall_prev = ck["wall_s"]
            eval_hist = ck.get("eval_hist", [])
            skips = ck.get("nonfinite_diagnostics", {}).get("skips", 0)
            last_sync_step = ck.get("sync_state", {}).get("last_sync_step", 0)
            if not legacy_resume:
                torch.set_rng_state(ck["rng_cpu"])
                if use_cuda:
                    torch.cuda.set_rng_state_all(ck["rng_cuda"])
                rng.set_state(ck["data_rng"])
            if use_cuda:
                torch.cuda.synchronize()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        finally:
            del ck
        load_errors = [local_error]
        if use_dist:
            load_errors = [None] * a.world
            dist.all_gather_object(load_errors, local_error)
        if any(error is not None for error in load_errors):
            raise RuntimeError("checkpoint resume failed: " + "; ".join(
                f"rank {rank}: {error}" for rank, error in enumerate(load_errors) if error is not None))
        if legacy_resume:
            print("RESUME inexact: legacy shared model/optimizer checkpoint; "
                  "using original per-rank cursors/counts where available; "
                  "private optimizer/model/RNG history is not recoverable", flush=True)
        print("RESUMED from step", start_step - 1, flush=True)

    os.makedirs(a.logdir, exist_ok=True)
    tmp_path = os.path.join(a.logdir, "rank%d_%s.jsonl" % (a.rank, a.tag))
    logf = open(tmp_path, "a")
    buf = []
    losses = []
    updates_start = opt.step_count
    skips_start = skips
    run_end = min(a.steps, start_step + a.max_run_steps - 1) if a.max_run_steps else a.steps
    last_step = start_step - 1
    stop_reason = "steps_complete" if last_step >= a.steps else "pilot_complete"
    checkpoint_skipped = False
    startup = {"event": "startup", "rank": a.rank, "start_step": start_step,
               "run_end": run_end, "schedule_steps": a.steps,
               "eval_protocol": eval_protocol, "eval_batches": 4,
               "eval_source": eval_src, "legacy_resume": legacy_resume,
               "eval_comparability": "external whole-file and legacy tail2% metrics are not comparable",
               "token_accounting": "per-rank tokens from successful optimizer updates"}
    print("EVAL PROTOCOL", eval_protocol,
          "external whole-file vs legacy tail2% metrics are NOT comparable", flush=True)
    logf.write(json.dumps(startup) + "\n"); logf.flush()
    if use_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    wall_window_start = t_start
    wall_window_tokens = tokens_done
    V = cfg.vocab_size

    def save_checkpoint(step, reason):
        nonlocal checkpoint_skipped, last_sync_step
        if use_dist and a.sync_bucket_mb > 0 and step % a.k != 0:
            with torch.no_grad():
                average_masters(opt._params_info, a.world, a.sync_bucket_mb)
            dist.barrier()
            last_sync_step = step
        try:
            free_gb = shutil.disk_usage(os.path.dirname(a.ckpt) or ".").free / 1e9
        except OSError:
            free_gb = float("inf")
        needed_gb = 9.0 if a.ckpt_save_optimizer else 1.2
        required_gb = max(a.ckpt_min_free_gb, needed_gb)
        can_save, free_gb = checkpoint_space_available(
            free_gb, required_gb, a.world, device)
        if not can_save:
            checkpoint_skipped = True
            buf.append(json.dumps({"event": "ckpt_skipped", "step": step,
                                   "rank": a.rank, "free_gb": round(free_gb, 2),
                                   "needed_gb": needed_gb}) + "\n")
            logf.write("".join(buf)); logf.flush(); buf.clear()
            return
        ck = {"step": step, "cursor": cursor, "resume_contract": resume_contract,
              "optimizer_step_count": opt.step_count, "scale": scale,
              "clean_streak": clean_streak, "tokens_done": tokens_done,
              "rng_cpu": torch.get_rng_state(),
              "rng_cuda": torch.cuda.get_rng_state_all() if use_cuda else [],
              "data_rng": rng.get_state(),
              "stop_reason": reason, "nonfinite_diagnostics": {"skips": skips},
              "wall_s": wall_prev + time.perf_counter() - t_start,
              "sync_state": {"world": a.world, "k": a.k, "last_sync_step": last_sync_step},
              "model": model.state_dict(),
              "eval_hist": eval_hist, "eval_protocol": eval_protocol,
              "optimizer_state_saved": bool(a.ckpt_save_optimizer)}
        if a.ckpt_save_optimizer:
            ck["opt_extra"] = [{"master": e["master"], "m": e["m"], "v": e["v"],
                                "output": e["output"]} for e in opt._params_info]
        try:
            save_rank_checkpoint(a.ckpt, ck, a.rank, a.world)
        except Exception as exc:
            buf.append(json.dumps({"event": "ckpt_fail", "step": step,
                                   "rank": a.rank, "err": str(exc)}) + "\n")
            logf.write("".join(buf)); logf.flush(); buf.clear()
            raise
        finally:
            del ck
        buf.append(json.dumps({"event": "ckpt", "step": step, "rank": a.rank,
                               "stop_reason": reason, "committed": True,
                               "optimizer_state_saved": bool(a.ckpt_save_optimizer)}) + "\n")
        logf.write("".join(buf)); logf.flush(); buf.clear()

    for step in range(start_step, run_end + 1):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        x = get_train_batch()
        logits = model(x[:, :-1])
        lg = logits.reshape(-1, V)
        lm = ChunkedCE.apply(lg, x[:, 1:].reshape(-1), a.z_loss)
        loss = lm * scale
        loss.backward()
        lval = float(lm.detach())
        losses.append(lval)
        del loss, lm, lg, logits, x

        grads = [p.grad for p in params if p.grad is not None]
        for grad in grads:
            grad.div_(scale)
        del grad
        grad_norm_tensor = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(grad, dtype=torch.float32)
            for grad in grads
        ]))
        grad_norm = float(grad_norm_tensor)
        bad = not math.isfinite(lval) or not math.isfinite(grad_norm)
        bad_name = "loss" if not math.isfinite(lval) else "grad_norm" if bad else None
        if use_dist:
            bad_flag = torch.tensor(int(bad), device=device)
            dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
            bad = bool(bad_flag.item())
            if bad and bad_name is None:
                bad_name = "remote_rank"
            del bad_flag
        sync_s = 0.0
        if bad:
            skips += 1
            scale = max(scale / 2.0, 1.0)
            clean_streak = 0
            diagnostic = {"step": step, "rank": a.rank, "tensor": bad_name,
                           "loss": lval, "scale": scale, "cursor": cursor,
                           "optimizer_step_count": opt.step_count,
                           "op_calls": get_op_call_counts()}
            diagnostic_path = os.path.join(
                a.logdir, "nonfinite_rank%d_step%d.json" % (a.rank, step))
            try:
                with open(diagnostic_path + ".tmp", "w") as dg:
                    json.dump(diagnostic, dg, default=str)
                os.replace(diagnostic_path + ".tmp", diagnostic_path)
            except OSError as exc:
                buf.append(json.dumps({"event": "diagnostic_write_fail", "step": step,
                                       "rank": a.rank, "err": str(exc)}) + "\n")
                diagnostic_path = None
            opt.zero_grad(set_to_none=True)
        else:
            clip_coef = (1.0 / (grad_norm_tensor + 1e-6)).clamp(max=1.0)
            for grad in grads:
                grad.mul_(clip_coef)
            del grad, clip_coef
            cur_lr = lr_at(step)
            opt.set_lr(cur_lr)
            if hasattr(opt, "set_muon_lr"):
                opt.set_muon_lr(a.muon_lr * (cur_lr / a.lr) if a.lr > 0 else a.muon_lr)
            opt.step()
            if use_dist and step % a.k == 0:
                if use_cuda:
                    torch.cuda.synchronize()
                sync_start = time.perf_counter()
                with torch.no_grad():
                    if a.sync_bucket_mb > 0:
                        average_masters(opt._params_info, a.world, a.sync_bucket_mb)
                    else:
                        for e in opt._params_info:
                            dist.all_reduce(e["master"], op=dist.ReduceOp.SUM)
                            e["master"].div_(a.world)
                            e["output"].copy_(e["master"])
                            e["parameter"].copy_(e["output"])
                dist.barrier()
                if use_cuda:
                    torch.cuda.synchronize()
                sync_s = time.perf_counter() - sync_start
                last_sync_step = step
            tokens_done += B * S
            clean_streak += 1
            if clean_streak >= 64 and scale < a.scale_max:
                scale = min(scale * 2.0, a.scale_max)
                clean_streak = 0
        del grads, grad_norm_tensor
        if use_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        record = {"step": step, "loss": round(lval, 4),
                  "lm": round(lval, 4), "scale": scale,
                  "step_s": round(dt, 4), "ppl": round(math.exp(min(lval, 20)), 4),
                  "grad_norm": round(grad_norm, 4), "tok_s": round((0 if bad else B*S)/dt),
                  "sync_s": round(sync_s, 6),
                  "memory_mb": round(torch.cuda.max_memory_allocated()/1048576, 1)
                               if use_cuda else 0,
                  "optimizer_step_count": opt.step_count, "op_calls": get_op_call_counts()}
        if bad:
            record.update(event="skip", tensor=bad_name, diagnostic=diagnostic_path)
        buf.append(json.dumps(record) + "\n")
        logf.write("".join(buf)); logf.flush(); buf = []
        if a.heartbeat_every > 0 and step % a.heartbeat_every == 0:
            heartbeat = os.path.join(a.logdir, "rank%d_%s.heartbeat.json" % (a.rank, a.tag))
            with open(heartbeat + ".tmp", "w") as heartbeat_file:
                json.dump({"rank": a.rank, "step": step, "cursor": cursor,
                           "optimizer_step_count": opt.step_count,
                           "timestamp": time.time()}, heartbeat_file)
                heartbeat_file.flush()
                os.fsync(heartbeat_file.fileno())
            os.replace(heartbeat + ".tmp", heartbeat)

        if not bad and a.eval_every > 0 and step % a.eval_every == 0:
            was_training = model.training
            model.eval()
            etot = 0.0; en = 0
            with torch.no_grad():
                for j in range(4):
                    xe = get_eval_batch(j)
                    le = model(xe[:, :-1])
                    etot += ChunkedCE.apply(le.reshape(-1, V),
                                            xe[:, 1:].reshape(-1), 0.0).item()
                    en += 1
                    del xe, le
            model.train(was_training)
            ev = etot / en
            eval_hist.append({"step": step, "eval_loss": round(ev, 4),
                              "eval_protocol": eval_protocol})
            buf.append(json.dumps(eval_hist[-1]) + "\n")

        if step % 32 == 0:
            logf.write("".join(buf)); logf.flush(); buf = []

        last_step = step
        stop_reason = ("steps_complete" if step >= a.steps else
                       "pilot_complete" if step == run_end else "periodic")
        save_now = (a.save_every > 0 and step % a.save_every == 0) or step == run_end
        if save_now:
            save_checkpoint(step, stop_reason)
        if (save_now or (a.eval_every > 0 and step % a.eval_every == 0)
                or (a.heartbeat_every > 0 and step % a.heartbeat_every == 0)):
            if use_cuda:
                torch.cuda.synchronize()
            wall_now = time.perf_counter()
            window_s = wall_now - wall_window_start
            window_tokens = tokens_done - wall_window_tokens
            buf.append(json.dumps({"event": "wall_throughput", "step": step,
                                   "wall_window_s": round(window_s, 6),
                                   "wall_tokens": window_tokens,
                                   "wall_tok_s_rank": round(window_tokens / window_s),
                                   "includes_eval_save": True}) + "\n")
            logf.write("".join(buf)); logf.flush(); buf = []
            wall_window_start = wall_now
            wall_window_tokens = tokens_done

    logf.write("".join(buf))
    wall = wall_prev + time.perf_counter() - t_start
    summary = {"rank": a.rank, "tag": a.tag, "steps_done": opt.step_count - updates_start,
               "optimizer_step_count": opt.step_count, "last_step": last_step,
               "run_end": run_end, "stop_reason": stop_reason,
               "status": "checkpoint_skipped" if checkpoint_skipped else "completed",
               "final_loss": round(losses[-1], 4) if losses else None,
               "avg_loss_last50": round(sum(losses[-50:]) / max(1, len(losses[-50:])), 4),
               "skips": skips - skips_start, "skips_total": skips, "scale_end": scale,
               "tokens": tokens_done, "wall_s": round(wall, 2),
               "tok_s_rank": round(tokens_done / wall)}
    print("FINAL", json.dumps(summary))
    logf.write("FINAL " + json.dumps(summary) + "\n")
    logf.close()
    if a.exit_file:
        exit_path = "%s.rank%d" % (a.exit_file, a.rank)
        with open(exit_path + ".tmp", "w") as exit_file:
            json.dump({"rank": a.rank, "exit_code": int(checkpoint_skipped), "step": last_step,
                       "stop_reason": stop_reason,
                       "optimizer_step_count": opt.step_count,
                       "timestamp": time.time()}, exit_file)
            exit_file.flush()
            os.fsync(exit_file.fileno())
        os.replace(exit_path + ".tmp", exit_path)
    if checkpoint_skipped:
        raise RuntimeError("training completed without a committed checkpoint")
    try:
        pass
    except Exception as e:
        print("logcopy failed", e)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "failed", "err": f"{type(exc).__name__}: {exc}"}),
              file=sys.stderr, flush=True)
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
