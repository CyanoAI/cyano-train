import argparse
import ctypes
import glob
import math
import mmap
import os
import struct
import time

import torch
import torch.nn.functional as F

KERN_DIR = os.environ.get("CYANO_KERNEL_DIR", "")
_adamw_lib = None


def _load_adamw_kernel():
    global _adamw_lib
    if _adamw_lib is None:
        if not KERN_DIR:
            raise RuntimeError("Set CYANO_KERNEL_DIR to the directory containing fused_adamw.so")
        _adamw_lib = ctypes.CDLL(f"{KERN_DIR}/fused_adamw.so")
        _adamw_lib.launch_fused_adamw.restype = None
        _adamw_lib.launch_fused_adamw.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_int64, ctypes.c_void_p,
        ]
    return _adamw_lib


class FusedAdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self._params_info = []
        for parameter in params:
            if parameter.requires_grad:
                self._params_info.append({
                    "parameter": parameter,
                    "master": parameter.detach().float().clone(),
                    "m": torch.zeros_like(parameter, dtype=torch.float32),
                    "v": torch.zeros_like(parameter, dtype=torch.float32),
                    "output": torch.empty_like(parameter),
                })

    @property
    def param_groups(self):
        return [{"params": [entry["parameter"] for entry in self._params_info],
                 "lr": self.lr}]

    def set_lr(self, lr):
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for entry in self._params_info:
            parameter = entry["parameter"]
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def unscale_and_check(self, scale):
        inverse_scale = 1.0 / float(scale)
        for index, entry in enumerate(self._params_info):
            parameter = entry["parameter"]
            if parameter.grad is None:
                continue
            parameter.grad.mul_(inverse_scale)
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(
                    f"non-finite gradient at optimizer entry {index}, "
                    f"shape={tuple(parameter.shape)}"
                )

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        adamw_lib = _load_adamw_kernel()
        for index, entry in enumerate(self._params_info):
            parameter = entry["parameter"]
            if parameter.grad is None:
                continue
            adamw_lib.launch_fused_adamw(
                entry["master"].data_ptr(), parameter.grad.data_ptr(),
                entry["m"].data_ptr(), entry["v"].data_ptr(),
                entry["output"].data_ptr(), parameter.numel(),
                self.lr, self.beta1, self.beta2, self.eps,
                self.weight_decay, self.step_count, stream,
            )
            parameter.copy_(entry["output"])
            for state_name in ("master", "m", "v"):
                if not torch.isfinite(entry[state_name]).all().item():
                    raise FloatingPointError(
                        f"non-finite optimizer state {state_name} at entry "
                        f"{index}, shape={tuple(parameter.shape)}"
                    )

    def assert_finite(self):
        for index, entry in enumerate(self._params_info):
            parameter = entry["parameter"]
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(
                    f"non-finite parameter at optimizer entry {index}, "
                    f"shape={tuple(parameter.shape)}"
                )
            for state_name in ("master", "m", "v"):
                if not torch.isfinite(entry[state_name]).all().item():
                    raise FloatingPointError(
                        f"non-finite {state_name} at optimizer entry {index}, "
                        f"shape={tuple(parameter.shape)}"
                    )


class TrainBinLoader:
    def __init__(self, data_path, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_path = data_path
        self.fd = open(data_path, "rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.n_tokens = len(self.mm) // 4
        required = batch_size * seq_len
        if self.n_tokens < required:
            raise RuntimeError(
                f"train.bin contains {self.n_tokens} tokens, but {required} are required"
            )

    def batch(self):
        total = self.batch_size * self.seq_len
        max_start = self.n_tokens - total
        start = torch.randint(0, max_start + 1, (1,)).item()
        buffer = self.mm[start * 4:(start + total) * 4]
        values = struct.unpack(f"{total}I", buffer)
        return torch.tensor(values, dtype=torch.long).view(
            self.batch_size, self.seq_len
        )


def learning_rate(step, max_steps, warmup=200, peak_lr=3e-4):
    if step < warmup:
        return peak_lr * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    from model_champion import (
        CONFIGS,
        Transformer,
        get_op_call_counts,
        reset_op_call_counts,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="small", choices=sorted(CONFIGS))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--data",
        default=os.environ.get("CYANO_TRAIN_DATA", "./train.bin"),
    )
    args = parser.parse_args()

    if not args.data.endswith("train.bin"):
        raise RuntimeError("The training test must use the real train.bin dataset")

    torch.manual_seed(1234)
    config = CONFIGS[args.config]
    config.max_seq_len = args.seq_len
    # The repaired FP32-master FusedAdamW kernel consumes FP16 model weights,
    # gradients, and output storage while retaining FP32 master/m/v states.
    # Keeping model parameters in FP32 causes the HIP kernel to reinterpret
    # FP32 gradients and output buffers as half storage, corrupting the master
    # weights on the first update.
    model = Transformer(config).cuda().half().train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = FusedAdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    loader = TrainBinLoader(args.data, args.batch_size, args.seq_len)
    tokens_per_step = args.batch_size * (args.seq_len - 1)
    peak_flops = 21.6e12

    print(
        f"config={args.config} params={parameter_count} seq_len={args.seq_len} "
        f"batch_size={args.batch_size} steps={args.steps} data={args.data}"
    )

    reset_op_call_counts()
    optimizer.assert_finite()

    for step in range(1, args.steps + 1):
        lr = learning_rate(step, args.steps)
        optimizer.set_lr(lr)

        batch = loader.batch().cuda(non_blocking=False)
        torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].contiguous().reshape(-1),
            )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"non-finite loss at step {step}: {loss.detach().float().item()}"
            )

        scaler.scale(loss).backward()
        scale_before = float(scaler.get_scale())
        optimizer.unscale_and_check(scale_before)

        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError(
                    f"non-finite parameter before optimizer step {step}: "
                    f"name={name}, shape={tuple(parameter.shape)}"
                )
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(
                    f"non-finite gradient at step {step}: "
                    f"name={name}, shape={tuple(parameter.shape)}"
                )

        optimizer.step()
        optimizer.assert_finite()
        scaler.update(new_scale=scale_before)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens_per_second = tokens_per_step / elapsed
        mfu = min(1.0, 6.0 * parameter_count * tokens_per_second / peak_flops)
        allocated_mb = torch.cuda.memory_allocated() / 1e6
        reserved_mb = torch.cuda.memory_reserved() / 1e6
        call_counts = get_op_call_counts()

        print(
            f"step {step:4d} | loss {loss.item():.6f} | lr {lr:.8e} | "
            f"tokens/s {tokens_per_second:.3f} | MFU {mfu * 100.0:.4f}% | "
            f"allocated {allocated_mb:.1f}MB | reserved {reserved_mb:.1f}MB | "
            f"scale {scaler.get_scale():.1f} | step_ms {elapsed * 1000.0:.3f} | "
            f"op_calls {call_counts}",
            flush=True,
        )

    required_calls = (
        "rmsnorm_residual_forward",
        "rmsnorm_residual_backward",
        "rope_qk_forward",
        "rope_qk_backward",
        "swiglu_forward",
        "swiglu_backward",
    )
    final_counts = get_op_call_counts()
    missing = [name for name in required_calls if final_counts.get(name, 0) == 0]
    if missing:
        raise RuntimeError(
            f"compound operator integration evidence missing calls: {missing}; "
            f"observed={final_counts}"
        )
    print(f"final_op_call_counts={final_counts}", flush=True)


if __name__ == "__main__":
    main()
