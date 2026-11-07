import ctypes
import math
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Optional fused kernels. Set CYANO_OPS_DIR to a directory holding
# rmsnorm_residual.so / rope_qk_layout.so / swiglu.so (and optionally
# CYANO_OPS_SO for the vectorized build) to use them. Without these the model
# runs entirely on native PyTorch ops, which is the portable default.
_OPS_DIR = os.environ.get("CYANO_OPS_DIR", "")
_OPS_SO = os.environ.get("CYANO_OPS_SO", "")
_MISSING = object()

def _try_load_v2():
    p = _OPS_SO
    if p and os.path.exists(p):
        lib = ctypes.CDLL(p)
        fns = {}
        for n, at in [("rmsnorm_residual_forward_v2",
                        [ctypes.c_void_p]*4 + [ctypes.c_int]*2 +
                        [ctypes.c_float, ctypes.c_void_p]),
                      ("rmsnorm_residual_backward_v2",
                        [ctypes.c_void_p]*6 + [ctypes.c_int]*2 +
                        [ctypes.c_void_p]),
                      ("rope_qk_forward_v2",
                        [ctypes.c_void_p]*6 + [ctypes.c_int]*4 +
                        [ctypes.c_void_p]),
                      ("rope_qk_backward_v2",
                        [ctypes.c_void_p]*6 + [ctypes.c_int]*4 +
                        [ctypes.c_void_p]),
                      ("swiglu_forward_v2",
                        [ctypes.c_void_p]*3 + [ctypes.c_int64,
                                               ctypes.c_void_p]),
                      ("swiglu_backward_v2",
                        [ctypes.c_void_p]*5 + [ctypes.c_int64,
                                               ctypes.c_void_p])]:
            try:
                f = getattr(lib, "launch_" + n)
                f.restype = None
                f.argtypes = at
                fns[n] = f
            except AttributeError:
                return None
        return fns
    return None

_V2 = _try_load_v2()
OP_CALL_COUNTS = Counter()
DEBUG_FINITE = os.environ.get("CYANO_DEBUG_FINITE", "0") == "1"


def reset_op_call_counts():
    OP_CALL_COUNTS.clear()


def get_op_call_counts():
    return dict(OP_CALL_COUNTS)


def _assert_finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        finite = tensor.detach().float()[torch.isfinite(tensor)]
        finite_min = finite.min().item() if finite.numel() else float("nan")
        finite_max = finite.max().item() if finite.numel() else float("nan")
        raise FloatingPointError(
            f"non-finite compound-op tensor: {name}; shape={tuple(tensor.shape)}; "
            f"dtype={tensor.dtype}; contiguous={tensor.is_contiguous()}; "
            f"finite_min={finite_min}; finite_max={finite_max}"
        )


def _ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _bind_kernels():
    _rms_lib = ctypes.CDLL(os.path.join(_OPS_DIR, "rmsnorm_residual.so"))
    _rope_lib = ctypes.CDLL(os.path.join(_OPS_DIR, "rope_qk_layout.so"))
    _swiglu_lib = ctypes.CDLL(os.path.join(_OPS_DIR, "swiglu.so"))

    _rms_lib.launch_rmsnorm_residual_forward.restype = None
    _rms_lib.launch_rmsnorm_residual_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_void_p,
    ]
    _rms_lib.launch_rmsnorm_residual_backward.restype = None
    _rms_lib.launch_rmsnorm_residual_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _rope_lib.launch_rope_qk_forward.restype = None
    _rope_lib.launch_rope_qk_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _rope_lib.launch_rope_qk_backward.restype = None
    _rope_lib.launch_rope_qk_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    _swiglu_lib.launch_swiglu_forward.restype = None
    _swiglu_lib.launch_swiglu_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p,
    ]
    _swiglu_lib.launch_swiglu_backward.restype = None
    _swiglu_lib.launch_swiglu_backward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ]

    if _V2 is not None:
        fwd = _V2["rmsnorm_residual_forward_v2"]
        fwd.argtypes = _rms_lib.launch_rmsnorm_residual_forward.argtypes
        bwd = _V2["rmsnorm_residual_backward_v2"]
        bwd.argtypes = _rms_lib.launch_rmsnorm_residual_backward.argtypes
        rfwd = _V2["rope_qk_forward_v2"]
        rfwd.argtypes = _rope_lib.launch_rope_qk_forward.argtypes
        rbwd = _V2["rope_qk_backward_v2"]
        rbwd.argtypes = _rope_lib.launch_rope_qk_backward.argtypes
        sfwd = _V2["swiglu_forward_v2"]
        sfwd.argtypes = _swiglu_lib.launch_swiglu_forward.argtypes
        sbwd = _V2["swiglu_backward_v2"]
        sbwd.argtypes = _swiglu_lib.launch_swiglu_backward.argtypes
        return fwd, bwd, rfwd, rbwd, sfwd, sbwd
    return (_rms_lib.launch_rmsnorm_residual_forward,
            _rms_lib.launch_rmsnorm_residual_backward,
            _rope_lib.launch_rope_qk_forward,
            _rope_lib.launch_rope_qk_backward,
            _swiglu_lib.launch_swiglu_forward,
            _swiglu_lib.launch_swiglu_backward)


# Without these libraries the autograd Functions below are simply never used;
# the modules all run native PyTorch ops.
if _OPS_DIR and os.path.isdir(_OPS_DIR):
    (_rms_fwd_l, _rms_bwd_l, _rope_fwd_l, _rope_bwd_l,
     _sw_fwd_l, _sw_bwd_l) = _bind_kernels()
else:
    _rms_fwd_l = _rms_bwd_l = _rope_fwd_l = _rope_bwd_l = None
    _sw_fwd_l = _sw_bwd_l = None


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        if _rms_fwd_l is None:
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return F.rms_norm(x, (x.shape[-1],), weight, eps)
        x = x.contiguous()
        # The HIP kernel consumes FP16 values. Under autocast, x is
        # FP16 while the module parameter remains FP32; passing the FP32
        # storage directly made the kernel reinterpret float bytes as halves
        # and produced NaNs in the first integrated training step.
        original_weight_dtype = weight.dtype
        weight_kernel = weight.to(dtype=x.dtype).contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        output = torch.empty_like(x)
        inv_rms = torch.empty(rows, device=x.device, dtype=torch.float32)
        _rms_fwd_l(
            _ptr(x), _ptr(weight_kernel), _ptr(output),
            _ptr(inv_rms), rows, hidden, eps, _stream_ptr())
        if DEBUG_FINITE:
            _assert_finite("rmsnorm_residual.output", output)
            _assert_finite("rmsnorm_residual.inv_rms", inv_rms)
        OP_CALL_COUNTS["rmsnorm_residual_forward"] += 1
        ctx.save_for_backward(x, weight_kernel, inv_rms)
        ctx.input_shape = x.shape
        ctx.original_weight_dtype = original_weight_dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if _rms_bwd_l is None:
            x, weight = ctx.saved_tensors
            with torch.enable_grad():
                x_ = x.detach().requires_grad_(True)
                weight_ = weight.detach().requires_grad_(True)
                output = F.rms_norm(x_, (x_.shape[-1],), weight_, ctx.eps)
                grad_x, grad_weight = torch.autograd.grad(
                    output, (x_, weight_), grad_output)
            return grad_x, grad_weight, None
        x, weight_kernel, inv_rms = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        grad_x = torch.empty_like(x)
        grad_weight = torch.zeros(
            hidden, device=x.device, dtype=torch.float32)
        _rms_bwd_l(
            _ptr(grad_output), _ptr(x), _ptr(weight_kernel),
            _ptr(inv_rms), _ptr(grad_x),
            _ptr(grad_weight), rows, hidden, _stream_ptr())
        OP_CALL_COUNTS["rmsnorm_residual_backward"] += 1
        return (
            grad_x,
            grad_weight.to(ctx.original_weight_dtype),
            None,
        )


class RoPEQKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos_table, sin_table):
        if _rope_fwd_l is None:
            batch, sequence, heads, head_dim = q.shape
            cos_t = cos_table[:sequence].to(q.dtype).view(1, sequence, 1, -1)
            sin_t = sin_table[:sequence].to(q.dtype).view(1, sequence, 1, -1)
            q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
            k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
            ctx.save_for_backward(cos_table, sin_table)
            ctx.shape = (batch, sequence, heads, head_dim)
            q_rot = torch.cat((q1 * cos_t - q2 * sin_t,
                               q2 * cos_t + q1 * sin_t), dim=-1)
            k_rot = torch.cat((k1 * cos_t - k2 * sin_t,
                               k2 * cos_t + k1 * sin_t), dim=-1)
            return q_rot.transpose(1, 2), k_rot.transpose(1, 2)
        q = q.contiguous()
        k = k.contiguous()
        cos_table = cos_table.contiguous().float()
        sin_table = sin_table.contiguous().float()
        batch, sequence, heads, head_dim = q.shape
        q_out = torch.empty(
            batch, heads, sequence, head_dim, device=q.device, dtype=q.dtype)
        k_out = torch.empty_like(q_out)
        _rope_fwd_l(
            _ptr(q), _ptr(k), _ptr(cos_table), _ptr(sin_table),
            _ptr(q_out), _ptr(k_out), batch, sequence, heads, head_dim,
            _stream_ptr())
        if DEBUG_FINITE:
            _assert_finite("rope_qk.q_out", q_out)
            _assert_finite("rope_qk.k_out", k_out)
        OP_CALL_COUNTS["rope_qk_forward"] += 1
        ctx.save_for_backward(cos_table, sin_table)
        ctx.shape = (batch, sequence, heads, head_dim)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out):
        if _rope_bwd_l is None:
            cos_table, sin_table = ctx.saved_tensors
            _, sequence, _, head_dim = ctx.shape
            cos_t = cos_table[:sequence].to(grad_q_out.dtype).view(1, sequence, 1, -1)
            sin_t = sin_table[:sequence].to(grad_q_out.dtype).view(1, sequence, 1, -1)

            def inverse(grad):
                grad = grad.transpose(1, 2)
                grad1, grad2 = grad[..., :head_dim // 2], grad[..., head_dim // 2:]
                return torch.cat((grad1 * cos_t + grad2 * sin_t,
                                  -grad1 * sin_t + grad2 * cos_t), dim=-1)

            return inverse(grad_q_out), inverse(grad_k_out), None, None
        cos_table, sin_table = ctx.saved_tensors
        batch, sequence, heads, head_dim = ctx.shape
        grad_q_out = grad_q_out.contiguous()
        grad_k_out = grad_k_out.contiguous()
        grad_q = torch.empty(
            ctx.shape, device=grad_q_out.device, dtype=grad_q_out.dtype)
        grad_k = torch.empty_like(grad_q)
        _rope_bwd_l(
            _ptr(grad_q_out), _ptr(grad_k_out), _ptr(cos_table),
            _ptr(sin_table), _ptr(grad_q), _ptr(grad_k), batch, sequence,
            heads, head_dim, _stream_ptr())
        OP_CALL_COUNTS["rope_qk_backward"] += 1
        return grad_q, grad_k, None, None


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if _sw_fwd_l is None:
            ctx.save_for_backward(gate, up)
            return F.silu(gate) * up
        gate = gate.contiguous()
        up = up.contiguous()
        output = torch.empty_like(gate)
        _sw_fwd_l(
            _ptr(gate), _ptr(up), _ptr(output), gate.numel(), _stream_ptr())
        if DEBUG_FINITE:
            _assert_finite("swiglu.output", output)
        OP_CALL_COUNTS["swiglu_forward"] += 1
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if _sw_bwd_l is None:
            gate, up = ctx.saved_tensors
            sigmoid = torch.sigmoid(gate)
            silu = gate * sigmoid
            grad_gate = grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
            grad_up = grad_output * silu
            return grad_gate, grad_up
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        _sw_bwd_l(
            _ptr(grad_output), _ptr(gate), _ptr(up), _ptr(grad_gate),
            _ptr(grad_up), gate.numel(), _stream_ptr())
        OP_CALL_COUNTS["swiglu_backward"] += 1
        return grad_gate, grad_up

@dataclass
class Config:
    vocab_size: int = 50257
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    hidden_dim: int = 1536
    max_seq_len: int = 4096

CONFIGS = {
    'smoke': Config(vocab_size=128, dim=32, n_layers=1, n_heads=4, hidden_dim=64),
    'dense30m': Config(dim=512, n_layers=8, n_heads=8, hidden_dim=2048),
    'tiny': Config(dim=384, n_layers=6, n_heads=6, hidden_dim=1536),
    'small': Config(dim=768, n_layers=12, n_heads=12, hidden_dim=3072),
    'base': Config(dim=1024, n_layers=18, n_heads=16, hidden_dim=4096),
}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, base=100000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cache', freqs.cos())
        self.register_buffer('sin_cache', freqs.sin())
        self.head_dim = head_dim

    def forward(self, x):
        B, H, T, D = x.shape
        cos_t = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)
        sin_t = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., :D//2], x[..., D//2:]
        return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)

class FusedQKV(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.dim = dim

    def forward(self, x):
        qkv = self.wqkv(x)
        return qkv.split(self.dim, dim=-1)

class FusedSwiGLUProj(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.split(self.hidden_dim, dim=-1)
        return self.w_down(SwiGLUFunction.apply(gate, up))

class Attention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=4096, chunk_size=0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = FusedQKV(dim)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len)
        self.chunk_size = chunk_size

    def _forward_chunked(self, q, k, v):
        B, H, T, D = q.shape
        out = torch.empty_like(q)
        scale = D ** -0.5
        chunk = self.chunk_size or T
        for i in range(0, T, chunk):
            end = min(i + chunk, T)
            qi = q[:, :, i:end, :]
            scores = torch.matmul(qi, k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(end - i, T, device=q.device, dtype=torch.bool),
                diagonal=i + 1)
            scores = scores.masked_fill(mask, float('-inf'))
            probs = F.softmax(scores, dim=-1)
            out[:, :, i:end, :] = torch.matmul(probs, v)
        return out

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = RoPEQKFunction.apply(
            q, k, self.rope.cos_cache[:T], self.rope.sin_cache[:T])
        if self.chunk_size and self.chunk_size < T:
            out = self._forward_chunked(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)

class Block(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, max_seq_len=4096, chunk_size=0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, max_seq_len, chunk_size)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = FusedSwiGLUProj(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(RMSNormFunction.apply(
            x, self.attn_norm.weight, self.attn_norm.eps))
        x = x + self.mlp(RMSNormFunction.apply(
            x, self.mlp_norm.weight, self.mlp_norm.eps))
        return x

class Transformer(nn.Module):
    def __init__(self, config, chunk_size=0):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList([
            Block(config.dim, config.n_heads, config.hidden_dim,
                  config.max_seq_len, chunk_size)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, idx):
        h = self.embed(idx)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.output(h)
