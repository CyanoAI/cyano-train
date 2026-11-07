import os, sys, time, torch, ctypes
sys.path.insert(0, '.')
import cyano_model as sm
if not sm._OPS_DIR:
    raise RuntimeError("Set CYANO_OPS_DIR before running fused-kernel verification")
print('v2 loaded:', sm._V2 is not None)
g = torch.randn(64, 512, device='cuda').half()
u = torch.randn(64, 512, device='cuda').half()
go = torch.randn(64, 512, device='cuda').half()
gg = torch.empty_like(g)
gu = torch.empty_like(u)
sm._sw_bwd_l(sm._ptr(go), sm._ptr(g), sm._ptr(u), sm._ptr(gg), sm._ptr(gu),
             g.numel(), sm._stream_ptr())
torch.cuda.synchronize()
print('sw bwd ok', gg.abs().mean().item())

# End-to-end tiny model smoke test through the module classes.
cfg = sm.CONFIGS['dense30m']
m = sm.Transformer(cfg).cuda().half().train()
x = torch.randint(0, cfg.vocab_size, (4, 64), device='cuda')
logits = m(x)
loss = logits.float().mean()
loss.backward()
torch.cuda.synchronize()
print('model fwd+bwd loss', float(loss))
print('ops', sm.get_op_call_counts())

# Throughput at training-scale shapes with v2 ON (measured separately):
# Batch8, seq1024 softmax rms path: rows=8192, hidden=768
rows, hidden = 8192, 768
xB = (torch.randn(rows, hidden, device='cuda') * 0.5).half()
w = torch.rand(hidden, device='cuda').half() * 2
out = torch.empty_like(xB)
inv = torch.empty(rows, device='cuda', dtype=torch.float32)

def t(fn, iters=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

lib1 = ctypes.CDLL(os.path.join(sm._OPS_DIR, 'rmsnorm_residual.so'))
lib1.launch_rmsnorm_residual_forward.argtypes = [
    ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
lib1.launch_rmsnorm_residual_forward.restype = None
lib1.launch_rmsnorm_residual_backward.argtypes = [
    ctypes.c_void_p]*6 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib1.launch_rmsnorm_residual_backward.restype = None
t1 = t(lambda: lib1.launch_rmsnorm_residual_forward(sm._ptr(xB), sm._ptr(w), sm._ptr(out), sm._ptr(inv), ctypes.c_int(rows), ctypes.c_int(hidden), ctypes.c_float(1e-6), sm._stream_ptr()))
t2 = t(lambda: sm._rms_fwd_l(sm._ptr(xB), sm._ptr(w), sm._ptr(out), sm._ptr(inv), ctypes.c_int(rows), ctypes.c_int(hidden), ctypes.c_float(1e-6), sm._stream_ptr()))
print('rms fwd: v1 %.1fus  v2 %.1fus  speedup %.2fx' % (t1, t2, t1 / t2))

gx = torch.empty_like(xB)
gw = torch.zeros(hidden, device='cuda', dtype=torch.float32)
t1b = t(lambda: lib1.launch_rmsnorm_residual_backward(sm._ptr(go.view(rows, hidden) if go.numel() == rows * hidden else go.repeat(rows, 6).view(-1, hidden)[:rows]), sm._ptr(xB), sm._ptr(w), sm._ptr(inv), sm._ptr(gx), sm._ptr(gw), rows, hidden, sm._stream_ptr()), 100) if False else 0.0
# SwiGLU at training shape: n = 8192*3072 = 25165824 (must be multiple of 8: yes)
n = 8192 * 3072
ga = (torch.randn(n, device='cuda') * 0.5).half()
ub = (torch.randn(n, device='cuda') * 0.5).half()
dout = (torch.randn(n, device='cuda') * 0.3).half()
oB = torch.empty_like(ga)
lib_sw = ctypes.CDLL(os.path.join(sm._OPS_DIR, 'swiglu.so'))
lib_sw.launch_swiglu_forward.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64, ctypes.c_void_p]
lib_sw.launch_swiglu_forward.restype = None
lib_sw.launch_swiglu_backward.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64, ctypes.c_void_p]
lib_sw.launch_swiglu_backward.restype = None
t1 = t(lambda: lib_sw.launch_swiglu_forward(sm._ptr(ga), sm._ptr(ub), sm._ptr(oB), ctypes.c_int64(n), sm._stream_ptr()))
t2 = t(lambda: sm._sw_fwd_l(sm._ptr(ga), sm._ptr(ub), sm._ptr(oB), n, sm._stream_ptr()))
print('swiglu fwd: v1 %.1fus  v2 %.1fus  speedup %.2fx' % (t1, t2, t1 / t2))
ggB = torch.empty_like(ga); guB = torch.empty_like(ub)
t1 = t(lambda: lib_sw.launch_swiglu_backward(sm._ptr(dout), sm._ptr(ga), sm._ptr(ub), sm._ptr(ggB), sm._ptr(guB), n, sm._stream_ptr()))
t2 = t(lambda: sm._sw_bwd_l(sm._ptr(dout), sm._ptr(ga), sm._ptr(ub), sm._ptr(ggB), sm._ptr(guB), n, sm._stream_ptr()))
print('swiglu bwd: v1 %.1fus  v2 %.1fus  speedup %.2fx' % (t1, t2, t1 / t2))
print('OK')
