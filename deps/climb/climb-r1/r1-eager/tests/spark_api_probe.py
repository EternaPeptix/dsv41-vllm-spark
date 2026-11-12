# Read-only API probe for torch 2.13 on a Spark. Build the stdin script as: SRC = repr(open("dsv41_dtgraph.py").read()) + this file;
# run: ssh spark-one 'CUDA_VISIBLE_DEVICES= nice -n 19 ~/venvs/vllm032/bin/python -B -' < full.py  (no files written, no GPU)
import os, sys, types, inspect
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
import torch
print("torch", torch.__version__)
G = torch.cuda.CUDAGraph
print("CUDAGraph.__new__", inspect.signature(G.__new__))
print("capture_begin", inspect.signature(G.capture_begin))
print("has", [n for n in ("instantiate", "raw_cuda_graph", "raw_cuda_graph_exec", "replay", "capture_end") if hasattr(G, n)])
print("graph_pool_handle", hasattr(torch.cuda, "graph_pool_handle"))
print("Event", [n for n in ("record", "wait", "synchronize", "query") if hasattr(torch.cuda.Event, n)],
      "Stream", [n for n in ("wait_stream", "wait_event", "record_event", "synchronize", "query") if hasattr(torch.cuda.Stream, n)])
mod = types.ModuleType("dsv41_dtgraph"); mod.__file__ = "dsv41_dtgraph.py"
sys.modules["dsv41_dtgraph"] = mod
exec(compile(SRC, "dsv41_dtgraph.py", "exec"), mod.__dict__)
D = mod; D.PIN_ARENA = False
META = lambda t: getattr(t, "device", None) is not None and t.device.type == "meta"
class B:
    def __init__(s): s.log = []
    def stream(s): return s.st
    def wait_slot(s, k): return False
    def begin(s): s.log.append("begin"); return "g"
    def end_and_launch(s, g, seg, slot, st): s.log.append("launch")
    def fence(s, slot, st): s.log.append("fence")
b = B(); b.st = types.SimpleNamespace(cuda_stream=16)
r = D.Region(b, arena=D.Arena(pin=False), is_dev=META)
seen = []
orig = r.dispatch
def spy(func, args, kwargs):
    seen.append((str(func), D.classify(func, args, kwargs, META, r.arena.owns)))
    return orig(func, args, kwargs)
r.dispatch = spy
r.begin(b.st)
out = torch.empty(5, dtype=torch.int64, device="meta")
with D._Mode.make(r):
    data = torch.tensor([3, 1, 4, 1, 5], dtype=torch.int64)
    t = torch.empty(data.shape, dtype=data.dtype, device="cpu", pin_memory=True).copy_(data)
    out.copy_(t, non_blocking=True)
    t2 = torch.from_numpy(__import__("numpy").array([1, 2, 3], dtype="int32"))
    out2 = t2.to(device="meta", non_blocking=True)
    x = torch.from_numpy(__import__("numpy").arange(4, dtype="int32"))
    y = torch.empty(4, dtype=torch.int32, device="meta").copy_(x, non_blocking=True)
r.end()
for s in seen: print("  ", s)
print("stats", dict(r.stats), "log", b.log, "t in arena", r.arena.owns(t), "t2 in arena", r.arena.owns(t2), "out2", out2.device)
print("cuda initialized:", torch.cuda.is_initialized())
from cuda.bindings import runtime as rt
print("cudaGraphExecUpdate doc:", (rt.cudaGraphExecUpdate.__doc__ or "").strip().splitlines()[0:3])
ret = [l for l in (rt.cudaGraphExecUpdate.__doc__ or "").splitlines() if "Returns" in l or "cudaGraphExecUpdateResultInfo" in l][:4]
print("  ", ret)
print("ctor", rt.cudaGraphExec_t(init_value=0), rt.cudaGraph_t(init_value=0), rt.cudaStream_t(init_value=0))
from triton import knobs
print("triton hooks", knobs.runtime.jit_cache_hook, type(knobs.runtime.kernel_load_start_hook).__name__)
print("cuda initialized:", torch.cuda.is_initialized())
