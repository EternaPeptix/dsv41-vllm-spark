#!/usr/bin/env python3
"""g130 fc1: route recorder on a stopped GPU, before it is trusted in an instance. The Triton kernel has never been compiled.
Records known layouts, eagerly and through a CUDA graph with the layout rewritten between replays, then checks the words read back.
PASS = every record decodes to the layout written. FAIL = do not launch an instance with the recorder.
Usage: source ~/venvs/vllm032/bin/activate; python tests/test_route_rec_gpu.py"""
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deallib  # noqa: E402

torch = deallib.start()
spec = importlib.util.spec_from_file_location("dsv41_fc1_route_rec", deallib.KIT / "site/fc1_route_rec.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
rec = mod.RouteRec()
gen = torch.Generator().manual_seed(11)
fails = 0
for tokens in (1, 2, 4, 6, 8, 24):
    M_sum = deallib.m_sum(tokens)
    nmb = M_sum // 32
    lay = torch.full((M_sum,), -1, dtype=torch.int32, device="cuda")
    d = torch.empty((M_sum, 576), dtype=torch.bfloat16, device="cuda")
    bw = torch.empty((384, 1), dtype=torch.uint8, device="cuda")
    args = ((None, None), (bw, None), d, lay)

    def body():
        rec.observe(args)
    gr = deallib.graph_of(torch, body)
    for r in range(40):
        G = r % (nmb + 1)
        want = deallib.make_layout(torch, M_sum, G, gen, rows=("few", "one", "any", "full")[r % 4])
        lay.copy_(want.cuda())
        torch.cuda.synchronize()
        n0 = int(rec.cursor[0])
        gr.replay()
        torch.cuda.synchronize()
        assert int(rec.cursor[0]) == n0 + 1
        got = rec.ring[(n0 % mod.CAP) * mod.REC:][:mod.REC].cpu()
        ok = int(got[0]) == nmb and int(got[1]) == min(G, (mod.REC - 2) // 2)
        for m in range(min(nmb, (mod.REC - 2) // 2)):
            e = int(want[m * 32])
            c = int((want[m * 32:(m + 1) * 32] >= 0).sum())
            ok = ok and int(got[2 + 2 * m]) == e and int(got[3 + 2 * m]) == c
        if not ok:
            fails += 1
            print("BAD tokens", tokens, "replay", r, "G", G, got[:8].tolist(), flush=True)
    del gr
# non-FC1 geometry must not launch anything
n0 = int(rec.cursor[0])
rec.observe(((None, None), (torch.empty((384, 1), device="cuda"), None), torch.empty((32, 5120), device="cuda"), torch.empty((32,), device="cuda")))
torch.cuda.synchronize()
assert int(rec.cursor[0]) == n0
print("ROUTE REC", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
