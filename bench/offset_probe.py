"""Probe: does DeepGEMM SM12x fp8_fp4_paged_mqa_logits honour a kv_cache view's
storage offset when it reads the per-token FP32 scales (PR #1788 bug class)?
Also checks vllm cp_gather_indexer_k_quant_cache (prefill gather) with an offset view.
Tiny GPU work; does not touch the running service."""
import sys, torch
from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits, get_paged_mqa_logits_metadata
from vllm import _custom_ops as ops

torch.manual_seed(0)
dev = "cuda"
H = int(sys.argv[1]) if len(sys.argv) > 1 else 64
D, PAGE, NB = 128, 64, 64
ROW = D + 4
page_bytes = PAGE * ROW


def make_pages():
    k = (torch.randn(NB, PAGE, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    s = torch.rand(NB, PAGE, device=dev, dtype=torch.float32) + 0.5
    raw = torch.empty(NB, page_bytes, dtype=torch.uint8, device=dev)
    raw[:, : PAGE * D] = k.view(torch.uint8).reshape(NB, -1)
    raw[:, PAGE * D:] = s.view(torch.uint8).reshape(NB, -1)
    return k, s, raw


k, s, raw = make_pages()
# base: offset 0
base = raw.clone().view(NB, PAGE, ROW)
# offset view inside shared storage, decoy pages before it with different scales
OFF_PAGES = 7
big = torch.empty((NB + OFF_PAGES) * page_bytes, dtype=torch.uint8, device=dev)
decoy_k, decoy_s, decoy_raw = make_pages()
big[: OFF_PAGES * page_bytes] = decoy_raw.reshape(-1)[: OFF_PAGES * page_bytes]
big[OFF_PAGES * page_bytes:] = raw.reshape(-1)
offv = big[OFF_PAGES * page_bytes:].view(NB, PAGE, ROW)
assert offv.storage_offset() == OFF_PAGES * page_bytes
# also overwrite scale region before view so a missing offset reads decoy scales
results = {}
B = 3
ctx = torch.tensor([[1000], [2500], [4000]], dtype=torch.int32, device=dev)
max_len = 4096
bt = torch.stack([torch.randperm(NB, device=dev)[: max_len // PAGE] for _ in range(B)]).to(torch.int32)
q = (torch.randn(B, 1, H, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
w = torch.rand(B, H, device=dev, dtype=torch.float32)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
meta = get_paged_mqa_logits_metadata(ctx, PAGE, nsm)

# reference
ref = torch.full((B, max_len), float("-inf"), device=dev)
for b in range(B):
    L = int(ctx[b, 0])
    pages = bt[b, : (L + PAGE - 1) // PAGE].long()
    kk = k[pages].reshape(-1, D)[:L].float()
    ss = s[pages].reshape(-1)[:L]
    sc = torch.relu((q[b, 0].float() @ kk.T) * ss[None, :])
    ref[b, :L] = (sc * w[b][:, None]).sum(0)

for name, cache in (("base", base), ("offset_view", offv)):
    out = fp8_fp4_paged_mqa_logits((q, None), cache.unsqueeze(-2), w, ctx, bt, meta, max_len, clean_logits=False)
    torch.cuda.synchronize()
    errs = []
    for b in range(B):
        L = int(ctx[b, 0])
        e = ((out[b, :L] - ref[b, :L]).abs().max() / ref[b, :L].abs().max()).item()
        errs.append(e)
    tail_ok = all(torch.isinf(out[b, int(ctx[b, 0]):]).all().item() for b in range(B))
    print(f"paged_mqa {name}: storage_offset={cache.storage_offset()} rel_err={['%.2e' % e for e in errs]} tail_neg_inf={tail_ok}")

# varlen (indices) mode as used live with DSV41_SM121_VARLEN_INDEXER=1
ind = torch.arange(B, dtype=torch.int32, device=dev)
try:
    meta2 = get_paged_mqa_logits_metadata(ctx, PAGE, nsm, indices=ind)
    for name, cache in (("base", base), ("offset_view", offv)):
        out = fp8_fp4_paged_mqa_logits((q, None), cache.unsqueeze(-2), w, ctx, bt, meta2, max_len, clean_logits=False, indices=ind)
        torch.cuda.synchronize()
        errs = [((out[b, :int(ctx[b, 0])] - ref[b, :int(ctx[b, 0])]).abs().max() / ref[b, :int(ctx[b, 0])].abs().max()).item() for b in range(B)]
        print(f"paged_mqa varlen {name}: rel_err={['%.2e' % e for e in errs]}")
except Exception as ex:  # noqa
    print("varlen probe error:", repr(ex)[:300])

# prefill gather with offset view
try:
    for name, cache in (("base", base), ("offset_view", offv)):
        L = 2500
        kq = torch.empty(L, D, dtype=torch.float8_e4m3fn, device=dev)
        ks = torch.empty(L, 4, dtype=torch.uint8, device=dev)
        cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
        ops.cp_gather_indexer_k_quant_cache(cache, kq, ks, bt[1:2].contiguous(), cu)
        torch.cuda.synchronize()
        pages = bt[1, : (L + PAGE - 1) // PAGE].long()
        okk = torch.equal(kq.view(torch.uint8), k[pages].reshape(-1, D)[:L].view(torch.uint8))
        oks = torch.equal(ks.view(torch.float32).squeeze(-1), s[pages].reshape(-1)[:L])
        print(f"gather {name}: k_equal={okk} scale_equal={oks}")
except Exception as ex:  # noqa
    print("gather probe error:", repr(ex)[:300])
