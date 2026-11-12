"""r6-step (climb round 6): step-start collectives (C3) and the E1 DSpark main_proj side stream (C4).

Every lever is default off. With none of the flags below set this module is never imported (the two patched
files only test a module-level bool), so the live code path is unchanged.

C3  step-start collective consolidation (exact: gathers are byte copies; the embedding table is the live
    vocab-parallel + all-reduce output, proved element by element at build time)

  DSV41_STEP_COLEMB=1
      The target embedding (eager, step start) and the DSpark embedding (first op of the drafter graph) stop
      doing vocab-parallel lookup + all-reduce (ar_kernel_pl AR, 25-32 us) and do a column-shard lookup + one
      column all-gather instead (AG, ~13-20 us). The column table [V, D/tp] is built once, eagerly, before any
      capture, FROM THE LIVE PATH: every id goes through VocabParallelEmbedding.forward (fused
      vocab_parallel_embedding kernel + rdma all-reduce) in chunks of <= 128 rows, and this rank keeps its
      D/tp columns. While building, each rank checks every row it owns against canon(W_local) (the AR's map
      bf16_rn(0.f + v): -0 -> +0, NaN -> 0x7FFF, all else unchanged), all 5120 columns, and the ranks compare
      checksums of every AR output; any mismatch on any rank -> the lever stays off on every rank.
      Engages only for 0 < T <= DSV41_STEP_COLEMB_MAX_T (default 128, capped so the stock AR would be the rdma
      path whose map was proved). Memory: + V x D/tp bf16 per rank (165 MB at 129280 x 640); the vocab-parallel
      table is kept for the stock path (T above the cap, profile shapes).

  DSV41_STEP_ENGRAM_PACK=look|all   (1 = look)
      look: the two Engram layers' row all-gathers (layer 1 and layer 14, [T,3,256] each) become ONE gather
            of [T,6,256] at the first layer's Engram.embed; both layers' lookup rows sit in one persistent
            staging buffer so the packed operand needs no copy; one copy kernel splits the gathered heads.
            The second layer's embed returns the stored rows (no kernel, no collective).
      all:  look, plus the two wkv column-parallel all-gathers ([T,3200] each) become ONE gather of [T,6400]
            at layer 1: layer 14's wkv GEMM (same input bytes, shape, weights, tactic) runs right after
            layer 1's on the main stream, one cat, one gather, one split copy. Layer 14 takes the stored kv.
            Refused (falls back to look) while r5-prio's DSV41_PRIO_ENGRAM_HOIST is on: the hoist owns
            both GEMMs then. If both help, keep one (README section "C3 vs r5-prio hoist").
      Decode-sized forwards only (0 < T <= DSV41_STEP_ENGRAM_PACK_MAX_T, default 128), never under a
      breakable (piecewise) capture, never under SP / EngramBatch / tp 1 / Engram DP > 1.

C4  DSV41_STEP_C4=1   E1: DSpark main_proj local GEMM on a side stream under refine / sampling
      On a CERT-armed step the r4 column-split main_proj (cat -> MXFP8 quant -> CUTLASS GEMM) is launched on a
      side stream right after the screen kernel (it reads only the target graph's aux taps, which nothing
      writes until the next target replay). combine_hidden_states (propose) then waits on the GEMM's event
      and runs the main_proj all-gather + main_norm on the main stream exactly where the stock path does, so
      the collective order is unchanged. No record_stream: the side stream's next use waits on an event
      recorded on main after this step's consumer. Refused while G12 DSV41_MAINPROJ_HOIST is on.
      DSV41_STEP_C4_AUDIT=N: for the first N consumed steps also recompute the stock local GEMM at combine
      time on main and compare bitwise on device; one host read at step N logs PASS/MISMATCH and a mismatch
      disables C4 for the process. During those N steps the STOCK y is what gets gathered and served, so an audit
      arm is exact whether or not C4 is (audit arms only: it adds the GEMM back on the critical path).
"""
from __future__ import annotations

import collections
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "false", "no")


def _pack_mode() -> str:
    raw = os.environ.get("DSV41_STEP_ENGRAM_PACK", "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "0"
    m = {"1": "look", "on": "look", "look": "look", "all": "all"}
    if raw not in m:
        raise ValueError(f"DSV41_STEP_ENGRAM_PACK must be one of 0|1|look|all, got {raw!r}")
    return m[raw]


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


COLEMB = _truthy("DSV41_STEP_COLEMB")
COLEMB_MAX_T = _int("DSV41_STEP_COLEMB_MAX_T", 128)
COLEMB_CHUNK = _int("DSV41_STEP_COLEMB_CHUNK", 128)
PACK = _pack_mode()
PACK_MAX_T = _int("DSV41_STEP_ENGRAM_PACK_MAX_T", 128)
C4 = _truthy("DSV41_STEP_C4")
C4_AUDIT = _int("DSV41_STEP_C4_AUDIT", 0)
PRIO_HOIST = _truthy("DSV41_PRIO_ENGRAM_HOIST")
G12_HOIST = _truthy("DSV41_MAINPROJ_HOIST")
COUNTS: collections.Counter = collections.Counter()

# Injectable facades (CPU tests replace them; production = torch.cuda / vllm.distributed).
_cu = torch.cuda


def _all_gather(x: torch.Tensor, dim: int) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x, dim=dim)


def _tp() -> tuple[int, int]:
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _tp_agree_min(ok: bool) -> bool:
    from vllm.models.deepseek_v41.nvidia.dsv41_r4_small import _tp_agree_min as f

    return f(ok)


def _tp_all_equal(vals: list[int]) -> bool:
    """Every TP rank holds the same integer list (CPU group all-gather; eager only)."""
    from vllm.distributed import get_tp_group

    grp = get_tp_group()
    if grp.world_size == 1:
        return True
    t = torch.tensor(vals, dtype=torch.int64)
    out = [torch.empty_like(t) for _ in range(grp.world_size)]
    torch.distributed.all_gather(out, t, group=grp.cpu_group)
    return all(torch.equal(o, out[0]) for o in out)


def _capturing() -> bool:
    return _cu.is_available() and _cu.is_current_stream_capturing()


def _breakable_capture() -> bool:
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        return bool(BreakableCUDAGraphCapture.is_active())
    except Exception:  # noqa: BLE001
        return False


# =====================================================================================================
# The all-reduce map (pure torch, CPU-testable)
# =====================================================================================================
BF16_NAN_AR = 0x7FFF  # F2FP.BF16.F32.PACK_AB of the canonical fp32 NaN that FADD returns


def canon_bits(w: torch.Tensor) -> torch.Tensor:
    """bf16 tensor -> int16 bits of bf16_rn(((0.f + 0) + ...) + v + ... + 0) as ar_kernel_pl computes it for an
    element with exactly one non-zero contributor: FADD without .FTZ (denormals kept; live .so 14d049dd SASS has
    FADD only, no FADD.FTZ), RN pack. -0 -> +0; NaN -> 0x7FFF; +-inf, denormals, everything else unchanged."""
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    nan = ((u & 0x7F80) == 0x7F80) & ((u & 0x007F) != 0)
    out = torch.where(u == 0x8000, torch.zeros_like(u), u)
    out = torch.where(nan, torch.full_like(u, BF16_NAN_AR), out)
    return ((out ^ 0x8000) - 0x8000).to(torch.int16)  # back to signed int16 bit patterns


def special_counts(w: torch.Tensor) -> dict:
    u = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp, man = u & 0x7F80, u & 0x007F
    return {
        "neg0": int((u == 0x8000).sum()),
        "nan": int(((exp == 0x7F80) & (man != 0)).sum()),
        "nan_not_canon": int(((exp == 0x7F80) & (man != 0) & (u != BF16_NAN_AR)).sum()),
        "denorm": int(((exp == 0) & (man != 0)).sum()),
        "inf": int(((exp == 0x7F80) & (man == 0)).sum()),
    }


# =====================================================================================================
# C3a  DSV41_STEP_COLEMB: column-sharded embedding (+ one column all-gather)
# =====================================================================================================
class ColTable:
    __slots__ = ("table", "rank", "tp", "n", "V", "D", "max_t", "lookup")

    def __init__(self, table, rank, tp, n, V, D, max_t):
        self.table, self.rank, self.tp, self.n, self.V, self.D, self.max_t = table, rank, tp, n, V, D, max_t
        self.lookup = None  # set by _colemb_lookup_fn on first use


_COL: dict[int, object] = {}  # weight data_ptr -> ColTable | False


def _rdma_max_reduce_rows(D: int) -> int:
    """Rows whose stock AR goes to the rdma all-reduce (whose map canon_bits describes); 0 = no rdma path."""
    try:
        from vllm.distributed import get_tp_group

        rc = getattr(get_tp_group().device_communicator, "rdma_comm", None)
        if rc is None or getattr(rc, "disabled", False):
            return 0
        return int(rc.max_reduce_bytes) // (2 * D)
    except Exception:  # noqa: BLE001
        return 0


def colemb_refusal(emb) -> str | None:
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    if not isinstance(emb, VocabParallelEmbedding) or isinstance(emb, ParallelLMHead):
        return "not a plain VocabParallelEmbedding"
    if getattr(emb, "tp_size", 1) <= 1 or not getattr(emb, "reduce_results", False):
        return "tp 1 or no reduce"
    if getattr(emb, "parallel_group", None) is not None:
        return "custom parallel group"
    if type(getattr(emb, "quant_method", None)).__name__ != "UnquantizedEmbeddingMethod":
        return "quantized embedding"
    w = emb.weight
    if w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_contiguous() or not w.is_cuda:
        return "weight is not a contiguous bf16 CUDA matrix"
    si = emb.shard_indices
    if int(si.num_added_elements) != 0 or int(getattr(emb, "num_added_embeddings", 0) or 0) != 0:
        return "added (LoRA) vocab"
    if emb.embedding_dim % emb.tp_size or (emb.embedding_dim // emb.tp_size) % 8:
        return "embedding_dim not split into 16 B-aligned column shards"
    return None


def build_coltable(emb, orig_forward, rank: int, tp: int, max_rows: int, chunk: int,
                   id_dtype=torch.int64):
    """Eager, every TP rank at the same point. Returns (ColTable | None, stats dict)."""
    V, D = int(emb.org_vocab_size), int(emb.embedding_dim)
    n = D // tp
    si = emb.shard_indices
    vs, ve = int(si.org_vocab_start_index), int(si.org_vocab_end_index)
    dev = emb.weight.device
    W = emb.weight.data
    table = torch.empty((V, n), dtype=torch.bfloat16, device=dev)
    mism = torch.zeros((), dtype=torch.int64, device=dev)
    ck = torch.zeros(2, dtype=torch.int64, device=dev)
    chunk = max(1, min(chunk, max_rows))
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        ids = torch.arange(s, e, dtype=id_dtype, device=dev)
        full = orig_forward(emb, ids)  # the live path: fused vocab-parallel kernel + all-reduce
        bits = full.view(torch.int16)
        table[s:e].copy_(full[:, rank * n:(rank + 1) * n])
        lo, hi = max(s, vs), min(e, ve)
        if lo < hi:  # rows this rank owns: every column must equal canon(W)
            mism += (bits[lo - s:hi - s] != canon_bits(W[lo - vs:hi - vs])).sum()
        b64 = bits.to(torch.int64)
        ck[0] += b64.sum()
        ck[1] += (b64 * (torch.arange(e - s, device=dev, dtype=torch.int64) + s + 1).unsqueeze(1)).sum()
    stats = special_counts(W[: max(0, min(ve, V) - vs)])
    stats["mismatch"] = int(mism.item())
    stats["checksum"] = [int(v) for v in ck.tolist()]
    return ColTable(table, rank, tp, n, V, D, max_rows), stats


def _colemb_lookup_fn():
    """One Triton kernel: out[t, :] = table[id] if 0 <= id < V else +0 (raw int16 copy, no float ops)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _colemb_kernel(IDS, TAB, OUT, V, N: tl.constexpr, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        i = tl.load(IDS + t).to(tl.int64)
        ok = (i >= 0) & (i < V)
        row = tl.load(TAB + tl.where(ok, i, 0) * N + c, mask=(c < N) & ok, other=0)
        tl.store(OUT + t * N + c, row, mask=c < N)

    def lookup(ids: torch.Tensor, ct: ColTable) -> torch.Tensor:
        T = ids.numel()
        out = torch.empty((T, ct.n), dtype=torch.bfloat16, device=ids.device)
        blk = 128
        _colemb_kernel[(T, triton.cdiv(ct.n, blk))](
            ids.reshape(-1), ct.table.view(torch.int16), out.view(torch.int16), ct.V, N=ct.n, BLOCK=blk,
            num_warps=1)
        return out

    return lookup


def colemb_forward(emb, input_, ct: ColTable) -> torch.Tensor:
    if ct.lookup is None:
        ct.lookup = _colemb_lookup_fn()
    local = ct.lookup(input_, ct)  # [T, D/tp]
    out = _all_gather(local, -1)  # [T, D]: rank r's columns at [r*n, (r+1)*n) == the AR output row
    COUNTS["colemb"] += 1
    return out.view(*input_.shape, ct.D)


def _install_colemb() -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    if getattr(VocabParallelEmbedding, "_dsv41_step_colemb", False):
        return
    orig = VocabParallelEmbedding.forward

    def forward(self, input_):
        key = id(self)
        ct = _COL.get(key)
        if ct is False:
            return orig(self, input_)
        if ct is None:
            if _capturing():
                return orig(self, input_)  # never build inside a capture; this graph keeps the stock op
            ct = _colemb_setup(self, orig, input_.dtype)
            _COL[key] = ct if ct is not None else False
            if ct is None:
                return orig(self, input_)
        T = input_.numel()
        if T == 0 or T > ct.max_t or input_.dtype not in (torch.int32, torch.int64) or not input_.is_cuda:
            return orig(self, input_)
        return colemb_forward(self, input_, ct)

    VocabParallelEmbedding.forward = forward
    VocabParallelEmbedding._dsv41_step_colemb = True
    logger.info("DSV41_STEP_COLEMB: VocabParallelEmbedding.forward hook installed (max_t %d)", COLEMB_MAX_T)


_TABLES: dict[int, ColTable] = {}  # weight data_ptr -> table (the DSpark embedding may alias the target's)


def _colemb_setup(emb, orig, id_dtype=torch.int64):
    reason = colemb_refusal(emb)
    rank, tp = _tp() if reason is None else (0, 1)
    D = int(getattr(emb, "embedding_dim", 0) or 0)
    max_rows = min(COLEMB_MAX_T, _rdma_max_reduce_rows(D)) if reason is None else 0
    if reason is None and max_rows <= 0:
        reason = "the stock all-reduce of these rows is not the rdma path"
    if not _tp_agree_min(reason is None):
        logger.warning("DSV41_STEP_COLEMB: refused (%s on this rank or a peer): stock embedding", reason)
        return None
    wkey = emb.weight.data_ptr()
    if wkey in _TABLES:
        return _TABLES[wkey]
    ct, stats = build_coltable(emb, orig, rank, tp, max_rows, min(COLEMB_CHUNK, max_rows), id_dtype)
    same = _tp_all_equal(stats["checksum"])  # every rank calls both collectives (no short-circuit)
    ok = _tp_agree_min(stats["mismatch"] == 0 and same)
    msg = (f"DSV41_STEP_COLEMB rank {rank}: full-table check {ct.V} x {ct.D} (owner rows of this rank vs "
           f"canon(W): mismatches {stats['mismatch']}; -0 {stats['neg0']}, NaN {stats['nan']} "
           f"(non-canonical {stats['nan_not_canon']}), denormal {stats['denorm']}, inf {stats['inf']}; "
           f"AR checksums equal on all ranks: {same})")
    if not ok:
        logger.warning(msg + "  FAIL -> stock embedding on every rank")
        return None
    logger.info(msg + f"  PASS: column table {ct.V} x {ct.n} bf16, lookup+all-gather for 1 <= T <= {ct.max_t}")
    _TABLES[wkey] = ct
    return ct


# =====================================================================================================
# C3b  DSV41_STEP_ENGRAM_PACK: one rows gather (+ one wkv gather) for both Engram layers
# =====================================================================================================
class _PackState:
    def __init__(self):
        self.layers: list = []  # Engram instances in construction (= layer) order
        self.plan = None  # (first, second) or False
        self.mode = "0"
        self.seq = 0
        self.rows = None  # (seq, T, rows2 [T, H, hd]) for the second layer
        self.kv = None  # (seq, T, kv2 [T, N]) for the second layer
        self.second_seq = -1  # seq whose rows the second layer's embed just took


PK = _PackState()


def engram_refusal(e) -> str | None:
    if getattr(e, "_batch", None) is not None:
        return "EngramBatch path"
    if getattr(e, "use_sequence_parallel", False):
        return "sequence parallel"
    t = getattr(e, "embed_tokens", None)
    if t is None or getattr(t, "tp_size", 1) <= 1 or getattr(t, "dp_size", 1) != 1:
        return "tp 1 or Engram DP > 1"
    if int(t.tp_size) * int(t.part_n_hash_cols) != int(t.n_hash_cols):
        return "head shards do not tile n_hash_cols"
    w = getattr(e, "wkv", None)
    if w is None or not getattr(w, "gather_output", False) or getattr(w, "skip_bias_add", False):
        return "wkv does not gather / skips bias"
    qn = type(w).forward.__qualname__
    if qn != "ColumnParallelLinear.forward" and not qn.endswith(".ColumnParallelLinear.forward"):
        return f"wkv forward is {qn}"
    return None


def _plan_pack():
    if PK.plan is not None:
        return PK.plan
    reason = None
    if len(PK.layers) != 2:
        reason = f"{len(PK.layers)} Engram layers (need exactly 2)"
    else:
        a, b = PK.layers
        reason = engram_refusal(a) or engram_refusal(b)
        if reason is None:
            ta, tb = a.embed_tokens, b.embed_tokens
            if (ta.part_n_hash_cols, ta.dim, ta.n_hash_cols, ta.tp_size) != (
                    tb.part_n_hash_cols, tb.dim, tb.n_hash_cols, tb.tp_size):
                reason = "the two tables differ in shape"
            elif a.staged_rows.shape != b.staged_rows.shape or a.staged_rows.dtype != b.staged_rows.dtype:
                reason = "staging buffers differ"
    if reason is not None:
        logger.warning("DSV41_STEP_ENGRAM_PACK: refused (%s): stock Engram gathers", reason)
        PK.plan = False
        return False
    a, b = PK.layers
    PK.mode = PACK
    if PK.mode == "all" and PRIO_HOIST:
        logger.warning("DSV41_STEP_ENGRAM_PACK=all: DSV41_PRIO_ENGRAM_HOIST is on and owns the wkv GEMMs; "
                       "running look (rows gather packed, wkv gathers stock)")
        PK.mode = "look"
    if PK.mode == "all":
        _wrap_wkv(a, b)
    PK.plan = (a, b)
    logger.info("DSV41_STEP_ENGRAM_PACK=%s: engram layers %d and %d share one rows gather%s (T <= %d)",
                PK.mode, a.layer_hash_index, b.layer_hash_index,
                " and one wkv gather" if PK.mode == "all" else "", PACK_MAX_T)
    return PK.plan


def _pack_ok(T: int) -> bool:
    return 0 < T <= PACK_MAX_T and not _breakable_capture()


def packed_rows_gather(a, b, T: int):
    """Both layers' local rows -> one [T, 2H, hd] operand (one cat) -> one all-gather -> ([T, tp*H, hd] x 2) in
    the stock head order (rank r's heads at r*H .. r*H+H-1; one split copy). Main waits on both lookups' events
    (_ready_rows, exactly the stock wait; the second lookup was started at forward start on the same prefetch
    stream as the first)."""
    ra = a._ready_rows(T)
    rb = b._ready_rows(T)
    H, hd = ra.shape[1], ra.shape[2]
    g = _all_gather(torch.cat([ra, rb], dim=1), 1)  # [T, tp*2H, hd]; rank r's [a heads | b heads] block
    tp = g.shape[1] // (2 * H)
    s = g.view(T, tp, 2, H * hd).permute(2, 0, 1, 3).contiguous()  # [2, T, tp, H*hd]: one copy
    return s[0].view(T, tp * H, hd), s[1].view(T, tp * H, hd)


def _install_engram(engram_cls) -> None:
    if getattr(engram_cls, "_dsv41_step_pack", False):
        return
    orig_init, orig_embed = engram_cls.__init__, engram_cls.embed

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        PK.layers.append(self)

    def embed(self, hash_ids):
        plan = _plan_pack()
        if not plan:
            return orig_embed(self, hash_ids)
        a, b = plan
        T = int(hash_ids.shape[0])
        n = self.embed_tokens.n_hash_cols
        if self is a:
            PK.seq += 1
            PK.rows = PK.kv = None
            if not _pack_ok(T):
                COUNTS["pack_skip"] += 1
                return orig_embed(self, hash_ids)
            r1, r2 = packed_rows_gather(a, b, T)
            PK.rows = (PK.seq, T, r2)
            COUNTS["pack_rows"] += 1
            return r1[:, :n]
        if self is b:
            rec, PK.rows = PK.rows, None
            if rec is None or rec[1] != T:
                PK.second_seq = -1
                return orig_embed(self, hash_ids)
            PK.second_seq = rec[0]
            return rec[2][:, :n]
        return orig_embed(self, hash_ids)

    engram_cls.__init__ = __init__
    engram_cls.embed = embed
    engram_cls._dsv41_step_pack = True
    logger.info("DSV41_STEP_ENGRAM_PACK=%s: Engram hooks installed", PACK)


def _wrap_wkv(a, b) -> None:
    wa, wb = a.wkv, b.wkv
    fa, fb = wa.forward, wb.forward  # bound ColumnParallelLinear.forward

    def wkv_first(x):
        rec = PK.rows
        T = int(x.shape[0])
        if rec is None or rec[0] != PK.seq or rec[1] != T:
            return fa(x)
        x2 = rec[2][:, : b.embed_tokens.n_hash_cols].flatten(-2)  # == layer b's operand
        # == ColumnParallelLinear.forward's matrix multiply (bias exactly as it passes it; Engram's wkv has none)
        y1 = wa.quant_method.apply(wa, x, wa.bias if not wa.skip_bias_add else None)
        y2 = wb.quant_method.apply(wb, x2, wb.bias if not wb.skip_bias_add else None)  # layer b's, run earlier
        kv1, kv2 = packed_cols_gather(y1, y2)
        PK.kv = (PK.seq, T, kv2)
        COUNTS["pack_wkv"] += 1
        return kv1

    def wkv_second(x):
        rec, PK.kv = PK.kv, None
        if rec is None or rec[0] != PK.second_seq or rec[1] != int(x.shape[0]):
            return fb(x)
        return rec[2]

    wa.forward, wb.forward = wkv_first, wkv_second


def packed_cols_gather(y1: torch.Tensor, y2: torch.Tensor):
    """[T, n] x 2 -> one column all-gather of [T, 2n] -> the two stock [T, tp*n] gathers (one cat, one copy)."""
    T, n = y1.shape
    g = _all_gather(torch.cat([y1, y2], dim=-1), -1)  # [T, tp*2n], rank r's [y1_r | y2_r] at [r*2n, (r+1)*2n)
    tp = g.shape[1] // (2 * n)
    s = g.view(T, tp, 2, n).permute(2, 0, 1, 3).contiguous()  # [2, T, tp, n]
    return s[0].view(T, tp * n), s[1].view(T, tp * n)


# =====================================================================================================
# C4  DSV41_STEP_C4: main_proj local GEMM on a side stream from the screen's end to combine_hidden_states
# =====================================================================================================
class _C4State:
    def __init__(self):
        self.stream = self.ev_screen = self.ev_done = None
        self.ctx = None  # (aux list, speculator) of the sample_tokens call in flight
        self.pending = None
        self.keep = None  # last consumed launch: its side-stream tensors stay referenced until the next fork
        self.disabled = False
        self.installed = False
        self.audit_ctr = None
        self.audit_n = 0


C4S = _C4State()


class _Pending:
    __slots__ = ("st", "M", "aux", "x", "y", "shape", "xshape")


def install_c4() -> None:
    if C4S.installed:
        return
    C4S.installed = True
    if G12_HOIST:
        C4S.disabled = True
        logger.warning("DSV41_STEP_C4: DSV41_MAINPROJ_HOIST (G12) is on and owns main_proj: C4 off")
        return
    try:
        from vllm.v1.worker.gpu import model_runner as mr

        cls = mr.GPUModelRunner
        if getattr(cls, "_dsv41_step_c4", False):
            return
        orig = cls.sample_tokens

        def sample_tokens(self, *args, **kwargs):
            C4S.ctx = _make_ctx(self)
            try:
                return orig(self, *args, **kwargs)
            finally:
                C4S.ctx = None
                if C4S.pending is not None:
                    c4_drop("not consumed by combine_hidden_states")

        cls.sample_tokens = sample_tokens
        cls._dsv41_step_c4 = True
        logger.info("DSV41_STEP_C4: GPUModelRunner.sample_tokens hook installed (audit %d)", C4_AUDIT)
    except Exception as e:  # noqa: BLE001 - fail closed: no ctx -> never forks
        C4S.disabled = True
        logger.warning("DSV41_STEP_C4: could not hook sample_tokens (%r): stock main_proj", e)


def _make_ctx(runner):
    try:
        ems = runner.execute_model_state
        if ems is None or not getattr(runner, "is_last_pp_rank", True):
            return None
        if getattr(runner, "pcp_manager", None) is not None or getattr(runner, "batch_sharder", None) is not None:
            return None
        aux = ems.aux_hidden_states
        spec = getattr(runner, "speculator", None)
        if not aux or not isinstance(aux, (list, tuple)) or spec is None:
            return None
        if type(spec).__name__ != "DSparkSpeculator":
            return None
        return (aux, spec)
    except Exception:  # noqa: BLE001
        return None


class _Launch:
    def __init__(self, st, M, aux):
        self.st, self.M, self.aux = st, M, aux

    def after_screen(self) -> None:
        """Called by certified_logits right after the screen launch (main stream, eager, armed step)."""
        if C4S.disabled or C4S.pending is not None:
            return
        if C4S.stream is None:
            C4S.stream, C4S.ev_screen, C4S.ev_done = _cu.Stream(), _cu.Event(), _cu.Event()
        S, main = C4S.stream, _cu.current_stream()
        C4S.ev_screen.record(main)  # after the screen; the aux taps were final at the target graph's end
        S.wait_event(C4S.ev_screen)
        p = _Pending()
        p.st, p.M, p.aux = self.st, self.M, self.aux
        with _cu.stream(S):
            p.x = torch.cat(list(self.aux), dim=-1)  # == the operand propose passes to combine_hidden_states
            p.y, p.shape = self.st.local(p.x)  # MXFP8 quant + CUTLASS GEMM, this rank's [M, N/tp] columns
        C4S.ev_done.record(S)
        p.xshape = tuple(p.x.shape)
        C4S.pending = p
        COUNTS["c4_launch"] += 1

    def after_ag(self) -> None:  # the argmax all-gather stays where it is; nothing to do
        return None


def c4_target_hook():
    """dsv41_r4_small.target_logits (CERT-armed step): a _Launch or None (stock)."""
    ctx = C4S.ctx
    if ctx is None or C4S.disabled or C4S.pending is not None or _capturing():
        return None
    aux, spec = ctx
    try:
        from vllm.models.deepseek_v41.nvidia import dsv41_r4_small as r4

        inner = spec.model.model
        if not r4.MAINPROJ_SPLIT or not getattr(type(inner), "_dsv41_r4_mainproj", False):
            return None
        st = getattr(inner.main_proj, "_dsv41_r4_split", None)
        if not st:
            return None
        rows = {int(t.shape[0]) for t in aux}
        if len(rows) != 1 or any(t.dim() != 2 for t in aux):
            return None
        M = rows.pop()
        if M == 0 or M > r4.MAINPROJ_MAX_M or st.checked.get(M) is not True:
            COUNTS["c4_skip"] += 1
            return None
    except Exception:  # noqa: BLE001
        return None
    return _Launch(st, M, aux)


def c4_take(st, x, M: int):
    """dsv41_r4_small.mainproj_split, before its verdict lookup: the gathered main_proj output from the hoisted
    local GEMM (join + all-gather on main, where the stock path gathers), or None (stock)."""
    p = C4S.pending
    if p is None:
        return None
    C4S.pending = None
    if p.st is not st or p.M != M or tuple(x.shape) != p.xshape:
        _join(p)
        _count_drop("operand mismatch")
        return None
    _cu.current_stream().wait_event(C4S.ev_done)  # join: the GEMM (and everything S did) before the gather
    y = p.y
    if C4_AUDIT and C4S.audit_n < C4_AUDIT:
        y = _audit(st, x, p)  # audit window: serve the stock local GEMM, so the audit arm is exact either way
    out = st.gather(y, p.shape)  # rdma all-gather on main: same op, same place in the collective order
    C4S.keep = p
    COUNTS["c4_take"] += 1
    n = COUNTS["c4_take"]
    if n == 1 or n % 20000 == 0:
        logger.info("DSV41_STEP_C4: consumed %d, launched %d, skipped %d, dropped %d (M=%d)", n,
                    COUNTS["c4_launch"], COUNTS["c4_skip"], COUNTS["c4_drop"], M)
    return out


def _join(p) -> None:
    _cu.current_stream().wait_event(C4S.ev_done)
    C4S.keep = p


def _count_drop(why: str) -> None:
    COUNTS["c4_drop"] += 1
    if COUNTS["c4_drop"] <= 4:
        logger.warning("DSV41_STEP_C4: hoisted main_proj dropped (%s); stock path", why)


def c4_drop(why: str) -> None:
    p, C4S.pending = C4S.pending, None
    if p is not None:
        _join(p)
        _count_drop(why)


def _audit(st, x, p):
    """Compare the hoisted y with the stock local GEMM on device; returns the stock y (served while auditing)."""
    y_ref, _ = st.local(x)  # the stock local GEMM, on main, from the operand propose built now
    if C4S.audit_ctr is None:
        C4S.audit_ctr = torch.zeros(2, dtype=torch.int64, device=y_ref.device)
    C4S.audit_ctr[0] += 1
    C4S.audit_ctr[1] += (y_ref.view(torch.int16) != p.y.view(torch.int16)).sum()
    C4S.audit_n += 1
    if C4S.audit_n == C4_AUDIT:
        steps, bad = (int(v) for v in C4S.audit_ctr.tolist())
        if bad:
            C4S.disabled = True
            logger.warning("DSV41_STEP_C4 audit: %d steps, %d mismatching elements: MISMATCH -> C4 off", steps, bad)
        else:
            logger.info("DSV41_STEP_C4 audit: %d steps bitwise identical to the stock local GEMM: PASS", steps)
    return y_ref


# =====================================================================================================
# Entry points (called from the patched live files; each only when its flag is on)
# =====================================================================================================
def install_c3(engram_cls) -> None:
    if COLEMB:
        _install_colemb()
    if PACK != "0":
        _install_engram(engram_cls)
