"""Opt-in target-only SM121 native backend; prepared, not serving-qualified.

pd-d05 (2026-10-25, default off): DSV41_MOE_FC2_D05 routes the target FC2
grouped GEMM (E384, N5120, K384, alignment 32, BF16 out) to the Codex D05
candidate native (native 6fea5091, manifest fc230f26). That kernel stages and
computes every 32-row tile exactly like the original, but its elected lane
skips the TMA store of an 8-row output group whose first grouped_layout entry
is -1 (scatter32 prefix contract: live rows of an expert form a prefix of its
32-aligned segment, so such a group is entirely dead). ep_gather never reads
dead rows, so served outputs are bitwise unchanged. FC1, all other shapes and
the align-64/128 paths stay on the original native/installed DeepGEMM.

  DSV41_MOE_FC2_D05=0       original behaviour (default; identical to 723fdf8d)
  DSV41_MOE_FC2_D05=1       target FC2 -> D05 candidate (dsv41_fc2_live_store=True)
  DSV41_MOE_FC2_D05=verify  serve the ORIGINAL FC2 output; also run the D05
                            kernel into a shadow buffer and count, per FC2
                            weight (= per target layer), live/written-group
                            mismatches and touched dead groups on the device.
                            Diagnostic only (slower). Report: touch
                            $DSV41_MOE_FC2_D05_VERIFY_DIR/d05-verify.trigger
                            (default ~/dsv41-logs) after startup; each worker
                            writes d05-verify.<host>.<pid>.json.
  DSV41_MOE_FC2_D05_NATIVE_DIR  default ~/dsv41-goal120-20261024/d05-shadow-v1/native
                                (immutable, manifest-checked at load; same path as
                                the Codex v2a shadow so its JIT cubin key matches)
  DSV41_MOE_FC2_D05_JIT_DIR     default ~/dsv41-stage/pd/d05/jit-cache

pd-pdl (default off): DSV41_MOE_PDL_EARLY_B=1 loads the SAME native binaries from
pd-pdl trees whose FP4 GEMM JIT header adds the FC2-only early-weight PDL prologue
(~/dsv41-stage/pd/pdl/native-base, manifest 5d5f2f70; native-d05, manifest 2009e011)
with their own JIT caches (cold on first start: FC1/FC2 re-JIT). FC1 SASS is
identical to the original (checked offline); only the FC2 instantiation changes.
Pair with DSV41_MOE_PDL_TRIGGER=start|end (fp8_utils.py SiLU trigger) for overlap.

g1/m32-bw (default off): DSV41_MOE_M32_G1=pair|pf|la|ds|all loads the EXACT bandwidth variants of the same native
binaries from ~/dsv41-stage/g1/m32-bw (trees/<v>-base for FC1, trees/<v>-d05 for target FC2 with ds/all; pinned
manifests below; pre-JIT caches jit/<tree>). Requires DSV41_MOE_FC2_D05=1, excludes DSV41_MOE_PDL_EARLY_B.
  pair  FC1 TMA leader issues k-block pairs back to back (whole 128-B weight lines per request pair)
  pf    FC1 B/SFB L2 prefetch cursor 8 k-blocks ahead in 4-k-block groups + EVICT_FIRST weight loads
  la    FC1 loads only live A rows (cp.async, producer warp 9) instead of the 32-row TMA box
  ds    FC1 + FC2 store only live rows (32-bit st.global of the same BF16 pairs; no smem staging/TMA store)
  all   FC1 pf+la+ds + D04 dead-walk stop; FC2 ds + D04 + EVICT_FIRST weight loads
Every live output row is bitwise identical (same MMA sequence per element); see the g1/m32-bw README.

g1 fc2c-d05 + g2 fc2c-on-m32all (2026-10-26, default off): DSV41_MOE_FC2C=1 (needs DSV41_MOE_FC2_D05=1) runs the
target FC2 on a compact native (D05 host code + FC2-compact v2 tail, _C 33d4eb19) for M_sum >= rows(
DSV41_MOE_FC2C_MIN_TOKENS, default 2 tokens = 384 aligned rows); smaller FC2 calls (M=1) keep the D05 native of the
active tree. The kernel reads k-blocks 0/1 of the PADDED w2 through the ordinary descriptor (bytes 0..127 of each
192-B row) and the last k-block from a per-weight TAIL tensor [E*5120, 32] uint8 = the .b4x16_p64 shared-memory
image of k 256..287 (TMA zero-fills k 288..383 exactly where w2 holds FP4 zeros): FC2 weight bytes 192 -> 160 per
row (-16.7%), same smem data bytes, same MMA sequence, so every stored output word is bitwise the D05 kernel's of
the same tree. Tails are built once per FC2 weight on its first eligible EAGER call (+62.9 MB per target layer,
+2.34 GiB/rank for 40 layers; the padded w2 stays for M=1, prefill and the 128-row path). A call during CUDA-graph
capture whose weight has no tail yet falls back to the D05 kernel (counted, warned once; still exact).
  The compact tree follows DSV41_MOE_M32_G1:
    all   g2 tree ~/dsv41-stage/g2/fc2c-on-m32all/tree: the m32-bw 'all' FC2 header (ef + ds + dw) + the tail
          (tools/make_fc2c_tree.py; JIT cache .../jit-cache). Like the live 'all' D05 kernel it stores live rows only.
    0     g1 tree ~/dsv41-stage/g1/fc2c-d05/native: plain D05 header + the tail (JIT cache .../jit-cache).
    other pair/pf/la/ds: refused (no qualified compact tree).

  DSV41_MOE_FC2C=0          off (default): code paths identical to db659d80
  DSV41_MOE_FC2C=1          FC2-compact for M_sum >= min rows
  DSV41_MOE_FC2C=verify     serve the D05 output; also run FC2-compact into a shadow and count mismatches per
                            weight on the device (sm121_d05_verify.compare semantics). With a g1 'ds'/'all' tree
                            (live-row stores) the served buffer is first filled with the shadow's poison, so rows
                            neither kernel writes compare equal (dead rows are never read). Diagnostic only.
                            Report: touch $DSV41_MOE_FC2_D05_VERIFY_DIR/fc2c-verify.trigger after startup.
  DSV41_MOE_FC2C_MIN_TOKENS default 2 (rows = 6t + min(6t, 384) * 31)
  DSV41_MOE_FC2C_NATIVE_DIR / DSV41_MOE_FC2C_JIT_DIR: override the selected tree / its JIT cache (manifest-pinned)
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

_LOCK = threading.RLock()
_MANIFEST_SHA256 = "5f939d0021986cf071741012d49e4ad5acaafa814597aba94649ad4207a83aaf"
_NATIVE_SHA256 = "b1c4eff850fc243ebaad12bd04bae00a7910f310dde20727649517ecf98105a9"
# pd-d05: Codex goal-120 D05 candidate native (build-v1; = baseline source + source.patch).
_D05_MANIFEST_SHA256 = "fc230f267293c94f101d48dffe78113276de017425e9923d120014dd1a69dd2c"
_D05_NATIVE_SHA256 = "6fea5091715fc33836ec32c9c1aabfe24f3733756d00ffab0af6f326ea479626"
# Target FC2 geometry the D05 kernel is qualified for.
_FC2_E, _FC2_N, _FC2_KP, _FC2_KB = 384, 5120, 384, 192
# pd-pdl trees (same native binaries, patched JIT header, regenerated manifests).
_PDL_BASE_MANIFEST_SHA256 = "5d5f2f70ecd9b423d18edeba31771d5cdd677d37d7c438a1248940d4e916165a"
_PDL_D05_MANIFEST_SHA256 = "2009e01179eacde44e10de2d2be2ca1b05dea72c04f9dbfd1bbcf57915c89ee9"
# g1/m32-bw variant trees (tools/make_tree.py over the two immutable trees above; same .so binaries).
_G1_ROOT = Path.home() / "dsv41-stage/g1/m32-bw"
_G1_MANIFESTS = {
    "all-base": "74d0d4f2a09247a668f876d4d5e0993afa9e93462f4d7f6da70276738a218f5c",
    "all-d05": "073232efb28247cc03c83b4198050cd2d9c0d292f5b980819ef7403b57cb9aee",
    "ds-base": "1f2e29eec1af50f2308522001f7a2eca150a37988138ea2c5d585ae60175f5f7",
    "ds-d05": "2da533496854072ad8897e9b7b1378a144dcfff973d8f1e36e0c7e8e19ad9fed",
    "la-base": "5deefff689ab452ac6c7608841750d7b8c0e7b8db99005e0375017ab19889a13",
    "pair-base": "1d67bf093c945b89113220ae3a0021f11fdba35b6b6da731705f1fdaeaef7061",
    "pf-base": "4787326b10c4e2dc02fa19c6085bc88bbcef45b28e89a3f3c895b36d8dc79ad9",
}
_G1_VARIANTS = ("pair", "pf", "la", "ds", "all")
_G1_HAS_D05 = ("ds", "all")
_G1_HAS_LA = ("la", "all")
# g2 fc2c-on-m32all: the g1 fc2c-d05 native (same _C 33d4eb19) with the m32-bw 'all' FC2 header + the compact tail.
_G2_FC2C_ROOT = Path.home() / "dsv41-stage/g2/fc2c-on-m32all"
_G2_FC2C_MANIFEST_SHA256 = "5a79c849510e7e1403dc81ffc9ec6b9e30e6a408444feea1a76807beceb9e95e"


def g1_variant() -> str:
    raw = os.environ.get("DSV41_MOE_M32_G1", "0").strip().lower()
    if raw in ("", "0", "off"):
        return ""
    assert raw in _G1_VARIANTS, f"DSV41_MOE_M32_G1={raw!r}: use 0 or one of {_G1_VARIANTS}"
    assert not pdl_early_b(), "DSV41_MOE_M32_G1 and DSV41_MOE_PDL_EARLY_B are exclusive"
    assert d05_mode() == "1", "DSV41_MOE_M32_G1 is qualified with DSV41_MOE_FC2_D05=1 only"
    return raw


def pdl_early_b() -> bool:
    raw = os.environ.get("DSV41_MOE_PDL_EARLY_B", "0").strip().lower()
    assert raw in ("", "0", "1", "off", "on"), f"DSV41_MOE_PDL_EARLY_B={raw!r}"
    return raw in ("1", "on")
_D05_MAX_ROWS = 384 * 32 + 96 * 6  # = 12864 > M_sum(96 tokens) = 12480
# g1 fc2c-d05: D05 source + FC2-compact v2 tail native (spark-four CPU build, 2026-10-26).
_FC2C_MANIFEST_SHA256 = "3b6f30bcf05bdb8d1255409cfea91dd711479e5102c5cfb1466917ad76c90cb9"
_FC2C_NATIVE_SHA256 = "33d4eb19f3dc7c281440afa6df80724c1f9adae0ed6bc8e57f835b9eea83889a"
_FC2C_TOPK, _FC2C_ALIGN = 6, 32
_FC2C_KREAL_B = 144  # packed bytes of k 0..287; bytes 144..191 of every w2 row are TP padding (zero)
_NO_TAIL = object()  # FC2Compact.run(): weight has no tail (first seen during capture) -> caller falls back


def fc2c_mode() -> str:
    mode = os.environ.get("DSV41_MOE_FC2C", "0").strip().lower()
    if mode in ("", "0", "off", "false"):
        return "0"
    if mode in ("1", "on", "true"):
        return "1"
    if mode == "verify":
        return "verify"
    raise ValueError(f"DSV41_MOE_FC2C={mode!r}: use 0, 1 or verify")


def fc2c_min_rows(tokens: int | None = None) -> int:
    """Aligned FC2 rows (M_sum) of a `tokens`-token batch on the M32 path (compute_aligned_M_and_alignment):
    round_up(6t + min(6t, E) * 31, 32); monotonic in t, so `M_sum >= fc2c_min_rows(t)` <=> M >= t."""
    if tokens is None:
        tokens = int(os.environ.get("DSV41_MOE_FC2C_MIN_TOKENS", "2"))
    assert tokens >= 1, tokens
    routes = _FC2C_TOPK * tokens
    rows = routes + min(routes, _FC2_E) * (_FC2C_ALIGN - 1)
    return -(-rows // _FC2C_ALIGN) * _FC2C_ALIGN


def make_fc2c_tail(w2):
    """[E, 5120, 192] packed FP4 (int8/uint8) -> TAIL [E*5120, 32] uint8.

    Row r = the .b4x16_p64 shared-memory image of k 256..287 of w2 row r: bytes 0-7 = w2 bytes 128..135
    (k 256..271), bytes 16-23 = w2 bytes 136..143 (k 272..287), pads 0. The kernel's tail TMA zero-fills
    bytes 32..127 (k 288..383), so the K padding of w2 (bytes 144..191) must be zero; checked here.
    """
    E, N, KB = w2.shape
    assert (E, N, KB) == (_FC2_E, _FC2_N, _FC2_KB), tuple(w2.shape)
    assert w2.is_cuda and w2.element_size() == 1 and w2.is_contiguous()
    return fc2c_tail_from_bytes(w2)


def fc2c_tail_from_bytes(w2):
    """Device-agnostic body of make_fc2c_tail (also used by the CPU test): [..., R, 192] bytes -> [R', 32]."""
    import torch

    u8 = w2.view(torch.uint8)
    assert u8.shape[-1] == _FC2_KB
    assert not bool(u8[..., _FC2C_KREAL_B:].any()), "FC2 K padding (k >= 288) is not zero"
    rows = u8.numel() // _FC2_KB
    tail = torch.zeros((rows, 32), dtype=torch.uint8, device=u8.device)
    tail[:, 0:8] = u8[..., 128:136].reshape(rows, 8)
    tail[:, 16:24] = u8[..., 136:144].reshape(rows, 8)
    return tail


def d05_mode() -> str:
    mode = os.environ.get("DSV41_MOE_FC2_D05", "0").strip().lower()
    if mode in ("", "0", "off", "false"):
        return "0"
    if mode in ("1", "on", "true"):
        return "1"
    if mode == "verify":
        return "verify"
    raise ValueError(f"DSV41_MOE_FC2_D05={mode!r}: use 0, 1 or verify")


def _verify_tree(root: Path, manifest_sha: str, native_sha: str) -> dict:
    manifest_bytes = (root / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha, root
    manifest = json.loads(manifest_bytes)
    assert manifest["native_sha256"] == native_sha
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    return manifest


def d05_eligible(args, kwargs) -> bool:
    """Python pre-filter: only the target FC2 call reaches the D05 native.

    The native host code re-checks the full geometry/config and falls back to
    the ordinary full-store kernel if anything differs; this filter keeps FC1
    and every other shape on the original native module.
    """
    if len(args) != 4 or set(kwargs) - {"recipe_a", "recipe_b"}:
        return False
    try:
        (aq, sfa), (bw, sfb), d, layout = args
    except (TypeError, ValueError):
        return False
    return (
        tuple(bw.shape) == (_FC2_E, _FC2_N, _FC2_KB)
        and aq.dim() == 2
        and aq.shape[1] == _FC2_KP
        and aq.shape[0] % 32 == 0
        and 0 < aq.shape[0] <= _D05_MAX_ROWS
        and tuple(d.shape) == (aq.shape[0], _FC2_N)
        and str(d.dtype) == "torch.bfloat16"
        and d.is_contiguous()
        and str(layout.dtype) == "torch.int32"
        and tuple(layout.shape) == (aq.shape[0],)
        and kwargs.get("recipe_a") == (1, 128)
        and kwargs.get("recipe_b") == (1, 32)
    )


class _D05Verifier:
    """Shadow compare for DSV41_MOE_FC2_D05=verify (diagnostics, not timing)."""

    POISON = -1  # int16 0xFFFF in skipped/untouched rows of the shadow

    def __init__(self, tag: str = "d05"):
        import torch

        assert tag in ("d05", "fc2c"), tag
        self.tag = tag

        assert not torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()
        self.shadow = torch.empty((_D05_MAX_ROWS, _FC2_N), dtype=torch.bfloat16, device="cuda")
        # per FC2 weight (target layer, first-call order): calls, live rows,
        # mismatching int16 words in written groups, touched words in dead groups
        self.counters = torch.zeros((64, 4), dtype=torch.int32, device="cuda")
        self.index: dict[int, int] = {}
        self._host = torch.zeros((64, 4), dtype=torch.int32, pin_memory=True)
        self._side = torch.cuda.Stream()
        from vllm.model_executor.layers.fused_moe.experts import sm121_d05_verify as kv

        self._kernels = kv
        # Compile the Triton kernel now (eager, outside any capture).
        layout = torch.full((32,), -1, dtype=torch.int32, device="cuda")
        layout[:3] = 0
        probe = torch.zeros((32, _FC2_N), dtype=torch.bfloat16, device="cuda")
        cnt = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
        kv.compare(probe, probe, layout, cnt, self.POISON)
        torch.cuda.synchronize()
        self._start_reporter()

    def check(self, fc2_native, args, kwargs) -> None:
        import torch

        (aq, sfa), (bw, sfb), d, layout = args
        m = aq.shape[0]
        idx = self.index.setdefault(bw.data_ptr(), len(self.index))
        assert idx < self.counters.shape[0], "D05 verify: more FC2 weights than counter rows"
        shadow = self.shadow[:m]
        shadow.view(torch.int16).fill_(self.POISON)
        fc2_native.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (aq, sfa), (bw, sfb), shadow, layout,
            ensure_zero_padding=False, dsv41_fc2_live_store=True, **kwargs)
        self._kernels.compare(d, shadow, layout, self.counters[idx:idx + 1], self.POISON)

    def _start_reporter(self) -> None:
        import socket
        import time

        root = Path(os.environ.get("DSV41_MOE_FC2_D05_VERIFY_DIR", str(Path.home() / "dsv41-logs")))
        delay = float(os.environ.get("DSV41_MOE_FC2_D05_VERIFY_DELAY_S", "900"))
        trigger = root / f"{self.tag}-verify.trigger"
        out = root / f"{self.tag}-verify.{socket.gethostname()}.{os.getpid()}.json"
        born = time.time()

        def loop():
            import torch

            seen = None
            while True:
                time.sleep(2.0)
                try:
                    mtime = trigger.stat().st_mtime
                except OSError:
                    continue
                # Never touch CUDA before startup/graph capture is long over.
                if mtime == seen or time.time() - born < delay:
                    continue
                seen = mtime
                try:
                    with torch.cuda.device(self.device), torch.cuda.stream(self._side):
                        self._host.copy_(self.counters, non_blocking=True)
                        self._side.synchronize()
                    rows = self._host.tolist()
                    per = [dict(index=i, calls=r[0], live_rows=r[1], mismatch_written=r[2],
                                touched_dead=r[3]) for i, r in enumerate(rows[: len(self.index)])]
                    bad = sum(p["mismatch_written"] + p["touched_dead"] for p in per)
                    report = dict(host=socket.gethostname(), pid=os.getpid(), time=time.time(),
                                  mode="verify", fc2_weights_seen=len(self.index),
                                  weights_with_calls=sum(1 for p in per if p["calls"] > 0),
                                  status="MISMATCH" if bad else "OK", per_weight=per)
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(report, indent=1))
                    tmp.replace(out)
                except Exception as exc:  # diagnostics must never kill serving
                    try:
                        out.with_suffix(".err").write_text(repr(exc))
                    except OSError:
                        pass

        threading.Thread(target=loop, name=f"{self.tag}-verify-report", daemon=True).start()

    def check_compact(self, fc2c, args, kwargs) -> None:
        """g1 fc2c-d05 verify: shadow = FC2-compact + D05 on the served inputs; compare with the served D05 D."""
        import torch

        (aq, sfa), (bw, sfb), d, layout = args
        m = aq.shape[0]
        idx = self.index.setdefault(bw.data_ptr(), len(self.index))
        assert idx < self.counters.shape[0], "FC2C verify: more FC2 weights than counter rows"
        shadow = self.shadow[:m]
        shadow.view(torch.int16).fill_(self.POISON)
        if fc2c.run(((aq, sfa), (bw, sfb), shadow, layout), kwargs) is not _NO_TAIL:
            self._kernels.compare(d, shadow, layout, self.counters[idx:idx + 1], self.POISON)


class FC2Compact:
    """g1 fc2c-d05: FC2-compact tail tensors per FC2 weight and the compact+D05 call (DSV41_MOE_FC2C)."""

    def __init__(self, native, mode: str = "1", min_rows: int | None = None, verifier=None):
        assert mode in ("1", "verify"), mode
        assert mode != "verify" or verifier is not None
        self.native = native
        self.mode = mode
        self.min_rows = fc2c_min_rows() if min_rows is None else int(min_rows)
        self.verifier = verifier
        # w2 data_ptr -> (w2 view kept alive, tail [E*N, 32] uint8, device descriptor table [6, 128] uint8)
        self.entries: dict[int, tuple] = {}
        self.tail_bytes = 0
        self.capture_misses = 0
        self.eager_calls = 0  # host-side count of eager/captured launches (graph replays are not counted)

    def descs(self, bw):
        entry = self.entries.get(bw.data_ptr())
        if entry is not None:
            return entry[2]
        import torch

        if torch.cuda.is_current_stream_capturing():
            # Never allocate or copy inside a capture: this graph keeps the (bitwise-identical) D05 kernel.
            self.capture_misses += 1
            if self.capture_misses == 1:
                from vllm.logger import init_logger

                init_logger(__name__).warning(
                    "DSV41_MOE_FC2C: FC2 weight %#x first seen during CUDA-graph capture; "
                    "that graph uses the D05 kernel (exact, no compact gain)", bw.data_ptr())
            return None
        with _LOCK:
            entry = self.entries.get(bw.data_ptr())
            if entry is None:
                tail = make_fc2c_tail(bw)
                descs = self.native.fc2_compact_tail_descs(tail).to(bw.device)
                torch.cuda.current_stream(bw.device).synchronize()
                entry = (bw, tail, descs)
                self.entries[bw.data_ptr()] = entry
                self.tail_bytes += tail.numel()
                if len(self.entries) == 1:
                    from vllm.logger import init_logger

                    init_logger(__name__).info(
                        "DSV41_MOE_FC2C=%s: FC2-compact tails on (min rows %d); %.1f MB per FC2 weight",
                        self.mode, self.min_rows, tail.numel() / 1e6)
        return entry[2]

    def run(self, args, kwargs):
        """Launch FC2-compact + D05 on args = ((aq, sfa), (bw, sfb), d, layout); _NO_TAIL = fall back."""
        (aq, sfa), (bw, sfb), d, layout = args
        descs = self.descs(bw)
        if descs is None:
            return _NO_TAIL
        self.eager_calls += 1
        return self.native.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (aq, sfa), (bw, sfb), d, layout, ensure_zero_padding=False, dsv41_fc2_live_store=True,
            dsv41_fc2c_tail_descs=descs, **kwargs)


class TargetM32Kernel:
    def __init__(self, native, fc2_native=None, d05: str = "0", verifier=None, fc2c=None):
        self.native = native
        self.fc2_native = fc2_native
        self.d05 = d05
        self.verifier = verifier
        self.fc2c = fc2c
        self.g1_la = False
        assert d05 == "0" or fc2_native is not None
        assert d05 != "verify" or verifier is not None
        assert fc2c is None or d05 == "1", "DSV41_MOE_FC2C needs DSV41_MOE_FC2_D05=1"

    @contextlib.contextmanager
    def alignment_scope(self):
        with _LOCK:
            previous = self.native.get_mk_alignment_for_contiguous_layout()
            self.native.set_mk_alignment_for_contiguous_layout(32)
            try:
                yield
            finally:
                self.native.set_mk_alignment_for_contiguous_layout(previous)

    def gemm(self, *args, **kwargs):
        # This adapter is private to the tested MoE path. Its reduction ignores
        # invalid routes; the public grouped-GEMM padding contract is unchanged.
        assert "ensure_zero_padding" not in kwargs
        if self.g1_la:  # g1-la cp.async's A rows at row stride K: contiguous, 16-B aligned A only
            aq = args[0][0]
            assert aq.dim() == 2 and aq.is_contiguous() and aq.data_ptr() % 16 == 0, "g1-la: bad A layout"
        if self.d05 != "0" and d05_eligible(args, kwargs):
            if self.d05 == "1":
                fc2c = self.fc2c
                if fc2c is not None and args[0][0].shape[0] >= fc2c.min_rows:
                    if fc2c.mode == "1":
                        result = fc2c.run(args, kwargs)
                        if result is not _NO_TAIL:
                            return result
                    else:
                        if getattr(fc2c, "poison_served", False):
                            # g2: a g1 'ds'/'all' D05 kernel stores live rows only, so the served buffer gets the
                            # shadow's poison first: rows neither kernel writes then compare equal. Never read.
                            import torch

                            args[2].view(torch.int16).fill_(_D05Verifier.POISON)
                        result = self.fc2_native.m_grouped_fp8_fp4_gemm_nt_contiguous(
                            *args, ensure_zero_padding=False, dsv41_fc2_live_store=True, **kwargs
                        )
                        fc2c.verifier.check_compact(fc2c, args, kwargs)
                        return result
                return self.fc2_native.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    *args, ensure_zero_padding=False, dsv41_fc2_live_store=True, **kwargs
                )
            result = self.native.m_grouped_fp8_fp4_gemm_nt_contiguous(
                *args, ensure_zero_padding=False, **kwargs
            )
            self.verifier.check(self.fc2_native, args, kwargs)
            return result
        return self.native.m_grouped_fp8_fp4_gemm_nt_contiguous(
            *args, ensure_zero_padding=False, **kwargs
        )


def load_d05_native(baseline_native, pdl: bool):
    """Load the D05 candidate native under its own module name and JIT cache."""
    if pdl_early_b():
        root = (Path.home() / "dsv41-stage/pd/pdl/native-d05").resolve()
        cache = (Path.home() / "dsv41-stage/pd/pdl/jit-cache-d05").resolve()
        manifest_sha = _PDL_D05_MANIFEST_SHA256
    elif g1_variant() in _G1_HAS_D05:
        root = (_G1_ROOT / f"trees/{g1_variant()}-d05").resolve()
        cache = (_G1_ROOT / f"jit/{g1_variant()}-d05").resolve()
        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-d05"]
    else:
        root = Path(os.environ.get(
            "DSV41_MOE_FC2_D05_NATIVE_DIR",
            str(Path.home() / "dsv41-goal120-20261024/d05-shadow-v1/native"))).resolve()
        cache = Path(os.environ.get(
            "DSV41_MOE_FC2_D05_JIT_DIR", str(Path.home() / "dsv41-stage/pd/d05/jit-cache"))).resolve()
        manifest_sha = _D05_MANIFEST_SHA256
    baseline_root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()
    assert root != baseline_root and not root.is_relative_to(baseline_root)
    assert not cache.is_relative_to(root) and cache != baseline_root.parent / "target-m32-jit-cache-v1"
    manifest = _verify_tree(root, manifest_sha, _D05_NATIVE_SHA256)
    with _LOCK:
        prior_cache = os.environ.get("DG_JIT_CACHE_DIR")
        os.environ["DG_JIT_CACHE_DIR"] = str(cache)
        try:
            name = "vllm_target_m32_d05._C"
            assert name not in sys.modules
            spec = importlib.util.spec_from_file_location(name, root / manifest["native"])
            assert spec is not None and spec.loader is not None
            native = importlib.util.module_from_spec(spec)
            sys.modules[name] = native
            spec.loader.exec_module(native)
            native.init(str(root))
            native.set_pdl(pdl)
            if hasattr(native, "set_tc_util") and hasattr(baseline_native, "get_tc_util"):
                native.set_tc_util(baseline_native.get_tc_util())
            assert native.get_num_sms() == baseline_native.get_num_sms() == 48
            base_alignment = baseline_native.get_mk_alignment_for_contiguous_layout()
            # Used only for target FC2 inside the M32 path: alignment 32 for good
            # (the D05 API asserts it); prove the two natives keep separate state.
            native.set_mk_alignment_for_contiguous_layout(32)
            assert baseline_native.get_mk_alignment_for_contiguous_layout() == base_alignment
            native.set_pdl(not pdl)
            assert baseline_native.get_pdl() == pdl
            native.set_pdl(pdl)
        finally:
            if prior_cache is None:
                os.environ.pop("DG_JIT_CACHE_DIR", None)
            else:
                os.environ["DG_JIT_CACHE_DIR"] = prior_cache
    return native


def fc2c_tree() -> tuple:
    """(tree root, JIT cache, pinned manifest sha) of the FC2-compact native for the active DSV41_MOE_M32_G1."""
    variant = g1_variant()
    if variant == "all":
        root, cache, manifest_sha = _G2_FC2C_ROOT / "tree", _G2_FC2C_ROOT / "jit-cache", _G2_FC2C_MANIFEST_SHA256
    else:
        assert variant == "", f"DSV41_MOE_FC2C is qualified with DSV41_MOE_M32_G1=all or 0, not {variant!r}"
        g1 = Path.home() / "dsv41-stage/g1/fc2c-d05"
        root, cache, manifest_sha = g1 / "native", g1 / "jit-cache", _FC2C_MANIFEST_SHA256
    root = Path(os.environ.get("DSV41_MOE_FC2C_NATIVE_DIR", str(root))).resolve()
    cache = Path(os.environ.get("DSV41_MOE_FC2C_JIT_DIR", str(cache))).resolve()
    return root, cache, manifest_sha


def load_fc2c_native(baseline_native, pdl: bool):
    """g1 fc2c-d05 / g2 fc2c-on-m32all: load the FC2-compact native under its own module name and JIT cache."""
    assert not pdl_early_b(), "DSV41_MOE_FC2C has no pd-pdl tree"
    root, cache, manifest_sha = fc2c_tree()
    baseline_root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()
    d05_root = Path(os.environ.get(
        "DSV41_MOE_FC2_D05_NATIVE_DIR",
        str(Path.home() / "dsv41-goal120-20261024/d05-shadow-v1/native"))).resolve()
    for other in (baseline_root, d05_root, (_G1_ROOT / "trees").resolve()):
        assert root != other and not root.is_relative_to(other) and not cache.is_relative_to(other)
    assert not cache.is_relative_to(root)
    manifest = _verify_tree(root, manifest_sha, _FC2C_NATIVE_SHA256)
    with _LOCK:
        prior_cache = os.environ.get("DG_JIT_CACHE_DIR")
        os.environ["DG_JIT_CACHE_DIR"] = str(cache)
        try:
            name = "vllm_target_m32_fc2c._C"
            assert name not in sys.modules
            spec = importlib.util.spec_from_file_location(name, root / manifest["native"])
            assert spec is not None and spec.loader is not None
            native = importlib.util.module_from_spec(spec)
            sys.modules[name] = native
            spec.loader.exec_module(native)
            native.init(str(root))
            native.set_pdl(pdl)
            if hasattr(native, "set_tc_util") and hasattr(baseline_native, "get_tc_util"):
                native.set_tc_util(baseline_native.get_tc_util())
            assert native.get_num_sms() == baseline_native.get_num_sms() == 48
            assert hasattr(native, "fc2_compact_tail_descs")
            base_alignment = baseline_native.get_mk_alignment_for_contiguous_layout()
            # Target FC2 inside the M32 path only: alignment 32 for good (the private API asserts it).
            native.set_mk_alignment_for_contiguous_layout(32)
            assert baseline_native.get_mk_alignment_for_contiguous_layout() == base_alignment
            native.set_pdl(not pdl)
            assert baseline_native.get_pdl() == pdl
            native.set_pdl(pdl)
        finally:
            if prior_cache is None:
                os.environ.pop("DG_JIT_CACHE_DIR", None)
            else:
                os.environ["DG_JIT_CACHE_DIR"] = prior_cache
    return native


@functools.cache
def get_target_m32_kernel() -> TargetM32Kernel:
    assert os.environ.get("DSV41_SM121_TARGET_M32") == "1"
    from vllm.platforms import current_platform
    from vllm.utils import deep_gemm as installed

    assert current_platform.is_device_capability(121)
    installed._lazy_init()
    old = installed._import_deep_gemm()
    assert old is not None
    if pdl_early_b():
        root = (Path.home() / "dsv41-stage/pd/pdl/native-base").resolve()
        jit_cache = root.parent / "jit-cache-base"
        manifest_sha = _PDL_BASE_MANIFEST_SHA256
    elif g1_variant():
        root = (_G1_ROOT / f"trees/{g1_variant()}-base").resolve()
        jit_cache = (_G1_ROOT / f"jit/{g1_variant()}-base").resolve()
        manifest_sha = _G1_MANIFESTS[f"{g1_variant()}-base"]
    else:
        root = Path(os.environ["DSV41_TARGET_M32_NATIVE_DIR"]).resolve()
        jit_cache = root.parent / "target-m32-jit-cache-v1"
        manifest_sha = _MANIFEST_SHA256
    manifest_bytes = (root / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha
    manifest = json.loads(manifest_bytes)
    assert manifest["native_sha256"] == _NATIVE_SHA256
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative

    with _LOCK:
        # DeepJIT freezes its cache configuration at first runtime creation.
        # Initialize the installed backend first and create the candidate's
        # runtime under a separate cache, then restore the process environment.
        prior_cache = os.environ.get("DG_JIT_CACHE_DIR")
        os.environ["DG_JIT_CACHE_DIR"] = str(jit_cache)
        try:
            name = "vllm_target_m32._C"
            assert name not in sys.modules
            spec = importlib.util.spec_from_file_location(name, root / manifest["native"])
            assert spec is not None and spec.loader is not None
            native = importlib.util.module_from_spec(spec)
            sys.modules[name] = native
            spec.loader.exec_module(native)
            native.init(str(root))
            pdl = old.get_pdl()
            native.set_pdl(pdl)
            old_alignment = old.get_mk_alignment_for_contiguous_layout()
            native.set_mk_alignment_for_contiguous_layout(32)
            assert old.get_mk_alignment_for_contiguous_layout() == old_alignment
            native.set_mk_alignment_for_contiguous_layout(128)
            native.set_pdl(not pdl)
            assert old.get_pdl() == pdl
            native.set_pdl(pdl)
        finally:
            if prior_cache is None:
                os.environ.pop("DG_JIT_CACHE_DIR", None)
            else:
                os.environ["DG_JIT_CACHE_DIR"] = prior_cache
    mode = d05_mode()
    fmode = fc2c_mode()
    if fmode != "0":
        assert mode == "1", "DSV41_MOE_FC2C needs DSV41_MOE_FC2_D05=1"
    if mode == "0":
        return TargetM32Kernel(native)
    fc2_native = load_d05_native(native, pdl)
    verifier = _D05Verifier() if mode == "verify" else None
    fc2c = None
    if fmode != "0":
        fc2c = FC2Compact(load_fc2c_native(native, pdl), fmode, fc2c_min_rows(),
                          _D05Verifier("fc2c") if fmode == "verify" else None)
        fc2c.poison_served = g1_variant() in _G1_HAS_D05  # g2: those D05 trees store live rows only
    kernel = TargetM32Kernel(native, fc2_native, mode, verifier, fc2c)
    kernel.g1_la = g1_variant() in _G1_HAS_LA
    return kernel
