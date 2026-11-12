#!/usr/bin/env python3
"""r5-glue device test: RQPAD and RPSPLIT kernels vs the live (production) kernels, bitwise, with must-fail mutants.

Run by the lead on a STOPPED node (no serving process on its GPU), e.g. spark-eight. The environment is MANDATORY
(a non-interactive ssh shell silently tests the wrong reference); the script refuses to start without it:

    ssh spark-eight
    source ~/venvs/vllm032/bin/activate; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
    cd ~/climb-scratch/r5-glue        # a copy of this kit folder (the kit need NOT be installed)
    python3 device_test.py [--quick] [--no-timing] [--part rq|rp|hook|all]

Expected runtime: about 8-15 min full (Triton JIT ~1 min; RQ 32 M x 5 sets x 2 poisons + 4 byte mutants x 32 M + the
PDL producer cases; RP 10 M x 5 routings x 2 x-kinds x 2 id dtypes x 2 modes x 3 FC1 fronts + Q1/Q2 sweeps + 7 byte
mutant runs x 10 M + the PDL producer cases; H in a fresh process; graphs; timings), --quick about 4 min (fewer sweep
rows / routings; mutants and every gated M still run).

Must-fail mutants (each must be caught at EVERY gated M, else FAIL):
  RQ  4 byte mutants (RQ-X, M 1..32) + 1 PDL-order mutant (RQ-P, M 1,2,4,5,6)                      = 5
  RP  5 byte-mutant definitions = 7 mode-runs: split 2 (last chunk, segment padding), hoist 1 (chunk 3/4 loads
      swapped), both modes 2 (packed-word shift, scale doubled) (RP-X, M 1..10)
      + 1 PDL-order mutant per mode (RP-P, M 1,2,4,6)                                                 = 9 mode-runs

Start-up assertions (any failure = exit 2, nothing measured):
  E1 python is ~/venvs/vllm032, CUDA_HOME=/usr/local/cuda, nvcc and ptxas resolve under /usr/local/cuda/bin
  E2 the GPU is idle (nvidia-smi --query-compute-apps empty), compute capability (12, 1)
  S0 production's launcher (~/dsv41-tp8.sh) is the expected one (sha 166d7aac, or --launcher-sha) and the serving env of
     this test is READ from it, not assumed (review N6.2): DSV41_ATTNOUT_OP contains rq, DSV41_MOE_ROUTEPREP=1,
     DSV41_MOE_FC1_FRONT absent or 0, sonar SPLITS/MAX_M; the geometry comes from its model's config.json
     (o_lora_rank = N_RQ with o_groups / 8 = 1 group per rank, hidden 5120, 384 experts, top-6)
  S1 the serving reference ops are the ones under test: live dsv41_attnout.py / dsv41_routeprep.py have the production
     shas (r5-glue tagged lines stripped), sonar SPLITS=32 / MAX_M=32, routeprep static check (UE8M0, e4m3 +-448,
     mfront=1) and PDL on, and a profiler pass shows the kernels that actually run: `_r2_reduce_quant`,
     the flashinfer CuTe-DSL MXFP8 quant (`kernel_cutlass_kernel_flashinferquantization...`, i.e. not a torch
     fallback), `_dsv41_rp_kernel`, and the kit's `_r5g_reduce_quant` / `_r5g_rp_kernel`.

Cases (raw-byte comparisons; first differing byte reported):
  RQ-B  M = 1..32 (every sonar M; production serves 1,2,4,5,6) x {normal, hostile, wide, ties+cancellation, ftz-corner}
        x poisons {0xFF, 0x00}: kit q + all PADDED_M*SF_COLS SF bytes == r2 kernel's; and == live reduce + flashinfer
        quant (ftz-corner listed apart: CuTe-DSL FTZ semantics, as in the r2-attnout harness).
  RQ-W  wrapper on the device: first eager call passes its self-check, bytes == r2; a mutant module's wrapper DISABLES
        itself and serves r2 bytes.
  RQ-G  CUDA graph of the wrapper call (after a pass) at M = 1, 2, 4, 5, 6: 50 replays with poisoned outputs == eager.
  RQ-X  must-fail mutants (pad row skipped / pad value 1 / pad stride M+1 / bf16 rounding dropped): each must be caught
        at EVERY M in 1..32, else FAIL.
  RQ-P  in-stream PDL producer (review N6.1): tools/pdl_producer.late_copy (gdc_launch_dependents at its start, spins,
        THEN writes the partials) followed by the wrapper, eager and as a CUDA graph, partials poisoned before every
        run: kit == r2 bytes at M = 1, 2, 4, 5, 6; the mutant that issues the partial loads before griddepcontrol.wait
        must be caught (eager or graph) at every one of these M.
  RP-B  hoist and split, M = 1..10 (the RPSPLIT scope; production serves 1,2,4,6) x routings {live dsv4_topk, distinct,
        same, hot, with dead ids} x x-kinds {normal, hostile} x ids {int32, int64} x FC1_FRONT {0, pdl, pf2} x poisons
        {0xA5/0x5A5A5A5A, 0x00/0}: rows, packed scale words, layout, inv_perm, expert ends byte-identical to r2.
  RP-Q  Q1 (every finite bf16 x at every reachable UE8M0 scale) and Q2 (every finite bf16 absmax) rows through both
        modes at M = 10 (and M = 1 for the first rows): byte-identical to r2.
  RP-W  wrapper: self-check passes (kit) / disables + serves r2 bytes (mutant); M = 11 and 16 stay on r2.
  RP-G  CUDA graph [dsv4_topk (PDL) -> wrapper] at M = 1, 2, 4, 6 per mode: 50 poisoned replays == eager r2 bytes.
  RP-X  must-fail mutants per mode (split: last chunk skipped, segment padding skipped; hoist: chunk 3/4 loads swapped;
        both: packed-word shift, scale doubled): each must be caught at EVERY M in 1..10, else FAIL.
  RP-P  in-stream PDL producer writing x late, then the wrapper (eager and graph, x poisoned before every run): kit ==
        r2 bytes at M = 1, 2, 4, 6 per mode; the mutant that issues the hoisted x loads before griddepcontrol.wait must
        be caught (eager or graph) at every one of these M.
  H     tools/hook_import_test.py in a fresh process (review N5): install.py on a symlink copy of site-packages/vllm,
        flags on, the real import sites (deep_gemm_utils, lazy dsv41_attnout import with the circular kit import),
        _launch_rq / _launch wrapped, dsv41_attnout._launch_rq and fused_permute end to end == r2, sweep INFO lines.
  T     timings: median of 200 CUDA-graph replays (1 launch per replay, and 40 launches per replay / 40), warm and
        cold-L2 (64 MiB flush): r2 vs kit at the served M. RP also with dsv4_topk in front (the PDL chain as served).
The wrappers' first eager call runs the synthetic sweep (RQ M 1..32, RP M 1..10 x 2 routings) before any real-data
check, so RQ-W / RP-W / the mutant wrappers exercise it on the device.
Exit code 0 only when every comparison is bit-identical AND every mutant is caught at every gated M.
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_RQ = os.path.join(HERE, "files/vllm/models/deepseek_v4/nvidia/ops/dsv41_r5g_rq.py")
KIT_RP = os.path.join(HERE, "files/vllm/model_executor/layers/fused_moe/dsv41_r5g_rp.py")
LIVE_SHA = {"dsv41_attnout.py": "308942ebbed1fc121f74ea5b9820a941b7fed895edf7aae4d6d63480ba1d1bfc",
            "dsv41_routeprep.py": "842f49fb5df830793e02c12ff0195fb853165e90fac35a16ff3f136cf7b0eda7"}
TAG = "# r5-glue"
FAILS = []
N_RQ, SPLITS = 1024, 32  # N_RQ is re-read from the model config in S0 (o_lora_rank) and asserted
E, TOPK, H = 384, 6, 5120  # asserted against the model config in S0
LAUNCHER = os.path.expanduser("~/dsv41-tp8.sh")
LAUNCHER_SHA = "166d7aac"
TP = 8

RQ_MUTANTS = [
    ("pad row PADDED_M-1 skipped", "mask=(p < PADDED_M)[:, None])", "mask=(p < PADDED_M - 1)[:, None])"),
    ("pad value 1", "tl.zeros([PAD_J, 8], tl.uint8)", "tl.full([PAD_J, 8], 1, tl.uint8)"),
    ("pad stride M+1", "p = M + r + j0 + jj * M", "p = M + r + j0 + jj * (M + 1)"),
    ("bf16 rounding of the split sum dropped", "    zf = zb.to(tl.float32)  #", "    zf = ev + od  #"),
]
# PDL-order mutants: the loads move above griddepcontrol.wait (bytes identical without an in-stream producer)
RQ_PDL_MUTANT = ("partial loads before gdc_wait", [
    ("        tl.extra.cuda.gdc_wait()\n    cols = ", "        pass\n    cols = "),
    ("    zb = (ev + od).to(tl.bfloat16)", "    if USE_PDL:\n        tl.extra.cuda.gdc_wait()\n    zb = (ev + od).to(tl.bfloat16)"),
])
RP_PDL_MUTANT = ("hoisted x loads before gdc_wait", [
    ("        tl.extra.cuda.gdc_wait()\n\n    # ---- r5-glue: issue", "        pass\n\n    # ---- r5-glue: issue"),
    ("\n    # ---- plan: the same integer formulas",
     "\n    if launch_pdl:\n        tl.extra.cuda.gdc_wait()\n    # ---- plan: the same integer formulas"),
])
RP_MUTANTS = [  # (name, modes, old, new)
    ("split: last chunk not quantized", ("split",),
     "            _r5g_rp_chunk(xa, y, orow,", "            if y < 4:\n                _r5g_rp_chunk(xa, y, orow,"),
    ("split: segment padding never written", ("split",), "    if first & is0:\n        r = start_e",
     "    if first & (y > 9):\n        r = start_e"),
    ("hoist: chunk 3/4 loads swapped", ("hoist",), "xa3 = tl.load(xrow + _r5g_offs(3, GB, GROUP))",
     "xa3 = tl.load(xrow + _r5g_offs(4, GB, GROUP))"),
    ("packed word v3 << 2", ("hoist", "split"), "(v3 << 1)\n", "(v3 << 2)\n"),
    ("scale doubled", ("hoist", "split"), "y = tl.math.div_rn(amax, fmax_v)", "y = tl.math.div_rn(amax * 2.0, fmax_v)"),
]


def say(*a):
    print(*a, flush=True)


def check(name, ok, detail=""):
    say(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


# ---------------------------------------------------------------------------------------------- environment rule
def env_guard(assume_stopped):
    bad = []
    venv = os.path.realpath(os.path.expanduser("~/venvs/vllm032"))
    if os.path.realpath(sys.prefix) != venv:
        bad.append(f"python prefix {sys.prefix} is not {venv}: run `source ~/venvs/vllm032/bin/activate`")
    if os.environ.get("CUDA_HOME") != "/usr/local/cuda":
        bad.append(f"CUDA_HOME={os.environ.get('CUDA_HOME')!r}: export CUDA_HOME=/usr/local/cuda")
    for tool in ("nvcc", "ptxas"):
        p = shutil.which(tool)
        if not p or not os.path.realpath(p).startswith(os.path.realpath("/usr/local/cuda")):
            bad.append(f"{tool} resolves to {p}: export PATH=/usr/local/cuda/bin:$PATH")
    if bad:
        say("E1 ENVIRONMENT RULE VIOLATED (nothing measured):\n  " + "\n  ".join(bad))
        sys.exit(2)
    say(f"PASS E1 environment: {sys.executable}, CUDA_HOME=/usr/local/cuda, nvcc/ptxas from /usr/local/cuda/bin")
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        apps = [ln for ln in r.stdout.splitlines() if ln.strip() and str(os.getpid()) not in ln]
        q_ok = r.returncode == 0
    except Exception as e:  # noqa: BLE001
        apps, q_ok = [], False
        say(f"WARN nvidia-smi query failed: {e!r}")
    if apps or (not q_ok and not assume_stopped):
        why = apps if apps else "query failed; pass --assume-stopped only if you checked by hand"
        say(f"E2 GPU NOT IDLE or not queryable (stopped node required): {why}")
        sys.exit(2)
    import torch
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    if cap != (12, 1):
        say(f"E2 device capability {cap} != (12, 1) (GB10)")
        sys.exit(2)
    say(f"PASS E2 GPU idle, {torch.cuda.get_device_name()} cc {cap}")


# ------------------------------------------------------------------------------------- S0: production's launcher
def launcher_env(path):
    """(sha, {VAR: default}) of the launcher's `export VAR=${VAR:-default}` / `export VAR=value` lines; the MODEL
    default too. Read-only."""
    import re
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    env = {}
    pat = re.compile(r"(?:^|\s)([A-Z][A-Z0-9_]*)=(?:\$\{[A-Z][A-Z0-9_]*:-([^}]*)\}|([^\s#$]*))")
    for ln in raw.decode("utf-8", "replace").splitlines():
        if not re.match(r"\s*(?:export\s+)?[A-Z][A-Z0-9_]*=", ln):
            continue  # only plain assignment / export lines (several per line allowed)
        for m in pat.finditer(ln.split(" #", 1)[0]):
            env[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return sha, env


def s0_launcher(want_sha):
    global N_RQ
    import json
    if not os.path.exists(LAUNCHER):
        check(f"S0 production launcher {LAUNCHER} present", False)
        return {}
    sha, env = launcher_env(LAUNCHER)
    check(f"S0 launcher sha {sha[:8]} == {want_sha} (production)", sha.startswith(want_sha), sha[:16])
    need = {"DSV41_ATTNOUT_OP": lambda v: "rq" in v.split(","), "DSV41_MOE_ROUTEPREP": lambda v: v == "1",
            "DSV41_SONAR_OPROJ": lambda v: v == "fp8", "DSV41_SONAR_OPROJ_SPLITS": lambda v: v == "32",
            "DSV41_SONAR_OPROJ_MAX_M": lambda v: v == "32"}
    for k, ok in need.items():
        check(f"S0 launcher {k}={env.get(k)!r} (the served configuration this test reproduces)",
              k in env and ok(env[k]))
    check("S0 launcher DSV41_MOE_FC1_FRONT absent or 0 (production front off; pdl/pf2 are tested as arms)",
          env.get("DSV41_MOE_FC1_FRONT", "0") in ("", "0"), str(env.get("DSV41_MOE_FC1_FRONT")))
    model = os.path.expandvars(os.path.expanduser(env.get("MODEL", "").replace("$HOME", os.path.expanduser("~"))))
    cfgp = os.path.join(model, "config.json")
    try:
        c = json.load(open(cfgp))
        c = c.get("text_config", c)
        n_rq = int(c["o_lora_rank"])
        geo = (int(c["o_groups"]) // TP, int(c["hidden_size"]), int(c["n_routed_experts"]),
               int(c["num_experts_per_tok"]))
    except Exception as e:  # noqa: BLE001
        check(f"S0 model config {cfgp} readable (o_lora_rank, o_groups, hidden_size, n_routed_experts, "
              "num_experts_per_tok)", False, repr(e))
        return {}
    check(f"S0 model config: o_lora_rank {n_rq} (= the rq N), o_groups/{TP} = 1, hidden {H}, {E} experts, top-{TOPK}",
          geo == (1, H, E, TOPK) and n_rq % 256 == 0, f"o_lora_rank {n_rq}, (groups/rank, H, E, topk) {geo}")
    N_RQ = n_rq
    keys = list(need) + ["DSV41_MOE_FUSED_PERMUTE", "DSV41_ROUTE_LIVE", "VLLM_USE_DEEP_GEMM"]
    return {k: env[k] for k in keys if k in env}


def stripped_sha(path):
    src = open(path, encoding="utf-8", newline="").read()
    src = "".join(ln for ln in src.splitlines(keepends=True) if not ln.rstrip("\r\n").endswith(TAG))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def load_path(name, path, env):
    for k in ("DSV41_GLUE_RQPAD", "DSV41_GLUE_RPSPLIT"):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mutant_module(path, name, old, new, env):
    """old/new: one replacement, or old = [(old, new), ...] and new = None for several."""
    src = open(path).read()
    for o, n in ([(old, new)] if new is not None else old):
        if src.count(o) != 1:
            raise SystemExit(f"mutant {name!r}: anchor found {src.count(o)}x (kit source changed?)")
        src = src.replace(o, n)
    d = tempfile.mkdtemp(prefix="r5g-mut-")
    p = os.path.join(d, os.path.basename(path).replace(".py", "_mut.py"))
    open(p, "w").write(src)
    return load_path("mut_" + "".join(c if c.isalnum() else "_" for c in name), p, env)


def kernel_names(fn):
    import torch
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if e.device_type.name == "CUDA"}


def u8(t):
    return t.contiguous().view(__import__("torch").uint8).flatten()


def first_diff(a, b):
    import torch
    a8, b8 = u8(a), u8(b)
    if a8.shape != b8.shape:
        return f"size {a8.numel()} vs {b8.numel()}"
    idx = torch.nonzero(a8 != b8)
    return None if idx.numel() == 0 else f"byte {int(idx[0])} of {a8.numel()} ({idx.numel()} differ)"


# -------------------------------------------------------------------------------------------------------------- RQ
def rq_sets(M, gen, ao):
    import torch
    sets = {"normal": torch.randn((SPLITS, 1, M, N_RQ), generator=gen) * 0.02,
            "hostile": ao._hostile_partials((SPLITS, 1, M, N_RQ), gen),
            "wide": torch.randn((SPLITS, 1, M, N_RQ), generator=gen) * torch.exp2(
                torch.randint(-40, 40, (SPLITS, 1, M, N_RQ), generator=gen).float())}
    t = torch.zeros((SPLITS, 1, M, N_RQ))
    t[0] = 1.0 + 2.0 ** -8  # bf16 ties on every element
    t[1, ..., ::3] = -(1.0 + 2.0 ** -8)  # exact cancellation on a third
    t[2, ..., 1::7] = 448.0 * 2.0 ** 10
    sets["ties+cancel"] = t
    sets["ftz-corner"] = torch.randn((SPLITS, 1, M, N_RQ), generator=gen) * 2.0 ** -126
    return {k: v.float().cuda() for k, v in sets.items()}


def rq_pair(kit, P, M, poison):
    import torch
    padded = (M + 127) // 128 * 128
    q_r = torch.full((M, N_RQ), poison, dtype=torch.uint8, device="cuda")
    sf_r = torch.full((padded * (N_RQ // 32),), poison, dtype=torch.uint8, device="cuda")
    q_n, sf_n = q_r.clone(), sf_r.clone()
    kit._launch_r2_into(P, q_r, sf_r, M, N_RQ, SPLITS)
    kit._launch_new_into(P, q_n, sf_n, M, N_RQ, SPLITS)
    torch.cuda.synchronize()
    return (q_r, sf_r), (q_n, sf_n)


def rq_diff(kit, P, M):
    for poison in (0xFF, 0x00):
        (q_r, sf_r), (q_n, sf_n) = rq_pair(kit, P, M, poison)
        for nm, a, b in (("q", q_r, q_n), ("sf", sf_r, sf_n)):
            d = first_diff(a, b)
            if d:
                return f"poison {poison:#x} {nm}: {d}"
    return None


def case_rq(ao, quick, timing):
    import torch
    say("== RQPAD")
    kit = load_path("r5g_rq_kit", KIT_RQ, {"DSV41_GLUE_RQPAD": "1"})
    gen = torch.Generator().manual_seed(20261109)
    names = kernel_names(lambda: rq_pair(kit, rq_sets(1, gen, ao)["normal"], 1, 0))
    check("S1 RQ kernels that run: _r2_reduce_quant and _r5g_reduce_quant",
          any("_r2_reduce_quant" in n for n in names) and any("_r5g_reduce_quant" in n for n in names), str(names))
    # B: bitwise vs r2 and vs flashinfer
    n = 0
    bad, ftz = [], []
    for M in range(1, 33):
        for sname, P in rq_sets(M, gen, ao).items():
            d = rq_diff(kit, P, M)
            n += 1
            if d:
                bad.append(f"M={M} {sname}: {d}")
            (_, _), (q_n, sf_n) = rq_pair(kit, P, M, 0xFF)
            z = torch.empty((M, 1, N_RQ), device="cuda", dtype=torch.bfloat16)
            ao._launch_live_reduce(P, z, M, 1, N_RQ, SPLITS)
            q_fi, sf_fi = ao._live_quant(z.flatten(1))
            dq, ds = first_diff(q_n, q_fi), first_diff(sf_n, sf_fi)
            if dq or ds:
                (ftz if sname == "ftz-corner" else bad).append(f"M={M} {sname} vs flashinfer: q {dq} sf {ds}")
    check(f"RQ-B kit == r2 kernel (q + every SF byte, 2 poisons) and == live reduce + flashinfer quant: {n} cases, "
          "M 1..32 x 5 sets", not bad, "; ".join(bad[:6]))
    if ftz:
        say(f"INFO RQ-B ftz-corner vs flashinfer differences (same in the r2 kernel; not an RQPAD effect): {ftz[:3]}")
    # W: wrapper self-check on device
    w = kit.wrap(ao._launch_rq.__wrapped__ if hasattr(ao._launch_rq, "__wrapped__") else ao._launch_rq)
    P = rq_sets(1, gen, ao)["normal"]
    q, sf = w(P, 1, N_RQ, SPLITS)
    q2, sf2 = (ao._launch_rq.__wrapped__ if hasattr(ao._launch_rq, "__wrapped__") else ao._launch_rq)(P, 1, N_RQ,
                                                                                                   SPLITS)
    torch.cuda.synchronize()
    check("RQ-W kit wrapper: eager self-check passes, bytes == r2", kit._ok is True and not first_diff(q, q2)
          and not first_diff(sf, sf2))
    # G: graph replays of the wrapper after a pass
    for M in (1, 2, 4, 5, 6):
        P = rq_sets(M, gen, ao)["hostile"]
        w(P, M, N_RQ, SPLITS)  # eager check at this M
        (q_r, sf_r), _ = rq_pair(kit, P, M, 0xFF)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            w(P, M, N_RQ, SPLITS)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            qg, sfg = w(P, M, N_RQ, SPLITS)
        ok = True
        for _ in range(50):
            qg.view(torch.uint8).fill_(0xFF)
            sfg.fill_(0xFF)
            g.replay()
            torch.cuda.synchronize()
            if first_diff(qg, q_r) or first_diff(sfg, sf_r):
                ok = False
                break
        check(f"RQ-G graph of the wrapper M={M}: 50 poisoned replays == r2 bytes", ok)
        del g
    # X: mutants, every M
    for name, old, new in RQ_MUTANTS:
        mut = mutant_module(KIT_RQ, "rq " + name, old, new, {"DSV41_GLUE_RQPAD": "1"})
        missed = [M for M in range(1, 33) if not rq_diff(mut, rq_sets(M, gen, ao)["normal"], M)]
        check(f"RQ-X mutant '{name}' caught at every M 1..32", not missed, f"NOT caught at M={missed}")
        if name == RQ_MUTANTS[0][0]:
            wm = mut.wrap(ao._launch_rq.__wrapped__ if hasattr(ao._launch_rq, "__wrapped__") else ao._launch_rq)
            P = rq_sets(2, gen, ao)["normal"]
            q, sf = wm(P, 2, N_RQ, SPLITS)
            (q_r, sf_r), _ = rq_pair(kit, P, 2, 0xFF)
            check("RQ-W mutant wrapper: self-check DISABLES RQPAD and serves r2 bytes",
                  mut._ok is False and not first_diff(q, q_r) and not first_diff(sf, sf_r))
    case_rq_pdl(ao, gen)
    if timing:
        flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        say("T RQ  M | r2 1-launch warm / cold | kit warm / cold | r2 40x / kit 40x (us per launch, graph replay)")
        for M in (1, 2, 3, 4, 5, 6, 16, 32):
            P = rq_sets(M, gen, ao)["normal"]
            (q, sf), _ = rq_pair(kit, P, M, 0)
            cells = []
            for fn in (lambda: kit._launch_r2_into(P, q, sf, M, N_RQ, SPLITS),
                       lambda: kit._launch_new_into(P, q, sf, M, N_RQ, SPLITS)):
                cells += [bench(fn, flush, 1, False), bench(fn, flush, 1, True)]
            for fn in (lambda: kit._launch_r2_into(P, q, sf, M, N_RQ, SPLITS),
                       lambda: kit._launch_new_into(P, q, sf, M, N_RQ, SPLITS)):
                cells.append(bench(fn, flush, 40, False))
            say(f"T RQ {M:2d} | {cells[0]:6.2f} {cells[1]:6.2f} | {cells[2]:6.2f} {cells[3]:6.2f} | "
                f"{cells[4]:6.2f} {cells[5]:6.2f}   delta(40x) {cells[5] - cells[4]:+6.2f}")


def case_rq_pdl(ao, gen):
    """RQ-P: [late-writing PDL producer -> wrapper] eager and as a graph; kit passes, PDL-order mutant is caught."""
    import torch
    from tools.pdl_producer import late_copy
    say("== RQ-P (in-stream PDL producer)")
    orig = getattr(ao._launch_rq, "__wrapped__", ao._launch_rq)
    kit = load_path("r5g_rq_pdl_kit", KIT_RQ, {"DSV41_GLUE_RQPAD": "1"})
    mut = mutant_module(KIT_RQ, "rq " + RQ_PDL_MUTANT[0], RQ_PDL_MUTANT[1], None, {"DSV41_GLUE_RQPAD": "1"})
    for label, mod in (("kit", kit), ("mutant", mut)):
        w = mod.wrap(orig)
        res = {}
        for M in (1, 2, 4, 5, 6):
            src = rq_sets(M, gen, ao)["hostile"]
            P = torch.empty_like(src)
            w(src, M, N_RQ, SPLITS)  # first eager call(s): sweep + real-data check at M
            (q_r, sf_r), _ = rq_pair(kit, src, M, 0xFF)
            P.view(torch.int32).fill_(-1)  # NaN poison: a premature read changes every byte it feeds
            late_copy(src, P)
            q, sf = w(P, M, N_RQ, SPLITS)
            torch.cuda.synchronize()
            eager_ok = not first_diff(q, q_r) and not first_diff(sf, sf_r)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                late_copy(src, P)
                w(P, M, N_RQ, SPLITS)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                late_copy(src, P)
                qg, sfg = w(P, M, N_RQ, SPLITS)
            graph_ok = True
            for _ in range(20):
                P.view(torch.int32).fill_(-1)
                qg.view(torch.uint8).fill_(0xFF)
                sfg.fill_(0xFF)
                g.replay()
                torch.cuda.synchronize()
                if first_diff(qg, q_r) or first_diff(sfg, sf_r):
                    graph_ok = False
                    break
            del g
            res[M] = (eager_ok, graph_ok)
        if label == "kit":
            check("RQ-P kit: [late PDL producer -> wrapper] == r2 bytes, eager and 20 graph replays, M 1,2,4,5,6",
                  mod._ok is True and all(e and g for e, g in res.values()), str(res))
        else:
            missed = [M for M, (e, g) in res.items() if e and g]
            check(f"RQ-P mutant '{RQ_PDL_MUTANT[0]}' caught (eager or graph) at every M 1,2,4,5,6", not missed,
                  f"NOT caught at M={missed}; (eager_ok, graph_ok) {res}")


def case_rp_pdl(r2_launch, rng):
    """RP-P: [late-writing PDL producer of x -> wrapper] eager and as a graph, per mode; PDL-order mutant caught."""
    import torch
    from tools.pdl_producer import late_copy
    say("== RP-P (in-stream PDL producer of x)")
    for mode in ("hoist", "split"):
        kit = load_path(f"r5g_rp_pdl_{mode}", KIT_RP, {"DSV41_GLUE_RPSPLIT": mode})
        mut = mutant_module(KIT_RP, f"rp {mode} {RP_PDL_MUTANT[0]}", RP_PDL_MUTANT[1], None,
                            {"DSV41_GLUE_RPSPLIT": mode})
        for label, mod in (("kit", kit), ("mutant", mut)):
            w = mod.wrap(r2_launch)
            res = {}
            for M in (1, 2, 4, 6):
                Ms = m_sum(M)
                src = hostile_x(M, rng)
                ids = synth_ids(M, "hot", rng, torch.int32)
                x = torch.empty_like(src)
                rp_launch(w, src, ids, M, Ms, rp_bufs(M, Ms, 0, 0))  # sweep + real-data check at M
                ref = rp_bufs(M, Ms, 0x11, 0x11111111)
                rp_launch(r2_launch, src, ids, M, Ms, ref)
                bufs = rp_bufs(M, Ms, 0x11, 0x11111111)
                x.view(torch.int16).fill_(-1)  # bf16 NaN poison
                late_copy(src, x)
                rp_launch(w, x, ids, M, Ms, bufs)
                torch.cuda.synchronize()
                eager_ok = not any(first_diff(bufs[k], ref[k]) for k in ("out", "sfp", "layout", "inv", "ends"))
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    late_copy(src, x)
                    rp_launch(w, x, ids, M, Ms, bufs)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    late_copy(src, x)
                    rp_launch(w, x, ids, M, Ms, bufs)
                graph_ok = True
                for _ in range(20):
                    x.view(torch.int16).fill_(-1)
                    for k in ("out", "sfp", "layout", "inv", "ends"):
                        bufs[k].fill_(0x11 if k == "out" else 0x11111111)
                    g.replay()
                    torch.cuda.synchronize()
                    if any(first_diff(bufs[k], ref[k]) for k in ("out", "sfp", "layout", "inv", "ends")):
                        graph_ok = False
                        break
                del g
                res[M] = (eager_ok, graph_ok)
            if label == "kit":
                check(f"RP-P {mode} kit: [late PDL producer of x -> wrapper] == r2 bytes, eager and 20 graph replays, "
                      "M 1,2,4,6", mod._ok is True and all(e and g for e, g in res.values()), str(res))
            else:
                missed = [M for M, (e, g) in res.items() if e and g]
                check(f"RP-P {mode} mutant '{RP_PDL_MUTANT[0]}' caught (eager or graph) at every M 1,2,4,6",
                      not missed, f"NOT caught at M={missed}; (eager_ok, graph_ok) {res}")


def case_hook():
    say("== H (installed hook path, fresh process)")
    for mode in ("split", "hoist"):
        r = subprocess.run([sys.executable, os.path.join(HERE, "tools/hook_import_test.py"), "--mode", mode],
                           capture_output=True, text=True, timeout=3600)
        for ln in r.stdout.splitlines():
            if ln.startswith(("PASS", "FAIL", "HOOK")):
                say("   " + ln)
        check(f"H installed hook path, DSV41_GLUE_RPSPLIT={mode} + RQPAD (tools/hook_import_test.py exit 0)",
              r.returncode == 0 and "HOOK ALL PASS" in r.stdout, (r.stdout + r.stderr)[-1500:])


# -------------------------------------------------------------------------------------------------------------- RP
def m_sum(M):
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M_and_alignment
    ms, al = compute_aligned_M_and_alignment(M, TOPK, E, 32, None)
    assert al == 32, al
    return ms


def rp_bufs(M, Ms, p8, p32, inv_like=None):
    import torch
    mn = (Ms + 3) // 4 * 4
    return dict(out=torch.full((Ms, H), p8, dtype=torch.uint8, device="cuda"),
                sfp=torch.full((10 * mn,), p32, dtype=torch.int32, device="cuda"),
                layout=torch.full((Ms,), p32, dtype=torch.int32, device="cuda"),
                inv=torch.full((M, TOPK), p32, dtype=torch.int32, device="cuda"),
                ends=torch.full((E,), p32, dtype=torch.int32, device="cuda"), mn=mn)


def rp_launch(fn, x, ids, M, Ms, b, w1=None, sf=None, **kw):
    return fn(x, x.stride(0), ids, ids.stride(0), ids.stride(1), b["out"], b["out"].stride(0), b["sfp"], b["mn"],
              b["layout"], b["inv"], b["inv"].stride(0), b["inv"].stride(1), b["ends"], w1, sf, M, Ms, **kw)


def rp_diff(r2_launch, kit, split, x, ids, M, w1=None, sf=None):
    import torch
    Ms = m_sum(M)
    for p8, p32 in ((0xA5, 0x5A5A5A5A), (0x00, 0)):
        a, b = rp_bufs(M, Ms, p8, p32), rp_bufs(M, Ms, p8, p32)
        rp_launch(r2_launch, x, ids, M, Ms, a, w1, sf)
        rp_launch(kit._launch_new, x, ids, M, Ms, b, w1, sf, split=split)
        torch.cuda.synchronize()
        for k in ("out", "sfp", "layout", "inv", "ends"):
            d = first_diff(a[k], b[k])
            if d:
                return f"poison {p8:#x} {k}: {d}"
    return None


def hostile_x(M, rng):
    import numpy as np
    import torch
    r = rng.standard_normal((M, H)).astype(np.float32)
    kinds = [r, (r * np.exp2(rng.integers(-60, 60, size=(M, 1)))).astype(np.float32),
             (r * np.exp2(rng.integers(-130, 120, size=(M, H)))).astype(np.float32)]
    z = np.zeros((M, H), np.float32)
    z[:, ::2] = -0.0
    kinds.append(z)
    kinds.append(np.repeat(448.0 * np.exp2(rng.integers(-30, 30, size=(M, 40))), 128, axis=1).astype(np.float32))
    kinds.append(np.full((M, H), 3.3895e38, np.float32) * rng.choice([-1, 1], size=(M, H)).astype(np.float32))
    kinds.append((rng.integers(1, 128, size=(M, H)) * 2.0 ** -133).astype(np.float32))
    k = kinds[rng.integers(len(kinds))]
    with np.errstate(all="ignore"):
        k = np.where(np.isfinite(k), k, 0).astype(np.float32)
    return torch.from_numpy(k).to(torch.bfloat16).cuda()


def synth_ids(M, kind, rng, dtype):
    import numpy as np
    import torch
    if kind == "same":
        ids = np.tile(rng.choice(E, TOPK, replace=False), (M, 1))
    elif kind == "distinct":
        p = np.concatenate([rng.permutation(E) for _ in range(M * TOPK // E + 1)])
        ids = p[:M * TOPK].reshape(M, TOPK)
    elif kind == "hot":
        ids = np.stack([rng.choice(np.r_[np.arange(8), rng.choice(np.arange(8, E), TOPK, replace=False)], TOPK,
                                   replace=False) for _ in range(M)])
    elif kind == "dead":
        ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(M)])
        ids[rng.random(ids.shape) < 0.3] = -1
    else:
        raise KeyError(kind)
    return torch.from_numpy(ids).to(dtype).cuda()


def topk_ids(M, dtype):
    import torch
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32) * 3
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.01
    pad = torch.zeros(M, dtype=torch.bool, device="cuda")
    _, ids = dsv4_topk(logits, bias, dtype, 2.5, is_padding=pad)
    return ids, (logits, bias, pad)


def q_rows():
    """Q1: every finite bf16 x at every reachable scale; Q2: every finite bf16 absmax (as in the r2-moefront harness)."""
    import numpy as np
    b = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    fin = b[((b >> 7) & 0xFF) != 0xFF]
    fv = (fin.astype(np.uint32) << 16).view(np.float32)
    pos = fin[(fv > 0) | (((fin & 0x8000) == 0) & (fv == 0))]
    pv = (pos.astype(np.uint32) << 16).view(np.float32)
    y = np.maximum(np.maximum(pv, np.float32(1e-10)) / np.float32(448.0), np.float32(1e-10)).astype(np.float32)
    yb = y.view(np.uint32)
    sc = np.where((yb & 0x7FFFFF) != 0, (yb & 0x7F800000) + 0x800000, yb & 0x7F800000)
    groups = []
    for s in np.unique(sc):
        A = pos[sc == s].max()
        Av = (np.array([A], dtype=np.uint32) << 16).view(np.float32)[0]
        xs = fin[np.abs(fv) <= Av]
        n = -(-xs.size // 127)
        groups.append(np.concatenate([np.full((n, 1), A, dtype=np.uint16), np.resize(xs, n * 127).reshape(n, 127)], 1))
    g1 = np.concatenate(groups, 0)
    g2 = np.zeros((fin.size, 128), dtype=np.uint16)
    g2[:, 5] = fin
    out = []
    for g in (g1, g2):
        r = -(-g.shape[0] // 40)
        g = np.concatenate([g, np.zeros((r * 40 - g.shape[0], 128), dtype=np.uint16)], 0)
        out.append(g.reshape(r, 5120))
    return out


def fake_fc1(dev="cuda"):
    import torch
    w1 = torch.randint(-128, 128, (E, 576, H // 2), device=dev, dtype=torch.int8)
    sf = torch.empty_strided((E, 576, H // 128), (576 * (H // 128), 1, 576), device=dev, dtype=torch.int32)
    sf.fill_(0x7F7F7F7F)
    return w1, sf


def case_rp(quick, timing):
    import numpy as np
    import torch
    say("== RPSPLIT")
    rng = np.random.default_rng(20261109)
    parents = {}
    for front in ("0", "pdl", "pf2"):
        os.environ["DSV41_MOE_ROUTEPREP"] = "1"
        os.environ["DSV41_MOE_FC1_FRONT"] = front
        import vllm.model_executor.layers.fused_moe as fm
        parents[front] = load_path(f"live_rp_{front}", os.path.join(os.path.dirname(fm.__file__),
                                                                    "dsv41_routeprep.py"), {})
    os.environ["DSV41_MOE_FC1_FRONT"] = "0"
    live = parents["0"]
    check("S1 RP routeprep static check (UE8M0, e4m3 +-448, mfront=1) and PDL on", live._static_ok() and live._PDL)
    r2_launch = getattr(live._launch, "__wrapped__", live._launch)
    kits = {mode: load_path(f"r5g_rp_{mode}", KIT_RP, {"DSV41_GLUE_RPSPLIT": mode}) for mode in ("hoist", "split")}
    w1, sf = fake_fc1()
    x = torch.randn(4, H, device="cuda").to(torch.bfloat16)
    ids = synth_ids(4, "distinct", rng, torch.int32)
    names = kernel_names(lambda: [rp_diff(r2_launch, kits[m], m == "split", x, ids, 4) for m in kits])
    check("S1 RP kernels that run: _dsv41_rp_kernel and _r5g_rp_kernel",
          any("_dsv41_rp_kernel" in n for n in names) and any("_r5g_rp_kernel" in n for n in names), str(names))
    # B
    for mode, kit in kits.items():
        bad, n = [], 0
        for front in ("0", "pdl", "pf2"):
            kit._rp = parents[front]  # the kit reads EARLY_TRIGGER / PF_KB / _pf_args from its parent module
            r2l = getattr(parents[front]._launch, "__wrapped__", parents[front]._launch)
            kinds = ["topk", "distinct", "same", "hot", "dead"] if not quick or front == "0" else ["topk", "same"]
            for M in range(1, 11):
                for kind in kinds:
                    for dtype in (torch.int32, torch.int64):
                        ids = topk_ids(M, dtype)[0] if kind == "topk" else synth_ids(M, kind, rng, dtype)
                        for xk in ("normal", "hostile"):
                            x = torch.randn(M, H, device="cuda").to(torch.bfloat16) if xk == "normal" else \
                                hostile_x(M, rng)
                            d = rp_diff(r2l, kit, mode == "split", x, ids, M,
                                        *((w1, sf) if front == "pf2" else (None, None)))
                            n += 1
                            if d:
                                bad.append(f"front={front} M={M} {kind} {dtype} {xk}: {d}")
        kit._rp = live
        check(f"RP-B {mode}: rows / scale words / layout / inv_perm / ends == r2 kernel, {n} cases (M 1..10, "
              "FC1_FRONT 0/pdl/pf2, 2 poisons)", not bad, "; ".join(bad[:5]))
    # Q sweeps
    q1, q2 = q_rows()
    lim = 200 if quick else None
    for mode, kit in kits.items():
        for qn, rows in (("Q1", q1), ("Q2", q2)):
            rows = rows[:lim] if lim else rows
            xa = torch.from_numpy(rows.astype(np.int16)).view(torch.bfloat16).cuda()
            bad = []
            for i in range(0, xa.shape[0], 10):
                x = xa[i:i + 10].contiguous()
                M = x.shape[0]
                d = rp_diff(r2_launch, kit, mode == "split", x, synth_ids(M, "distinct", rng, torch.int32), M)
                if d:
                    bad.append(f"rows {i}..{i + M}: {d}")
            d1 = rp_diff(r2_launch, kit, mode == "split", xa[:1].contiguous(), synth_ids(1, "distinct", rng,
                                                                                         torch.int32), 1)
            check(f"RP-Q {mode} {qn}: {xa.shape[0]} rows byte-identical", not bad and not d1, "; ".join(bad[:3]))
    # W: wrapper on device (kit passes; mutant disables and serves r2); out-of-scope M stays r2
    for mode in ("hoist", "split"):
        kit = load_path(f"r5g_rp_w_{mode}", KIT_RP, {"DSV41_GLUE_RPSPLIT": mode})
        w = kit.wrap(r2_launch)
        M = 4
        Ms = m_sum(M)
        x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
        ids = synth_ids(M, "hot", rng, torch.int32)
        a, b = rp_bufs(M, Ms, 0, 0), rp_bufs(M, Ms, 0, 0)
        rp_launch(w, x, ids, M, Ms, a)
        rp_launch(r2_launch, x, ids, M, Ms, b)
        torch.cuda.synchronize()
        check(f"RP-W {mode} wrapper: eager self-check passes, bytes == r2",
              kit._ok is True and all(not first_diff(a[k], b[k]) for k in ("out", "sfp", "layout", "inv", "ends")))
        for M in (11, 16):
            Ms = m_sum(M)
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            ids = synth_ids(M, "distinct", rng, torch.int32)
            names = kernel_names(lambda: rp_launch(w, x, ids, M, Ms, rp_bufs(M, Ms, 0, 0)))
            check(f"RP-W {mode} M={M} (out of scope) runs the r2 kernel only",
                  any("_dsv41_rp_kernel" in n for n in names) and not any("_r5g_rp_kernel" in n for n in names))
        # G: graph [dsv4_topk -> wrapper] after the pass
        from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
        for M in (1, 2, 4, 6):
            Ms = m_sum(M)
            ids_e, (lg, bias, pad) = topk_ids(M, torch.int32)
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            ref = rp_bufs(M, Ms, 0x11, 0x11111111)
            rp_launch(r2_launch, x, ids_e, M, Ms, ref)
            rp_launch(w, x, ids_e, M, Ms, rp_bufs(M, Ms, 0, 0))  # eager check at this M
            bufs = rp_bufs(M, Ms, 0x11, 0x11111111)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _, i2 = dsv4_topk(lg, bias, torch.int32, 2.5, is_padding=pad)
                rp_launch(w, x, i2, M, Ms, bufs)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _, ig = dsv4_topk(lg, bias, torch.int32, 2.5, is_padding=pad)
                rp_launch(w, x, ig, M, Ms, bufs)
            ok = True
            for _ in range(50):
                for k in ("out", "sfp", "layout", "inv", "ends"):
                    bufs[k].fill_(0x11 if k == "out" else 0x11111111)
                ig.fill_(0)
                g.replay()
                torch.cuda.synchronize()
                if not torch.equal(ig, ids_e) or any(first_diff(bufs[k], ref[k]) for k in
                                                     ("out", "sfp", "layout", "inv", "ends")):
                    ok = False
                    break
            check(f"RP-G {mode} graph [dsv4_topk -> wrapper] M={M}: 50 poisoned replays == r2 bytes", ok)
            del g
    # X: mutants, every M in scope
    for name, modes, old, new in RP_MUTANTS:
        for mode in modes:
            mut = mutant_module(KIT_RP, f"rp {mode} {name}", old, new, {"DSV41_GLUE_RPSPLIT": mode})
            missed = []
            for M in range(1, 11):
                x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
                if not rp_diff(r2_launch, mut, mode == "split", x, synth_ids(M, "distinct", rng, torch.int32), M):
                    missed.append(M)
            check(f"RP-X {mode} mutant '{name}' caught at every M 1..10", not missed, f"NOT caught at M={missed}")
            if name == "packed word v3 << 2":
                wm = mut.wrap(r2_launch)
                M = 2
                Ms = m_sum(M)
                x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
                ids = synth_ids(M, "distinct", rng, torch.int32)
                a, b = rp_bufs(M, Ms, 0, 0), rp_bufs(M, Ms, 0, 0)
                rp_launch(wm, x, ids, M, Ms, a)
                rp_launch(r2_launch, x, ids, M, Ms, b)
                torch.cuda.synchronize()
                check(f"RP-W {mode} mutant wrapper: self-check DISABLES RPSPLIT and serves r2 bytes",
                      mut._ok is False and all(not first_diff(a[k], b[k]) for k in ("out", "sfp", "layout", "inv",
                                                                                      "ends")))
    case_rp_pdl(r2_launch, rng)
    if timing:
        from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
        flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        say("T RP  M | 1 launch warm/cold: r2 | hoist | split || 40x: r2 | hoist | split || with dsv4_topk: r2 | "
            "hoist | split  (us, graph replay median of 200)")
        for M in (1, 2, 3, 4, 5, 6, 8, 10):
            Ms = m_sum(M)
            ids, (lg, bias, pad) = topk_ids(M, torch.int32)
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            b = rp_bufs(M, Ms, 0, 0)
            fns = [lambda: rp_launch(r2_launch, x, ids, M, Ms, b),
                   lambda: rp_launch(kits["hoist"]._launch_new, x, ids, M, Ms, b, split=False),
                   lambda: rp_launch(kits["split"]._launch_new, x, ids, M, Ms, b, split=True)]
            one = [f"{bench(f, flush, 1, False):5.2f}/{bench(f, flush, 1, True):5.2f}" for f in fns]
            forty = [f"{bench(f, flush, 40, False):5.2f}" for f in fns]

            def tk(f):
                def run():
                    dsv4_topk(lg, bias, torch.int32, 2.5, is_padding=pad)
                    f()
                return run
            withtk = [f"{bench(tk(f), flush, 1, False):5.2f}" for f in fns]
            say(f"T RP {M:2d} | {' | '.join(one)} || {' | '.join(forty)} || {' | '.join(withtk)}")


# ---------------------------------------------------------------------------------------------------------- timing
def bench(fn, flush, reps, cold, n=200):
    import torch
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(n):
        if cold:
            flush.add_(1)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / reps)
    del g
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--part", default="all", choices=("rq", "rp", "hook", "all"))
    ap.add_argument("--launcher-sha", default=LAUNCHER_SHA, help="expected production launcher sha prefix")
    ap.add_argument("--assume-stopped", action="store_true",
                    help="only when nvidia-smi cannot be queried and you checked by hand that no server runs")
    args = ap.parse_args()
    t0 = time.time()
    env_guard(args.assume_stopped)
    for k in ("DSV41_GLUE_RQPAD", "DSV41_GLUE_RPSPLIT"):
        os.environ.pop(k, None)
    os.environ.update({"DSV41_SONAR_OPROJ": "fp8", "DSV41_SONAR_OPROJ_SPLITS": "32", "DSV41_SONAR_OPROJ_MAX_M": "32",
                       "DSV41_MOE_FUSED_PERMUTE": "1", "DSV41_ROUTE_LIVE": "skip", "VLLM_USE_DEEP_GEMM": "1",
                       "DSV41_MOE_ROUTEPREP": "1", "DSV41_ATTNOUT_OP": "rope,rq"})
    prod = s0_launcher(args.launcher_sha)  # S0: the launcher's own values override the defaults above
    os.environ.update(prod)
    os.environ.pop("DSV41_MOE_FC1_FRONT", None)
    say(f"INFO serving env from the launcher: {prod}")
    if FAILS:
        say("S0 launcher / model-config assertions failed: nothing measured")
        sys.exit(2)
    sys.path.insert(0, HERE)  # tools.pdl_producer
    import torch
    import triton
    torch.cuda.set_device(0)
    from vllm.model_executor.layers.fused_moe import dsv41_routeprep as rp_live
    from vllm.models.deepseek_v4.nvidia.ops import dsv41_attnout as ao
    from vllm.models.deepseek_v4.nvidia.ops import sonar_bf16_bmm as sonar
    say(f"torch {torch.__version__} triton {triton.__version__}; kit modules: rq {hashlib.sha256(open(KIT_RQ, 'rb').read()).hexdigest()[:16]} "
        f"rp {hashlib.sha256(open(KIT_RP, 'rb').read()).hexdigest()[:16]}")
    for mod in (ao, rp_live):
        base = os.path.basename(mod.__file__)
        check(f"S1 live {base} is the production module (sha {LIVE_SHA[base][:8]}, r5-glue lines stripped)",
              stripped_sha(mod.__file__) == LIVE_SHA[base], stripped_sha(mod.__file__)[:16])
    check("S1 sonar o_proj serving config: SPLITS=32 at every M, MAX_M=32",
          sonar._splits_for(1) == 32 and sonar._splits_for(32) == 32 and getattr(sonar, "MAX_M", 32) == 32)
    check(f"S1 attnout rq part on (DSV41_ATTNOUT_OP={os.environ.get('DSV41_ATTNOUT_OP')!r} from the launcher)",
          ao.RQ is True)
    z = torch.randn((4, N_RQ), device="cuda").to(torch.bfloat16)
    names = kernel_names(lambda: ao._live_quant(z))
    check("S1 reference quant is flashinfer's CuTe-DSL MXFP8 kernel (not a torch fallback)",
          any("flashinferquantization" in n.replace("_", "").lower() or "flashinfer" in n.lower() for n in names),
          str(sorted(names))[:300])
    if FAILS:
        say("serving-op assertions failed: nothing measured")
        sys.exit(2)
    if args.part in ("rq", "all"):
        case_rq(ao, args.quick, not args.no_timing)
    if args.part in ("rp", "all"):
        case_rp(args.quick, not args.no_timing)
    if args.part in ("hook", "all"):
        case_hook()
    say(f"{'ALL BIT-IDENTICAL, EVERY MUTANT CAUGHT' if not FAILS else 'FAILED: ' + ', '.join(FAILS[:20])} "
        f"({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
