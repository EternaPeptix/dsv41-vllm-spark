#!/usr/bin/env python3
"""acqcheck.py: the r7-comm2 N1 / N2 binary checks over arbitrary kernels (CPU only; SASS from audit/collect_sass.py).

    python3 -I acqcheck.py --sass audit/out/t2_sass.jsonl --pdl audit/out/t2_pdl.json [--json out.json] [--selftest]

Per kernel function (every variant found: each Triton cache entry / binary is checked on its own):
  A1 (N2, "ACQBULK in every CTA"): the kernel contains an unpredicated ACQBULK (griddepcontrol.wait) and no EXIT / RET
     is reachable from the entry on any CFG path that has not executed one (an early-exiting CTA would complete without
     waiting, so its grid could complete before its predecessor: the transitivity C2 relies on, review N2);
  A2 (N1, "waits before reading"): A1, and no global read (LDG / LD / LDGSTS / ATOMG / ATOM / RED / UTMALDG / UBLKCP /
     LDGDEPBAR-less TMA) is reachable before the wait either, so a PDL consumer launched early by an AR trigger cannot
     read the AR output before the AR grid has completed and flushed;
  NONE: no ACQBULK and no PREEXIT at all (a kernel compiled without griddepcontrol).
A kernel name passes N2 when every variant passes A1, or when it carries no griddepcontrol and has no PDL launch
evidence (it is then stream-ordered: it cannot start before its predecessor completes). The empirical PDL list (kernels
seen starting before their same-stream predecessor ended in the t2 traces, audit/t2_pdl.py) must all pass A1.
--selftest builds must-fail mutants from real SASS (ACQBULK predicated, ACQBULK removed, an EXIT / LDG moved before
the wait) and requires every one to be caught: a check that cannot fail is not a check."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sassir as S  # noqa: E402

READS = ("LDG", "LD", "LDGSTS", "ATOMG", "ATOM", "RED", "REDG", "UTMALDG", "UBLKCP", "UTMAPF", "LDGMC")
EXITS = ("EXIT", "RET")


def pre_wait(K):
    """instructions reachable from the entry on some path that has not executed an unpredicated ACQBULK."""
    seen, out, work = set(), [], [0]
    while work:
        bi = work.pop()
        if bi in seen:
            continue
        seen.add(bi)
        stopped = False
        for i in K.blocks[bi]:
            x = K.ins[i]
            if x.base == "ACQBULK" and x.gpred == "PT" and not x.gneg:
                stopped = True
                break
            out.append(x)
        if not stopped:
            work.extend(K.succ[bi])
    return out


def check(name, body):
    r = {"acq": body.count("ACQBULK"), "preexit": body.count("PREEXIT"), "errors_a1": [], "errors_a2": [], "notes": []}
    if r["acq"] == 0:
        r["class"] = "NONE" if r["preexit"] == 0 else "PREEXIT-ONLY"
        if r["preexit"]:
            r["errors_a1"].append("PREEXIT (launch_dependents) without any ACQBULK (wait)")
            r["errors_a2"].append("no ACQBULK")
        else:
            r["errors_a1"].append("no ACQBULK")
            r["errors_a2"].append("no ACQBULK")
        return r
    try:
        ins = [(int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2)).strip()) for m in map(S.INS_RE.search, body.splitlines()) if m]
        if not ins:
            raise ValueError("no instructions")
        K = S.Kernel(name, ins)
    except Exception as e:  # noqa: BLE001 - fail closed
        r["class"] = "UNPARSED"
        r["errors_a1"].append(f"SASS not parsed ({type(e).__name__}: {e})")
        r["errors_a2"].append("SASS not parsed")
        return r
    acq = [x for x in K.ins if x.base == "ACQBULK"]
    for x in acq:
        if x.gpred != "PT" or x.gneg:
            r["errors_a1"].append(f"predicated ACQBULK /*{x.addr:04x}*/ {x.text}")
    early = pre_wait(K)
    for x in early:
        if x.base in EXITS:
            r["errors_a1"].append(f"{x.text} /*{x.addr:04x}*/ reachable before the wait")
        if x.base in READS:
            r["errors_a2"].append(f"{x.text} /*{x.addr:04x}*/ reachable before the wait")
        if x.base == "CALL":
            r["errors_a1"].append(f"CALL /*{x.addr:04x}*/ before the wait (callee not followed)")
            r["errors_a2"].append(f"CALL /*{x.addr:04x}*/ before the wait")
    r["errors_a2"] = r["errors_a1"] + r["errors_a2"]
    pre_ix = {x.idx for x in early}
    r["notes"].append(f"{len(acq)} ACQBULK, {r['preexit']} PREEXIT, {len(early)} instructions before the wait, "
                      f"first wait /*{acq[0].addr:04x}*/" + ("; PREEXIT before the wait" if any(
                          x.base == "PREEXIT" for x in early) else ""))
    r["class"] = "WAITS" if not r["errors_a1"] else "EARLY-EXIT"
    r["pre_wait_ops"] = sorted({x.base for x in early})
    return r


# GB10 (sm_121) loads sm_121 / sm_121a cubins and, by binary compatibility within 12.x, sm_120 ones. Variants for other
# architectures in the same fat binary never run here and are not checked (listed only).
GB10_ARCHS = ("sm_120", "sm_121")

# A1g (grid-level wait, for kernels that fail the strict per-thread A1 only through warp-role or row-range exits):
# a concrete WITNESS thread (tid 0, CTA 0, lane 0) must reach the unpredicated ACQBULK on every CFG path the strict
# evaluator cannot decide, under the stated assumptions. CTA 0 is always launched, so if its thread 0 waits, every
# execution of the grid contains a wait: the grid cannot complete before its predecessor grid has completed and
# flushed, which is the transitivity C2 needs (a waiting CTA orders the whole grid's completion; trigger timing does
# not matter because every dependent waits for full completion). SHFL.IDX from lane RZ is read as the identity for
# this lane-0 witness. Assumptions are named per kernel and printed.
WITNESS_ASSUME = {
    "topkGatingSoftplusSqrt": ({(0, 0x398): 1}, "num_rows (c[0x0][0x398]) >= 1: the gating kernel is never launched empty"),
    "DeviceGemmMxfp8": ({}, "none (the exit at /*09b0*/ is warp-group 3+ of tid.x >> 7: a role exit, not a CTA exit)"),
}


# Launch-site evidence for kernels whose t2 "overlap" count is a CUDA-graph stream-attribution artifact (nsys maps graph
# branches onto streams, so two parallel branch nodes can share a reported stream). A kernel listed here has no
# griddepcontrol in its SASS AND a plain <<<>>> launch at its only launch site, so it is never PDL-launched.
LAUNCH_EVIDENCE = {
    "mhc_proj_kernel": "moet b0dc737 tools/dsv41_sm120/mhc_overlap/mhc_post_norm_sm120.cu:508 `kernel<<<dim3(S, ...), "
                       "kProjThreads, 0, stream>>>` (snapshot ~/Documents/Codex/2026-10-28/x20-okc-readme-md-3/work/"
                       "r12-source-snapshot); 2/27720 t2 overlaps of <= 0.42 us",
    "l2pf_kernel": "libdsv41_l2pf_g2.so l2pf_g2.cu:35 `l2pf_kernel<<<...>>>` (climb-r7/l2pf/ref/l2pf_g2.cu); 0 t2 overlaps",
    "l2pf_paced_kernel": "l2pf_g2.cu:102 `l2pf_paced_kernel<<<1, 32, 0, stream>>>`; 0 t2 overlaps",
    "rms_norm_kernel": "UNVERIFIED: vLLM csrc layernorm is a <<<>>> launch as far as known, source and SASS not read here; 0 t2 overlaps",
    "elementwise_kernel": "UNVERIFIED: ATen TensorIterator <<<>>> launch as far as known, not read here; 0 t2 overlaps",
    "index_elementwise_kernel": "UNVERIFIED: ATen index kernel <<<>>> launch as far as known, not read here; 0 t2 overlaps",
}
# Prior SASS audits of kernels this collection did not reach (named, not re-run here)
PRIOR_AUDIT = {
    "per_token_group_quant_8bit_register_kernel": "climb-r1/r1-pdl/README.md section 4 item 3: production variant "
        "<BFloat16, Float8_e4m3fn, false, true, 128> (_C_stable_libtorch.abi3.so 7f9fe0af..., sm_120 cubin) has ACQBULK at "
        "/*0050*/ before the early @P0 PREEXIT; @P0 EXIT at /*00c0*/ and before the first LDG (A1 and A2 by inspection)",
}


def witness(name, body, key):
    import sassir as S2
    cb, why = WITNESS_ASSUME.get(key, ({}, "none"))
    txt = re.sub(r"SHFL\.IDX PT, (R\d+), (R\d+), RZ, 0x1f", r"MOV \1, \2", body)
    ins = [(int(m.group(1), 16), re.sub(r"\s+", " ", m.group(2)).strip()) for m in map(S2.INS_RE.search, txt.splitlines()) if m]
    K = S2.Kernel(name, ins)
    sr = {"SR_TID.X": 0, "SR_TID.Y": 0, "SR_TID.Z": 0, "SR_CTAID.X": 0, "SR_CTAID.Y": 0, "SR_CTAID.Z": 0, "SR_LANEID": 0}
    ev = S2.Evaluator(K, S2.Env(dict(cb), sr, seed=1))
    seen, stack, bad = set(), [0], []
    while stack:
        bi = stack.pop()
        if bi in seen:
            continue
        seen.add(bi)
        waited = False
        for i in K.blocks[bi]:
            x = K.ins[i]
            if x.base == "ACQBULK" and x.gpred == "PT" and not x.gneg:
                waited = True
                break
            if x.base in EXITS:
                g = ev.strict(x.vals["guard"]) if x.vals else None
                if g != 0:
                    bad.append(f"{x.text} /*{x.addr:04x}*/ ({'taken' if g else 'undecided'} for the witness)")
                    if g:
                        break
        if waited or (bad and bad[-1].endswith("(taken for the witness)")):
            continue
        for s_ in K.succ[bi]:
            if ev.edge(bi, s_) != 0:
                stack.append(s_)
    return (not bad), bad, why


def base_name(mangled):
    for k in ("_proj_rows", "_q_kv_norm_quant_kernel", "_dsv4_topk_kernel", "_dsv41_rp_kernel", "_r2_reduce_quant",
              "mhc_pre_big_fuse_with_norm_tilelang_kernel", "mhc_fused_tilelang_kernel", "mhc_post_tilelang_kernel",
              "mhc_post_norm_kernel", "mhc_proj_kernel", "l2pf_paced_kernel", "l2pf_kernel", "topkGatingSoftplusSqrt",
              "fusedDeepseekV4QNormRopeKV", "act_and_mul_kernel", "per_token_group_quant_8bit_register_kernel",
              "sm120_fp8_paged_mqa_logits", "sm120_fp8_fp4_gemm_1d1d_impl", "sm120_tf32_hc_prenorm_gemm_impl",
              "DeviceGemmMxfp8", "flashinferquantization", "Kernel2", "rms_norm_kernel", "_fused_engram_post_wkv_kernel",
              "_gather_lookback_kernel", "_mk_winner_kernel", "_sf_onehot_kernel", "ar_kernel_pl"):
        if k in mangled:
            return k
    return mangled[:60]


# names of the t2 PDL-observed list (audit/out/t2_pdl.txt) -> base_name keys
PDL_OBS_MAP = {"Copy_Atom": "DeviceGemmMxfp8", "kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_qua":
               "flashinferquantization"}

# N1: the AR consumers named by the r6 comm review, plus every AR successor seen launched with PDL in t2
N1_CONSUMERS = ["mhc_fused_tilelang_kernel", "mhc_post_tilelang_kernel", "mhc_pre_big_fuse_with_norm_tilelang_kernel",
                "_proj_rows", "_q_kv_norm_quant_kernel",
                # every other immediate successor of an AR in t2 (audit/out/t2_kernels.txt, AR same-stream neighbours)
                "l2pf_kernel", "mhc_post_norm_kernel", "flashinferquantization", "_fused_engram_post_wkv_kernel",
                "_gather_lookback_kernel", "_mk_winner_kernel", "_sf_onehot_kernel", "ar_kernel_pl", "rms_norm_kernel",
                "elementwise_kernel", "index_elementwise_kernel"]


def selftest(entries):
    """must-fail mutants built from a real waiting kernel."""
    src = None
    for e in entries:
        if e.get("name") == "_q_kv_norm_quant_kernel" and "ACQBULK" in e["sass"]:
            src = e
            break
    if src is None:
        return [("no _q_kv_norm_quant_kernel SASS with ACQBULK to build mutants from", False)]
    body = src["sass"]
    out = []
    base = check(src["name"], body)
    out.append(("control: real _q_kv_norm_quant_kernel passes A1 and A2", not base["errors_a1"] and not base["errors_a2"]))
    lines = body.splitlines()
    ia = next(i for i, l in enumerate(lines) if "ACQBULK" in l)
    # M1: predicate the wait
    m1 = lines[:]
    m1[ia] = m1[ia].replace("ACQBULK", "@P0 ACQBULK")
    out.append(("M1 predicated ACQBULK caught by A1", bool(check(src["name"], "\n".join(m1))["errors_a1"])))
    # M2: wait removed (NOP)
    m2 = lines[:]
    m2[ia] = m2[ia].replace("ACQBULK", "NOP")
    out.append(("M2 ACQBULK removed caught by A1", bool(check(src["name"], "\n".join(m2))["errors_a1"])))
    # M3: a global load placed before the wait: the instruction right before ACQBULK is replaced by the text of the
    #     first LDG after the wait (an early read of predecessor data); A1 must still pass, A2 must fail
    il = next((i for i in range(ia + 1, len(lines)) if re.search(r"\bLDG\b", lines[i])), None)
    ib = next((i for i in range(ia - 1, -1, -1) if re.search(r"/\*[0-9a-f]{4,}\*/\s+\S", lines[i])), None)
    if il is not None and ib is not None:
        m3 = lines[:]
        tb = re.search(r"(/\*[0-9a-f]{4,}\*/\s+)(.*?;)", m3[ib])
        tl = re.search(r"(/\*[0-9a-f]{4,}\*/\s+)(.*?;)", m3[il])
        m3[ib] = m3[ib][:tb.start(2)] + re.sub(r"^@!?P\d\s+", "", tl.group(2)) + m3[ib][tb.end(2):]
        r3 = check(src["name"], "\n".join(m3))
        out.append(("M3 LDG before the wait caught by A2 (not by A1)", bool(r3["errors_a2"]) and not r3["errors_a1"]))
    else:
        out.append(("M3 could not be built", False))
    # M4: an EXIT before the wait, guarded by a predicate (an early-exiting CTA)
    m4 = lines[:ia] + [re.sub(r"(/\*[0-9a-f]{4,}\*/\s+).*?;", r"\1@P1 EXIT ;", lines[ia])] + lines[ia:]
    m4 = [l for l in m4]
    out.append(("M4 predicated EXIT before the wait caught by A1", bool(check(src["name"], "\n".join(m4))["errors_a1"])))
    # M5 (A1g): the witness must fail when the early exit is taken by thread 0 of CTA 0 (unpredicated)
    for e in entries:
        if "topkGatingSoftplusSqrt" in e["name"] and e["arch"].rstrip("af") in GB10_ARCHS:
            ok0, _, _ = witness(e["name"], e["sass"], "topkGatingSoftplusSqrt")
            body5 = re.sub(r"@P0 EXIT", "EXIT", e["sass"], count=1)
            ok5, bad5, _ = witness(e["name"], body5, "topkGatingSoftplusSqrt")
            out.append(("M5 A1g control (real topkGating sm_120 witness waits)", ok0))
            out.append(("M5 A1g unpredicated early EXIT caught by the witness", not ok5))
            body6 = e["sass"]
            ok6, _, _ = witness(e["name"], body6, "none-assumed")
            out.append(("M6 A1g without the num_rows >= 1 assumption is undecided (fails closed)", not ok6))
            break
    else:
        out.append(("M5/M6 no topkGating sm_120 SASS to build the witness mutants from", False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sass", required=True)
    ap.add_argument("--pdl", required=True)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    entries, missing = [], []
    import gzip
    for l in (gzip.open(a.sass, "rt") if a.sass.endswith(".gz") else open(a.sass)):
        j = json.loads(l)
        if "name" in j:
            entries.append(j)
        elif "missing" in j:
            missing = j["missing"]
    pdl = json.load(open(a.pdl))
    by, other_arch = {}, {}
    for e in entries:
        if e["arch"].rstrip("af") not in GB10_ARCHS:
            other_arch.setdefault(base_name(e["name"]), set()).add(e["arch"])
            continue
        r = check(e["name"], e["sass"])
        r.update(file=e["file"], arch=e["arch"], launch_pdl=e["meta"].get("launch_pdl"), file_sha256=e["file_sha256"])
        if r["class"] == "EARLY-EXIT":
            try:
                r["witness_ok"], r["witness_bad"], r["witness_assume"] = witness(e["name"], e["sass"], base_name(e["name"]))
            except Exception as ex:  # noqa: BLE001 - fail closed
                r["witness_ok"], r["witness_bad"] = False, [f"witness walk failed: {ex}"]
        by.setdefault(base_name(e["name"]), []).append(r)
    observed = {PDL_OBS_MAP.get(k, base_name(k)) for k in pdl["overlap"]}
    rep = {"kernels": {}, "missing_mangled": missing}
    ok_n2, ok_n1 = True, True
    print("== N2: ACQBULK in every CTA, per kernel name (all variants found)")
    for k in sorted(set(by) | observed):
        vs = by.get(k, [])
        a1 = [not v["errors_a1"] for v in vs]
        cls = sorted({v["class"] for v in vs})
        tri_pdl = sorted({str(v["launch_pdl"]) for v in vs if v["launch_pdl"] is not None})
        obs = k in observed
        if not vs:
            verdict = "NO-SASS" + (" (PDL-observed: OPEN)" if obs else "")
            if obs and k in PRIOR_AUDIT:
                verdict = "PASS-PRIOR-AUDIT (not re-collected: " + PRIOR_AUDIT[k] + ")"
            elif obs:
                ok_n2 = False
        elif all(a1):
            verdict = "PASS (every variant waits in every CTA)"
        elif all(v["class"] in ("WAITS", "EARLY-EXIT") for v in vs) and all(v.get("witness_ok") for v in vs
                                                                              if v["class"] == "EARLY-EXIT"):
            verdict = "PASS-A1g (every variant: thread 0 of CTA 0 waits on every path; " + \
                WITNESS_ASSUME.get(k, ({}, "none"))[1] + ")"
        elif all(v["class"] == "NONE" for v in vs) and not obs and "True" not in tri_pdl:
            verdict = "PASS (no griddepcontrol; no PDL launch evidence: stream-ordered)"
        elif all(v["class"] == "NONE" for v in vs) and k in LAUNCH_EVIDENCE and "True" not in tri_pdl:
            verdict = "PASS-LAUNCH-SITE (no griddepcontrol; plain launch: " + LAUNCH_EVIDENCE[k] + ")"
        else:
            verdict = "FAIL"
            ok_n2 = False
        rep["kernels"][k] = {"pdl_observed": obs, "variants": len(vs), "classes": cls, "triton_launch_pdl": tri_pdl,
                             "verdict_n2": verdict, "variants_detail": vs}
        print(f"{k:46s} pdl-obs {'Y' if obs else '-'} variants {len(vs):3d} classes {cls} triton launch_pdl {tri_pdl} -> {verdict}")
        for v in vs:
            if v["errors_a1"] and v["class"] != "NONE":
                print(f"    {v['file'][-70:]} {v['arch']}: {v['errors_a1'][0]}" +
                      (f"  | A1g witness {'waits' if v.get('witness_ok') else 'FAILS: ' + str(v.get('witness_bad'))[:160]}"
                       if "witness_ok" in v else ""))
        if k in other_arch:
            print(f"    (not loadable on GB10, not checked: {sorted(other_arch[k])})")
    print("== N1: AR consumers wait before any global read (A2)")
    for k in N1_CONSUMERS:
        vs = by.get(k, [])
        pdl_launched = k in observed or any(v["launch_pdl"] for v in vs)
        a2 = [not v["errors_a2"] for v in vs]
        if not vs and not pdl_launched and k in LAUNCH_EVIDENCE:
            weak = LAUNCH_EVIDENCE[k].startswith("UNVERIFIED")
            verdict = ("OPEN-WEAK" if weak else "PASS-NOT-PDL") + " (launch site: " + LAUNCH_EVIDENCE[k] + ")"
            if weak:
                ok_n1 = False
        elif not vs:
            verdict = "NO-SASS: OPEN" + (" (PDL-observed)" if pdl_launched else "")
            ok_n1 = False
        elif all(a2):
            verdict = "PASS (waits before every global read, every variant)"
        elif (not pdl_launched or k in LAUNCH_EVIDENCE) and all(v["class"] == "NONE" for v in vs):
            verdict = "PASS-NOT-PDL (no griddepcontrol, not PDL-launched: starts only after the AR grid completes)"
        else:
            verdict = "FAIL"
            ok_n1 = False
        rep["kernels"].setdefault(k, {})["verdict_n1"] = verdict
        print(f"{k:46s} variants {len(vs):3d} pdl-launched {pdl_launched} -> {verdict}")
        for v in vs:
            if v["errors_a2"] and v["class"] != "NONE":
                print(f"    {v['file'][-70:]}: {v['errors_a2'][0]}")
    if a.selftest:
        st = selftest(entries)
        for m, good in st:
            print(f"selftest {m}: {'OK' if good else 'NOT CAUGHT'}")
        rep["selftest"] = st
        if not all(g for _, g in st):
            ok_n1 = ok_n2 = False
    print(f"ACQCHECK N2 {'PASS' if ok_n2 else 'FAIL/OPEN'}  N1 {'PASS' if ok_n1 else 'FAIL/OPEN'}")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1, sort_keys=True, default=str)
    return 0 if ok_n1 and ok_n2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
