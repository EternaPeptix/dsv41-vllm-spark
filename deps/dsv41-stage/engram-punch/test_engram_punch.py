#!/usr/bin/env python3
"""Synthetic end-to-end test of engram_punch.py (tiny tables, ~12 MB). Linux: fallocate; macOS/APFS: F_PUNCHHOLE.
usage: ENGRAM_PUNCH_TEST=1 python3 -I test_engram_punch.py <tool.py> <empty work dir>"""
import hashlib, importlib.util, json, os, random, struct, subprocess, sys

tool, work = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("ep", tool)
ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)
model = os.path.join(work, "m", "uncensored-fp8"); os.makedirs(model)
tc = {"engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, "engram_vocab_size": 1000, "engram_n_heads": 8,
      "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99}
hs = ep.head_sizes(tc); tc["engram_num_embeddings"] = [sum(x) for x in hs]
json.dump({"text_config": tc}, open(os.path.join(model, "config.json"), "w"))
rng = random.Random(7); wmap = {}; shas = []
for j, (layer, name) in enumerate(((1, "model-00047-of-00048.safetensors"), (14, "model-00048-of-00048.safetensors"))):
    N = tc["engram_num_embeddings"][j]
    ents, blobs, off = {}, [], 0
    for key, dt, shape, nb in ((f"layers.{layer}.engram.embed.weight", "F8_E4M3", [N, 256], N * 256),
                               (f"layers.{layer}.engram.embed.scale", "F8_E8M0", [N, 8], N * 8),
                               (f"layers.{layer}.engram.q_weight", "BF16", [4, 64], 512),
                               (f"layers.{layer}.engram.wkv.weight", "F8_E4M3", [64, 96], 6144)):
        ents[key] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        blobs.append(rng.randbytes(nb)); off += nb; wmap[key] = name
    raw = json.dumps(ents).encode(); raw += b" " * (-len(raw) % 8)
    data = struct.pack("<Q", len(raw)) + raw + b"".join(blobs)
    open(os.path.join(model, name), "wb").write(data)
    shas.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
json.dump({"weight_map": wmap}, open(os.path.join(model, "model.safetensors.index.json"), "w"))
open(os.path.join(work, "m", "uncensored.sha"), "w").writelines(shas)
orig = {n: open(os.path.join(model, n), "rb").read() for n in sorted(set(wmap.values()))}

def run(*args, ok=True):
    p = subprocess.run([sys.executable, "-I", tool, "--model", model, "--state-dir", os.path.join(work, "state"), *args],
                       capture_output=True, text=True, env={**os.environ, "ENGRAM_PUNCH_TEST": "1"})
    if (p.returncode == 0) != ok:
        print(p.stdout, p.stderr); raise SystemExit(f"FAIL {args}: rc {p.returncode}")
    return p.stdout + p.stderr

print(run("--rank", "1", "--dry-run").splitlines()[1])
for r in range(8): run("--rank", str(r), "--check")
ref = os.path.join(work, "ref.json")
run("--rank", "0", "--ref-all", ref, "--test-host", "spark-four")
run("--rank", "1", "--ref-all", ref + "x", "--test-host", "spark-two", ok=False)                 # not pristine
run("--rank", "3", "--punch-in-place", "--yes-punch", "--ref-manifest", ref, "--test-host", "spark-four", ok=False)
run("--rank", "1", "--punch-in-place", "--yes-punch", "--ref-manifest", ref, "--test-host", "spark-three", ok=False)  # host map
run("--rank", "1", "--punch-in-place", "--ref-manifest", ref, "--test-host", "spark-two", ok=False)  # no --yes-punch
run("--rank", "1", "--punch-in-place", "--yes-punch", "--test-host", "spark-two", ok=False)          # no ref
out = run("--rank", "1", "--punch-in-place", "--yes-punch", "--ref-manifest", ref, "--test-host", "spark-two")
print(out.strip().splitlines()[-1])
run("--rank", "1", "--check")
for r in (0, 2, 7): run("--rank", str(r), "--check", ok=False)
run("--rank", "0", "--punch-in-place", "--yes-punch", "--ref-manifest", ref, "--test-host", "spark-one", ok=False)
run("--rank", "1", "--punch-in-place", "--yes-punch", "--ref-manifest", ref, "--test-host", "spark-two")  # idempotent
# byte-level proof: every kept byte equal, every punched byte zero, header/tail intact
for j, layer in enumerate((1, 14)):
    t = ep.find_tables(model, tc)[layer]; a, b = ep.rank_rows(tc, 1, 8)[layer]
    now = open(t["path"], "rb").read(); o = orig[t["file"]]
    for k, (lo, hi) in ep.kept_segments(t, a, b).items():
        assert now[lo:hi] == o[lo:hi], (layer, k)
    for lo, hi in ep.punch_ranges(t, a, b, ep.fs_block(model)):
        assert now[lo:hi] == bytes(hi - lo), (layer, lo, hi)
    assert len(now) == len(o)
    # every row the vLLM loader narrows for rank 1 is verbatim
    w0, s0 = t["w"][0], t["s"][0]
    assert now[w0 + a * 256:w0 + b * 256] == o[w0 + a * 256:w0 + b * 256]
    assert now[s0 + a * 8:s0 + b * 8] == o[s0 + a * 8:s0 + b * 8]
# --test-host is refused outside the synthetic test
p = subprocess.run([sys.executable, "-I", tool, "--model", model, "--rank", "1", "--dry-run", "--test-host", "spark-two"],
                   capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "ENGRAM_PUNCH_TEST"})
assert p.returncode != 0, "--test-host accepted without ENGRAM_PUNCH_TEST=1"
# zeros written as DATA (an rsync from a punched node): no holes, but --check must still fail for that rank
import shutil
model2 = os.path.join(work, "m2", "uncensored-fp8")
# copyfile, not copy2: on Linux copy2 also copies xattrs (the rank-1 punch record); an rsync without -X does not
shutil.copytree(model, model2, copy_function=shutil.copyfile)
for n in orig:
    open(os.path.join(model2, n), "wb").write(orig[n])
    try: os.remove(os.path.join(model2, n + ".engram_punch.json"))
    except OSError: pass
t = ep.find_tables(model2, tc)[1]; a, b = ep.rank_rows(tc, 5, 8)[1]
with open(t["path"], "r+b") as fh:
    fh.seek(t["w"][0] + a * 256); fh.write(bytes((b - a) * 256))
    fh.seek(t["s"][0] + a * 8); fh.write(bytes((b - a) * 8))
def run2(*args, ok=True):
    p = subprocess.run([sys.executable, "-I", tool, "--model", model2, *args], capture_output=True, text=True,
                       env={**os.environ, "ENGRAM_PUNCH_TEST": "1"})
    if (p.returncode == 0) != ok:
        print(p.stdout, p.stderr); raise SystemExit(f"FAIL model2 {args}: rc {p.returncode}")
run2("--rank", "5", "--check", ok=False)
for r in (0, 4, 6, 7): run2("--rank", str(r), "--check")
print("synthetic test PASS")
