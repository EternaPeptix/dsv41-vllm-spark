"""Time the server's tokenize call (base.py:637) on a dumped synthetic prompt,
and split it into AddedVocabulary+pre-tokenizer vs BPE.  usage:
  CUDA_VISIBLE_DEVICES= nice -n 19 taskset -c N python tok_bench.py prompt_300000.txt [reps]"""
import os
import statistics
import sys
import time

for line in open("/proc/meminfo"):
    if line.startswith("MemAvailable:") and int(line.split()[1]) < 8_000_000:
        sys.exit("ABORT: MemAvailable " + line.strip())
from vllm.tokenizers.registry import get_tokenizer  # noqa: E402

tok = get_tokenizer(os.path.expanduser("~/models/dsv41/uncensored-fp8"), tokenizer_mode="deepseek_v41")
s = open(sys.argv[1]).read()
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
be = tok._tokenizer
out = {}
for name, fn in (("hf_call", lambda: tok(s, truncation=True, max_length=1048577, add_special_tokens=False)["input_ids"]),
                 ("backend_encode", lambda: be.encode(s, add_special_tokens=False).ids),
                 ("pre_tokenize_only", lambda: be.pre_tokenizer.pre_tokenize_str(s))):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t) * 1e3)
    out[name] = (round(statistics.median(ts), 1), round(min(ts), 1))
n = len(be.encode(s, add_special_tokens=False).ids)
try:
    import fastokens  # noqa: F401
    fk = "installed"
except Exception:  # noqa: BLE001
    fk = "not installed"
print(f"cpu_affinity={sorted(os.sched_getaffinity(0))} chars={len(s)} tokens={n} "
      f"ms(median,min)={out} us/token={out['backend_encode'][0] * 1e3 / n:.2f} fastokens={fk}", flush=True)

if os.environ.get("CONTENT_MIX"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import synth
    gens = {"prose": lambda: synth.words(400), "code": lambda: synth.code_block(60), "logs": lambda: synth.log_lines(30),
            "json": lambda: __import__("json").dumps({"k": [synth.R.randint(0, 10**6) for _ in range(50)], "m": synth.words(30)}, indent=2),
            "cjk": lambda: synth.cjk(400), "tool_result_mix": lambda: synth.tool_result(4000)}
    for name, g in gens.items():
        parts, n = [], 0
        while n < 300_000:
            x = g(); parts.append(x); n += len(x) + 1
        text = "\n".join(parts)
        t = time.perf_counter(); ids = be.encode(text, add_special_tokens=False).ids; dt = time.perf_counter() - t
        print(f"content={name:16} chars={len(text)} tokens={len(ids)} chars/tok={len(text)/len(ids):.2f} "
              f"us/token={dt*1e6/len(ids):.2f} us/char={dt*1e6/len(text):.3f}", flush=True)

if os.environ.get("GIL_TEST"):
    import threading
    done = threading.Event()
    def work():
        be.encode(s, add_special_tokens=False); done.set()
    gaps, last = [], time.perf_counter()
    th = threading.Thread(target=work); t0 = time.perf_counter(); th.start()
    while not done.is_set():
        time.sleep(0.001); now = time.perf_counter(); gaps.append(now - last); last = now
    th.join()
    print(f"GIL test: encode wall {(time.perf_counter()-t0)*1e3:.0f} ms, main-thread ticks={len(gaps)}, "
          f"max gap={max(gaps)*1e3:.1f} ms (1 ms sleep loop: max gap ~1 ms => GIL released; ~encode time => GIL held)", flush=True)
    # same with encode_batch (tokenizers releases the GIL there?)
    done.clear()
    def work2():
        be.encode_batch([s], add_special_tokens=False); done.set()
    gaps, last = [], time.perf_counter()
    th = threading.Thread(target=work2); t0 = time.perf_counter(); th.start()
    while not done.is_set():
        time.sleep(0.001); now = time.perf_counter(); gaps.append(now - last); last = now
    th.join()
    print(f"GIL test encode_batch([s]): wall {(time.perf_counter()-t0)*1e3:.0f} ms, ticks={len(gaps)}, max gap={max(gaps)*1e3:.1f} ms", flush=True)
    # what the server actually calls (base.py:637)
    done.clear()
    def work3():
        tok(s, truncation=True, max_length=1048577, add_special_tokens=False)["input_ids"]; done.set()
    gaps, last = [], time.perf_counter()
    th = threading.Thread(target=work3); t0 = time.perf_counter(); th.start()
    while not done.is_set():
        time.sleep(0.001); now = time.perf_counter(); gaps.append(now - last); last = now
    th.join()
    print(f"GIL test HF __call__ (server path): wall {(time.perf_counter()-t0)*1e3:.0f} ms, ticks={len(gaps)}, max gap={max(gaps)*1e3:.1f} ms", flush=True)
