#!/usr/bin/env python3
"""gsm8k_sampled_gate.py — sampled-decoding (temperature > 0) GSM8K gate for the dsv41 server (stdlib only).

Run (on spark-one, server up):
  python3 gsm8k_sampled_gate.py run <prefix> [--n 50] [--conc 8] [--temp 0.6] [--top-p 1.0] [--runs 2] [--seed 1000]
      -> <prefix>.r<k>.jsonl per run (same per-item schema as gsm8k_gate.py) + <prefix>.summary.json
  python3 gsm8k_sampled_gate.py compare <control_prefix> <arm_prefix> [--tol-mean 2.5] [--tol-trunc 1]
      -> prints PASS/FAIL, exit code 0/1

Seeds: request seed = seed + 100000*run + idx, identical across arms, so arm and control see the same seed per item
(streams still diverge once the sampler consumes randomness differently; the pairing only removes some variance).

Gate rule (compare):
  PASS iff  mean_correct(arm) >= mean_correct(control) - tol_mean          (default 2.5 items of 50, i.e. 5 over 2 runs)
       and  mean_truncated(arm) <= mean_truncated(control) + tol_trunc     (default 1)
       and  mean_no_answer_line(arm) <= mean_no_answer_line(control) + 1   (garbage detector)
  Rationale: at ~96% accuracy one 50-item run has SD sqrt(50*.96*.04) = 1.39 items; the difference of two 2-run
  means has SD 1.39 = 1.8 sigma at 2.5 items (false-fail ~4%). This is a coarse garbage/regression gate: a lossy
  sampler that biases token frequencies by a few % will NOT show here; that is what test_spec_sampling.py is for.
  Promotion-grade: --n 200 --conc 16 --runs 1 and --tol-mean 6 (SD of difference 3.9 items).
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request

PROMPT = ("Solve the following math problem. End your reply with a final line of the form "
          "'Answer: <number>'.\n\n{q}")


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def run(args):
    items = [json.loads(l) for l in open(os.path.expanduser("~/gsm8k_test.jsonl"))][: args.n]

    def ask(i, k):
        it = items[i]
        body = {"model": args.model, "temperature": args.temp, "top_p": args.top_p, "max_tokens": 4096,
                "seed": args.seed + 100000 * k + i,
                "messages": [{"role": "user", "content": PROMPT.format(q=it["question"])}]}
        req = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=1800))
        dt = time.time() - t
        text = r["choices"][0]["message"].get("content") or ""
        final = text.split("</think>")[-1]
        m = re.findall(r"Answer:\s*([^\n]+)", final)
        pred = num(m[-1]) if m else num(final)
        gold = num(it["answer"].split("####")[-1])
        return {"idx": i, "gold": gold, "pred": pred, "ok": pred is not None and abs(pred - gold) < 1e-6,
                "has_answer_line": bool(m), "completion_tokens": r["usage"]["completion_tokens"],
                "seconds": round(dt, 2), "finish": r["choices"][0]["finish_reason"], "text": text}

    summ = {"n": len(items), "temp": args.temp, "top_p": args.top_p, "conc": args.conc, "seed": args.seed, "runs": []}
    for k in range(args.runs):
        t0 = time.time()
        with cf.ThreadPoolExecutor(args.conc) as ex:
            res = sorted(ex.map(lambda i: ask(i, k), range(len(items))), key=lambda r: r["idx"])
        wall = time.time() - t0
        with open(f"{args.prefix}.r{k}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        toks = sum(r["completion_tokens"] for r in res)
        row = {"run": k, "correct": sum(r["ok"] for r in res), "truncated": sum(r["finish"] == "length" for r in res),
               "no_answer_line": sum(not r["has_answer_line"] for r in res), "completion_tokens": toks,
               "wall_s": round(wall, 1), "agg_tok_s": round(toks / wall, 1),
               "wrong": [r["idx"] for r in res if not r["ok"]]}
        summ["runs"].append(row)
        print(json.dumps(row), flush=True)
    for key in ("correct", "truncated", "no_answer_line", "agg_tok_s"):
        summ["mean_" + key] = round(sum(r[key] for r in summ["runs"]) / len(summ["runs"]), 2)
    json.dump(summ, open(f"{args.prefix}.summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "runs"}))


def compare(args):
    c = json.load(open(f"{args.control}.summary.json"))
    a = json.load(open(f"{args.arm}.summary.json"))
    assert (c["n"], c["temp"], c["top_p"]) == (a["n"], a["temp"], a["top_p"]), "control/arm settings differ"
    checks = {
        "accuracy": a["mean_correct"] >= c["mean_correct"] - args.tol_mean,
        "truncation": a["mean_truncated"] <= c["mean_truncated"] + args.tol_trunc,
        "answer_lines": a["mean_no_answer_line"] <= c["mean_no_answer_line"] + 1,
    }
    ok = all(checks.values())
    print(json.dumps({"control_mean_correct": c["mean_correct"], "arm_mean_correct": a["mean_correct"],
                      "n": c["n"], "control_agg_tok_s": c["mean_agg_tok_s"], "arm_agg_tok_s": a["mean_agg_tok_s"],
                      "checks": checks, "verdict": "PASS" if ok else "FAIL"}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("prefix")
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--conc", type=int, default=8)
    r.add_argument("--temp", type=float, default=0.6)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--runs", type=int, default=2)
    r.add_argument("--seed", type=int, default=1000)
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default="dsv41")
    c = sub.add_parser("compare")
    c.add_argument("control")
    c.add_argument("arm")
    c.add_argument("--tol-mean", type=float, default=2.5)
    c.add_argument("--tol-trunc", type=float, default=1.0)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
