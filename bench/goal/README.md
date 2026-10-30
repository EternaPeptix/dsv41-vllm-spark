# dsv41 goal metrics (G1 / G4), 2026-10-26

Goal: decode 120 tok/s for a single user (G1) OR 50 tok/s per session across 4 concurrent individual sessions (G4),
greedy output bitwise identical on the anchor (exact.py a8c3394af9d2, AV profile pinned in A/B arms).

- `goal_single.py` (G1): bench.py's code / prose / math prompts (verbatim), 1024 max tokens, temperature 0,
  top_p 1.0, thinking at the server default, each x2, serial. Reports each type's decode tok/s and the mean
  of the three (= G1), per-type AL, and sha256 of every output (identical outputs across arms = exact).
- `goal_conc4.py` (G4): 4 DIFFERENT prompts (code, prose, math = the G1 prompts; mixed = a fixed RSA
  prose+math+code prompt) started together, 512 max tokens each, one excluded warm-up batch, then 3 measured
  batches. Reports per-session decode tok/s min and mean per batch and their means over the 3 batches (= G4),
  plus the common-window rate (tokens produced while all 4 sessions were decoding / window length).
- Decode tok/s = (completion_tokens - 1) / (last content chunk - first content chunk), as bench.py.
- The prompt sets are pinned: each script hashes its canonical prompt JSON and refuses to run on a mismatch.
  G1 prompts sha256 dc744bc29273cc072ffd3f9e29786f6c854141c0d4083001bebfb8ebee1291e2,
  G4 prompts sha256 42c93c19844208cfba09d51266b32ee9af339362eeba6a6266a3a6dbd41cf2b3.
  `SHA256SUMS` pins the script files; the directory is a git repo (`git log`).
- Run on spark-one: `python3 -I ~/dsv41-bench/goal/goal_single.py --out g1.jsonl --label X` and
  `python3 -I ~/dsv41-bench/goal/goal_conc4.py --out g4.jsonl --label X`. Stdlib only.
