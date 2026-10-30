# DeepSeek-V4.1 vLLM on 8× DGX Spark — backup snapshot

Code/text-only backup of the production DSv4.1 serving stack as it sits on the Sparks on 2026-10-30
(production launcher `dsv41-tp8.sh` sha256 `5b0a889c…`, supervisor `ops/rel/dsv41-relaunch.sh` `7bea1cf3…`).
Taken from spark-one; the patched vLLM files and the launcher are identical on all 8 nodes.

No model weights, compiled kernels (.so/.cubin/.ptx), JIT caches, traces or logs are included.

## Layout

| Path | What | Lives on the Sparks at |
|---|---|---|
| `prod/dsv41-tp8.sh` | Production per-rank launcher (TP8, DSpark, 1M ctx, all `DSV41_*` flags) | `~/dsv41-tp8.sh` |
| `prod/dsv41-tp8.sh.bak-*` | Every earlier launcher version, named after the change that replaced it | `~/` |
| `prod/ops-rel/` | Supervised relaunch (`dsv41-relaunch.sh --expect-launcher 5b0a889c`), status, replay detector | `~/dsv41-opt-20261021/ops/rel/` |
| `prod/pins/` | Adaptive-verification cost pins (prod uses `av-pin-k5-g3FU.json`) | `~/dsv41-opt-20261021/pins/` |
| `prod/lanproxy/` | LAN/tailnet TCP forwarder for the API + its systemd user unit | `~/dsv41-lanproxy/` |
| `vllm-site/` | Every file in the `vllm` 0.32.0 / `flashinfer-python` 0.7.0.post1 installs that differs from the wheel (`M`) or was added (`A`); see `MANIFEST.tsv` (with sha256) and `pip-freeze.txt` | `~/venvs/vllm032/lib/python3*/site-packages/` |
| `deps/` | Serving dependencies referenced by the launcher: `dsv41-sched` (custom scheduler), `rdma_ar_dsv41` (RDMA all-reduce source), `dsv41-stage/*` kits (M32 MoE, fc2c, engram-punch, g5det/g5mhc installers, layer3 tool-stream patch, g3 pin), MoET mHC overlap | `~/` same relative paths |
| `bench/` | Benchmark / goal / exactness harness scripts | `~/dsv41-bench/` |
| `notes/` | READMEs and result notes from every `~/dsv41-*` experiment folder | `~/dsv41-*/` |

## Restore sketch

1. Fresh `vllm==0.32.0` venv at `~/venvs/vllm032`; copy `vllm-site/` over its site-packages (check `MANIFEST.tsv` hashes).
2. Put `deps/*` back at `~/…` (same relative paths); rebuild `rdma_ar_dsv41` (`build-k1pl/rdma_ar_ext.so`) and let the M32/fc2c kits JIT.
3. Copy `prod/dsv41-tp8.sh` to `~/`, pins to `~/dsv41-opt-20261021/pins/`, supervisor to `~/dsv41-opt-20261021/ops/rel/`.
4. Weights: `~/models/dsv41/uncensored-fp8` (not in this repo). Engram rows are punched on spark-one/two/three — never rsync model files from a punched node.
5. Launch via the supervisor: `dsv41-relaunch.sh --expect-launcher 5b0a889c`.
