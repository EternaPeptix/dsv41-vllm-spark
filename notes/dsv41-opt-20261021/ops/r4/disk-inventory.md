# Disk inventory: 8 Sparks (read-only, 2026-10-22 04:47 UTC)

Nothing was deleted or changed. Raw outputs: `r4/disk-raw/spark-<host>.txt` (pass 1) and `spark-<host>.2.txt` (pass 2). The scripts are `inv.sh` and `inv2.sh` in the same folder; both are read-only and were run at `nice -n 19`.
There is no NOPASSWD sudo, so `/var/lib/docker` could not be measured with du. Docker sizes below come from `docker system df -v`.

Note on du numbers: `~/.cache/uv` reports 13-20 GB per host when measured alone. Most of that is hard-linked into the venvs (whole-home du counts it only once), so pruning it frees only a few GB.

## 1. Free space

| host | size | used | free | use% | home du | non-home (swap 16G + OS + docker + logs) |
|---|---|---|---|---|---|---|
| spark-one (rank0) | 916G | 857G | **13G** | **99%** | 728G | ~129G (docker 55G) |
| spark-two | 916G | 854G | **16G** | **99%** | 698G | ~156G (docker bfs images ~27G + broken container) |
| spark-three | 916G | 737G | 132G | 85% | 674G | ~63G |
| spark-four | 916G | 689G | 181G | 80% | 625G | ~64G |
| spark-five | 916G | 707G | 162G | 82% | 630G | ~77G |
| spark-six | 916G | 811G | **59G** | 94% | 727G | ~84G (docker 34.7G) |
| spark-seven | 916G | 752G | 118G | 87% | 668G | ~84G (docker 34.7G) |
| spark-eight | 916G | 738G | 132G | 85% | 655G | ~83G (docker 34.7G) |

Every host has a single 916G nvme (`/run/sparkos/host`, which also serves `/home` and `/tmp`), a 16G `swap.img`, the spark-os erofs (1.9G) and `var/log` (~2.5G on spark-one).

## 2. Per-host tables (sizes from du)

### Common to all 8 (each host)
| item | size | status |
|---|---|---|
| ~/models/dsv41/uncensored-fp8 | 476G | **LIVE model**: keep. It is the only model copy on each host; no duplicates. HF caches hold only small stubs (<200M). |
| ~/venvs/vllm032 | 7.7G | **LIVE**: keep |
| ~/rdma_ar_dsv41 | ~1-3M | **LIVE** (launcher + rdma_all_reduce.py load `~/rdma_ar_dsv41/build/rdma_ar_ext.so`): keep |
| ~/dsv41-stage | ~120M | **LIVE**: launcher sets `DSV41_TARGET_M32_NATIVE_DIR=$HOME/dsv41-stage/m32/target-m32-native-v1`. r3 also stages fc2c here: keep |
| ~/dsv41-opt-20261020 | 116M (spark-one 1.2G) | Codex/r1-r2 experiment dirs, baselines and rollback launchers. Keep until r3/r4 finish. |
| ~/dsv41-opt-20261021 | 29M (spark-one only) | README/ops for rollbacks: keep |
| ~/dsv41-r2prof | 88M | r2 profile source: keep for now |
| ~/.cache/vllm | 148M-954M | torch.compile/graph cache for the live service. Keep: deleting it slows the next restart. |
| ~/queen (see §3) | 92-150G | Queen project |
| ~/voiceform | 19-20G (spark-six 133G) | voice retraining project, last mtime 2026-09-18/19 |
| ~/sparkos-prebuilt/0.9.5-1019 | 2.0G | spark-os rollback material: keep |
| ~/.local/lib | 9.0G (one-five) | user pip site-packages. Unknown dependents (Queen?), so keep. |
| ~/.nv/ComputeCache | 0.4-1.0G | CUDA JIT cache; regenerates |
| ~/.rustup | 1-3.9G | toolchains (ECDSA work) |
| rdma_ar, rdma_ar2..5, mimo-*, devimg*, mimo-tp8.sh | <3M each | MiMo-era leftovers, negligible. The launcher only mentions mimo-tp8.sh in a comment. |

### spark-one (13G free): host-specific
| item | size | last mtime | notes |
|---|---|---|---|
| docker: 6 dangling `<none>` images (MiMo sglang dev builds, created 2026-10-05, `COPY mimo_v2.py ... /sgl-workspace/sglang`) | **33.7G shared** | 10-05 | 0 containers. These are leftovers of the 10-20 mimo-gb10 cull. |
| docker: altfreezing-aarya:unlimited + its untagged base eea3fe49 (11 months old, 19.5G shared) | 19.7G | 3 wk | 1 exited container (altfreezing-aarya-unlimited). User project. |
| docker: brainicism/bgutil-ytdlp-pot-provider | 0.4G | 6 wk | exited container bgutil-provider. Pairs with clipfarmer. |
| ~/imprint (clipfarmer 60G, parity 1.3G) | 61G | 10-05 | user project |
| ~/ComfyUI (models 41G, .venv 7G) | 48G | 10-05 | user project |
| ~/queen | 93G | | worker copy (§3) |
| ~/.cache/uv | 14G apparent / ~6.4G unique | | `uv cache prune` frees little |
| ~/exo-venv-9b23.tar.zst | 1.6G | | exo/K3 venv tarball; K3 instance killed 10-13 |
| ~/r3s | 1.8G | 10-08 | ECDSA/r3 rows |
| ECDSA worktrees (wt-*, verify-*, int, grind2, lamx, exl3-arm64-v26) | ~2.5G total | | ECDSA repo moved to the Mac |
| ~/.cache/glm52-exl3-tr3 | 346M | | GLM-5.2 era |
| ~/spbm-work | 11M | | spbm power logs: keep |

### spark-two (16G free)
| item | size | last mtime | notes |
|---|---|---|---|
| ~/bfs-spark (downloads 41G, runpod-full-build 39G) | **80G** | 10-09 | user project (BFS face swap) |
| docker bfs-ltx25:latest 12.7G, bfs-spark:20261001 12.5G, moby/buildkit 0.36G, tonistiigi/binfmt 0.1G | ~25.7G (some layers shared) | 12 d / 2 wk | same project |
| docker broken container d8be405a… (rw layer snapshot not found) | ? | | `docker ps -a` errors on it; it was left alone on 10-20 |
| ~/queen | 92G | | worker copy |
| ~/sq | 3.5G | 10-08 | ECDSA |
| ~/exo-venv-9b23.tar.zst | 1.6G | | exo leftover |
| ~/.rustup | 3.9G | | |
| ECDSA misc (q1263-*, fx-fa-genmul.*, ecdsa-*, wt-*) | ~1.3G | | |
| ~/k3.sh, ~/r3A_k3check.py | 4K | | kept on 10-20 as ambiguous |

### spark-three (132G free): Queen coordinator
| item | size | notes |
|---|---|---|
| ~/queen/models/Qwen3.8-27B | **52G** | kept on 10-20 "for redistribution". Referenced by Queen consolidate configs (local Qwen teacher option, still pending the user's teacher decision). |
| ~/queen/models/Qwen3.8-27B-FP8 | **29G** | same |
| ~/queen/queen_pawn-9-s0002500 / s0005000 / s0007500 | 7.5G each | s2500 = round-10 init/miner: keep. s5000/s7500 are used in the strength comparison. |
| ~/queen/queen-src | 24G | runs 15G (release 14G); data/self_distill moved to S2 |
| ~/queen/queen_pawn-8 | 16G | PAWN-8 + venv + lc0 |
| ~/.cache/uv | 13G apparent / ~0.4G unique | |
| ~/r3s 704M, sphere_beat 683M, wt-superopt 392M, r3-A-fold, r3fold, tgt_trap, scratch_pp, sopt_run | ~2.5G | ECDSA |
| ~/queen-move.sh, ~/queen-move.log | 4K | move finished (§4) |
| systemd --user queen-fleet-health.timer | | still **active**, runs the read-only watchdog every 5 min. queen-server.service is disabled. |
| docker kindling-spark-os (4 tags, 1 image) | 1.67G | keep |

### spark-four (181G free)
| item | size | notes |
|---|---|---|
| ~/queen | 100G | worker copy, plus s5000 (7.5G) |
| ~/queen-park | 12K | **empty skeleton** (`queen-src/data/`) left after the move |
| ~/sphere_beat 690M, scr 550M, r3sq 305M, r3-A-square, wt-* | ~2G | ECDSA |
| ~/.cache/uv | 13G apparent | |
| ~/exo-venv-9b23.tar.zst | 1.6G | |

### spark-five (162G free)
| item | size | notes |
|---|---|---|
| ~/queen | 99G | worker copy, plus queen_hce-4 8.4G |
| ~/queen-park | 12K | **empty skeleton** (`queen-src/runs/`) |
| ~/samuel (.venv 5.5G) + ~/samuel-data (LibriSpeech) + ~/hf-cache | 6.2G + 1.4G + 1.2G | mtime 2026-08-18, voice project precursor |
| ~/src/glm52-spark-runtime | 922M | GLM-5.2 era (07-27) |
| ~/exl3-arm64, ~/exl3-arm64-v26 | 209M each | EXL3 builds |
| ECDSA (grind2 466M, wt-grind, ecdsa-1007, grindtools, wt-rows2, r2, vp, ve) | ~1.4G | |

### spark-six (59G free)
| item | size | notes |
|---|---|---|
| ~/voiceform (runs 76G, data 49G, ckpts 3.1G) | **133G** | voice project main training host. Backbone v3 frozen 09-25; per memory, decisions go through the user. |
| docker: 6 dangling MiMo sglang dev images | **33.7G shared** | 0 containers |
| ~/queen | 100G | worker copy, plus s7500 |
| ~/.cache/uv 20G apparent / ~7.4G unique, ~/.cache/pip 0.9G | | |
| ~/wt-loans 832M, ~/lam 809M, ~/scr 474M, cryptominisat 115M, … | ~2.5G | ECDSA |

### spark-seven (118G free)
| item | size | notes |
|---|---|---|
| docker: 6 dangling MiMo sglang dev images | **33.7G shared** | 0 containers |
| ~/eve (proj 14G, .mamba 13G) | 29G | user project, mtime 10-05 (the Hemingway snapshot was already removed 10-13) |
| ~/imprint/spike-fakegen | 17G | user project, 09-29 |
| ~/cisco8101-backup | 5.4G | **switch recovery images: keep** |
| ~/ztp (kx 2.8G, www) | 5.0G | switch ZTP/iPXE: keep |
| ~/queen | 100G | worker copy, plus s7500 |
| ~/models/dsv41: base.sha, base.revision, dl/verify logs, small *.sh, mimo-dl/get-mimo scripts | KBs | base-fp8 itself was deleted 10-20; the hashes are kept for the eval re-download |
| ~/lam 1.4G, wt-loans 515M, wt-lambda, payfloor, rad-metric | ~2.5G | ECDSA |
| ~/.cache/pip | 2.4G | |

### spark-eight (132G free)
| item | size | notes |
|---|---|---|
| docker: 6 dangling MiMo sglang dev images | **33.7G shared** | 0 containers |
| ~/eve (proj 11G, .mamba 6.7G, venv 5.2G) | 25G | user project, mtime 10-16 (recent) |
| ~/heesch-curate (repo 5.7G, raw 5.2G) | 12G | Heesch campaign is archived to a private repo; fleet ~/heesch removed 10-05 |
| ~/envdata | 11G | mtime 10-07, owner unclear |
| ~/queen | 101G | worker copy, plus s5000 |
| ~/models/*.sh, missing*.txt, mimo-half-dl.log | KBs | MiMo download leftovers |

## 3. Queen footprint (all 8)
- Worker copy on each of the 7 non-coordinator nodes is 92-101G. It contains:
  - `queen-src/runs/self_distill/iteration/step_{1250..7500}`, about 31G per node. These are the **per-rank FSDP shards** of the round-9 training (`random_states_<rank>.pkl`, `optimizer_0`, `pytorch_model_fsdp_0`). The S2 archive holds only spark-three's rank-0 shards (`random_states_0.pkl`). So the worker shards are **unique**: deleting them gives up any optimizer-state resume of round 9. Round 10 inits from the HF export `queen_pawn-9-s0002500`, so it does not need them.
  - `queen-src/runs/release/pawn-8` (14G per node) and `queen_pawn-8` (16G incl. 7.5G venv and lc0). This may be a duplicate; compare before treating it as one.
  - `queen-src/data/self_distill` (7-9.6G: `iteration` 6.3G and `round10` 2.8G mined partials).
  - `queen-src/.venv` 5.1G, `models/SmolLM3-3B` 5.8G.
- Coordinator spark-three: 150G, which includes 81G of Qwen3.8-27B (+FP8).
- Round-10 rebalanced data and spark-three's `runs/self_distill` and `pilot-train` now live **only on 512S2 `~/queen-archive`** (294G). Training needs it copied back (this is already in memory).

## 4. Queen move status and the temp key
- systemd --user `queen-move` on spark-three is **finished and no longer loaded**. `~/queen-move.log` ends with:
  - `2026-10-20 14:39 A DONE` (park on q4/q5 over the fabric), then `14:40 B VERIFIED runs/pilot-train on S2`
  - `22:31 C VERIFIED data/self_distill on S2 (3197 files); deleting park copy on q4`
  - `2026-10-21 01:03:44 C VERIFIED runs/self_distill on S2 (78 files); deleting park copy on q5`
  - `01:03:44 temp key removed from S2`, then `ALL DONE`
- `~/queen-park` on spark-four and spark-five: only empty directory skeletons are left (12K).
- **512S2 `~/.ssh/authorized_keys`: "queenmove-temp" is NOT present** (grep count 0). The script removed it at 01:03:44 PDT on 10-21.
  - The file has 7 lines. Their comments are jeweled@512S1, ultor@olympusmons, kevinvegda@gmail.com, sufithaire@gmail.com and three keys with no comment.
  - This is reported only; nothing was changed. Worth a glance from the user if any of those keys are unexpected.
- 512S2 `~/queen-archive` = 294G.

## 5. Ranked deletion PROPOSAL (nothing deleted; each item needs user approval)

Ranking is by safety, then by benefit to the near-full hosts (spark-one/two at 99%: a log or compile-cache burst on rank 0 could fill the disk mid-run).

| # | what | where | size freed | why safe | depends on it | how (when approved) |
|---|---|---|---|---|---|---|
| 1 | 6 dangling MiMo sglang dev images (created 10-05, `/sgl-workspace/sglang` + `mimo_v2.py`) | one, six, seven, eight | **~33.7G each, ~135G total** | untagged, 0 containers; MiMo weights deleted 10-08 and image tags culled 10-20; layer-shared only with each other (2/85 layers in common with altfreezing) | nothing | `docker image prune` (dangling only, no `-a`) |
| 2 | `~/queen-park` empty skeletons | four, five | ~0 | the move verified and deleted the payload | nothing | `rm -r ~/queen-park` |
| 3 | `~/exo-venv-9b23.tar.zst` | one, two, three, four | 1.6G each (6.4G) | exo/K3 instance killed and daemons disabled 10-13 | nothing known | rm |
| 4 | Small dead project files: rdma_ar..rdma_ar5 (NOT rdma_ar_dsv41), mimo-*, devimg*, ~/models/{mimo,missing}*, `~/.cache/glm52-exl3-tr3`, `~/src/glm52-spark-runtime` (five), exl3-arm64* | various | ~2G total | MiMo/GLM eras over | the launcher only mentions mimo-tp8.sh in a comment | rm |
| 5 | Caches: `uv cache prune`, `~/.cache/pip`, `~/.nv/ComputeCache` (NOT `~/.cache/vllm`, NOT `~/.triton`/deep_gemm JIT) | all | ~1-8G per host (spark-one ~6G, six ~8G) | regenerable | none live | `uv cache prune`; rm pip cache |
| 6 | Queen pawn-9 s5000/s7500 copies on workers (keep spark-three's) | four, eight (s5000); six, seven (s7500) | 7.5G each, ~30G | the canonical copies stay on spark-three; round 10 uses s2500 | strength chart re-runs (those can use spark-three) | user call |
| 7 | spark-two BFS project: `~/bfs-spark` + images bfs-ltx25/bfs-spark/buildkit/binfmt | two | **~105G** | inactive since 10-09; fixes spark-two at 99% | user project | **user decision** (archive downloads first?) |
| 8 | spark-one user projects: ComfyUI 48G, imprint/clipfarmer 60G, altfreezing image+base ~20G, bgutil 0.4G (+2 exited containers) | one | **up to ~128G** | inactive since 10-05; fixes rank-0 at 99% | user projects | **user decision**; at minimum consider moving ComfyUI/models to a node with free space |
| 9 | Queen round-9 per-rank FSDP shards `queen-src/runs/self_distill/iteration` | 7 workers | ~31G each, ~217G | round 10 inits from the HF export | gives up optimizer-state resume of round 9; S2 archive has rank 0 only, so these shards are unique | **user decision** |
| 10 | spark-three Qwen3.8-27B (52G) + FP8 (29G) | three | 81G | re-downloadable from HF | possible local-Qwen consolidation teacher (pending decision); "redistribution" note from 10-20 | user decision, after the teacher choice |
| 11 | voiceform on spark-six (runs 76G, data 49G) and the 19G copy on every other node | six (+7 nodes) | up to ~133G + ~130G | backbone frozen 09-25 | voice project; decisions through user | **user decision**: do not touch by default |
| 12 | Other user projects: eve (seven 29G, eight 25G, active 10-16), imprint/spike-fakegen (seven 17G), heesch-curate (eight 12G), envdata (eight 11G), samuel* (five 8.8G) | | ~100G | mostly inactive | user | user decision |
| 13 | ECDSA worktrees/scratch (wt-*, verify-*, grind2, r3s, sq, lam, scr, r3-stack, r7-*, r8-*, …) | all | ~2-5G per host | repo lives on the Mac | grind results may be unmerged | low value; user decision |

**Do NOT delete:** `~/models/dsv41/uncensored-fp8`, `~/venvs/vllm032`, `~/rdma_ar_dsv41`, `~/dsv41-stage` (live M32 native dir), `~/dsv41-opt-20261021` and `~/dsv41-opt-20261020` (rollback/baselines), `~/dsv41-logs`, `~/balloon.py`, `~/.cache/vllm`, `~/sparkos-prebuilt`, `~/spbm-*`, `spark-seven ~/cisco8101-backup` and `~/ztp`, `~/models/dsv41/*.sha` (verification hashes), Queen `queen_pawn-9-s0002500` / `queen_pawn-8` / `queen-src` code+venv on all nodes, and the spark-two broken docker container (needs a separate look).

**Minimum to get spark-one/two out of the danger zone:**
- spark-one: #1 (+33.7G) and #3 (+1.6G) take it from 13G to ~48G free with no project impact.
- spark-two: it has no dangling images, so only #3 (+1.6G) and #5 are free of project impact. Real relief needs #7 (BFS, ~105G).
