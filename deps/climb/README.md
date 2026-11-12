# Climb kits that production uses (2026-11-08 to 2026-11-11)

These are the master copies of the overlay kits behind every lever the G1 climb promoted. Each kit's `install.py`
patches the `~/venvs/vllm032` site-packages from its `KIT.json` (base sha256 → patched sha256). What the patches
produce is checked in under `vllm-site/`, and `vllm-site/MANIFEST.tsv` is authoritative for what runs. The masters
live on the operator's Mac under `~/Documents/dsv41-offline/<round>/<kit>`. Each kit's `README.md`, `EXACTNESS.md`,
`REVIEW.md` and `TEST-RESULTS.txt` describe the exactness argument, the gates and the review.

| Kit | Launcher flag(s) in `prod/dsv41-tp8.sh` | Promoted |
|---|---|---|
| `climb-r1/r1-pf` | `DSV41_L2_PREFETCH_R1_SHW2=1`, `DSV41_L2_PREFETCH_R1_KVSRC=1` | 11-08, `dcd09a30` |
| `climb-r1/r1-attn` | `DSV41_IDXPAR=1` | 11-08, `dcd09a30` |
| `climb-r1/r1-eager` | `DSV41_DTGRAPH=1` | 11-08, `dcd09a30` |
| `climb-r2/r2-skinny` (kernel: `deps/dsv41-r2/skinny_mx_v2`) | `DSV41_SKINNY_V2=1` | 11-08, `b22e4aa9` |
| `climb-r2/r2-attnout` | `DSV41_ATTNOUT_OP=rope,rq` | 11-08, `b22e4aa9` |
| `climb-r2/r2-draftmoe` | `DSV41_DRAFT_MOE_G17=1` | 11-08, `b22e4aa9` |
| `climb-r2/r2-moeback` | `DSV41_FC2_GATHER_ADD=1` | 11-08, `b22e4aa9` |
| `climb-r2/r2-moefront` | `DSV41_MOE_ROUTEPREP=1` | 11-08, `b22e4aa9` |
| `climb-r3/r3-xsplit` (kernel source in `src/`) | `DSV41_ATTN_XSPLIT=1`, `DSV41_ATTN_XSPLIT_ON_MISMATCH=disable` | 11-08, `166d7aac` |
| `climb-r4/r4-xsretime` | `DSV41_L2_PREFETCH_PACE_GBPS_ATTN=200` | 11-08, `166d7aac` |
| `climb-r5/glue` | `DSV41_GLUE_RQPAD=1`, `DSV41_GLUE_RPSPLIT=split` | 11-09, `4f3cc1c6` |
| `climb-r5/prio` | `DSV41_PRIO_CMP=prio` | 11-10, `b1306620` |
| `climb-r6/step` + `climb-r8/fixb/step` (live `dsv41_step.py` = fixb) | `DSV41_STEP_COLEMB=1` | 11-10, `b1306620` |
| `climb-r6/head` + `climb-r8/fixb/head` (installer re-anchored) | `DSV41_R6_HEAD_SCREEN=1`, `_SCREEN_DQ=2`, `_SCREEN_CFG=64,256,4,2`, `_SELFCHECK=1`, `_REFINE=row` | 11-11, `4a04b175` |
| `climb-r8/comm2fix` (docs only; the kit as installed is `deps/dsv41-stage/r7comm2`) | `RDMA_AR_EXT=…/r7comm2/build/rdma_ar_ext.so`, `DSV41_RDMA_TRIM=1` | 11-11, `58899745` |

Kits that are installed but inert in production are left out of this directory: r1-pdl, r1-drafter, r2-oblit,
r5-ffnfront, r6-skew and r7-l2pf. Their site-packages files are still listed in `vllm-site/` because they differ
from the wheel. The launcher does not set their enabling flags (for example `DSV41_GLUE_PDL` or
`DSV41_OBLIT_FOLD`). The same holds for the g130 stage kit `pdl-fc2c` (`DSV41_MOE_FC2C_PDL`, default off in the
live `sm121_target_m32.py`): its kit is not in `deps/`, and its record is in `notes/dsv41-stage/g130/pdl-fc2c/`.

Left out of every kit: `base/`, `ref/` and `live/` copies of upstream files, compiled objects (`.so`, `.cubin`,
`.ptx`, SASS dumps), full build logs (`build.log`) and result dumps. Each kit keeps its short test and compile
summaries (`TEST-RESULTS.txt`, `COMPILE-LOG.txt`, and small `.txt` outputs under `tests/` and `scripts/`) as
evidence.
