# CPU wake-chain test (2026-10-24 UTC), DSv4.1 TP8 live 1M config (launcher 80c749ca, CANON=1)

Root came only from throwaway `docker run --rm --privileged` containers (kindling-spark-os:0.9.5). Every change was runtime sysfs/procfs or a held `/dev/cpu_dma_latency` fd. Nothing in /etc was touched, nothing was persisted, and nothing was rebooted or restarted. All 8 nodes were reverted and verified afterwards (mac/vf_*.txt): all 80 cpuidle `disable` values match the originals in orig/, IRQ smp/effective affinity match, cpu_dma_latency=2000000000, and no wake-qos container is left.

## Step 0 (root read-only)
- ASPM is already disabled on every link: the CX-7 x4 functions report "ASPM not supported", and LnkCtl shows "ASPM Disabled" on the root ports and on the GPU 000f:01:00.0, with L1 substates off. Policy is [default]. ASPM is not a lever here.
- Idle states (acpi_idle, menu governor): LPI-0 WFI, LPI-1 (42 us exit / 1930 us residency), LPI-2 (231/2542, never used), LPI-3 (433/2542).
- The hot GPU IRQ (nvidia 487/483/434) has affinity 0-19 and effective cpu0, which is a 2.8 GHz A725 core (the big X925 cores are 5-9 and 15-19). It is not managed and can be steered. The mlx5 comp IRQs are pinned one per CPU, not managed, and can also be steered. irqbalance is not running.
- The hot threads (VLLM::Worker ~47% of a core, Worker_TP, EngineCore) are not pinned (0-19) and migrate between samples, so no core is reliably busy.

## Arms (live toggles, all 8 nodes in parallel, ~12 s settle + warmup excluded)
S1 = PM-QoS cpu_dma_latency=0 (WFI only). S2a = LPI-2/3 off on cpu0. S2b = LPI-2/3 off on all cores. S3 = hot nvidia IRQ + top-2 mlx5 comp IRQs steered to cpu0, with cpu0 LPI-2/3 off. S4q = S1 + the S3 IRQ steering.
Randomized paired ABAB: 6 pairs per arm (A/A 6+3). Metrics are matched_bench C1 coding/prose ms/step (3 trials each), bench.py single code 1024 tok/s, and conc8 512 tok/s.

## Results: paired delta (T-C)/C %, mean +- 95% t-CI (ms/step: negative = faster)
| arm | n | C1 coding ms/step | C1 prose ms/step | single tok/s | conc8 tok/s | LPI-3/s on spark-two cpu0 / all cores (C -> T) |
|---|---|---|---|---|---|---|
| A/A | 6 | +0.23 +-0.65 | -0.18 +-1.51 | +0.04 +-0.13 | +0.78 +-2.10 | 204 / 1505 -> 212 / 1537 |
| A/A (calib) | 3 | +0.32 +-1.08 | +0.16 +-1.05 | +0.06 +-0.19 | -0.25 +-2.89 | |
| S1 QoS=0 | 6 | -0.44 +-0.39 | -0.25 +-0.68 | +0.47 +-0.09 | +0.47 +-1.33 | 186 / 1473 -> 12 / 156 |
| S2a cpu0 shallow | 6 | +0.40 +-0.68 | -0.84 +-0.82 | -0.07 +-0.17 | -0.71 +-2.04 | 197 / 1513 -> 41 / 1555 |
| S2b all shallow | 6 | +0.23 +-0.75 | -0.04 +-0.76 | -0.06 +-0.03 | +0.70 +-1.36 | 214 / 1534 -> 0 / 0 |
| S3 IRQ sink cpu0 | 6 | +0.01 +-0.26 | +0.27 +-0.83 | +0.00 +-0.16 | -0.50 +-1.56 | 210 / 1512 -> 0 / 1272 |
| S4q QoS+steer | 6 | -0.16 +-0.42 | -0.54 +-0.46 | +0.46 +-0.05 | +2.08 +-1.43 | 207 / 1487 -> 10 / 181 |
Draft acceptance length was identical in every pair (2.551/2.551), so all deltas are step time.

## Power (spark-one spbm only; the other nodes have no power telemetry)
Under load, mean over the block: S1 +4.8 W sys_total (cpu_p +3.9 W), S4q +2.7 W (cpu_p +3.1 W). S2a, S2b and S3 are within +-0.5 W.
At idle, 2 x 40 s each: S0 37.2 W, S1 41.1 W (cpu_p +3.0 W), S4q 39.8 W (cpu_p +2.7 W). S2a/S2b/S3 are 35.5-36.4 W, within noise of S0, with cpu_p unchanged.
Expect roughly +25-40 W fleet-wide for QoS=0 on all 8 nodes.

## Verdict
No toggle reaches the 1.5% C1 ms/step gate. The best C1 result is S1 at -0.44% coding (CI just excludes 0) and S4q at -0.54% prose, against an expected ceiling of about 3%. Killing LPI-3 entirely (S2b, 0 entries/s) changes nothing, so the 433 us LPI-3 exit is not on the critical path at C1. The only effect at or above 1.5% is S4q conc8 +2.1 +-1.4% (single arm, 6 pairs; the A/A conc8 CI is +-2.1), which costs about 3 W per node. Nothing should be persisted. If it is ever wanted, re-test conc8 with S4q alone at >=12 pairs first.
