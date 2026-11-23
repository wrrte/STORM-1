ALE/Qbert-v5, seed 3333; 100 updates per threshold; score = add.

| Threshold | Updates | Mean anchors/update | Within-run SD | 95% CI of mean | Status |
|---:|---:|---:|---:|---|---|
| 3.5 | 100 | 6.320 | 2.117 | [5.930, 6.730] | completed |
| 3.75 | 100 | 6.290 | 1.945 | [5.980, 6.590] | completed |
| 4 | 100 | 6.030 | 2.007 | [5.690, 6.380] | completed |
| 4.25 | 100 | 5.650 | 1.877 | [5.300, 6.020] | completed |
| 4.5 | 100 | 5.340 | 1.805 | [5.050, 5.620] | completed |

Every threshold restores the same model, optimizer, replay, EMA and RNG checkpoint in a fresh process. Models and EMA statistics continue updating during each run; retrieval differences can cause the subsequent trajectories to diverge.

Each 95% CI uses a circular moving-block bootstrap (block size 10, 10000 resamples). These are per-run intervals, not simultaneous intervals or estimates of uncertainty across games/seeds. Zero-anchor updates and repeated replay positions are included. See summary.csv and logs for failures.
