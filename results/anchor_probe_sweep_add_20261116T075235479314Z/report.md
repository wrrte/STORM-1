ALE/Qbert-v5, seed 3333; 100 updates per threshold; score = add.

| Threshold | Updates | Mean anchors/update | Within-run SD | 95% CI of mean | Status |
|---:|---:|---:|---:|---|---|
| 3.5 | | | | | interrupted |
| 3.75 | | | | | pending |
| 4 | | | | | pending |
| 4.25 | | | | | pending |
| 4.5 | | | | | pending |

Every threshold restores the same model, optimizer, replay, EMA and RNG checkpoint in a fresh process. Models and EMA statistics continue updating during each run; retrieval differences can cause the subsequent trajectories to diverge.

Each 95% CI uses a circular moving-block bootstrap (block size 10, 10000 resamples). These are per-run intervals, not simultaneous intervals or estimates of uncertainty across games/seeds. Zero-anchor updates and repeated replay positions are included. See summary.csv and logs for failures.
