# TD-error-only anchor insertions per step

Metric: `Retrieval/triggered_anchors_step`. Counts include zero-anchor updates and repeated insertions of the same replay state.
One step is one world-model update (also one environment step for these runs).

Confidence level: 95%; bootstrap replicates: 10000.

| Campaign | Games | Runs | Mean anchors/step | SE | CI |
|---|---:|---:|---:|---:|---|
| b3f54884d761 | 8 | 8 | 4.876850 | 0.697422 | [3.619805, 6.341522] |

Run-level SE and CI use a circular moving-block bootstrap of consecutive updates.
Game-level SE and CI use independent run means; they are undefined for a single run.
Overall SE and CI resample game means with equal game weights, not individual updates.
Within-run intervals describe temporal resampling of the observed training trajectory; learning can be nonstationary.
They do not estimate variability across independently trained seeds.
Exact observed windows and partial-window flags are stored in per_run.csv.
Skipped jobs: 18; invalid/unreadable runs: 0.
