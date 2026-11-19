# FLASH hyperparameter sensitivity

Exploratory historical runs; see metadata.json and selected_runs.csv before citing.
Matching: CSV metadata only; evaluator and core code may differ.
Confidence: 95.0%; 50,000 bootstrap replicates; bootstrap seed=2027.
All deltas are variant minus reference. Intervals are pointwise and conditional on the selected games/seeds.

| Comparison | Variant / reference | Games | Seed pairs | Mean HNS reference → variant | Delta [CI] | CI within supplied margin |
|---|---|---:|---:|---|---|---|
| threshold | 3.4 / 3.5 | 9 | 26 | 0.734867 → 0.734733 | -0.000134 [-0.021968, +0.021872] | not assessed |
| threshold | 3.55 / 3.5 | 9 | 26 | 0.734867 → 0.702229 | -0.032638 [-0.066873, +0.001347] | not assessed |
| hash_bits | 9 / 10 | 5 | 12 | 0.957136 → 0.915126 | -0.042010 [-0.083121, -0.000933] | not assessed |

Check per_game.csv and the mean/max absolute game deltas in summary.csv for cancellation across games.
A CI containing zero does not establish that performance changes are small.
Code matching covers four files; train.py differences and unrecorded settings still require review.
