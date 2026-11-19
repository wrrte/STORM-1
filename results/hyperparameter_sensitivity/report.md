# FLASH hyperparameter sensitivity

Exploratory historical runs; see metadata.json and selected_runs.csv before citing.
Matching: CSV metadata plus identical evaluator/core code in four files; train.py still needs review.
Confidence: 95.0%; 50,000 bootstrap replicates; bootstrap seed=2027.
All deltas are variant minus reference. Intervals are pointwise and conditional on the selected games/seeds.

| Comparison | Variant / reference | Games | Seed pairs | Mean HNS reference → variant | Delta [CI] | CI within supplied margin |
|---|---|---:|---:|---|---|---|
| threshold | 3.4 / 3.5 | 3 | 7 | 0.314546 → 0.317427 | +0.002881 [-0.015530, +0.022238] | not assessed |
| threshold | 3.55 / 3.5 | 3 | 7 | 0.314546 → 0.283415 | -0.031131 [-0.085399, +0.025590] | not assessed |

No eligible matched panel for hash_bits; no statistics computed.

Check per_game.csv and the mean/max absolute game deltas in summary.csv for cancellation across games.
A CI containing zero does not establish that performance changes are small.
Code matching covers four files; train.py differences and unrecorded settings still require review.
