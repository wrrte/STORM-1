| Setting | Mean anchors per update | Within-run SD across updates | 95% CI of the mean |
|---|---:|---:|---|
| Additive (threshold 3.75) | 6.290 | 1.945 | [5.980, 6.590] |

ALE/Qbert-v5, seed 3333, 100 updates at environment steps [50193, 50293). Sample SD (ddof=1); circular block bootstrap, block size 10, 10000 resamples. Zero-anchor updates are included.

Counts are new anchor queue insertions, including repeated replay positions. The environment resets on resume, as in train.py. The interval describes only this short continuation under a locally stationary block-bootstrap approximation; it does not estimate variation across games or training seeds. For the rebuttal, compare both score rules using the same checkpoint and probe window.
