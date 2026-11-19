| Setting | Mean anchors per update | Within-run SD across updates | SE | 95% CI of the mean |
|---|---:|---:|---:|---|
| TD-error only (threshold 3.5625) | 2.210 | 1.387 | 0.107 | [2.010, 2.420] |

ALE/Qbert-v5, seed 6010, 100 measured updates at environment steps [50559, 50659); 0 preceding updates excluded. Sample SD (ddof=1); circular block bootstrap, block size 10, 10000 resamples. Zero-anchor updates are included.

Counts are new anchor queue insertions, including repeated replay positions. The environment resets on resume, as in train.py. The interval describes only this short continuation under a locally stationary block-bootstrap approximation; it does not estimate variation across games or training seeds. For the rebuttal, compare both score rules using the same checkpoint and probe window.
