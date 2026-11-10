# Main-performance paired permutation p-values

Open `/mnt/hdd_4A/choemj/main_performance_pvalues.xlsx` to see STORM and DRAMA together.

- **Metric P-values**: Mean, Median, IQM, and Optimality Gap; original point estimates and difference CIs; unadjusted two-sided p-values; Holm-adjusted p-values for all eight aggregate tests.
- **Game P-values**: exact paired tests of each game's mean score difference, including N and the selected seeds; Holm adjustment is a separate family of 52 game tests.
- **Definitions / Provenance**: test definitions, assumptions, source paths, and SHA256 hashes.

The inputs are exactly those used for the original CI calculation: STORM 103 paired observations and DRAMA 102, across 26 fixed games each. The script checks every score against the frozen audit and verifies the audit hash saved with the earlier CI. No hypothetical mean-valued seeds or additional Frostbite screening runs are used.

## Test

Within every game/training-seed pair, independently either retain or exchange the baseline and FLASH scores. Recompute the original metric and its difference `T = metric(FLASH) - metric(baseline)` after every swap. Game membership and seed pairing remain fixed. The null assumption is exchangeability of method labels within pairs; this is conditional on the selected paper seeds and fixed benchmark games.

For the four aggregate statistics, 200,000 assignments are sampled with NumPy RNG seed 2027. The two-sided absolute-tail p-value is:

```text
p = (1 + number of assignments with abs(T_swap) >= abs(T_observed) - tolerance)
    / (200000 + 1)
```

Ties use tolerance `100 * float64_eps * max(1, abs(T_observed))`. The absolute-tail convention is explicit; two-sided conventions need not give identical finite Monte Carlo estimates. The paired-swap null distribution is symmetric under exchanging all method labels. Saved null samples allow alternative conventions to be checked without rerunning the test.

For each game, all `2**N` assignments are enumerated and `p = extreme_count / 2**N`; exact tests do not use the Monte Carlo +1 correction. Raw-score and HNS tests are verified to agree.

Method reference: [SciPy's paired sample-label permutation test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), including within-pair swaps, inclusive tails, floating-point ties, and the Monte Carlo correction. The implementation here uses NumPy and retains the repository's four metric definitions; it does not require SciPy.

## Interpretation

All eight aggregate unadjusted p-values exceed 0.05 under this specified test. This does not establish that the methods are equivalent. Holm adjustment across eight tests is also reported.

The earlier percentile bootstrap CI and this paired-label-swap test use different constructions; they are not exact inverses. In particular, DRAMA Mean's earlier percentile CI excludes zero, while its paired-permutation p-value is approximately 0.0706. Both results are retained rather than selecting an inference method by its outcome. The bootstrap distribution was not treated as a null distribution.

`Monte Carlo SE Approx` measures numerical error from sampling a finite number of assignments. It is not the seed SD, the metric's bootstrap SE, or a confidence interval for the effect.

The game-level exact p-values are coarse because each game has only 2–5 pairs. Their complete assignment spaces are retained in the counts; zero differences and ties are included.

## Reproduce

```bash
/mnt/hdd_4A/choemj/miniconda3/envs/storm/bin/python \
  /mnt/hdd_4A/choemj/STORM-1/results/main_performance_pvalues.py
```

CSV results, metadata, and all eight null distributions (`permutation_null_samples.npz`) are saved beside this README. Seven numerical tests cover known exact p-values, independent scalar calculations of all masks, ties, global method-label reversal, batch-invariant random draws, and Holm adjustment:

```bash
cd /mnt/hdd_4A/choemj/STORM-1/results
../../miniconda3/envs/storm/bin/python -m unittest test_main_performance_pvalues -v
```
