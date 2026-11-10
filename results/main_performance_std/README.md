# Main-performance seed standard deviations

Open `/mnt/hdd_4A/choemj/main_performance_std.xlsx` to see STORM and DRAMA together.

- **Game SD**: raw-score mean and sample SD for baseline, Full FLASH, and their paired difference, for every game. N and the exact selected seeds are included.
- **Game HNS SD**: the same statistics in human-normalized score units.
- **Metric Bootstrap SE**: estimates and bootstrap standard errors for Mean, Median, IQM, and Optimality Gap, using the original saved 50,000 paired-bootstrap replicates.
- **Paired Seed Scores**: all 205 original game/seed pairs and their raw scores and HNS.
- **Definitions / Provenance**: definitions, original source paths, and SHA256 hashes.

Sample SD uses `ddof=1`: `sqrt(sum((x - mean(x))**2) / (N - 1))`.
Each observation is one training seed's evaluation episode mean. This is not the SD across evaluation episodes.
The difference column uses `SD(FLASH_seed - baseline_seed)`; it is not the difference of the two methods' SDs.

IQM and the other aggregate metrics have one observed estimate each. Their bootstrap SE is the sample SD of their saved bootstrap estimates; this differs from the game-level sample SD.

Inputs are the frozen selections used for the original CI calculation: STORM 103 pairs and DRAMA 102 pairs. No hypothetical mean-valued seeds or new Frostbite screening runs are included.
The script checks every selected score against the original audit, verifies the audit SHA256 recorded by the CI run, and reproduces the existing point estimates and CI endpoints from the saved bootstrap samples before reporting their SEs.

Reproduce from any working directory:

```bash
/mnt/hdd_4A/choemj/miniconda3/envs/storm/bin/python \
  /mnt/hdd_4A/choemj/STORM-1/results/main_performance_std.py
```

CSV copies and metadata are saved beside this README. Existing paper and CI reports are not modified.
