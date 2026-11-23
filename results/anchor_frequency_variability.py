"""Describe update dispersion separately from uncertainty in mean frequency.

Reads the validated cache produced by anchor_frequency.py; no network access.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import zlib

import numpy as np

import anchor_frequency as frequency


def interval(draws):
    return np.quantile(draws, [.025, .975]).tolist()


def distribution(probabilities):
    """Moments/quantiles of an explicit discrete update-count distribution."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Expected a normalized count distribution")
    counts = np.arange(len(probabilities))
    mean = float(counts @ probabilities)
    return {
        "mean": mean,
        "update_sd": float(np.sqrt((counts - mean)**2 @ probabilities)),
        **{f"p{int(100*q):02}": int(np.searchsorted(np.cumsum(probabilities), q))
           for q in (.05, .25, .5, .75, .95)},
        "zero_fraction": float(probabilities[0]),
    }


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty, finite count series")
    if np.any(values < 0) or np.any(values > 16) or np.any(values != np.floor(values)):
        raise ValueError("Expected integer anchor counts in [0, 16]")
    probabilities = np.bincount(values.astype(int), minlength=17) / len(values)
    result = distribution(probabilities)
    result.update(updates=len(values), min=int(values.min()), max=int(values.max()),
                  sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None)
    if len(values) > 2 and values[:-1].std() > 0 and values[1:].std() > 0:
        result["lag1_correlation"] = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    else:
        result["lag1_correlation"] = None
    return result, probabilities


def circular_block_means(values, block_size, resamples, rng):
    """Circular moving-block bootstrap, retaining exactly N updates per draw.

    Overlapping blocks begin at a uniform index; a final shorter block keeps
    the requested sample length. Prefix sums avoid materializing B*N values.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty finite series")
    if block_size < 1 or resamples < 1:
        raise ValueError("Block size and resamples must be positive")
    n = len(values)
    length = min(block_size, n)
    extended = np.concatenate((values, values[:length-1]))
    prefix = np.concatenate(([0.], extended.cumsum()))
    block_sums = prefix[length:length+n] - prefix[:n]
    complete, remainder = divmod(n, length)
    # Batch the random indices so short block lengths cannot exhaust memory.
    means = np.empty(resamples)
    for left in range(0, resamples, 256):
        right = min(left + 256, resamples)
        starts = rng.integers(n, size=(right-left, complete))
        totals = block_sums[starts].sum(axis=1)
        if remainder:
            partial_sums = prefix[remainder:remainder+n] - prefix[:n]
            totals += partial_sums[rng.integers(n, size=right-left)]
        means[left:right] = totals / n
    return means


def summarize(run_statistics, histograms, bootstrap_samples):
    """Choose game, seed, then update uniformly, preserving original weights."""
    groups = defaultdict(lambda: defaultdict(list))
    for record in run_statistics:
        groups[record["game"]][record["method"]].append(record)
    game_rows, means, probabilities = [], defaultdict(list), defaultdict(list)
    within_sample_variances, within_variances, between_seed_variances = [defaultdict(list) for _ in range(3)]
    for game, methods in sorted(groups.items()):
        row = {"game": game, "paired_seeds": len(methods["full"])}
        for kind in ("full", "add"):
            records = methods[kind]
            if len(records) != row["paired_seeds"] or not records:
                raise ValueError("Expected paired seeds in both methods")
            hist = np.mean([histograms[r["run_id"]] for r in records], axis=0)
            stats = distribution(hist)
            seed_means = [r["mean"] for r in records]
            rms = float(np.sqrt(np.mean([r["sample_sd"]**2 for r in records])))
            row.update({f"{kind}_{key}": value for key, value in stats.items()})
            row[f"{kind}_within_run_sd_rms"] = rms
            row[f"{kind}_seed_mean_sd"] = float(np.std(seed_means, ddof=1)) if len(records) > 1 else None
            means[kind].append(stats["mean"])
            probabilities[kind].append(hist)
            within_sample_variances[kind].append(rms**2)
            within_variances[kind].append(np.mean([r["update_sd"]**2 for r in records]))
            between_seed_variances[kind].append(np.var(seed_means))
        row["mean_difference_add_minus_full"] = row["add_mean"] - row["full_mean"]
        game_rows.append(row)
    if not game_rows:
        raise ValueError("No completed paired games")
    summary = {"games": len(game_rows), "pairs": len(run_statistics)//2,
               "weighting": "equal games; equal paired seeds within game; equal updates within run",
               "bootstrap_unit": "paired game means; observed seeds averaged within each game",
               "bootstrap_samples": bootstrap_samples, "methods": {}}
    histograms_out = {}
    for kind in ("full", "add"):
        histogram = np.mean(probabilities[kind], axis=0)
        histograms_out[kind] = histogram
        summary["methods"][kind] = {
            **distribution(histogram),
            "within_run_sd_rms": float(np.sqrt(np.mean(within_sample_variances[kind]))),
            "game_mean_sample_sd": float(np.std(means[kind], ddof=1)) if len(game_rows) > 1 else None,
            "variance_within_runs": float(np.mean(within_variances[kind])),
            "variance_between_seeds_within_game": float(np.mean(between_seed_variances[kind])),
            "variance_between_game_means": float(np.var(means[kind])),
            "mean_ci95_game_bootstrap": None,
        }
    full, add = np.asarray(means["full"]), np.asarray(means["add"])
    summary["mean_difference_add_minus_full"] = float(np.mean(add-full))
    summary["mean_difference_ci95_game_bootstrap"] = None
    summary["ratio_add_over_full"] = float(add.mean()/full.mean()) if full.mean() else None
    summary["ratio_ci95_game_bootstrap"] = None
    if len(game_rows) > 1:
        indices = np.random.default_rng(0).integers(len(game_rows), size=(bootstrap_samples, len(game_rows)))
        f, a = full[indices].mean(axis=1), add[indices].mean(axis=1)
        summary["methods"]["full"]["mean_ci95_game_bootstrap"] = interval(f)
        summary["methods"]["add"]["mean_ci95_game_bootstrap"] = interval(a)
        summary["mean_difference_ci95_game_bootstrap"] = interval(a-f)
        if np.all(f > 0):
            summary["ratio_ci95_game_bootstrap"] = interval(a/f)
    return game_rows, summary, histograms_out


def plots(output, games, histograms, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, max(6, .34*len(games))))
    y = np.arange(len(games))
    for offset, kind, label in ((-.15, "full", "Full FLASH (3.5)"), (.15, "add", "Additive (4.0)")):
        left.errorbar([r[f"{kind}_mean"] for r in games], y+offset,
                      xerr=[r[f"{kind}_within_run_sd_rms"] for r in games],
                      fmt="o", markersize=4, capsize=2, alpha=.8, label=label)
        right.plot(np.arange(17), histograms[kind], "o-", label=label)
    left.set(yticks=y, yticklabels=[r["game"] for r in games],
             xlabel="Mean anchors/update; bars = within-run SD (RMS), not CI")
    left.invert_yaxis()
    left.legend()
    right.set(xlabel="Anchor insertions in one update", ylabel="Probability (equal game/seed weights)", xticks=range(0, 17, 2))
    right.legend()
    fig.tight_layout()
    fig.savefig(output/"frequency_variability.png", dpi=180)
    fig.savefig(output/"frequency_variability.pdf")
    plt.close(fig)
    columns = min(4, len(games))
    fig, axes = plt.subplots(int(np.ceil(len(games)/columns)), columns,
                             figsize=(4*columns, 2.8*int(np.ceil(len(games)/columns))), squeeze=False)
    for ax, game in zip(axes.flat, games):
        for kind, label in (("full", "Full FLASH"), ("add", "Additive")):
            series = sorted([r for r in phases if r["game"] == game["game"] and r["method"] == kind],
                            key=lambda r:r["env_step_bin_start"])
            x = [r["env_step_bin_start"] for r in series]
            line, = ax.plot(x, [r["mean"] for r in series], label=label)
            ax.fill_between(x, [r["p05"] for r in series], [r["p95"] for r in series],
                            color=line.get_color(), alpha=.15)
        ax.set(title=game["game"], xlabel="Environment step (bin start)", ylabel="Anchors/update", ylim=(0, 16))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    for ax in list(axes.flat)[len(games):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Lines: means; shading: empirical update P05-P95 (not a confidence interval)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output/"frequency_variability_over_time.png", dpi=180)
    plt.close(fig)


def report(output, summary, games, block_sizes, time_resamples):
    def ci(bounds):
        return "N/A" if bounds is None else f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"
    lines = ["# Anchor frequency: dispersion and uncertainty", "",
             f"Same selection as the main analysis: **{summary['games']} games, {summary['pairs']} paired seeds**.", "",
             "| Method | Mean | Within-run SD (RMS) | Total update SD | Mean 95% CI | Update P05-P95 |",
             "|---|---:|---:|---:|---|---|"]
    for kind, name in (("full", "Full FLASH (3.5)"), ("add", "Additive (4.0)")):
        s = summary["methods"][kind]
        lines.append(f"| {name} | {s['mean']:.3f} | {s['within_run_sd_rms']:.3f} | {s['update_sd']:.3f} | "
                     f"{ci(s['mean_ci95_game_bootstrap'])} | {s['p05']}-{s['p95']} |")
    lines.extend(["", f"Additive minus Full FLASH: **{summary['mean_difference_add_minus_full']:.3f} anchors/update**, "
                  f"95% CI {ci(summary['mean_difference_ci95_game_bootstrap'])}.",
                  f"Additive / Full FLASH: **{summary['ratio_add_over_full']:.3f}**, "
                  f"95% CI {ci(summary['ratio_ci95_game_bootstrap'])}.", "",
                  "## What the quantities mean", "",
                  "- **Within-run SD (RMS)**: sample SD (ddof=1) across updates in each run, "
                  "then square-root of the mean variance with equal game/seed weights. This measures temporal dispersion within a run.",
                  "- **Total update SD**: exact empirical SD of a mixture that first chooses a game uniformly, "
                  "then an observed paired seed uniformly, then an update uniformly. It includes within-run, seed, and game differences. "
                  "It uses population moments (ddof=0) of the observed distribution; variance components are saved in variability_summary.json.",
                  "- **Mean 95% CI**: percentile bootstrap of the paired game means, with observed seeds averaged within each game. "
                  "The same game indices are used for both methods and for their difference/ratio. "
                  "It measures sensitivity to game composition, conditional on observed seeds; "
                  "it does not turn correlated updates into independent replicates or estimate uncertainty over future training seeds.",
                  "- **P05-P95**: central empirical update-count range, not a CI for the mean. "
                  "Overlap of update distributions or SD bars does not demonstrate equivalence of means.", "",
                  "## Per-run and seed uncertainty", "",
                  f"per_run_variability.csv includes mean, sample SD, P05/P25/P50/P75/P95, range, zeros, lag-1 correlation, "
                  f"and a conditional circular moving-block bootstrap CI at block length 1000. "
                  f"mean_ci_block_sensitivity.csv repeats it with lengths {block_sizes}, using {time_resamples} resamples and seed 0.", "",
                  "Blocks preserve local time dependence; the final block is truncated so every draw has exactly the observed update count. "
                  "These time-series CIs assume a representative approximately stationary trajectory and adequate block length. "
                  "Learning drift can violate that assumption: use the phase plots and block-length sensitivity, "
                  "and treat these CIs as conditional diagnostics, not confidence intervals across independent training seeds. "
                  "For a fixed completed log its observed mean itself is known exactly.", "",
                  "per_game_variability.csv reports the sample SD of run means across paired seeds. "
                  "Games with one seed have no seed SD (blank), and games with two seeds have weak evidence about seed variability. "
                  "No seed-bootstrap CI is manufactured from one or two seeds.", "",
                  "## Per-game results", "",
                  "Entries are **mean ± within-run SD (RMS)**, not mean confidence intervals.", "",
                  "| Game | Paired seeds | Full FLASH | Additive |",
                  "|---|---:|---:|---:|"])
    for r in games:
        lines.append(f"| {r['game']} | {r['paired_seeds']} | {r['full_mean']:.3f} ± {r['full_within_run_sd_rms']:.3f} | "
                     f"{r['add_mean']:.3f} ± {r['add_within_run_sd_rms']:.3f} |")
    lines.extend(["", "![Mean and dispersion](frequency_variability.png)", "",
                  "![Time evolution and empirical ranges](frequency_variability_over_time.png)", "",
                  "Bootstrap background: [Künsch (1989), stationary observations](https://doi.org/10.1214/aos/1176347265).", ""])
    (output/"variability_report.md").write_text("\n".join(lines))


def analyze(output, bootstrap_samples=10000, block_sizes=(250, 1000, 5000), time_resamples=2000, make_plots=True):
    output = Path(output)
    with (output/"per_run.csv").open() as stream:
        selected = list(csv.DictReader(stream))
    statistics, histograms, block_records = [], {}, []
    phase_groups = defaultdict(list)
    for record in selected:
        directory = output/"raw"/record["run_id"]
        raw = json.loads((directory/"run.json").read_text())
        start, stop, _ = frequency.expected_updates(raw)
        with gzip.open(directory/"anchors.csv.gz", "rt") as stream:
            rows = [{"_step": int(r["_step"]), frequency.METRIC: int(r[frequency.METRIC])}
                    for r in csv.DictReader(stream)]
        frequency.validate_history(rows, start, stop, raw["config"]["JointTrainAgent"]["BatchSize"])
        lo, hi = int(record["env_step_start"]), int(record["env_step_stop_exclusive"])
        values = np.array([r[frequency.METRIC] for r in rows[lo-start:hi-start]])
        if len(values) != int(record["updates"]) or values.sum() != int(record["anchor_insertions"]):
            raise ValueError(f"Cache and selected analysis interval differ: {record['run_id']}")
        stats, probabilities = describe(values)
        row = {"game": record["game"], "seed": int(record["seed"]), "method": record["method"],
               "run_id": record["run_id"], "env_step_start": lo, "env_step_stop_exclusive": hi, **stats}
        for block_size in sorted(set(block_sizes) | {1000}):
            rng = np.random.default_rng(np.random.SeedSequence([0, zlib.crc32(record["run_id"].encode()), block_size]))
            draws = circular_block_means(values, block_size, time_resamples, rng)
            bounds = interval(draws)
            block_records.append({"game": record["game"], "seed": int(record["seed"]),
                                  "method": record["method"], "run_id": record["run_id"], "mean": stats["mean"],
                                  "block_size_requested": block_size, "block_size_used": min(block_size, len(values)),
                                  "resamples": time_resamples, "bootstrap_se": float(draws.std(ddof=1)),
                                  "mean_ci95_low": bounds[0], "mean_ci95_high": bounds[1]})
            if block_size == 1000:
                row.update(mean_ci95_block1000_low=bounds[0], mean_ci95_block1000_high=bounds[1])
        statistics.append(row)
        histograms[record["run_id"]] = probabilities
        for left in range((lo//5000)*5000, hi, 5000):
            a, b = max(left, lo), min(left+5000, hi)
            counts = values[a-lo:b-lo]
            if len(counts):
                phase_groups[record["game"], record["method"], left].append(
                    (np.bincount(counts, minlength=17)/len(counts), len(counts)))
    games, summary, overall_histograms = summarize(statistics, histograms, bootstrap_samples)
    summary.update(time_bootstrap={"method": "circular moving-block bootstrap; observed seed/trajectory conditional",
                                   "block_sizes": sorted(set(block_sizes) | {1000}), "resamples": time_resamples, "rng_seed": 0})
    phases = [{"game": game, "method": kind, "env_step_bin_start": left,
               "seeds_observed": len(records), "updates_observed": sum(r[1] for r in records),
               **distribution(np.mean([r[0] for r in records], axis=0))}
              for (game, kind, left), records in sorted(phase_groups.items())]
    frequency.write_csv(output/"per_run_variability.csv", statistics)
    frequency.write_csv(output/"per_game_variability.csv", games)
    frequency.write_csv(output/"mean_ci_block_sensitivity.csv", block_records)
    frequency.write_csv(output/"per_phase_variability.csv", phases)
    frequency.write_csv(output/"update_count_distribution.csv", [
        {"anchor_count": i, "full_probability": float(overall_histograms["full"][i]),
         "add_probability": float(overall_histograms["add"][i])} for i in range(17)])
    frequency.write_json(output/"variability_summary.json", summary)
    report(output, summary, games, sorted(set(block_sizes) | {1000}), time_resamples)
    if make_plots:
        plots(output, games, overall_histograms, phases)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=frequency.ROOT/"results/anchor_frequency")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--time-resamples", type=int, default=2000)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[250, 1000, 5000])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 2 or args.time_resamples < 2 or min(args.block_sizes) < 1:
        parser.error("Resample counts must be at least two; block sizes must be positive")
    result = analyze(args.output, args.bootstrap_samples, args.block_sizes, args.time_resamples, not args.no_plots)
    print(json.dumps(result, indent=2))
