"""Offline, matched-seed sensitivity analysis of historical FLASH development runs.

Python standard library only. Run from any directory; see the adjacent README.
Selection never uses score magnitude. This is an exploratory historical analysis,
not a claim of equivalence or a replacement for a controlled hyperparameter sweep.
"""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import csv


HERE = Path(__file__).resolve().parent
MISSING = {"", "n/a", "none", "nan", "null"}
METRICS = ("mean_hns", "median_hns", "iqm_hns", "optimality_gap")
# Unknown values must match other unknown values; they are never silently filled
# with today's defaults. Inactive dynamic-warmup knobs are deliberately omitted:
# this analysis requires a fixed, observed 50k warmup in every run.
MATCH_FIELDS = (
    "Logic", "Warmup Steps", "Calculated Warmup Steps", "Batch Size Reduction",
    "Retrieval Target", "Anchor Weight", "Anchor Weight R", "Anchor Offset",
    "Warmup Offset Pairs", "Score Combination", "Value Signal", "Neighbor Source",
    "Eval Checkpoint Step",
)
CODE_FILES = ("eval.py", "agents.py", "replay_buffer.py", "utils.py", "train.py")
COMPARISONS = {
    "threshold": ("Z Score Threshold", (3.5, 3.4, 3.55), {"Hash Bits": 10}),
    "hash_bits": ("Hash Bits", (10, 9), {"Z Score Threshold": 3.5}),
}


def canonical(value):
    text = str(value if value is not None else "").strip()
    if text.lower() in MISSING:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return text


def finite_number(value):
    number = canonical(value)
    return number if isinstance(number, float) else None


def load_references(path):
    """Read the existing literal without importing its pandas/Excel dependencies."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "REFERENCE_SCORES"
            for target in node.targets
        ):
            references = {game: tuple(map(float, values[:2]))
                          for game, values in ast.literal_eval(node.value).items()}
            if not references or any(not all(map(math.isfinite, values))
                                     or values[1] <= values[0]
                                     for values in references.values()):
                raise ValueError("Invalid Random/Human normalization references")
            return references
    raise ValueError(f"REFERENCE_SCORES literal not found in {path}")


@lru_cache(maxsize=None)
def code_versions(repo, commit):
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", str(commit)):
        return {}
    versions = {}
    for filename in CODE_FILES:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}:{filename}"],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            return {}
        if result.returncode == 0:
            versions[filename] = result.stdout.strip()
    return versions


def read_runs(path, references):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"Run Name", "Run ID", "State", "Commit", "Eval Return",
                    "Retrieval Enable", "Seed", "Created At", "Z Score Threshold",
                    "Hash Bits", *MATCH_FIELDS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        runs = []
        seen_ids = set()
        for line, row in enumerate(reader, 2):
            run_id = row["Run ID"].strip()
            if not run_id or run_id in seen_ids:
                raise ValueError(f"CSV row {line}: missing or duplicate Run ID")
            seen_ids.add(run_id)
            games = [game for game in references if row["Run Name"].startswith(
                (game + "_", game + "-"))]
            seed = finite_number(row["Seed"])
            try:
                created = datetime.fromisoformat(row["Created At"].replace("Z", "+00:00"))
                timestamp = created.timestamp() if created.tzinfo else None
            except (ValueError, TypeError):
                timestamp = None
            runs.append({"raw": row, "line": line,
                         "game": games[0] if len(games) == 1 else None,
                         "seed": int(seed) if seed is not None and seed >= 0
                         and seed.is_integer() else None,
                         "score": finite_number(row["Eval Return"]),
                         "timestamp": timestamp})
    return runs


def exclusion_reason(run):
    row = run["raw"]
    if row["State"].lower() != "finished":
        return "not_finished"
    if run["score"] is None:
        return "missing_or_nonfinite_evaluation"
    if row["Retrieval Enable"].lower() not in {"true", "1"}:
        return "retrieval_not_enabled"
    if run["game"] is None or run["seed"] is None or run["timestamp"] is None:
        return "invalid_game_seed_or_creation_time"
    expected = {"Warmup Steps": 50000, "Calculated Warmup Steps": 50000,
                "Batch Size Reduction": "retrieved", "Retrieval Target": 16,
                "Anchor Offset": -2}
    for field, value in expected.items():
        if canonical(row[field]) != value:
            return "wrong_" + field.lower().replace(" ", "_")
    for field, allowed in {
        "Score Combination": {None, "multiply"},
        "Value Signal": {None, "value_diff"},
        "Neighbor Source": {"retrieval"},
    }.items():
        if canonical(row[field]) not in allowed:
            return "wrong_" + field.lower().replace(" ", "_")
    return None


def select_panel(runs, comparison, repo, min_seeds=2, metadata_only=False):
    """Intersect all settings on game, seed, metadata and evaluator/core code.

    For duplicate runs in one exact stratum, use the latest creation time, then
    Run ID. For multiple complete strata, use the most recent complete stratum.
    Neither operation inspects the return, beyond requiring a finite value.
    """
    parameter, values, fixed = COMPARISONS[comparison]
    audit, buckets = [], defaultdict(lambda: defaultdict(list))
    for run in runs:
        row = run["raw"]
        entry = {"comparison": comparison, "csv_row": run["line"],
                 "run_id": row["Run ID"], "run_name": row["Run Name"],
                 "game": run["game"], "seed": run["seed"],
                 "setting": canonical(row[parameter]), "status": "excluded", "reason": ""}
        audit.append(entry)
        reason = exclusion_reason(run)
        if reason is None and (canonical(row[parameter]) not in values or any(
            canonical(row[field]) != value for field, value in fixed.items()
        )):
            reason = "outside_comparison_settings"
        versions = {}
        if reason is None:
            versions = code_versions(str(repo), row["Commit"])
            if not metadata_only and set(versions) != set(CODE_FILES):
                reason = "source_version_unavailable"
        if reason:
            entry["reason"] = reason
            continue
        signature = tuple(canonical(row[field]) for field in MATCH_FIELDS)
        # train.py changed during development (including an inactive 'half'
        # branch); expose that difference in the audit, rather than claiming
        # identical training code. The other four files must match by default.
        if not metadata_only:
            signature += tuple(versions[name] for name in CODE_FILES if name != "train.py")
        candidate = {**run, "setting": canonical(row[parameter]),
                     "versions": versions, "audit": entry}
        buckets[(run["game"], run["seed"], signature)][candidate["setting"]].append(candidate)
        entry["reason"] = "no_common_metadata_or_code_across_all_settings"

    complete = defaultdict(list)
    for (game, seed, _), by_setting in buckets.items():
        if set(by_setting) != set(values):
            continue
        picked = []
        for value in values:
            ordered = sorted(by_setting[value], key=lambda run: (
                run["timestamp"], run["raw"]["Run ID"]), reverse=True)
            picked.append(ordered[0])
            for duplicate in ordered[1:]:
                duplicate["audit"]["reason"] = "duplicate_older_created_run"
        complete[(game, seed)].append(picked)

    panel = defaultdict(list)
    for (game, seed), strata in sorted(complete.items()):
        strata.sort(key=lambda group: (min(run["timestamp"] for run in group),
                    tuple(run["raw"]["Run ID"] for run in group)), reverse=True)
        for group in strata[1:]:
            for run in group:
                run["audit"]["reason"] = "older_complete_metadata_stratum"
        panel[game].append(strata[0])
    for game in list(panel):
        if len(panel[game]) < min_seeds:
            for group in panel.pop(game):
                for run in group:
                    run["audit"]["reason"] = "insufficient_common_seeds_in_game"
    for groups in panel.values():
        for group in groups:
            for run in group:
                run["audit"].update(status="selected", reason="common_game_seed_and_configuration")
    return dict(panel), audit


def quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def interval(samples, confidence):
    ordered = sorted(samples)
    tail = (1 - confidence) / 2
    return quantile(ordered, tail), quantile(ordered, 1 - tail)


def metrics(games):
    """Match repository conventions; game means receive equal weight.

    IQM pools seed HNS and removes floor(N/4) from each tail, so unequal seed
    counts give games unequal weight for IQM only. No scores are clipped.
    """
    means = [statistics.fmean(scores) for scores in games]
    pooled = sorted(score for scores in games for score in scores)
    trim = len(pooled) // 4
    return (statistics.fmean(means), statistics.median(means),
            statistics.fmean(pooled[trim:len(pooled) - trim]),
            statistics.fmean(max(0, 1 - mean) for mean in means))


def analyze_panel(panel, comparison, references, reps=50000, confidence=0.95,
                  random_seed=2027, equivalence_margin=None):
    """Paired percentile bootstrap of training seeds within fixed games."""
    if not panel:
        return [], []
    if reps < 2 or not 0 < confidence < 1:
        raise ValueError("Require reps >= 2 and 0 < confidence < 1")
    if any(len(groups) < 2 for groups in panel.values()):
        raise ValueError("At least two common training seeds per game are needed for CIs")
    _, values, _ = COMPARISONS[comparison]
    game_names = sorted(panel)
    arrays = []
    for game in game_names:
        baseline, human = references[game]
        arrays.append([[(run["score"] - baseline) / (human - baseline)
                        for run in group] for group in panel[game]])
    points = [metrics([[row[index] for row in game] for game in arrays])
              for index in range(len(values))]
    boot_delta = [[[] for _ in METRICS] for _ in values[1:]]
    boot_game_delta = [[[] for _ in game_names] for _ in values[1:]]
    rng = random.Random(random_seed)
    for _ in range(reps):
        sampled = [[game[rng.randrange(len(game))] for _ in game] for game in arrays]
        sampled_metrics = [metrics([[row[index] for row in game] for game in sampled])
                           for index in range(len(values))]
        for variant in range(1, len(values)):
            for metric_index in range(len(METRICS)):
                boot_delta[variant - 1][metric_index].append(
                    sampled_metrics[variant][metric_index] - sampled_metrics[0][metric_index])
            for game_index, game in enumerate(sampled):
                boot_game_delta[variant - 1][game_index].append(
                    statistics.fmean(row[variant] - row[0] for row in game))

    summary, per_game = [], []
    for variant in range(1, len(values)):
        deltas = [statistics.fmean(row[variant] - row[0] for row in game) for game in arrays]
        for metric_index, metric in enumerate(METRICS):
            low, high = interval(boot_delta[variant - 1][metric_index], confidence)
            summary.append({
                "comparison": comparison, "reference": values[0], "variant": values[variant],
                "metric": metric, "reference_value": points[0][metric_index],
                "variant_value": points[variant][metric_index],
                "delta_variant_minus_reference": points[variant][metric_index] - points[0][metric_index],
                "ci_low": low, "ci_high": high, "confidence": confidence,
                "ci_excludes_zero": low > 0 or high < 0,
                "equivalence_margin_hns": equivalence_margin if metric == "mean_hns" else None,
                "ci_within_equivalence_margin": (
                    low > -equivalence_margin and high < equivalence_margin
                    if equivalence_margin is not None and metric == "mean_hns" else None),
                "games": len(panel), "seed_pairs": sum(map(len, panel.values())),
                "mean_abs_game_delta_hns": statistics.fmean(map(abs, deltas)),
                "max_abs_game_delta_hns": max(map(abs, deltas)),
            })
        for game_index, game in enumerate(game_names):
            groups = panel[game]
            reference_scores = [group[0]["score"] for group in groups]
            variant_scores = [group[variant]["score"] for group in groups]
            raw_deltas = [other - base for base, other in zip(reference_scores, variant_scores)]
            low, high = interval(boot_game_delta[variant - 1][game_index], confidence)
            per_game.append({
                "comparison": comparison, "reference": values[0], "variant": values[variant],
                "game": game, "seed_pairs": len(groups),
                "seeds": ";".join(str(group[0]["seed"]) for group in groups),
                "reference_return_mean": statistics.fmean(reference_scores),
                "variant_return_mean": statistics.fmean(variant_scores),
                "reference_return_sd": statistics.stdev(reference_scores),
                "variant_return_sd": statistics.stdev(variant_scores),
                "paired_return_delta_mean": statistics.fmean(raw_deltas),
                "paired_return_delta_sd": statistics.stdev(raw_deltas),
                "reference_hns": statistics.fmean(row[0] for row in arrays[game_index]),
                "variant_hns": statistics.fmean(row[variant] for row in arrays[game_index]),
                "delta_hns": deltas[game_index], "ci_low": low, "ci_high": high,
                "confidence": confidence,
            })
    return summary, per_game


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields or ["comparison", "status"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "wandb_runs_classification.csv")
    parser.add_argument("--references", type=Path, default=HERE / "convert_csv_to_excel.py")
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--output-dir", type=Path, default=HERE / "hyperparameter_sensitivity")
    parser.add_argument("--comparison", choices=("all", *COMPARISONS), default="all")
    parser.add_argument("--min-seeds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=50000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--equivalence-margin-hns", type=float,
                        help="User-chosen absolute mean-HNS tolerance; never inferred from results")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Exploratory fallback: do not require matching evaluator/core file versions")
    parser.add_argument("--audit-only", action="store_true", help="Select and audit runs without statistics")
    args = parser.parse_args(argv)
    if args.min_seeds < 2 or args.reps < 2 or not 0 < args.confidence < 1 or args.seed < 0:
        parser.error("Require min-seeds >= 2, reps >= 2, 0 < confidence < 1, seed >= 0")
    if args.equivalence_margin_hns is not None and (
        not math.isfinite(args.equivalence_margin_hns) or args.equivalence_margin_hns <= 0
    ):
        parser.error("Equivalence margin must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    references = load_references(args.references)
    runs = read_runs(args.csv, references)
    comparisons = list(COMPARISONS) if args.comparison == "all" else [args.comparison]
    summaries, per_game, selected, audits = [], [], [], []
    coverage = {}
    for name in comparisons:
        panel, audit = select_panel(runs, name, args.repo, args.min_seeds, args.metadata_only)
        audits.extend(audit)
        coverage[name] = {"games": {game: len(groups) for game, groups in panel.items()},
                          "exclusion_counts": dict(Counter(row["reason"] for row in audit
                                                           if row["status"] == "excluded"))}
        for game, groups in panel.items():
            for group in groups:
                for run in group:
                    selected.append({"comparison": name, "setting": run["setting"],
                                     "game": game, "seed": run["seed"], "csv_row": run["line"],
                                     **run["raw"], **{f"source_{key}": value
                                                     for key, value in run["versions"].items()}})
        print(f"{name}: {len(panel)} common games, {sum(map(len, panel.values()))} common seeds", flush=True)
        if not args.audit_only:
            summary, games = analyze_panel(panel, name, references, args.reps,
                                          args.confidence, args.seed, args.equivalence_margin_hns)
            summaries.extend(summary)
            per_game.extend(games)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "selected_runs.csv", selected)
    write_csv(output / "selection_audit.csv", audits)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.csv, args.references, Path(__file__))},
        "coverage": coverage,
        "normalization": references,
        "notes": [
            "Exploratory historical development-run analysis; all reported differences are variant minus reference.",
            "All settings in each comparison share exactly the same games and seed IDs; groups need >= min-seeds.",
            "One observation is a training run's evaluation mean, never an independent evaluation episode.",
            "Duplicate runs: latest creation time, then lexicographically largest Run ID; never highest return.",
            "Fixed games; jointly resample matched training seeds within each game; pointwise percentile CIs.",
            "With two seeds bootstrap CIs may be unstable; CIs do not account for selection of development settings.",
            "Mean/Median/Gap use game mean HNS; IQM pools seed HNS and trims floor(N/4) from each tail.",
            "Missing metadata match only other missing values and remain unverified; current defaults are not imputed.",
            "Default matching requires identical eval.py, agents.py, replay_buffer.py and utils.py Git blobs.",
            "train.py may differ; its Git blob is exported for manual review. CSV omits other training/evaluation knobs and dirty working-tree changes.",
            "Dynamic-warmup settings are inactive: fixed and calculated warmup must both equal 50000.",
            "11-bit target=1 runs are excluded because retrieval target must equal 16.",
            "A CI containing zero is not evidence of equivalence. Optional margin applies only to mean HNS.",
            "A post-hoc equivalence margin is descriptive, not a prespecified confirmatory equivalence test.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    if not args.audit_only:
        write_csv(output / "summary.csv", summaries)
        write_csv(output / "per_game.csv", per_game)
        mode = ("CSV metadata only; evaluator and core code may differ."
                if args.metadata_only else
                "CSV metadata plus identical evaluator/core code in four files; train.py still needs review.")
        report = ["# FLASH hyperparameter sensitivity", "", "Exploratory historical runs; see metadata.json and selected_runs.csv before citing.",
                  f"Matching: {mode}",
                  f"Confidence: {args.confidence:.1%}; {args.reps:,} bootstrap replicates; bootstrap seed={args.seed}.",
                  "All deltas are variant minus reference. Intervals are pointwise and conditional on the selected games/seeds.", "",
                  "| Comparison | Variant / reference | Games | Seed pairs | Mean HNS reference → variant | Delta [CI] | CI within supplied margin |",
                  "|---|---|---:|---:|---|---|---|"]
        for row in summaries:
            if row["metric"] == "mean_hns":
                report.append(f"| {row['comparison']} | {row['variant']:g} / {row['reference']:g} | {row['games']} | {row['seed_pairs']} | "
                              f"{row['reference_value']:.6f} → {row['variant_value']:.6f} | "
                              f"{row['delta_variant_minus_reference']:+.6f} [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | "
                              f"{row['ci_within_equivalence_margin'] if row['ci_within_equivalence_margin'] is not None else 'not assessed'} |")
        for name in comparisons:
            if not coverage[name]["games"]:
                report.extend(["", f"No eligible matched panel for {name}; no statistics computed."])
        report.extend(["", "Check per_game.csv and the mean/max absolute game deltas in summary.csv for cancellation across games.",
                       "A CI containing zero does not establish that performance changes are small.",
                       "Code matching covers four files; train.py differences and unrecorded settings still require review.", ""])
        (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved {'selection audit' if args.audit_only else 'statistics and audit'}: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
