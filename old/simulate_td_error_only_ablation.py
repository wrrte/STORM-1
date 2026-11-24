#!/usr/bin/env python3
"""Complete TD-only ablation with sampled value-ablation pairs (simulation only).

    ../miniconda3/envs/storm/bin/python results/simulate_td_error_only_ablation.py

Keep completed, matched Full FLASH/TD-only runs fixed. For each missing slot,
sample WITH replacement from that game's matched Full FLASH/Absolute value
inputs to tab:value_signal_ablation. Copy both scores from the same donor seed;
use Absolute value as a hypothetical TD-only score. No noise, shifting, or
best-score selection is applied. These are simulated slots, not new runs or
distinct training seeds. Even donor seeds already observed may be drawn again.

Default: 26 games, three slots per game, 10,000 completions, RNG seed 2027.
Games with no completed TD-only pair are filled entirely from donors. A separate
comparison on the originally observed games isolates completion from changes in
game coverage. Intervals are completion-simulation percentiles, NOT confidence
intervals for actual TD-only training. Sources and the paper are never written.

Requires numpy, pandas, openpyxl (already in the local storm environment).
"""

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

from update_tex import extract_float, load_results, main_table_rows, metric_name
from update_tex_ablation import ABLATIONS


HERE = Path(__file__).resolve().parent
DEFAULT_EXCEL = HERE / "converted_results.xlsx"
DEFAULT_TEX = HERE.parent.parent / "iclr2027_conference.tex"
DEFAULT_OUTPUT = HERE / "td_error_only_simulation"
METHODS = ("Full FLASH", "TD-error only (observed + proxy)")
METRICS = ("#Superhuman", "Mean", "Median", "IQM", "Optimality Gap")


@dataclass(frozen=True)
class Pair:
    seed: int
    full: float
    variant: float


def load_inputs(excel, tex):
    """Reuse the table loader's exclusions, exact configs, and duplicate parsing."""
    references = {}
    for _, parts, _ in main_table_rows(tex.read_text(encoding="utf-8").splitlines(True)):
        if metric_name(parts[0]):
            continue
        game = parts[0].strip()
        random, human = (extract_float(parts[i]) for i in (1, 2))
        if (game in references or random is None or human is None
                or not np.isfinite([random, human]).all() or human <= random):
            raise ValueError(f"Invalid/duplicate Random/Human reference: {game}")
        references[game] = (random, human)
    if not references:
        raise ValueError("No game references in tab:main_performance")

    loaded = []
    for key in ("td_error_only", "value"):
        ablation = ABLATIONS[key]
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seeds = load_results(
                excel, configs=ablation.configs, method_names=ablation.method_names,
                include_seeds=True,
            )
        pairs = {
            game: tuple(Pair(seed, full, variant)
                        for seed, full, variant in zip(seeds[game], *values, strict=True))
            for game, values in scores.items()
        }
        unknown = pairs.keys() - references.keys()
        if unknown:
            raise ValueError(f"Games missing Random/Human references: {sorted(unknown)}")
        for game, values in pairs.items():
            if not all(np.isfinite([p.full, p.variant]).all() for p in values):
                raise ValueError(f"Non-finite input score: {key}/{game}")
        loaded.append(pairs)
    return references, *loaded


def complete(games, observed, donors, target, trials, rng_seed):
    """Return [trial, method, game, slot] raw scores plus first-trial provenance."""
    if target < 1 or trials < 1:
        raise ValueError("target and trials must be positive")
    rng = np.random.default_rng(rng_seed)
    scores = np.empty((trials, 2, len(games), target), dtype=float)
    example = []
    for game_index, game in enumerate(games):
        real = observed.get(game, ())
        pool = donors.get(game, ())
        missing = target - len(real)
        if missing < 0:
            raise ValueError(f"{game}: {len(real)} observed pairs exceed target {target}; "
                             "increase --target-seeds to retain every observed run")
        if missing and not pool:
            raise ValueError(f"{game}: needs {missing} slots but has no value-ablation donors")
        draws = rng.integers(len(pool), size=(trials, missing)) if missing else None
        for slot in range(target):
            is_observed = slot < len(real)
            if is_observed:
                pair = real[slot]
                scores[:, :, game_index, slot] = (pair.full, pair.variant)
            else:
                donor_scores = np.array([(p.full, p.variant) for p in pool])
                chosen = draws[:, slot - len(real)]
                scores[:, :, game_index, slot] = donor_scores[chosen]
                pair = pool[int(chosen[0])]
            example.append({
                "game": game, "slot": slot + 1,
                "is_synthetic": not is_observed,
                "source_table": "td_error_only" if is_observed else "value",
                "source_variant": "TD-error only" if is_observed else "Absolute value",
                "source_training_seed": pair.seed,
                "full_flash_score": pair.full, "td_only_or_proxy_score": pair.variant,
            })
    return scores, example


def calculate_metrics(game_means, pooled_scores):
    """Unrounded HNS inputs; pooled IQM trims floor(N/4) from each tail."""
    pooled = np.sort(pooled_scores, axis=-1)
    trim = pooled.shape[-1] // 4
    middle = pooled[..., trim:pooled.shape[-1] - trim]
    return {
        "#Superhuman": np.sum(game_means > 1.0, axis=-1),
        "Mean": np.mean(game_means, axis=-1),
        "Median": np.median(game_means, axis=-1),
        "IQM": np.mean(middle, axis=-1),
        "Optimality Gap": np.maximum(0.0, 1.0 - game_means).mean(axis=-1),
    }


def completed_metrics(scores, games, references):
    random = np.array([references[g][0] for g in games])[None, None, :, None]
    scale = np.array([references[g][1] - references[g][0]
                      for g in games])[None, None, :, None]
    hns = (scores - random) / scale
    return calculate_metrics(hns.mean(axis=-1), hns.reshape(*hns.shape[:2], -1))


def observed_metrics(games, observed, references):
    means, pooled = [[], []], [[], []]
    for game in games:
        random, human = references[game]
        for method in (0, 1):
            values = [(getattr(p, "full" if method == 0 else "variant") - random)
                      / (human - random) for p in observed[game]]
            means[method].append(np.mean(values))
            pooled[method].extend(values)
    return calculate_metrics(np.array(means)[None, ...], np.array(pooled)[None, ...])


def summary_rows(metrics, scope, game_count):
    rows = []
    for metric, values in metrics.items():
        delta = values[:, 0] - values[:, 1]
        lower_is_better = metric == "Optimality Gap"
        for method, samples in (
            (METHODS[0], values[:, 0]), (METHODS[1], values[:, 1]),
            ("Full FLASH minus TD-only/proxy", delta),
        ):
            rows.append({
                "scope": scope, "games": game_count, "metric": metric, "method": method,
                "example_trial": float(samples[0]), "simulation_mean": float(samples.mean()),
                "simulation_p2_5": float(np.quantile(samples, 0.025)),
                "simulation_p97_5": float(np.quantile(samples, 0.975)),
                "p_full_better": float(np.mean(delta < 0 if lower_is_better else delta > 0)),
                "p_tie": float(np.mean(np.isclose(delta, 0, rtol=0, atol=1e-12))),
            })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-seeds", type=positive_int, default=3,
                        help="number of simulated slots per game (default: 3)")
    parser.add_argument("--trials", type=positive_int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=2027)
    parser.add_argument("--observed-games-only", action="store_true",
                        help="omit games without a completed, matched TD-only pair")
    args = parser.parse_args(argv)
    if args.rng_seed < 0:
        parser.error("--rng-seed must be nonnegative")

    references, observed, donors = load_inputs(args.excel, args.tex)
    original_games = [g for g in references if observed.get(g)]
    if not original_games:
        raise ValueError("No completed, matched TD-only pairs")
    games = original_games if args.observed_games_only else list(references)
    scores, example = complete(games, observed, donors, args.target_seeds,
                               args.trials, args.rng_seed)
    example_games = []
    for i, game in enumerate(games):
        example_games.append({
            "game": game, "observed_pairs": len(observed.get(game, ())),
            "synthetic_slots": args.target_seeds - len(observed.get(game, ())),
            "donor_pairs": len(donors.get(game, ())),
            "full_flash_mean": float(scores[0, 0, i].mean()),
            "td_only_or_proxy_mean": float(scores[0, 1, i].mean()),
        })

    original = observed_metrics(original_games, observed, references)
    scopes = {"completed_observed_games": (
        completed_metrics(scores[:, :, [games.index(g) for g in original_games]],
                          original_games, references), len(original_games)),
    }
    if games != original_games:
        scopes["completed_all_games"] = (completed_metrics(scores, games, references), len(games))
    summary = [row for scope, (metrics, count) in scopes.items()
               for row in summary_rows(metrics, scope, count)]
    original_rows = [{"games": len(original_games), "metric": metric,
                      "full_flash": float(values[0, 0]), "td_error_only": float(values[0, 1]),
                      "full_minus_td": float(values[0, 0] - values[0, 1])}
                     for metric, values in original.items()]
    trials = [{
        "scope": scope, "trial": i + 1,
        **{f"{metric}_{key}": float(values[i, method])
           for metric, values in metrics.items()
           for method, key in enumerate(("full_flash", "td_only_proxy"))},
    } for scope, (metrics, _) in scopes.items() for i in range(args.trials)]

    # Fail before writing if the user accidentally selects a source directory
    # and filename combination that would overwrite an input.
    output = args.output_dir.resolve()
    names = ("example_slots.csv", "example_games.csv", "observed_metrics.csv",
             "simulation_summary.csv", "trials.csv", "inputs.json", "README.md")
    sources = {args.excel.resolve(), args.tex.resolve()}
    if any(output / name in sources for name in names):
        raise ValueError("Output files would overwrite a source file")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("example_slots.csv", example), ("example_games.csv", example_games),
                       ("observed_metrics.csv", original_rows), ("simulation_summary.csv", summary),
                       ("trials.csv", trials)):
        write_csv(output / name, rows)
    metadata = {
        "simulation_only": True, "rng_seed": args.rng_seed, "trials": args.trials,
        "target_slots_per_game": args.target_seeds, "sampling": "paired, with replacement",
        "proxy_assumption": "Absolute value scores stand in for missing TD-only scores",
        "observed_games": original_games,
        "fully_synthetic_games": [g for g in games if not observed.get(g)],
        "sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (args.excel, args.tex)},
        "input_pairs": {key: {g: [asdict(p) for p in values] for g, values in data.items()}
                        for key, data in (("td_error_only", observed), ("value", donors))},
        "configurations": {key: ABLATIONS[key].configs for key in ("td_error_only", "value")},
    }
    (output / "inputs.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")

    lines = [
        "# TD-error-only 가상 시드 보충 시뮬레이션", "",
        f"RNG={args.rng_seed}, 반복={args.trials:,}, 게임별 슬롯={args.target_seeds}.", "",
        "기존 Full FLASH–TD-error-only 공통 시드 결과는 고정합니다. 부족한 슬롯마다 같은 게임의 "
        "Value Difference vs. Absolute Value 표 입력에서 시드 쌍을 균등 복원추출합니다. "
        "Full FLASH 값은 Full FLASH로, Absolute value 값은 가상의 TD-error-only 값으로 사용합니다. "
        "같은 공여 시드를 여러 번 뽑거나 기존 시드와 중복해 뽑을 수 있습니다. "
        "슬롯 수가 3이어도 독립적인 실제 학습 시드 3개를 확보한 것은 아닙니다.", "",
        "구간은 위 가정에서 누락 슬롯을 채우는 무작위성의 2.5–97.5 백분위수입니다. "
        "실제 TD-error-only 성능의 신뢰구간이나 실제 학습 결과가 아닙니다. "
        "기존 관측값은 재표집하지 않으며 관측값과 공여값의 불확실성을 추정하지 않습니다.", "",
        "관측 TD 결과가 전혀 없는 게임: "
        + (", ".join(metadata["fully_synthetic_games"]) or "없음") + ".", "",
        "Mean/Median/Optimality Gap/#Superhuman은 반올림 전 게임별 평균 HNS를 사용합니다. "
        "IQM은 반올림 전 시드 슬롯별 HNS를 모아 양끝 floor(N/4)개씩 제거합니다. "
        "표시는 소수 셋째 자리까지이며, 기존 표의 표시값 반올림과 경계에서 차이가 날 수 있습니다.", "",
        f"기존 {len(original_games)}게임 (보충 전):", "",
        "| Metric | Full FLASH | TD-error only |", "|---|---:|---:|",
    ]
    for row in original_rows:
        lines.append(f"| {row['metric']} | {row['full_flash']:.3f} | {row['td_error_only']:.3f} |")
    for scope, (_, count) in scopes.items():
        lines.extend(["", f"{scope}: {count}게임, 각 {args.target_seeds}슬롯", "",
                      "| Metric | FLASH simulation mean [2.5%, 97.5%] | TD/proxy simulation mean [2.5%, 97.5%] | P(FLASH better) |",
                      "|---|---:|---:|---:|"])
        for metric in METRICS:
            rows = [r for r in summary if r["scope"] == scope and r["metric"] == metric]
            cells = [f"{r['simulation_mean']:.3f} [{r['simulation_p2_5']:.3f}, "
                     f"{r['simulation_p97_5']:.3f}]" for r in rows[:2]]
            lines.append(f"| {metric} | {cells[0]} | {cells[1]} | {rows[0]['p_full_better']:.1%} |")
    lines.extend(["", "- `example_slots.csv`: 첫 반복의 슬롯별 관측/합성 구분·출처 시드·원점수",
                  "- `example_games.csv`: 첫 반복의 게임별 평균·실제 시드 수·추가 슬롯 수",
                  "- `observed_metrics.csv`: 보충 전 관측 게임만의 지표",
                  "- `simulation_summary.csv`: 각 지표·방법·차이의 평균/백분위수/우세 빈도",
                  "- `trials.csv`: 모든 반복의 지표",
                  "- `inputs.json`: 입력 경로·SHA256·전체 원점수·설정", ""])
    report = "\n".join(lines)
    (output / "README.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
