"""Extract paired query data and statistics from exactly three completed seeds.

Uses NumPy and closed-form Student t distribution formulas for df=2; no SciPy
installation is needed. Existing evaluation files are read only.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

try:
    from .aggregate import aggregate
except ImportError:
    from aggregate import aggregate


NAMES = ("retrieval_similarity", "random_similarity", "paired_similarity_gain")
SOURCES = {
    "paired_t_test": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html",
    "t_confidence_interval": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm",
    "exact_sign_flip": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
}


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {name: {"mean": float(values[:, i].mean()),
                   "sample_sd": float(values[:, i].std(ddof=1))}
            for i, name in enumerate(NAMES)}


def seed_statistics(seed_means):
    values = np.asarray(seed_means, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("Expected three finite seed means for each of three metrics")
    stats = describe(values)
    # For df=2, F(t) = 1/2 + t/(2*sqrt(t*t+2)).
    # Therefore t_.975 = sqrt(2)*.95/sqrt(1-.95**2).
    critical = math.sqrt(2) * .95 / math.sqrt(1 - .95**2)
    for metric in stats.values():
        se = metric["sample_sd"] / math.sqrt(3)
        metric["standard_error"] = se
        metric["t_ci95"] = [metric["mean"] - critical * se,
                            metric["mean"] + critical * se]
    difference = stats["paired_similarity_gain"]
    if difference["standard_error"] == 0:
        raise ValueError("Zero between-seed variance: paired t-test is undefined for this report")
    t = difference["mean"] / difference["standard_error"]
    radius = math.sqrt(t * t + 2)
    # Stable evaluation of 1 - abs(t)/sqrt(t*t+2).
    p = 2 / (radius * (radius + abs(t)))
    signs = list(itertools.product((-1, 1), repeat=3))
    null = np.asarray(signs) @ values[:, 2] / 3
    observed = float(values[:, 2].mean())
    tolerance = 1e-14 * max(1, abs(observed))
    extreme = np.abs(null) >= abs(observed) - tolerance
    return dict(
        metrics=stats, sample_count=3, aggregation="equal weight per rollout seed",
        sd_definition="sample standard deviation across three seed means; ddof=1",
        ci_method="two-sided 95% Student t interval; df=2",
        t_critical_975=critical,
        paired_t_test=dict(statistic=t, df=2, alternative="two-sided",
                           null_hypothesis="mean paired seed similarity gain is zero",
                           p_value=p,
                           assumption="independent, normally distributed seed-level paired differences"),
        exact_sign_flip_test=dict(
            statistic=observed, alternative="two-sided", p_value=float(extreme.mean()),
            extreme_permutations=int(extreme.sum()), total_permutations=len(signs),
            minimum_possible_two_sided_p_value=2 / len(signs),
            assumption="independent seed-level differences with exchangeable signs under the null"),
    ), [dict(signs=list(s), mean_gain=float(v), extreme=bool(e))
        for s, v, e in zip(signs, null, extreme)]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new analysis directory")
    # Reuse the evaluator's game/model/config checks and original estimator.
    original = aggregate(args.inputs, bootstrap_samples=5000, seed=2027)
    per_seed, paired, arrays, provenance = [], [], [], []
    for directory in args.inputs:
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["status"] != "complete" or not metadata.get("parameters_unchanged"):
            raise ValueError(f"{directory}: expected a complete frozen-model evaluation")
        query_path = directory / "queries.jsonl"
        content = query_path.read_bytes()
        queries = [json.loads(line) for line in content.splitlines() if line.strip()]
        valid = [q for q in queries if q.get("retrieval_distance") is not None
                 and q.get("random_distance") is not None]
        if not valid or len({q["query_id"] for q in queries}) != len(queries):
            raise ValueError(f"{directory}: missing paired queries or duplicate query IDs")
        values = np.array([[1 - q["retrieval_distance"], 1 - q["random_distance"],
                            q["random_distance"] - q["retrieval_distance"]] for q in valid])
        if not np.isfinite(values).all():
            raise ValueError(f"{directory}: nonfinite query values")
        stats = describe(values)
        row = dict(evaluation_seed=metadata["evaluation_seed"], query_count=len(queries),
                   paired_query_count=len(valid), excluded_query_count=len(queries) - len(valid),
                   anchor_episode_count=len({q["anchor_episode"] for q in valid}))
        for name, metric in stats.items():
            row[f"{name}_mean"] = metric["mean"]
            row[f"{name}_query_sample_sd"] = metric["sample_sd"]
        per_seed.append(row)
        arrays.append(values)
        for query, value in zip(valid, values):
            if query["evaluation_seed"] != metadata["evaluation_seed"]:
                raise ValueError("Query/metadata seed mismatch")
            paired.append(dict(game=metadata["game"], **query,
                               **{name: float(v) for name, v in zip(NAMES, value)}))
        saved = json.loads((directory / "summary.json").read_text())
        if saved["paired_query_count"] != len(valid):
            raise ValueError("Saved summary/query count mismatch")
        for name in NAMES[:2]:
            key = name.replace("similarity", "mean_similarity")
            if not math.isclose(saved[key], stats[name]["mean"], abs_tol=1e-12):
                raise ValueError("Saved summary/query mean mismatch")
        provenance.append(dict(
            directory=str(directory.resolve()), evaluation_seed=metadata["evaluation_seed"],
            query_sha256=hashlib.sha256(content).hexdigest(),
            metadata_sha256=hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()))

    seed_means = np.array([a.mean(axis=0) for a in arrays])
    inference, permutations = seed_statistics(seed_means)
    report = dict(
        game=original["game"], evaluation_seeds=[r["evaluation_seed"] for r in per_seed],
        paired_query_count=len(paired), query_count=sum(r["query_count"] for r in per_seed),
        primary_seed_level=inference,
        descriptive_pooled_queries=dict(
            sample_count=len(paired), metrics=describe(np.concatenate(arrays)),
            interpretation="SD describes anchor-to-anchor variation; queries are not assumed independent"),
        existing_query_weighted_protocol=original,
        per_seed=per_seed, source_files=provenance, method_references=SOURCES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            "Three independent evaluation rollouts of one fixed trained model; training-seed variability is not estimated.",
            "Paired t-test and t intervals assume normal seed-level differences; three seeds cannot establish this assumption.",
            "The exact sign-flip test enumerates eight outcomes; its smallest attainable two-sided p-value is 0.25.",
            "The original bootstrap resamples only three seed clusters; its interval has limited resolution.",
            "Original query-weighted and equal-seed-weighted estimators differ slightly because seed query counts differ.",
            "This analysis retains the original metric, including same-episode and overlapping-context pairs.",
        ])
    args.output.mkdir(parents=True)
    (args.output / "statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "existing_protocol_summary.json").write_text(json.dumps(original, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "per_seed_metrics.csv", per_seed)
    write_csv(args.output / "paired_queries.csv", paired)
    write_csv(args.output / "exact_sign_flip_distribution.csv", [
        {**{f"seed_{seed}_sign": sign for seed, sign in zip(report["evaluation_seeds"], p["signs"])},
         "mean_gain": p["mean_gain"], "extreme": p["extreme"]} for p in permutations])

    lines = [f"# {original['game'].capitalize()}: 평가 seed 세 개의 유사도 통계", "",
             f"완료된 seed {report['evaluation_seeds']}; paired query {len(paired):,}개.", "",
             "주 분석은 seed마다 앵커 평균을 구한 후 세 seed를 동일 가중으로 평균한다.",
             "SD는 seed 평균 세 개의 표본 표준편차(ddof=1), CI는 df=2인 95% t 구간이다.", "",
             "| 지표 | 평균 | seed 간 SD | 95% t CI |", "|---|---:|---:|---|" ]
    for name, label in zip(NAMES, ("Retrieval 유사도", "Random 유사도", "차이: Retrieval − Random")):
        m = inference["metrics"][name]
        lines.append(f"| {label} | {m['mean']:.9f} | {m['sample_sd']:.9f} | [{m['t_ci95'][0]:.9f}, {m['t_ci95'][1]:.9f}] |")
    ttest, exact = inference["paired_t_test"], inference["exact_sign_flip_test"]
    lines += ["", f"- 양측 paired t-test: t={ttest['statistic']:.6f}, df=2, p={ttest['p_value']:.9g}.",
              f"- 양측 seed 단위 exact sign-flip: p={exact['p_value']:.3f} ({exact['extreme_permutations']}/{exact['total_permutations']}가지).", "",
              "t-test와 t CI는 seed별 차이의 정규성 가정을 사용한다. 세 seed로 이 가정을 확인하기 어렵다.",
              "정확 부호반전 검정은 귀무가설 아래 독립 seed 차이의 부호가 교환 가능하다는 가정을 사용한다.",
              "세 seed에서는 양측 최소 p-value가 0.25여서, 이 검정으로 5% 유의성을 확보할 수 없다.",
              "모든 seed에서 retrieval 우위가 관찰되지만, 가정에 덜 의존하는 유의성 판단에는 독립 반복이 더 필요하다.", "",
              "## 앵커별 기술 통계", "",
              "paired_queries.csv에는 모든 유효 앵커의 원래 query 필드와 두 유사도 및 차이를 저장했다.",
              "per_seed_metrics.csv에는 seed별 평균과 앵커 간 표준편차를 저장했다.",
              f"statistics.json의 descriptive_pooled_queries는 앵커 {len(paired):,}개 수준의 기술 통계다. 앵커를 독립 표본으로 간주한 검정은 하지 않았다.", "",
              "## 기존 출력과의 비교", "",
              f"기존 query 동일 가중 평균: retrieval={original['retrieval_mean_similarity']:.9f}, random={original['random_mean_similarity']:.9f}, 차이={original['paired_distance_reduction']:.9f}.",
              f"기존 seed-cluster bootstrap 차이 CI: {original['paired_distance_reduction_ci95']} (5,000회).",
              "이 값은 existing_protocol_summary.json에 저장했다. seed별 query 수가 달라 주 분석과 평균이 조금 다르다.",
              "지정한 완료 seed 세 개만 포함했다. 이 결과는 한 학습 모델에 대한 평가 rollout 변동을 나타낸다.",
              "기존 원자료와 summary 파일은 수정하지 않았다.", "", "## 재현", "", "```bash",
              "python probing/ram_retrieval_eval/analyze_three_seed_statistics.py \\",
              "  --inputs " + " ".join(str(p) for p in args.inputs) + " \\",
              "  --output 새_분석_폴더", "```", "", "## 방법 출처", ""]
    lines += [f"- [{name}]({url})" for name, url in SOURCES.items()]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(output=str(args.output), seeds=report["evaluation_seeds"],
                          paired_queries=len(paired), primary=inference), indent=2))


if __name__ == "__main__":
    main()
