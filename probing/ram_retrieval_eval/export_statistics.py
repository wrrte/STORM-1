"""Build an Excel report from completed RAM retrieval evaluations, without GPU.

python probing/ram_retrieval_eval/export_statistics.py
python probing/ram_retrieval_eval/export_statistics.py --watch 60

Dependencies: numpy, openpyxl, mpmath. Source evaluations are read only.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

try:
    import numpy as np
    import mpmath as mp
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
except ModuleNotFoundError as error:
    raise SystemExit(f"Missing {error.name}. Install with: python -m pip install numpy openpyxl mpmath")


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("retrieval_similarity", "random_similarity", "similarity_gain")
EXCEL_DATA_ROWS = 1_048_575
EXACT_SIGN_LIMIT = 16
COMPLETE = {"complete", "complete_no_valid_pairs"}
METHOD_SOURCES = {
    "paired_t": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html",
    "t_ci": "https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm",
    "sign_flip": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def group_identity(metadata):
    """Conservative grouping: never pool different models or protocols."""
    required = ("game", "model_hash_before", "metric_sha256", "retrieval_sha256", "config")
    for field in required:
        if field not in metadata:
            raise ValueError(f"Missing grouping field {field}")
    identity = {key: metadata[key] for key in required}
    identity.update({key: metadata.get(key) for key in (
        "protocol", "scores", "effective_model_dtype", "replay_capacity", "actual_steps")})
    identity["analysis_settings"] = {
        k: v for k, v in metadata.get("analysis_settings", {}).items() if k != "seed"}
    return identity


@dataclass
class Run:
    path: Path
    metadata: dict
    identity: dict
    record: dict
    queries: list
    valid: list
    values: np.ndarray
    query_hash: str

    @property
    def seed(self):
        return int(self.metadata["evaluation_seed"])


def discover(inputs):
    found = set()
    for item in inputs:
        item = item.resolve()
        if not item.exists():
            raise ValueError(f"Input does not exist: {item}")
        if item.is_file():
            if item.name != "metadata.json":
                raise ValueError(f"Expected an input directory or metadata.json: {item}")
            found.add(item)
        else:
            found.update(p.resolve() for p in item.rglob("metadata.json"))
    return sorted(found)


def read_queries(path, metadata):
    data = path.read_bytes()
    queries, valid, values, ids = [], [], [], set()
    repeats = int(metadata.get("analysis_settings", {}).get("random_repeats", 1))
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        for key in ("query_id", "evaluation_seed", "anchor_step", "anchor_episode", "neighbor_count",
                    "valid_retrieval_pairs", "valid_random_pairs", "retrieval_distance", "random_distance"):
            if key not in q:
                raise ValueError(f"Line {number}: missing {key}")
        if int(q["evaluation_seed"]) != int(metadata["evaluation_seed"]):
            raise ValueError(f"Line {number}: query/metadata seed mismatch")
        if q["query_id"] in ids:
            raise ValueError(f"Duplicate query_id {q['query_id']}")
        ids.add(q["query_id"])
        queries.append(q)
        r, b = q["retrieval_distance"], q["random_distance"]
        if r is None or b is None:
            continue
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in (r, b)):
            raise ValueError(f"Line {number}: invalid paired distances")
        count = int(q["neighbor_count"])
        if count <= 0 or q["valid_retrieval_pairs"] != count or q["valid_random_pairs"] != count * repeats:
            raise ValueError(f"Line {number}: incomplete matched neighbor counts")
        valid.append(q)
        values.append((1 - r, 1 - b, b - r))
    return queries, valid, np.asarray(values, dtype=np.float64).reshape(-1, 3), hashlib.sha256(data).hexdigest()


def collect(paths, games=None):
    records, candidates = [], []
    for path in paths:
        record = dict(directory=str(path.parent), game=None, evaluation_seed=None, status=None,
                      included=False, reason=None, group_id=None, query_sha256=None)
        records.append(record)
        try:
            raw = path.read_bytes()
            metadata = json.loads(raw)
            record.update(game=metadata.get("game"), evaluation_seed=metadata.get("evaluation_seed"),
                          status=metadata.get("status"), metadata_sha256=hashlib.sha256(raw).hexdigest())
            if games and metadata.get("game") not in games:
                record["reason"] = "game_filter"
                continue
            if metadata.get("status") not in COMPLETE:
                record["reason"] = f"not_complete: {metadata.get('status')}"
                continue
            if metadata.get("parameters_unchanged") is not True:
                raise ValueError("Frozen model verification is missing or false")
            if metadata.get("model_hash_after") != metadata.get("model_hash_before"):
                raise ValueError("Model hash before/after differs")
            identity = group_identity(metadata)
            record["group_id"] = f"{metadata['game']}_{hashlib.sha256(canonical(identity).encode()).hexdigest()[:12]}"
            queries, valid, values, digest = read_queries(path.parent / "queries.jsonl", metadata)
            record.update(query_sha256=digest, query_count=len(queries), paired_query_count=len(valid))
            if not len(values):
                record["reason"] = "no_valid_paired_queries"
                continue
            summary = json.loads((path.parent / "summary.json").read_text())
            if summary["query_count"] != len(queries) or summary["paired_query_count"] != len(valid):
                raise ValueError("Saved summary counts do not match raw queries")
            for column, key in enumerate(("retrieval_mean_similarity", "random_mean_similarity", "paired_distance_reduction")):
                if not math.isclose(summary[key], float(values[:, column].mean()), rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Saved summary disagrees with raw queries: {key}")
            candidates.append(Run(path.parent, metadata, identity, record, queries, valid, values, digest))
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["reason"] = f"invalid: {error}"

    # A repeated rollout seed is not a new independent replicate.
    by_seed = defaultdict(list)
    for run in candidates:
        by_seed[(run.record["group_id"], run.seed)].append(run)
    accepted = []
    for runs in by_seed.values():
        if len({r.query_hash for r in runs}) > 1:
            for run in runs:
                run.record["reason"] = "conflicting_duplicate_seed: all copies excluded; select one input root"
            continue
        chosen, *duplicates = runs
        chosen.record.update(included=True, reason="included")
        accepted.append(chosen)
        for duplicate in duplicates:
            duplicate.record["reason"] = f"identical_duplicate_seed: using {chosen.path}"
    return sorted(accepted, key=lambda r: (r.record["group_id"], r.seed)), records


def sample_sd(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else None


def t_two_sided_p(statistic, df):
    with mp.workdps(40):
        t = mp.mpf(abs(statistic))
        x = mp.mpf(df) / (df + t * t)
        return float(mp.betainc(mp.mpf(df) / 2, mp.mpf("0.5"), 0, x, regularized=True))


@lru_cache(maxsize=None)
def t_critical(df):
    lo, hi = 0., 1.
    while t_two_sided_p(hi, df) > .05:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > .05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_interval(values):
    sd = sample_sd(values)
    if sd is None or sd == 0:
        return (None, None)
    half = t_critical(len(values) - 1) * sd / math.sqrt(len(values))
    mean = float(np.mean(values))
    return mean - half, mean + half


def paired_t(values):
    sd = sample_sd(values)
    if sd is None:
        return dict(t=None, df=None, p=None, status="requires_at_least_two_seeds")
    if sd == 0:
        return dict(t=None, df=len(values) - 1, p=None, status="zero_seed_variance")
    t = float(np.mean(values) / (sd / math.sqrt(len(values))))
    return dict(t=t, df=len(values) - 1, p=t_two_sided_p(t, len(values) - 1), status="ok")


def sign_flip(values, rng, permutations):
    n = len(values)
    if n < 2:
        return dict(p=None, method="requires_at_least_two_seeds", samples=0, minimum_exact_p=None)
    nonzero = np.count_nonzero(values)
    minimum = min(1., 2. ** (1 - int(nonzero)))
    exact = n <= EXACT_SIGN_LIMIT
    samples = 2**n if exact else permutations
    threshold = abs(float(np.mean(values))) - 1e-14 * max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    extreme = 0
    batch = max(1, min(2048, 1_000_000 // n))
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        if exact:
            bits = (np.arange(start, start + count, dtype=np.uint64)[:, None]
                    >> np.arange(n, dtype=np.uint64)[None]) & 1
            signs = 2 * bits.astype(np.float64) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(count, n)) - 1
        null = np.sum(signs * values, axis=1) / n
        extreme += int(np.count_nonzero(np.abs(null) >= threshold))
    p = extreme / samples if exact else (extreme + 1) / (samples + 1)
    return dict(p=float(p), method="exact" if exact else "monte_carlo_plus_one",
                samples=samples, minimum_exact_p=minimum)


def cluster_ci(values, labels, samples, rng):
    """Resample whole clusters; preserve paired columns and query weights."""
    _, ids = np.unique(labels, return_inverse=True)
    count = int(ids.max()) + 1
    if count < 2 or not samples:
        return np.full((2, 3), np.nan)
    sums = np.stack([np.bincount(ids, weights=values[:, i], minlength=count) for i in range(3)], axis=1)
    sizes = np.bincount(ids, minlength=count)
    boot = np.empty((samples, 3))
    batch = max(1, min(128, 200_000 // count))
    for start in range(0, samples, batch):
        indexes = rng.integers(0, count, (min(batch, samples - start), count))
        boot[start:start + len(indexes)] = sums[indexes].sum(1) / sizes[indexes].sum(1)[:, None]
    return np.quantile(boot, [.025, .975], axis=0)


def analyze(runs, bootstrap_samples=5000, permutations=100000, random_seed=2027):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.record["group_id"]].append(run)
    summaries, metrics, seeds, group_info = [], [], [], []
    for group_id, members in sorted(grouped.items()):
        # Stable per-group streams make results independent of other groups' arrival.
        entropy = [random_seed, int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16)]
        seed_rng, query_rng, permutation_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(entropy).spawn(3)]
        means = np.stack([r.values.mean(axis=0) for r in members])
        pooled = np.concatenate([r.values for r in members])
        n = len(members)
        seed_ci = cluster_ci(means, np.arange(n), bootstrap_samples, seed_rng)
        if n >= 2:
            labels = np.concatenate([np.full(len(r.values), r.seed) for r in members])
            unit = "rollout_seed"
        else:
            labels = np.array([q["anchor_episode"] for q in members[0].valid])
            unit = "anchor_episode_conditional_on_one_rollout"
        query_ci = cluster_ci(pooled, labels, bootstrap_samples, query_rng)
        ttest = paired_t(means[:, 2])
        flip = sign_flip(means[:, 2], permutation_rng, permutations)
        game = members[0].metadata["game"]
        note = ""
        if n == 1:
            note = "독립 seed 1개: seed 간 SD·CI·p-value 계산 불가"
        elif n < 5:
            note = "독립 seed 수가 적음; t-test·t CI의 정규성 가정 확인이 어려움"
        if flip["minimum_exact_p"] is not None and flip["minimum_exact_p"] > .05:
            note += f"; seed 부호반전 검정의 최소 양측 p={flip['minimum_exact_p']:.4g}"
        total_queries = sum(len(r.queries) for r in members)
        point = means.mean(axis=0)
        delta_t_ci = t_interval(means[:, 2])
        summaries.append(dict(
            group_id=group_id, game=game, n_seeds=n, evaluation_seeds=", ".join(str(r.seed) for r in members),
            paired_queries=len(pooled), query_count=total_queries,
            coverage=sum(q["neighbor_count"] > 0 for r in members for q in r.queries) / total_queries,
            retrieval_mean=float(point[0]), retrieval_seed_sd=sample_sd(means[:, 0]),
            random_mean=float(point[1]), random_seed_sd=sample_sd(means[:, 1]),
            similarity_gain=float(point[2]), gain_seed_sd=sample_sd(means[:, 2]),
            gain_t_ci95_low=delta_t_ci[0], gain_t_ci95_high=delta_t_ci[1],
            gain_bootstrap_ci95_low=seed_ci[0, 2], gain_bootstrap_ci95_high=seed_ci[1, 2],
            paired_t_p=ttest["p"], sign_flip_p=flip["p"], sign_flip_method=flip["method"],
            relative_distance_reduction=float(point[2] / (1 - point[1])) if point[1] < 1 else None,
            seeds_retrieval_better=int(np.count_nonzero(means[:, 2] > 0)),
            query_win_rate=float(np.mean(pooled[:, 2] > 0)), notes=note))
        for i, name in enumerate(METRICS):
            ci = t_interval(means[:, i])
            sd = sample_sd(means[:, i])
            metrics.append(dict(
                group_id=group_id, game=game, metric=name, n_seeds=n,
                seed_equal_mean=float(point[i]), seed_sample_sd=sd,
                seed_standard_error=sd / math.sqrt(n) if sd is not None else None,
                seed_t_ci95_low=ci[0], seed_t_ci95_high=ci[1],
                seed_bootstrap_ci95_low=seed_ci[0, i], seed_bootstrap_ci95_high=seed_ci[1, i],
                query_weighted_mean=float(pooled[:, i].mean()), query_sample_sd=sample_sd(pooled[:, i]),
                query_weighted_ci95_low=query_ci[0, i], query_weighted_ci95_high=query_ci[1, i],
                query_bootstrap_unit=unit, query_bootstrap_clusters=len(np.unique(labels)),
                paired_t=ttest["t"] if i == 2 else None, t_df=ttest["df"] if i == 2 else None,
                paired_t_p=ttest["p"] if i == 2 else None, t_test_status=ttest["status"] if i == 2 else None,
                sign_flip_p=flip["p"] if i == 2 else None,
                sign_flip_method=flip["method"] if i == 2 else None,
                sign_flip_samples=flip["samples"] if i == 2 else None,
                minimum_exact_two_sided_p=flip["minimum_exact_p"] if i == 2 else None))
        for run, mean in zip(members, means):
            seed_row = dict(group_id=group_id, game=game, evaluation_seed=run.seed,
                            query_count=len(run.queries), paired_queries=len(run.values),
                            excluded_queries=len(run.queries) - len(run.values),
                            anchor_episode_count=len({q["anchor_episode"] for q in run.valid}),
                            retrieval_pairs=sum(q["valid_retrieval_pairs"] for q in run.valid),
                            random_pairs=sum(q["valid_random_pairs"] for q in run.valid),
                            query_win_rate=float(np.mean(run.values[:, 2] > 0)), directory=str(run.path))
            for i, name in enumerate(METRICS):
                seed_row[f"{name}_mean"] = float(mean[i])
                seed_row[f"{name}_query_sd"] = sample_sd(run.values[:, i])
            seeds.append(seed_row)
        group_info.append(dict(group_id=group_id, game=game,
                               protocol_identity_json=canonical(members[0].identity)))
    return summaries, metrics, seeds, group_info


def method_rows(args):
    items = [
        ("갱신 시각", "generated_at", datetime.now(timezone(timedelta(hours=9))).isoformat(), ""),
        ("입력", "inputs", "; ".join(str(p.resolve()) for p in args.inputs), ""),
        ("정의", "similarity", "similarity = 1 - distance. 높을수록 가깝다. 게임별 점수를 직접 평균하지 않는다.", ""),
        ("정의", "similarity_gain", "같은 앵커의 retrieval similarity − random similarity. 양수면 retrieval이 더 가깝다.", ""),
        ("집계", "Summary / seed_equal_mean", "각 seed 안에서 앵커 평균을 구한 뒤, 독립 평가 seed를 동일 가중으로 평균한다.", ""),
        ("집계", "query_weighted_mean", "유효 앵커 전체를 동일 가중으로 평균한다. 기존 summary.json 방식이며 seed별 query 수에 따라 seed 가중치가 달라진다.", ""),
        ("변동", "seed_sample_sd", "seed 평균 사이의 표본 표준편차(ddof=1). query_sample_sd는 앵커 사이의 기술적 변동이다.", ""),
        ("CI", "seed_t_ci95", "seed 평균의 95% Student t 구간. seed별 값의 정규성·독립성 가정. 작은 seed 수에서는 가정 확인이 어렵다. 1개 seed 또는 분산 0이면 빈칸.", METHOD_SOURCES["t_ci"]),
        ("CI", "seed_bootstrap_ci95", "독립 seed 전체를 복원 추출한 percentile 95% CI. 같은 재표집을 retrieval/random에 적용. seed 수가 적으면 해상도가 낮다.", ""),
        ("CI", "query_weighted_ci95", "여러 seed이면 seed cluster 전체, 하나이면 anchor episode cluster 전체를 재표집한다. 단일 rollout의 episode CI는 해당 replay에 조건부인 근사치.", ""),
        ("검정", "paired_t_p", "양측 paired t-test: seed별 평균 유사도 차이의 평균=0. seed 차이의 정규성 가정. 이웃 쌍이나 앵커를 독립 표본으로 세지 않는다.", METHOD_SOURCES["paired_t"]),
        ("검정", "sign_flip_p", "seed별 평균 차이의 부호를 바꾸는 양측 검정. 귀무가설 아래 독립 seed 차이 부호의 교환 가능성을 가정한다.", METHOD_SOURCES["sign_flip"]),
        ("검정", "sign_flip_method", f"seed ≤ {EXACT_SIGN_LIMIT}: 모든 2^n 부호 조합을 열거. 그 이상: Monte Carlo, (extreme+1)/(samples+1).", ""),
        ("검정", "minimum_exact_two_sided_p", "0이 아닌 차이를 가진 seed가 3개면 최소 양측 p=0.25. 작은 t-test p만으로 가정에 덜 의존하는 유의성을 단정하지 않는다.", ""),
        ("표본", "evaluation_seed", "한 고정 학습 모델의 독립 평가 rollout seed. 여러 training seed에 대한 일반화를 추정하지 않는다.", ""),
        ("필터", "completed_only", "완료 상태·모델 불변 검증을 통과한 실행만 포함. 실행 중/중단/오류 파일의 제외 사유는 Runs 시트에 기록한다.", ""),
        ("필터", "duplicates", "동일 모델·프로토콜의 같은 seed는 한 번만 집계한다. 원자료가 같은 복사본은 하나만 사용, 다른 복사본들은 모두 제외한다.", ""),
        ("필터", "complete_pairs", "retrieval/random 거리와 이웃 개수가 완전한 앵커만 추론에 포함한다. 제외 앵커도 Queries 시트에 남긴다.", ""),
        ("필터", "temporal_pairs", "기존 평가의 동일 episode·context 중첩 이웃을 그대로 포함한다. 해당 효과를 제거한 별도 실험은 아니다.", ""),
        ("그룹", "group_id", "게임·모델 hash·metric hash·retrieval hash·config·precision·replay 크기·실행 길이·평가 설정이 같은 실행끼리만 묶는다.", ""),
        ("재현", "random_seed", str(args.random_seed), ""),
        ("재현", "bootstrap_samples", str(args.bootstrap_samples), ""),
        ("재현", "permutation_samples", str(args.permutations), ""),
        ("재현", "script_sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), ""),
        ("읽기", "empty_cells", "seed가 부족하거나 통계가 정의되지 않는 경우 빈칸. 0이라는 뜻이 아니다. 여러 게임·그룹의 p-value는 다중 비교 보정 전 값이다.", ""),
        ("읽기", "raw_data", "Queries에는 앵커 평균을 기록한다. 개별 이웃 및 128-byte RAM 원본은 각 실행 폴더의 neighbors.csv / pairs.jsonl에 보존된다.", ""),
    ]
    return [dict(category=a, field=b, explanation=c, source=d) for a, b, c, d in items]


def cell(sheet, value, header=False, field=""):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    if isinstance(value, str):
        # Store paths/labels as literal text, including strings beginning with '='.
        value = "".join(c for c in value if ord(c) >= 32 or c in "\n\t")[:32767]
    result = WriteOnlyCell(sheet, value=value)
    if isinstance(value, str):
        result.data_type = "s"
    if header:
        result.font = Font(name="Calibri", bold=True, color="FFFFFF")
        result.fill = PatternFill("solid", fgColor="24476B")
        result.alignment = Alignment(wrap_text=True, vertical="center")
    elif isinstance(value, float):
        if field.endswith("_p") or field == "minimum_exact_two_sided_p":
            result.number_format = "0.000E+00"
        elif field in ("coverage", "query_win_rate", "relative_distance_reduction"):
            result.number_format = "0.00%"
        else:
            result.number_format = "0.000000"
    return result


def table(workbook, name, rows, fields=None):
    iterator = iter(rows)
    first = next(iterator, None)
    if fields is None:
        fields = list(first) if first is not None else ["information"]
    sheet = workbook.create_sheet(name)
    sheet.freeze_panes = "C2" if len(fields) > 2 else "A2"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 44
    for index, field in enumerate(fields, 1):
        width = 70 if field in ("notes", "reason", "explanation", "directory", "protocol_identity_json", "source") else 23
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.append([cell(sheet, f, header=True) for f in fields])
    count = 0
    def append(row):
        sheet.append([cell(sheet, row.get(field), field=field) for field in fields])
    if first is not None:
        append(first)
        count += 1
    for row in iterator:
        append(row)
        count += 1
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(fields))}{count + 1}"
    return count


QUERY_FIELDS = ["group_id", "game", "evaluation_seed", "query_id", "included_pair", "exclusion_reason",
                "evaluation_step", "anchor_step", "anchor_episode", "anchor_pointer", "neighbor_count",
                "valid_retrieval_pairs", "valid_random_pairs", "retrieval_distance", "random_distance",
                "retrieval_similarity", "random_similarity", "similarity_gain", "lazy_hash_hit_rate"]


def query_rows(runs):
    for run in runs:
        for q in run.queries:
            r, b = q["retrieval_distance"], q["random_distance"]
            valid = r is not None and b is not None
            row = dict(q, group_id=run.record["group_id"], game=run.metadata["game"],
                       included_pair=valid,
                       exclusion_reason="" if valid else "no_neighbors_or_incomplete_mapping_pairs")
            row.update(retrieval_similarity=1 - r if r is not None else None,
                       random_similarity=1 - b if b is not None else None,
                       similarity_gain=b - r if valid else None)
            yield row


def export_workbook(args, paths):
    runs, records = collect(paths, set(args.games) if args.games else None)
    if not runs:
        raise ValueError("No eligible completed evaluations; existing workbook was left unchanged")
    summary, metrics, seeds, groups = analyze(runs, args.bootstrap_samples, args.permutations, args.random_seed)
    workbook = Workbook(write_only=True)
    workbook.properties.title = "RAM retrieval vs random similarity statistics"
    workbook.properties.description = "Paired evaluation; independent rollout seeds are the primary inference unit."
    table(workbook, "Summary", summary)
    table(workbook, "Statistics", metrics)
    table(workbook, "Seeds", seeds)
    fields = ["directory", "game", "evaluation_seed", "status", "included", "reason", "group_id",
              "query_count", "paired_query_count", "metadata_sha256", "query_sha256"]
    table(workbook, "Runs", records, fields)
    table(workbook, "Groups", groups)
    table(workbook, "Methods", method_rows(args))
    if not args.no_queries:
        iterator = iter(query_rows(runs))
        index = 1
        while True:
            # Each worksheet stays inside Excel's row limit, including its header.
            batch = itertools.islice(iterator, EXCEL_DATA_ROWS)
            first = next(batch, None)
            if first is None:
                break
            table(workbook, f"Queries_{index:03d}", itertools.chain([first], batch), QUERY_FIELDS)
            index += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".xlsx", prefix=".retrieval_stats_", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        workbook.save(temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved: {args.output.resolve()}", flush=True)
    for row in summary:
        print(f"  {row['game']} [{row['group_id']}]: seeds={row['n_seeds']}, paired_queries={row['paired_queries']}, "
              f"retrieval={row['retrieval_mean']:.6f}, random={row['random_mean']:.6f}, gain={row['similarity_gain']:.6f}, "
              f"t_p={row['paired_t_p']}, sign_flip_p={row['sign_flip_p']}", flush=True)
    excluded = Counter(r["reason"] for r in records if not r["included"])
    if excluded:
        print("Excluded runs (details in Runs sheet): " + str(dict(excluded)), flush=True)
    return summary


def fingerprint(paths):
    entries = []
    for path in paths:
        for source in (path, path.parent / "queries.jsonl", path.parent / "summary.json"):
            # In watch mode, in-progress queries do not trigger a heavy rebuild.
            if source != path:
                try:
                    if json.loads(path.read_text()).get("status") not in COMPLETE:
                        continue
                except (OSError, ValueError):
                    continue
            try:
                stat = source.stat()
                entries.append((str(source), stat.st_mtime_ns, stat.st_size))
            except OSError:
                entries.append((str(source), None, None))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, default=[ROOT / "results/ram_verification"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/ram_verification/retrieval_statistics.xlsx")
    parser.add_argument("--games", nargs="+", help="Optional game filter, e.g. seaquest frostbite pong")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=2027)
    parser.add_argument("--no-queries", action="store_true", help="Omit anchor-level worksheets for a smaller report")
    parser.add_argument("--watch", type=float, default=0, metavar="SECONDS", help="Refresh when input changes, e.g. --watch 60")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".xlsx":
        parser.error("--output must end in .xlsx")
    if args.bootstrap_samples < 0 or args.permutations < 1 or args.random_seed < 0:
        parser.error("bootstrap samples / random seed must be nonnegative; permutations must be positive")
    if args.watch and args.watch < 5:
        parser.error("--watch must be zero or at least five seconds")
    last = None
    try:
        while True:
            paths = discover(args.inputs)
            current = fingerprint(paths)
            if current != last or not args.output.exists():
                try:
                    export_workbook(args, paths)
                    last = current
                except (ValueError, OSError) as error:
                    if not args.watch:
                        parser.exit(1, f"Error: {error}\n")
                    print(f"Report not updated: {error}", file=sys.stderr, flush=True)
            if not args.watch:
                return 0
            print(f"Watching for completed evaluations; check again in {args.watch:g}s. Ctrl+C to stop.", flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
