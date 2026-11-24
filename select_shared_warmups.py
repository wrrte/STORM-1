"""Rank shared warmups and export a workbook with editable HNS tolerances."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from math import isfinite
from pathlib import Path
from statistics import fmean, median, stdev
from tempfile import TemporaryDirectory

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from collect_warmup_scores import TARGET_STEP
from queue_train import ROOT, SEED_RANGES, SEED_DIR, canonical_game, read_seed_files


BASELINE = "Retrieval off"
RETRIEVAL = "Retrieval on (t=16)"
SUMMARY = "게임별 추천"
OPTIONS = "시드 수별 조합"
SEED_COUNTS = (2, 3, 4)
FINAL_EVAL_STEP = 100_000
SUMMARY_HEADER = 5
RETRIEVAL_KEYS = ("hash_bits", "anchor_offset", "anchor_weight", "anchor_weight_r",
                  "score_combination", "value_signal", "trigger_mode", "z_score_threshold",
                  "td_error_z_score_threshold", "max_anchors")


@dataclass(frozen=True)
class Reference:
    random: float
    human: float
    paper: float
    reproduced: float | None = None

    def normalize(self, score):
        return (score - self.random) / (self.human - self.random)


@dataclass(frozen=True, order=True)
class Snapshot:
    game: str
    seed: int
    gpu: str
    machine: str
    path: str

    @classmethod
    def from_row(cls, row):
        return cls(canonical_game(row["game"]), row["seed"], row["gpu_type"],
                   row["machine"], row["warmup_source"])


@dataclass(frozen=True)
class Candidate:
    snapshot: Snapshot
    score: float
    hns: float
    run_ids: tuple[str, ...]
    eval_step: int | None
    eval_precision: str | None
    owner_note: str
    method: str
    profile: str

    def evidence_order(self):
        return self.eval_step is None, -len(self.run_ids), self.snapshot, self.run_ids


@dataclass(frozen=True)
class SeedOption:
    candidates: tuple[Candidate, ...]
    mean: float
    delta: float
    min_mean: float
    max_mean: float


def paper_references(path):
    def score(text):
        return float(text.split(":", 1)[1].strip().replace(",", ""))

    references, reproduced = {}, {}
    game = None
    with path.open("rb") as handle:
        workbook = load_workbook(handle, read_only=True, data_only=True)
        try:
            rows = workbook["Results"].iter_rows(max_col=2, values_only=True)
            if next(rows) != ("Game", "Config"):
                raise ValueError("Expected 'Game' and 'Config' columns in Results")
            for label, config in rows:
                if label:
                    name, random, human = label.splitlines()
                    game = canonical_game(name.strip())
                    random, human = score(random), score(human)
                if isinstance(config, str) and config.startswith("STORM (논문):"):
                    paper = score(config)
                    if (game is None or game in references or human <= random or
                            not all(map(isfinite, (random, human, paper)))):
                        raise ValueError(f"Invalid or duplicate paper reference for {game}")
                    references[game] = Reference(random, human, paper)
                elif isinstance(config, str) and config.startswith("STORM (reproduced):"):
                    value = score(config)
                    if game is None or game in reproduced or not isfinite(value):
                        raise ValueError(f"Invalid or duplicate reproduced reference for {game}")
                    reproduced[game] = value
        finally:
            workbook.close()
    if not references:
        raise ValueError("No STORM paper rows in the Results sheet")
    return {game: Reference(ref.random, ref.human, ref.paper, reproduced.get(game))
            for game, ref in references.items()}


def read_reports(directory):
    reports, snapshots, runs = [], set(), {}
    for path in sorted(directory.glob("*/*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path, report))
        included = {Snapshot.from_row(row) for row in report["warmups"] if row["warmup_target_step"] == TARGET_STEP}
        snapshots.update(included)
        for row in report["runs"]:
            if row["warmup_target_step"] != TARGET_STEP:
                continue
            if Snapshot.from_row(row) not in included:
                raise ValueError(f"Run {row['run_id']} has no matching warmup in {path}")
            key = row["wandb_project"], row["run_id"]
            if key in runs and runs[key] != row:
                raise ValueError(f"Conflicting records for W&B run {key}")
            runs[key] = row
    if not reports:
        raise ValueError(f"No per-machine reports under {directory}")
    if len({report["project"] for _, report in reports}) != 1:
        raise ValueError("Score reports must use the same W&B project")
    return reports, snapshots, runs.values()


def owner_notes(snapshots):
    observed = defaultdict(set)
    for snapshot in snapshots:
        observed[snapshot.seed].add(snapshot.gpu)
    assigned = {entry["seed"]: entry["gpu_type"] for entry in read_seed_files(SEED_DIR).values()}

    def note(snapshot):
        seed = snapshot.seed
        owner = assigned.get(seed) or next((gpu for gpu, (lo, hi) in SEED_RANGES.items() if lo <= seed <= hi), None)
        if owner and owner != snapshot.gpu:
            return f"기존 큐 소유 GPU {owner}"
        if len(observed[seed]) > 1:
            return "여러 GPU에 존재 · " + ", ".join(sorted(observed[seed]))
        return ""

    return {snapshot: note(snapshot) for snapshot in snapshots}


def eligible_run(row, strict):
    score = row.get("final_score")
    if type(score) not in (int, float) or not isfinite(score):
        return False
    if row.get("sample_max_steps") != 102_000 or row.get("eval_episodes") != 20 or row.get("eval_seed") not in (None, 0):
        return False
    if row.get("eval_precision") not in (None, "auto", "bf16", "both"):
        return False
    step = row.get("eval_checkpoint_step")
    if step is not None or strict:
        return step == FINAL_EVAL_STEP
    configured = row["wandb_config"]["JointTrainAgent"].get("FinalEvalStep")
    return configured in (None, FINAL_EVAL_STEP) and row.get("score_status") == "final_only_finished"


def study_settings(row):
    """Identify comparable settings for comparisons across seeds."""
    config = row["wandb_config"]
    training = config["JointTrainAgent"]
    retrieval = training["Retrieval"]
    enabled = row.get("retrieval_enable")
    branch = row.get("branch")
    target = retrieval.get("target")
    if enabled is False:
        method = BASELINE
    elif (enabled is True and branch not in ("randomneighor", "randomneighbor") and
          retrieval.get("neighbor_source", "retrieval") == "retrieval"):
        if branch in (None, "retrieval") and target == 16 and retrieval.get("score_combination") in (None, "multiply"):
            method = RETRIEVAL
        else:
            variant = branch or retrieval.get("score_combination") or "설정 미기록"
            method = f"Retrieval on (t={target if target is not None else '?'}, {variant})"
    else:
        return None

    settings = {"demo": training.get("UseDemonstration"), "models": config.get("Models", {})}
    if enabled:
        settings.update({field: retrieval.get(field) for field in RETRIEVAL_KEYS})
    return method, json.dumps(settings, sort_keys=True, ensure_ascii=False)


def setting_description(profile):
    settings = json.loads(profile)
    return ", ".join(f"{key}={value}" for key, value in settings.items() if key != "models" and value is not None)


def candidate_pools(runs, references, notes, strict):
    repeats = defaultdict(list)
    for row in runs:
        snapshot = Snapshot.from_row(row)
        if snapshot.game not in references or not eligible_run(row, strict):
            continue
        study = study_settings(row)
        if study is None:
            continue
        config = {k: v for k, v in row["wandb_config"].items() if k != "RunMetadata"}
        run_config = json.dumps(config, sort_keys=True, ensure_ascii=False)
        repeats[snapshot, study, row.get("eval_checkpoint_step"), row.get("eval_precision"), run_config].append(row)

    per_snapshot = defaultdict(list)
    for (snapshot, study, step, precision, _), rows in repeats.items():
        method, profile = study
        score = median(row["final_score"] for row in rows)
        ids = tuple(sorted(row["run_id"] for row in rows))
        candidate = Candidate(snapshot, score, references[snapshot.game].normalize(score), ids, step,
                              precision, notes[snapshot], method, profile)
        per_snapshot[snapshot, study].append(candidate)

    pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (snapshot, (method, profile)), choices in per_snapshot.items():
        pools[snapshot.game][method][profile].append(min(choices, key=Candidate.evidence_order))
    return pools


def distinct_seeds(candidates):
    by_seed = {}
    for candidate in sorted(candidates, key=Candidate.evidence_order):
        by_seed.setdefault(candidate.snapshot.seed, candidate)
    return by_seed


def best_profile(profiles):
    def rank(item):
        profile, candidates = item
        seeds = distinct_seeds(candidates).values()
        return -len(seeds), -sum(c.eval_step is not None for c in seeds), profile

    return min(profiles.items(), key=rank, default=(None, []))


def supplement(candidates, profiles):
    """Add observations for new seeds while retaining existing scores for overlapping seeds."""
    existing = distinct_seeds(candidates)
    additions = distinct_seeds(candidate for profile in profiles for candidate in profile
                               if candidate.snapshot.seed not in existing)
    return [*candidates, *additions.values()]


def selection_pools(profiles, primary):
    """Widen from one setting to t=16 variants, other retrieval settings, then baseline."""
    t16 = supplement(primary, profiles.get(RETRIEVAL, {}).values())
    other = supplement(t16, (candidates for method, settings in profiles.items()
                             if method not in (BASELINE, RETRIEVAL) for candidates in settings.values()))
    baseline = supplement(other, profiles.get(BASELINE, {}).values())
    return (("동일 설정", primary), ("t=16 설정 보충", t16),
            ("다른 retrieval 설정 보충", other), ("Retrieval off 보충", baseline))


def best_option(candidates, count, target):
    best = None
    lo, hi = float("inf"), float("-inf")
    for group in combinations(sorted(candidates, key=lambda c: c.snapshot), count):
        if len({c.snapshot.seed for c in group}) != count:
            continue
        mean = fmean(c.hns for c in group)
        lo, hi = min(lo, mean), max(hi, mean)
        delta = mean - target
        if abs(delta) < 1e-12:
            delta = 0.0
        rank = (delta < 0, abs(delta), sum(c.eval_step is None for c in group),
                sum(bool(c.owner_note) for c in group), tuple(c.snapshot for c in group))
        if best is None or rank < best[0]:
            best = rank, group, mean, delta
    if best is None:
        return None
    _, group, mean, delta = best
    return SeedOption(group, mean, delta, lo, hi)


def make_sheet(workbook, name, headers, widths, *, header_row=1):
    sheet = workbook.create_sheet(name)
    sheet.sheet_view.showGridLines = False
    for col, (header, width) in enumerate(zip(headers, widths, strict=True), 1):
        cell = sheet.cell(header_row, col, header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="24425A")
        sheet.column_dimensions[get_column_letter(col)].width = width
    sheet.row_dimensions[header_row].height = 23
    sheet.freeze_panes = f"B{header_row + 1}"
    return sheet


def number_format(sheet, row, *columns):
    for column in columns:
        sheet[f"{column}{row}"].number_format = "0.0000"


def game_rows(summary, row, game, method, baseline_count, retrieval_count, count, sd, ref, max_seeds, last_option):
    summary.cell(row, 1, game)
    final_row = (f"INDEX('{OPTIONS}'!${{col}}$2:${{col}}${last_option},"
                 f"MATCH($A{row}&\"|\"&$R{row},'{OPTIONS}'!$A$2:$A${last_option},0))")

    def option(col):
        return f'=IF($R{row}="","",{final_row.format(col=col)})'

    values = [
        f'=IF(ISNUMBER(E{row}),IF(E{row}/SQRT(2)<=$B$1,2,IF(E{row}/SQRT(3)<=$B$1,3,4)),"")',
        method, count, sd,
        f'=IF(B{row}="","",E{row}/SQRT(B{row}))',
        f'=IF(ISNUMBER(Q{row}),IF(J{row}-Q{row}>$E$1,"경고 · 낮은 시드 포함",""),"")',
        baseline_count, retrieval_count, ref.normalize(ref.paper), option("F"), option("G"),
        (f'=IF(ISNUMBER(L{row}),IF(L{row}>=0,"논문 이상",IF(-L{row}<=$E$1,"허용 오차 이내",'
         f'IF(AND(ISNUMBER(U{row}),K{row}>=U{row}),"재현 성능 이상","경고 · 논문보다 낮음"))),"조합 자료 부족")'),
        option("K"), option("L"), option("M"), option("N"),
        f'=IF(OR(B{row}="",T{row}<2),"",MIN(B{row},T{row}))',
        f'=IF(R{row}="","",{final_row.format(col="O")}&IF(R{row}<B{row}," · "&R{row}&"/"&B{row},""))', max_seeds,
        ref.normalize(ref.reproduced) if ref.reproduced is not None else None,
    ]
    for column, value in enumerate(values, 2):
        summary.cell(row, column, value)
    number_format(summary, row, "E", "F", "J", "K", "L", "Q", "U")


def append_hns_totals(summary):
    last_game = summary.max_row
    game_means = f"K{SUMMARY_HEADER + 1}:K{last_game}"
    for row, (label, function) in enumerate((("Mean HNS", "AVERAGE"), ("Median HNS", "MEDIAN")), last_game + 2):
        label_cell = summary.cell(row, 10, label)
        value_cell = summary.cell(row, 11, f'=IF(COUNT({game_means})>0,{function}({game_means}),"")')
        number_format(summary, row, "K")
        for cell in (label_cell, value_cell):
            cell.font = Font(bold=True, color="24425A")
            cell.fill = PatternFill("solid", fgColor="DFEBF3")
    summary.cell(last_game + 2, 4, "집계 게임 수")
    summary.cell(last_game + 2, 5, f"=COUNT({game_means})")


def write_workbook(path, references, pools, reports, args):
    workbook = Workbook()
    workbook.remove(workbook.active)
    summary = make_sheet(workbook, SUMMARY, (
        "게임", "권장 시드 수", "점수 기준", "기준 시드 수", "시드 HNS SD", "평균 표준오차 추정", "시드별 하방 확인",
        "Off 시드 수", "t=16 비교 설정 시드 수", "논문 HNS", "선택 평균 HNS", "차이 (평균 − 논문)", "논문 비교 판정",
        "후보 조합 (GPU:시드)", "평가 스텝 미기록 수", "GPU 소유권 확인 수", "조합 최저 시드 HNS",
        "실제 조합 시드 수", "조합 설정 범위", "전체 후보 시드 수", "STORM 재현 HNS",
    ), (21, 14, 42, 14, 16, 20, 26, 14, 25, 16, 17, 21, 26, 66, 21, 21, 22, 22, 36, 22, 22), header_row=SUMMARY_HEADER)
    summary["A1"], summary["B1"] = "편차 허용치 (HNS)", args.stability_tolerance
    summary.merge_cells("C1:D1")
    summary["C1"], summary["E1"] = "논문 대비 하방 허용 오차 (HNS)", args.paper_tolerance
    summary["B1"].fill = summary["E1"].fill = PatternFill("solid", fgColor="FFE598")
    summary["B1"].font = summary["E1"].font = Font(bold=True, color="613E10")
    summary["B1"].number_format = summary["E1"].number_format = "0.0000"
    summary.merge_cells("A3:H3")
    summary["A3"] = "노란색 두 값을 바꾸면 후보 조합, 비교 판정과 하단의 Mean·Median HNS가 갱신됩니다."
    validation = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1=0, showErrorMessage=True)
    validation.error = "0 이상의 HNS 허용치를 입력하세요."
    summary.add_data_validation(validation)
    validation.add("B1")
    validation.add("E1")

    options = make_sheet(workbook, OPTIONS, (
        "조회 키", "게임", "시드 수", "점수 기준", "논문 HNS", "최선 평균 HNS", "차이 (평균 − 논문)",
        "절대 차이", "가능한 평균 최저", "가능한 평균 최고", "후보 조합 (GPU:시드)",
        "평가 스텝 미기록 수", "GPU 소유권 확인 수", "조합 최저 시드 HNS", "조합 설정 범위",
    ), (20, 21, 12, 24, 14, 17, 21, 16, 19, 19, 66, 21, 21, 22, 33))
    detail = make_sheet(workbook, "조합 체크포인트", (
        "게임", "시드 수", "실제 run 설정", "GPU", "시드", "원점수", "HNS", "평가 스텝", "평가 정밀도",
        "GPU 소유권", "머신", "체크포인트", "W&B run ID", "run 설정값",
    ), (21, 12, 36, 15, 12, 16, 16, 14, 15, 38, 19, 81, 32, 120))
    sources = make_sheet(workbook, "기준", ("항목", "설명"), (27, 124))
    last_option = 1 + len(references) * len(SEED_COUNTS)
    for row, (game, ref) in enumerate(references.items(), SUMMARY_HEADER + 1):
        profiles = pools.get(game, {})
        _, baseline = best_profile(profiles.get(BASELINE, {}))
        profile, retrieval = best_profile(profiles.get(RETRIEVAL, {}))
        baseline_count = len(distinct_seeds(baseline))
        retrieval_count = len(distinct_seeds(retrieval))
        method = BASELINE if baseline_count >= args.min_baseline_seeds else RETRIEVAL
        selected = baseline if method == BASELINE else retrieval
        tiers = selection_pools(profiles, selected)
        counts = [(label, candidates, len(distinct_seeds(candidates))) for label, candidates in tiers]
        label, variance_pool, _ = next((tier for tier in counts if tier[2] >= 2), counts[-1])
        seed_scores = [c.hns for c in distinct_seeds(variance_pool).values()]
        variance_label = method if label == "동일 설정" else f"{method} · {label}"
        sd = stdev(seed_scores) if len(seed_scores) >= 2 else None
        target = ref.normalize(ref.paper)
        game_rows(summary, row, game, variance_label, baseline_count, retrieval_count, len(seed_scores), sd,
                  ref, counts[-1][2], last_option)
        for count in SEED_COUNTS:
            key = f"{game}|{count}"
            tier = next((tier for tier in counts if tier[2] >= count), None)
            if tier is None:
                options.append((key, game, count, method, target))
            else:
                label, candidates, _ = tier
                result = best_option(candidates, count, target)
                group = result.candidates
                labels = ", ".join(f"{c.snapshot.gpu}:{c.snapshot.seed}" for c in group)
                options.append((key, game, count, method, target, result.mean, result.delta, abs(result.delta),
                                result.min_mean, result.max_mean, labels, sum(c.eval_step is None for c in group),
                                sum(bool(c.owner_note) for c in group), min(c.hns for c in group), label))
                for c in group:
                    s = c.snapshot
                    detail.append((game, count, c.method, s.gpu, s.seed, c.score, c.hns, c.eval_step or "미기록",
                                   c.eval_precision or "미기록", c.owner_note, s.machine, s.path, ", ".join(c.run_ids),
                                   setting_description(c.profile)))
                    number_format(detail, detail.max_row, "F", "G")
            number_format(options, options.max_row, "E", "F", "G", "H", "I", "J", "N")
        if method == RETRIEVAL and profile:
            sources.append((game, f"t=16 비교 설정 · {setting_description(profile)}"))

    sources.append(("편차 허용치", "시드 HNS 표본 SD / √k가 허용치 이내인 최소 k를 2·3·4 중에서 선택합니다."))
    sources.append(("조합 순위", "선택한 k에서 논문 HNS 이상인 평균을 우선합니다. 같은 방향에서는 논문과 가장 가까운 평균을 표시합니다."))
    sources.append(("하방 허용 오차", "선택된 평균 또는 개별 시드가 논문 HNS보다 허용 오차를 초과해 낮을 때 각각 경고합니다."))
    sources.append(("재현 성능 이상", "선택 평균 HNS가 논문보다 허용 오차를 초과해 낮지만 STORM (reproduced) HNS 이상이면 논문 비교 판정에 표시합니다."))
    sources.append(("HNS", "(원점수 − Random) / (Human − Random). 논문 원점수와 두 참조 점수는 Results 시트에서 읽습니다."))
    sources.append(("Mean·Median HNS", "선택 조합의 평균 HNS를 게임당 하나씩 집계합니다. 조합이 있는 게임은 같은 비중으로 반영합니다."))
    sources.append(("점수 기준", f"같은 baseline 설정으로 관측한 시드가 {args.min_baseline_seeds}개 미만이면 t=16 retrieval-on에서 시작합니다."))
    sources.append(("t=16 설정 선택", "게임마다 모델과 주요 retrieval 설정이 같은 후보 중, 서로 다른 시드 수가 가장 많은 설정을 사용합니다."))
    sources.append(("후보 보충", "같은 설정 → 다른 t=16 설정 → 다른 retrieval 설정 → retrieval-off 순서로 새 시드를 추가합니다."))
    sources.append(("실제 조합 시드 수", "권장 개수에 모자라면 확보 가능한 최대 개수의 조합과 그 성능을 표시합니다."))
    sources.append(("반복 관측", "한 스냅샷의 같은 W&B 설정과 평가 조건의 반복은 중앙값을 사용합니다. 조합에서 동일 숫자 시드는 한 번만 셉니다."))
    sources.append(("평가 조건", "102,000 학습 스텝 · 평가 20회 · 100,000 평가 체크포인트. 스텝 미기록 legacy 포함: "
                    + ("아니요" if args.strict_100k else "예")))
    sources.append(("GPU 소유권", "큐의 기존 소유권 또는 인벤토리의 중복 GPU를 표시합니다. 숫자 시드의 GPU 배정은 검토 후 확정할 수 있습니다."))
    sources.append(("논문 기준 파일", str(args.reference)))
    for report_path, report in reports:
        sources.append((report["gpu_type"], f"{report_path} · 스캔 {report.get('collected_at', '미기록')}"))
    options.column_dimensions["A"].hidden = True
    first_data = SUMMARY_HEADER + 1
    paper_status_range = f"M{first_data}:M{summary.max_row}"
    for status, color in (("논문 이상", "FFDCEFE2"), ("재현 성능 이상", "FFDCEFE2"),
                          ("조합 자료 부족", "FFECEFF3")):
        summary.conditional_formatting.add(paper_status_range, FormulaRule(
            formula=[f'$M{first_data}="{status}"'], fill=PatternFill(bgColor=color)))
    summary.conditional_formatting.add(paper_status_range, FormulaRule(
        formula=[f'$M{first_data}="경고 · 논문보다 낮음"'], fill=PatternFill(bgColor="FFFFC7CE"),
        font=Font(color="FF9C0006")))
    summary.conditional_formatting.add(f"G{first_data}:G{summary.max_row}", FormulaRule(
        formula=[f'$G{first_data}="경고 · 낮은 시드 포함"'], fill=PatternFill(bgColor="FFFCE7CF")))
    for sheet, first in ((summary, SUMMARY_HEADER), (options, 1), (detail, 1)):
        sheet.auto_filter.ref = f"A{first}:{get_column_letter(sheet.max_column)}{sheet.max_row}"
        for cells in sheet.iter_rows(min_row=first + 1):
            if cells[0].row % 2 == 0:
                for cell in cells:
                    cell.fill = PatternFill("solid", fgColor="F0F5F9")
            for cell in cells:
                cell.alignment = Alignment(vertical="center")
    append_hns_totals(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        workbook.save(temporary)
        temporary.replace(path)


def tolerance(value):
    number = float(value)
    if not isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("the HNS tolerance must be a finite, nonnegative number")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "training_queue/warmup_scores")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/converted_results.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "results/shared_warmup_selection.xlsx")
    parser.add_argument("--stability-tolerance", type=tolerance, default=0.05, help="HNS standard error tolerance (default: 0.05)")
    parser.add_argument("--paper-tolerance", type=tolerance, default=0.05, help="Allowed HNS shortfall below the paper (default: 0.05)")
    parser.add_argument("--min-baseline-seeds", type=int, default=4, help="Use t=16 below this many baseline seeds (default: 4)")
    parser.add_argument("--strict-100k", action="store_true", help="Require a logged evaluation checkpoint step of exactly 100,000")
    args = parser.parse_args(argv)
    if args.min_baseline_seeds < 2:
        parser.error("--min-baseline-seeds must be at least 2")
    if args.output.suffix.lower() != ".xlsx" or args.output.resolve() == args.reference.resolve():
        parser.error("--output must be an .xlsx file distinct from the reference workbook")
    try:
        refs = paper_references(args.reference)
        reports, snapshots, runs = read_reports(args.reports_dir)
        pools = candidate_pools(runs, refs, owner_notes(snapshots), args.strict_100k)
        write_workbook(args.output, refs, pools, reports, args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot create selection workbook: {error}\n")
    print(f"{len(refs)} games from {len(reports)} GPU reports → {args.output}")


if __name__ == "__main__":
    main()
