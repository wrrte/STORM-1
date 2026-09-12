"""최신 run을 선택해 미사용/anchor-only 두 기준의 비교 결과를 XLSX에 저장한다.

실행: python compare_target_results.py
      python compare_target_results.py --csv results/wandb_runs_classification.csv \
          --output results/target_comparison.xlsx

의존성: pandas, numpy, openpyxl
기본 입력/출력은 이 스크립트가 있는 디렉터리 기준이다.
Random/Human 점수는 상위 디렉터리의 TeX에서 읽으며, TeX는 수정하지 않는다.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


ROOT = Path(__file__).resolve().parent
BASELINE = "Retrieval 미사용"
ANCHOR_ONLY = "target: 1 (anchor 미설정)"
REFERENCES = {"Base": BASELINE, "Anchor": ANCHOR_ONLY}
EMA_DATE = pd.Timestamp("2026-08-24T11:05:43Z")

DEFAULT_TEX = ROOT.parent / "iclr2027_conference.tex"
METRICS = ["#Superhuman", "Mean", "Median", "IQM", "Optimality Gap"]


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_normalization(tex_path):
    """main_performance 표의 게임별 Random/Human 원점수만 읽는다."""
    text = Path(tex_path).read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    table = next((block for block in re.findall(
        r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, flags=re.S
    ) if r"\label{tab:main_performance}" in block), None)
    if table is None:
        raise ValueError("TeX에서 tab:main_performance 표를 찾을 수 없습니다.")
    result = {}
    columns = None
    for line in table.splitlines():
        parts = [part.strip() for part in line.split("&")]
        if parts[0] == "Game" and "Random" in parts and "Human" in parts:
            columns = (parts.index("Random"), parts.index("Human"))
            continue
        if columns is None or len(parts) <= max(columns):
            continue
        game = parts[0]
        if not re.fullmatch(r"[A-Za-z]+", game) or game in METRICS:
            continue
        values = tuple(number(parts[i].replace(",", "")) for i in columns)
        if None in values or values[0] == values[1]:
            raise ValueError(f"TeX의 Random/Human 점수가 유효하지 않습니다: {game}")
        if game in result:
            raise ValueError(f"TeX에 중복 게임 행이 있습니다: {game}")
        result[game] = values
    if not result:
        raise ValueError("TeX 표에서 Random/Human 점수를 읽지 못했습니다.")
    return result


def load_latest(csv_path):
    """기존 실험 필터를 적용하고 (Game, Config, Seed)별 최신 유효 점수를 선택한다.

    hash_bits 우선순위는 적용하지 않는다. 날짜 동률이면 Run ID 오름차순으로
    선택하며, 그것도 같으면 CSV에서 먼저 나온 행을 선택한다.
    """
    source = pd.read_csv(csv_path)
    required = {
        "Run Name", "Run ID", "Eval Return", "Retrieval Enable", "Warmup Steps",
        "Batch Size Reduction", "Z Score Threshold", "Retrieval Target",
        "Anchor Weight", "Seed", "Created At",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"CSV 필수 열 누락: {', '.join(sorted(missing))}")

    records = []
    for _, row in source.iterrows():
        if row.get("State") in ("running", "killed"):
            continue
        score = number(row["Eval Return"])
        if score is None:
            continue
        parts = str(row["Run Name"]).split("_")
        seed = number(row["Seed"])
        if seed is None:
            if len(parts) >= 4 and parts[-1].upper() in ("O", "X"):
                seed = number(parts[-2])
            elif len(parts) >= 3:
                seed = number(parts[2])
        if seed is None or seed < 0 or not seed.is_integer():
            continue
        enabled = str(row["Retrieval Enable"]).strip().lower() in ("true", "1", "1.0", "t")
        target = number(row["Retrieval Target"])
        anchor = number(row["Anchor Weight"])
        if enabled:
            if (number(row["Warmup Steps"]) != 50000
                    or str(row["Batch Size Reduction"]).strip() != "retrieved"
                    or number(row["Z Score Threshold"]) != 3.5):
                continue
            if target is None:
                raise ValueError(f"Retrieval Target 누락: {row['Run ID']}")
        created = pd.to_datetime(row["Created At"], utc=True, errors="coerce")
        if pd.isna(created):
            raise ValueError(f"최신 실행을 판별할 수 없는 Created At: {row['Run ID']}")
        # Retrieval 미사용은 과거 데이터도 유지한다.
        if enabled and created < EMA_DATE:
            continue
        config = BASELINE
        if enabled:
            suffix = "anchor 미설정" if anchor is None else f"anchor: {anchor:g}"
            config = f"target: {target:g} ({suffix})"
        records.append({
            "Game": parts[0], "Config": config, "Seed": int(seed),
            "Eval Return": score, "Created At": created,
            "Run ID": str(row["Run ID"]), "Run Name": row["Run Name"],
            "Target": target if enabled else None,
            "Anchor Weight": anchor if enabled else None,
            "Hash Bits": number(row.get("Hash Bits")),
            "Warmup Steps": number(row["Warmup Steps"]),
            "Calculated Warmup Steps": number(row.get("Calculated Warmup Steps")),
            "Source": "CSV",
        })

    # 기존 convert_csv_to_excel.py의 수동 복구 데이터. 최신 CSV 결과가 있으면 대체된다.
    for seed, score, date in (
        (10, 2068.0, "2026-08-11T04:36:29Z"),
        (3710, 1904.0, "2026-08-11T04:36:35Z"),
    ):
        records.append({
            "Game": "Frostbite", "Config": BASELINE, "Seed": seed,
            "Eval Return": score, "Created At": pd.Timestamp(date),
            "Run ID": f"legacy-frostbite-{seed}", "Run Name": "Manual recovery",
            "Source": "Legacy manual recovery",
        })

    runs = pd.DataFrame(records).sort_values(
        ["Created At", "Source", "Run ID"], ascending=[False, True, True], kind="stable"
    )
    keys = ["Game", "Config", "Seed"]
    runs["Candidate Count"] = runs.groupby(keys)["Seed"].transform("size")
    latest = runs.drop_duplicates(keys, keep="first").copy()
    return latest.sort_values(keys).reset_index(drop=True)


def config_order(runs, reference=BASELINE):
    retrieval = runs[runs["Config"] != BASELINE].drop_duplicates("Config")
    retrieval = retrieval.sort_values(["Target", "Anchor Weight"], na_position="first")
    ordered = [BASELINE] + list(retrieval["Config"])
    return [reference] + [c for c in ordered if c != reference]


def compare(runs, normalization, reference=BASELINE):
    """선택한 기준에 대해 기존 공통 seed 보정식을 적용한다."""
    if reference not in set(runs["Config"]):
        raise ValueError(f"필터 적용 후 기준 설정 '{reference}'의 유효 점수가 없습니다.")
    configs = config_order(runs, reference)
    games = sorted(set(normalization) | set(runs["Game"]))
    scores = pd.DataFrame(np.nan, index=games, columns=configs)
    counts = pd.DataFrame(0, index=games, columns=configs)
    details = []
    for game, group in runs.groupby("Game"):
        values = group.pivot(index="Seed", columns="Config", values="Eval Return")
        if reference not in values:
            continue
        ref = values[reference].dropna()
        pairs = {}
        for config in configs:
            if config == reference or config not in values:
                continue
            paired = values[[reference, config]].dropna()
            counts.loc[game, config] = len(paired)
            if not paired.empty:
                pairs[config] = paired
        counts.loc[game, reference] = len(ref)
        if not pairs:
            # 비교할 공통 seed가 없으면 보정 점수는 비워 둔다. 실제 평균은 Raw Means에 남긴다.
            continue
        # 동률이면 미사용, 이후 target/anchor 숫자 오름차순을 우선한다.
        base_config = max(pairs, key=lambda c: len(pairs[c]))
        base = float(pairs[base_config][reference].mean())
        scores.loc[game, reference] = base
        for config, paired in pairs.items():
            ref_mean = float(paired[reference].mean())
            other_mean = float(paired[config].mean())
            delta = other_mean - ref_mean
            scores.loc[game, config] = base + delta
            details.append({
                "Game": game, "Reference": reference, "Config": config, "Common Seeds": len(paired),
                "Seed IDs": ", ".join(str(s) for s in paired.index),
                "Reference Paired Mean": ref_mean, "Config Paired Mean": other_mean,
                "Delta (Config - Reference)": delta,
                "Relative Delta (%)": delta / abs(ref_mean) * 100 if ref_mean != 0 else np.nan,
                "Reference Base": base, "Base Selected From": base_config,
                "Base Seed IDs": ", ".join(str(s) for s in pairs[base_config].index),
                "Adjusted Score": base + delta,
            })
    raw = runs.pivot_table(index="Game", columns="Config", values="Eval Return", aggfunc="mean")
    raw = raw.reindex(index=games, columns=configs)
    for frame in (scores, counts, raw):
        frame.index.name = "Game"
        frame.columns.name = None
    detail_columns = [
        "Game", "Reference", "Config", "Common Seeds", "Seed IDs", "Reference Paired Mean",
        "Config Paired Mean", "Delta (Config - Reference)", "Relative Delta (%)",
        "Reference Base", "Base Selected From", "Base Seed IDs", "Adjusted Score",
    ]
    return scores, raw, counts, pd.DataFrame(details, columns=detail_columns)


def summarize(scores, normalization):
    """모든 설정에 보정 점수가 있는 동일 게임 집합에서 기존 HNS 지표를 계산한다."""
    complete = scores.dropna()
    missing = set(complete.index) - set(normalization)
    if missing:
        raise ValueError(f"TeX에 정규화 점수가 없는 게임: {', '.join(sorted(missing))}")
    rows = []
    for config in scores.columns:
        hns = np.array([
            (row[config] - normalization[game][0])
            / (normalization[game][1] - normalization[game][0])
            for game, row in complete.iterrows()
        ])
        n = len(hns)
        trim = n // 4
        rows.append({
            "Config": config, "Games": n,
            "#Superhuman": int((hns > 1).sum()) if n else np.nan,
            "Mean": hns.mean() if n else np.nan,
            "Median": np.median(hns) if n else np.nan,
            "IQM": np.sort(hns)[trim:n-trim].mean() if n else np.nan,
            "Optimality Gap": np.maximum(0, 1 - hns).mean() if n else np.nan,
            "Included Games": ", ".join(complete.index),
        })
    return pd.DataFrame(rows)


def metric_label(metric):
    return f"{metric} {'↓' if metric == 'Optimality Gap' else '↑'}"


def paired_summary(comparisons, normalization):
    """각 직접 비교의 동일 게임/seed 평균으로 지표와 차이를 계산한다. 보정 없음."""
    columns = [
        "Config", "Reference", "Games", "Seed Pairs", "Min Seeds per Game",
        "Wins", "Ties", "Losses", "Metric", "Reference Value", "Config Value",
        "Delta (Config - Reference)", "Improvement (%)", "Included Games",
    ]
    rows = []
    for (config, reference), group in comparisons.groupby(["Config", "Reference"], sort=False):
        scores = group.set_index("Game")[["Reference Paired Mean", "Config Paired Mean"]]
        scores.columns = [reference, config]
        summary = summarize(scores, normalization).set_index("Config")
        delta = group["Config Paired Mean"] - group["Reference Paired Mean"]
        for metric in METRICS:
            ref_value = summary.loc[reference, metric]
            value = summary.loc[config, metric]
            difference = value - ref_value
            improvement = -difference if metric == "Optimality Gap" else difference
            rows.append({
                "Config": config, "Reference": reference, "Games": len(group),
                "Seed Pairs": int(group["Common Seeds"].sum()),
                "Min Seeds per Game": int(group["Common Seeds"].min()),
                "Wins": int((delta > 0).sum()), "Ties": int((delta == 0).sum()),
                "Losses": int((delta < 0).sum()), "Metric": metric_label(metric),
                "Reference Value": ref_value, "Config Value": value,
                "Delta (Config - Reference)": difference,
                "Improvement (%)": improvement / abs(ref_value) * 100 if ref_value != 0 else np.nan,
                "Included Games": ", ".join(scores.index),
            })
    return pd.DataFrame(rows, columns=columns)


def dual_comparisons(runs):
    """후보/미사용/anchor-only 세 설정 모두에 있는 동일 seed만 사용한다."""
    rows = []
    candidates = [c for c in config_order(runs) if c not in REFERENCES.values()]
    for game, group in runs.groupby("Game"):
        values = group.pivot(index="Seed", columns="Config", values="Eval Return")
        if BASELINE not in values or ANCHOR_ONLY not in values:
            continue
        for config in candidates:
            if config not in values:
                continue
            shared = values[[BASELINE, ANCHOR_ONLY, config]].dropna()
            if shared.empty:
                continue
            for reference in REFERENCES.values():
                ref_mean = float(shared[reference].mean())
                config_mean = float(shared[config].mean())
                rows.append({
                    "Game": game, "Config": config, "Reference": reference,
                    "Common Seeds": len(shared),
                    "Seed IDs": ", ".join(str(s) for s in shared.index),
                    "Reference Paired Mean": ref_mean, "Config Paired Mean": config_mean,
                    "Delta (Config - Reference)": config_mean - ref_mean,
                })
    return pd.DataFrame(rows, columns=[
        "Game", "Config", "Reference", "Common Seeds", "Seed IDs",
        "Reference Paired Mean", "Config Paired Mean", "Delta (Config - Reference)",
    ])


def write_report(runs, output, csv_path, tex_path=DEFAULT_TEX):
    normalization = load_normalization(tex_path)
    # 두 기준 모두 검증/계산을 마친 뒤에만 출력 파일을 연다.
    reports = {prefix: compare(runs, normalization, reference)
               for prefix, reference in REFERENCES.items()}
    notes = [
        ("Input CSV", str(Path(csv_path).resolve())),
        ("Normalization source", str(Path(tex_path).resolve())),
        ("Base sheets", f"{BASELINE} 기준으로 나머지 모든 설정 비교."),
        ("Anchor sheets", f"{ANCHOR_ONLY} 기준으로 나머지 모든 설정 비교. 이 설정은 retrieval 없는 anchor-only ablation."),
        ("Latest", "Game/Config/Seed별 Created At이 가장 최신인 유효 점수. hash_bits 우선순위 없음."),
        ("Timestamp ties", "같은 생성 시각이면 CSV 우선, Run ID 오름차순, 이후 원본 행 순서."),
        ("Filters", "Retrieval 사용: 2026-08-24 11:05:43 UTC 이후, warmup=50000, BSR=retrieved, z=3.5."),
        ("Baseline", "Retrieval 미사용은 과거 데이터도 포함. Frostbite 수동 복구 2건은 기존 변환기에서 유지."),
        ("States", "기존 분류기와 동일하게 running/killed 제외. 점수가 유한한 실행만 사용."),
        ("Scores", "공통 seed 수가 최대인 비교의 기준 평균 B를 선택. 보정 점수 = B + mean(config - reference)."),
        ("Base ties", "공통 seed 수 동률이면 미사용 우선, 이후 target/anchor 오름차순(미설정 우선)."),
        ("Raw Means", "설정별 선택된 모든 seed의 실제 평균. 보정 점수와 다를 수 있음."),
        ("Seed Counts", "각 비교 설정과 기준의 공통 seed 수. 기준 열은 기준의 전체 유효 seed 수."),
        ("Comparisons", "각 설정과 기준의 공통 seed 평균 및 차이. Delta 양수는 비교 설정의 점수가 더 높음."),
        ("Summary", "기준별 모든 설정의 보정 점수가 존재하는 게임 교집합에서 HNS 계산. 기준별 Games/Included Games가 다를 수 있음."),
        ("Paired Summary", "보정 없이 각 기준/설정 쌍의 공통 seed 평균으로 지표 비교. 비교 쌍마다 게임/seed 집합이 다를 수 있음."),
        ("Dual Summary", "각 후보/미사용/anchor-only 세 설정의 공통 seed만 사용. 한 후보의 두 기준 비교는 동일 게임/seed. 후보 간에는 집합이 다를 수 있음."),
        ("Dual Comparisons", "Dual Summary에 실제 사용된 게임별 평균과 seed 목록. 두 기준에 대한 후보 평균은 동일."),
        ("Improvement (%)", "Paired/Dual Summary에서 양수는 개선. Optimality Gap은 감소율, 나머지는 증가율. 기준값 0이면 빈칸."),
        ("Interpretation", "지표는 점추정치이며 신뢰구간/유의성 검정 없음. ICLR 제출 충분성이나 retrieval 필수성을 자동 판정하지 않음."),
        ("Score footer", "Scores 하단에 #Superhuman, Mean, Median, IQM, Optimality Gap 표시. Summary와 동일한 값."),
        ("HNS", "(보정 점수 - Random) / (Human - Random). TeX의 게임별 원점수를 사용."),
        ("Metric definitions", "#Superhuman: HNS > 1인 게임 수. Mean/Median: HNS 평균/중앙값. Optimality Gap: mean(max(0, 1-HNS))."),
        ("IQM", "기존 코드와 동일: 게임별 HNS 정렬 후 양 끝 floor(N/4)개 제거."),
        ("Missing", "공통 seed 또는 기준 점수가 없으면 Scores는 빈칸. 기준 평균 0이면 상대 차이 %는 빈칸."),
        ("Colors", "Scores에서 기준보다 높으면 파랑, 낮으면 빨강(Optimality Gap은 반대). 절대 변화율 15% 이상은 굵게."),
        ("Precision", "수치 계산은 반올림하지 않음. Excel 숫자 서식만 표시 자릿수 제한."),
    ]
    selected = runs.copy()
    selected["Created At"] = selected["Created At"].map(lambda t: t.isoformat())
    normalization_table = pd.DataFrame.from_dict(normalization, orient="index", columns=["Random", "Human"])
    normalization_table.index.name = "Game"
    sheets = {"Readme": pd.DataFrame(notes, columns=["Item", "Description"])}
    for prefix, (scores, raw, counts, comparisons) in reports.items():
        summary = summarize(scores, normalization)
        metric_rows = summary.set_index("Config")[METRICS].T.reindex(columns=scores.columns)
        metric_rows.index.name = "Game"
        sheets.update({
            f"{prefix} Scores": pd.concat([scores, metric_rows]).reset_index(),
            f"{prefix} Summary": summary.rename(columns={m: metric_label(m) for m in METRICS}),
            f"{prefix} Seed Counts": counts.reset_index(),
            f"{prefix} Comparisons": comparisons,
        })
    all_comparisons = pd.concat([report[3] for report in reports.values()], ignore_index=True)
    dual = dual_comparisons(runs)
    sheets.update({
        "Paired Summary": paired_summary(all_comparisons, normalization),
        "Dual Summary": paired_summary(dual, normalization),
        "Dual Comparisons": dual,
        "Raw Means": reports["Base"][1].reset_index(), "Selected Runs": selected,
        "Normalization": normalization_table.reset_index(),
    })
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="234E70")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[1].height = 42
            for column in ws.columns:
                width = max(len(str(c.value or "")) for c in column)
                ws.column_dimensions[column[0].column_letter].width = min(max(width + 2, 14), 48)
                for cell in column[1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000" if name.endswith("Summary") else "0.0#"
                    elif isinstance(cell.value, str):
                        # CSV 문자열이 Excel 수식으로 해석되지 않도록 한다.
                        cell.data_type = "s"
            if name == "Readme":
                ws.column_dimensions["B"].width = 105
                for row in ws.iter_rows(min_row=2):
                    row[1].alignment = Alignment(wrap_text=True, vertical="top")
                    ws.row_dimensions[row[0].row].height = 32
        for prefix, reference in REFERENCES.items():
            scores = reports[prefix][0]
            ws = writer.sheets[f"{prefix} Scores"]
            ws.auto_filter.ref = f"A1:{ws.cell(len(scores) + 1, ws.max_column).coordinate}"
            ref_col = list(scores.columns).index(reference) + 2
            for row in ws.iter_rows(min_row=2):
                label = row[0].value
                if label in METRICS:
                    for cell in row:
                        cell.fill = PatternFill("solid", fgColor="E8EFF5")
                        cell.number_format = "0" if label == "#Superhuman" else "0.000"
                    row[0].font = Font(bold=True)
                    row[0].value = metric_label(label)
                base = number(row[ref_col - 1].value)
                if base is None:
                    continue
                for cell in row[1:]:
                    value = number(cell.value)
                    if cell.column == ref_col or value is None:
                        continue
                    delta = value - base
                    bold = abs(delta) >= abs(base) * 0.15 if base else delta != 0
                    improvement = -delta if label == "Optimality Gap" else delta
                    cell.font = Font(color="0000FF" if improvement > 0 else "FF0000" if improvement < 0 else "000000", bold=bold)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "wandb_runs_classification.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "target_comparison.xlsx")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Random/Human 점수를 읽을 TeX 파일 (읽기 전용)")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".xlsx":
        parser.error("--output은 .xlsx 파일이어야 합니다.")
    if args.output.resolve() == args.csv.resolve():
        parser.error("입력 파일과 출력 파일은 달라야 합니다.")
    try:
        runs = load_latest(args.csv)
        reports = write_report(runs, args.output, args.csv, args.tex)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"오류: {exc}\n")
    print(f"저장 완료: {args.output.resolve()}")
    print(f"선택된 점수 {len(runs)}개")
    for prefix, reference in REFERENCES.items():
        scores, _, _, comparisons = reports[prefix]
        print(f"{prefix}: {reference} 기준, 설정 {len(scores.columns)}개, 공통 seed 비교 {len(comparisons)}개")


if __name__ == "__main__":
    main()
