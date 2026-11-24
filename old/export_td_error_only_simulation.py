#!/usr/bin/env python3
"""Export the saved first simulation trial in final_results.xlsx's layout.

Uses example_slots.csv without drawing another sample. The workbook contains
two method rows per game, three simulated slots, means, aggregate metrics, and
a Sources sheet. Yellow cells are borrowed value-ablation observations; they
do not represent actual TD-only runs. The template is never overwritten.
"""

import argparse
from copy import copy
import csv
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, PatternFill
from openpyxl.utils import get_column_letter

from simulate_td_error_only_ablation import DEFAULT_OUTPUT, DEFAULT_TEX, completed_metrics
from update_tex import extract_float, main_table_rows, metric_name


ROOT = Path(__file__).resolve().parents[2]


def export(simulation_dir, template_path, output_path, tex_path):
    source_csv = simulation_dir / "example_slots.csv"
    source_metadata = simulation_dir / "inputs.json"
    inputs = {p.resolve() for p in (template_path, tex_path, source_csv, source_metadata)}
    if output_path.resolve() in inputs:
        raise ValueError("Output must differ from all source files")
    with source_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if metadata["target_slots_per_game"] != 3:
        raise ValueError("This workbook layout expects three simulated slots per game")
    by_game = {}
    for row in rows:
        game, slot = row["game"], int(row["slot"])
        if slot in by_game.setdefault(game, {}):
            raise ValueError(f"Duplicate game/slot: {game}/{slot}")
        by_game[game][slot] = row
    if not by_game or any(set(slots) != {1, 2, 3} for slots in by_game.values()):
        raise ValueError("Every game must have exactly slots 1, 2, 3")
    games = list(by_game)
    scores = np.array([[[float(by_game[g][s][field]) for s in (1, 2, 3)] for g in games]
                       for field in ("full_flash_score", "td_only_or_proxy_score")])[None]
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    references = {parts[0].strip(): (extract_float(parts[1]), extract_float(parts[2]))
                  for _, parts, _ in main_table_rows(tex_path.read_text(encoding="utf-8").splitlines(True))
                  if not metric_name(parts[0])}
    metrics = completed_metrics(scores, games, references)

    workbook = load_workbook(template_path)
    template = workbook["STORM"]
    sheet = workbook.create_sheet("TD_error_only")
    sources = workbook.create_sheet("Sources")
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "C4"
    sheet.sheet_properties.pageSetUpPr = copy(template.sheet_properties.pageSetUpPr)
    sheet.page_setup = copy(template.page_setup)
    sheet.page_margins = copy(template.page_margins)
    column_map = (1, 2, 3, 4, 5, 8)
    for column, original in enumerate(column_map, 1):
        letter = get_column_letter(column)
        sheet.column_dimensions[letter].width = 25 if column == 2 else (
            24 if column == 1 else 16)

    def styled_cell(row, column, value, template_row, template_column=None):
        cell = sheet.cell(row, column, value)
        cell._style = copy(template.cell(template_row, template_column or column_map[column-1])._style)
        return cell

    for row in (1, 2):
        styled_cell(row, 1, "", row)
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        sheet.row_dimensions[row].height = template.row_dimensions[row].height
    sheet["A1"] = "STORM — TD-error-only simulation"
    sheet["A2"] = (f"Trial 1; RNG {metadata['rng_seed']} | 노랑: 가상 보충값 | "
                   "1–3은 슬롯 번호. 실제 출처 seed는 셀 메모·Sources 참조.")
    sheet["A2"].alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[2].height = 34
    for column, label in enumerate(("Game", "Method", 1, 2, 3, "Mean"), 1):
        cell = styled_cell(3, column, label, 3)
        if column in (3, 4, 5):
            cell.comment = Comment("Simulated slot number, not a shared training seed ID. "
                                   "Each game has its own source seeds; see cell comments and Sources.", "Simulation")
    sheet.row_dimensions[3].height = template.row_dimensions[3].height
    yellow = PatternFill("solid", fgColor="FFFFF2CC")
    for game_index, game in enumerate(games):
        first_row = 4 + 2*game_index
        for method, name in enumerate(("Full FLASH", "TD-error only")):
            row_index = first_row + method
            sheet.row_dimensions[row_index].height = template.row_dimensions[4+method].height
            styled_cell(row_index, 1, game if method == 0 else None, 4+method)
            styled_cell(row_index, 2, name, 4+method)
            for slot in (1, 2, 3):
                provenance = by_game[game][slot]
                synthetic = provenance["is_synthetic"] == "True"
                cell = styled_cell(row_index, slot+2, float(scores[0, method, game_index, slot-1]), 4+method)
                cell.number_format = "#,##0.##"
                if synthetic:
                    cell.fill = copy(yellow)
                cell.comment = Comment(
                    f"Game: {game}\nSimulation trial: 1; slot: {slot}\n"
                    f"Source training seed: {provenance['source_training_seed']}\n"
                    f"Source table: {provenance['source_table']}\n"
                    f"Data type: {'borrowed pair (simulated completion)' if synthetic else 'observed TD-only pair'}\n"
                    f"Source method: {'Full FLASH' if method == 0 else provenance['source_variant']}",
                    "Simulation",
                )
            styled_cell(row_index, 6, float(scores[0, method, game_index].mean()), 4+method)
        sheet.merge_cells(start_row=first_row, start_column=1, end_row=first_row+1, end_column=1)

    metric_header = 4+2*len(games)+2
    for offset, labels in enumerate((
            ("Metric", "Full FLASH", "TD-error only"),
            *[(name, float(values[0, 0]), float(values[0, 1])) for name, values in metrics.items()])):
        row_index = metric_header+offset
        sheet.row_dimensions[row_index].height = template.row_dimensions[58+offset].height
        for column in range(1, 7):
            styled_cell(row_index, column, None, 58+offset, column)
        for column, value in zip((1, 3, 5), labels):
            sheet.cell(row_index, column, value)
            sheet.merge_cells(start_row=row_index, start_column=column,
                              end_row=row_index, end_column=column+1)
            if offset > 0:
                sheet.cell(row_index, column).number_format = (
                    "General" if column == 1 else ("0" if offset == 1 else "0.000"))
    sheet.print_options.horizontalCentered = True
    sheet.print_area = f"A1:F{metric_header+5}"
    sheet.print_title_rows = "1:3"

    sources.sheet_view.showGridLines = False
    sources.freeze_panes = "D5"
    sources.merge_cells("A1:H1")
    sources["A1"] = f"Simulation trial 1 / RNG {metadata['rng_seed']} / {len(games)} games × 3 slots"
    sources["A1"]._style = copy(template["A1"]._style)
    sources.row_dimensions[1].height = 40
    sources.merge_cells("A2:H2")
    sources["A2"] = "Synthetic slots reuse Absolute value pairs as TD-only proxies; source seeds may repeat."
    sources["A2"].alignment = Alignment(wrap_text=True, vertical="center")
    sources.row_dimensions[2].height = 30
    headers = ("Game", "Slot", "Source seed", "Data type", "Source variant", "Full FLASH", "TD-only / proxy", "Source table")
    for column, label in enumerate(headers, 1):
        cell = sources.cell(4, column, label)
        cell._style = copy(template["A3"]._style)
        sources.column_dimensions[get_column_letter(column)].width = 27 if column in (1, 4, 5) else 19
    for row_index, row in enumerate(rows, 5):
        synthetic = row["is_synthetic"] == "True"
        values = (row["game"], int(row["slot"]), int(row["source_training_seed"]),
                  "Synthetic completion" if synthetic else "Observed pair", row["source_variant"],
                  float(row["full_flash_score"]), float(row["td_only_or_proxy_score"]), row["source_table"])
        for column, value in enumerate(values, 1):
            cell = sources.cell(row_index, column, value)
            if column in (6, 7):
                cell.number_format = "#,##0.##"
            if synthetic:
                cell.fill = copy(yellow)
    sources.auto_filter.ref = f"A4:H{4+len(rows)}"
    for old_sheet in list(workbook):
        if old_sheet not in (sheet, sources):
            workbook.remove(old_sheet)
    workbook.active = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=ROOT / "final_results.xlsx")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--output", type=Path, default=ROOT / "final_results_td_error_only_simulation.xlsx")
    args = parser.parse_args()
    print(export(args.simulation_dir, args.template, args.output, args.tex))


if __name__ == "__main__":
    main()
