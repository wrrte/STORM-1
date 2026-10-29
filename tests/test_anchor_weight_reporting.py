"""Offline checks that relative-weight runs never merge into fixed-weight results."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REPORT_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None
                            for name in ("openpyxl", "pandas", "wandb", "yaml"))


def load_report_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "results" / filename)
    module = importlib.util.module_from_spec(spec)
    # The classifier adjusts sys.path to avoid results/old/wandb.py shadowing W&B.
    with mock.patch.object(sys, "path", list(sys.path)):
        spec.loader.exec_module(module)
    return module


if REPORT_DEPS_AVAILABLE:
    from openpyxl import load_workbook
    classifier = load_report_module("anchor_weight_classifier", "classify_wandb_runs.py")
    excel = load_report_module("anchor_weight_excel", "convert_csv_to_excel.py")
    scheduler = load_report_module("anchor_weight_scheduler", "schedule_experiments.py")


def run(ratio=None, *, enabled=True, score=10, source="retrieval", created=None,
        warmup=50000, offset=None, actual_step=None):
    settings = {
        "enable": enabled, "warmup_steps": warmup, "batch_size_reduction": "retrieved",
        "target": 16, "anchor_weight": .0625, "z_score_threshold": 3.5,
        "hash_bits": 10, "neighbor_source": source,
    }
    if ratio is not None:
        settings["anchor_weight_r"] = ratio
    if offset is not None:
        settings["anchor_offset"] = offset
    summary = {"eval/episode_avg_return": score}
    if actual_step is not None:
        summary["Retrieval/shared_warmup_ended_at_step"] = actual_step
    return SimpleNamespace(
        name=f"Pong_weight{score}_17_{'O' if enabled else 'X'}", id=f"weight{score}",
        state="finished", commit=None, created_at=created or "2026-10-28T00:00:00Z",
        summary=summary,
        config={"JointTrainAgent": {"Retrieval": settings}},
    )


@unittest.skipUnless(REPORT_DEPS_AVAILABLE, "reporting dependencies are not installed")
class AnchorWeightReportingTests(unittest.TestCase):
    def test_classifier_exports_optional_ratio_without_changing_fixed_weight(self):
        for ratio in (None, 1, 2.5):
            with self.subTest(ratio=ratio):
                row = classifier.classify_run(run(ratio))
                self.assertEqual(row["Anchor Weight"], .0625)
                self.assertEqual(row["Anchor Weight R"], "N/A" if ratio is None else ratio)

    def test_excel_separates_ratios_and_preserves_old_baseline_data(self):
        runs = [run(score=10), run(1, score=20), run(2, score=30),
                run(1.0000001, score=40), run(2, score=50, source="random"),
                run(enabled=False, score=60, created="2001-01-01T00:00:00Z"),
                run(2, score=70, warmup=20000, offset=-4, actual_step=20590),
                run(2, score=80, offset=0),
                run(enabled=False, score=90, warmup=20000, actual_step=20590)]
        rows = [classifier.classify_run(item) for item in runs]
        original_read = Path.read_text

        def read_fixture(path, *args, **kwargs):
            if path.name == "manual_results.json":
                return "[]"
            return original_read(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            csv_path, output = Path(directory) / "runs.csv", Path(directory) / "results.xlsx"
            with csv_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with mock.patch.object(excel, "load_excluded_seeds", return_value={}), \
                    mock.patch.object(excel, "read_queued_runs", return_value=[]), \
                    mock.patch.object(Path, "read_text", new=read_fixture), \
                    contextlib.redirect_stdout(io.StringIO()):
                excel.main(csv_path, output, Path(directory))
            workbook = load_workbook(output, read_only=True)
            self.addCleanup(workbook.close)
            sheet_rows = list(workbook["Results"].values)
            seed_column = sheet_rows[0].index(17)
            values = {row[1]: row[seed_column] for row in sheet_rows[1:]}
            expected = {
                "target: 16 (anchor 미설정)": 10,
                "target: 16 (anchor r: 1)": 20,
                "target: 16 (anchor r: 2)": 30,
                "target: 16 (anchor r: 1.0000001)": 40,
                "target: 16 (anchor r: 2) [RandomNeighor]": 50,
                "Retrieval 미사용": 60,
                "target: 16 (anchor r: 2) [warmup: 20000, offset: -4]": 70,
                "target: 16 (anchor r: 2) [offset: 0]": 80,
                "Retrieval 미사용 [warmup: 20000]": 90,
            }
            for label, score in expected.items():
                self.assertIn(label, values)
                self.assertEqual(excel.parse_score(values[label]), score)

    def test_queued_yaml_and_cli_ratio_overrides_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config_files").mkdir()
            (root / "config_files/STORM.yaml").write_text(
                "JointTrainAgent:\n  Retrieval:\n    enable: True\n"
                "    anchor_weight: 0.0625\n    anchor_weight_r: 2\n")
            command = "python train.py -n Pong-17 -seed 17 -env_name ALE/Pong-v5"
            row = excel.queue_command_row(command, root)
            self.assertEqual(row["Anchor Weight R"], 2)
            row = excel.queue_command_row(command + " JointTrainAgent.Retrieval.anchor_weight_r 4", root)
            self.assertEqual(row["Anchor Weight R"], 4)
            row = excel.queue_command_row(command + " JointTrainAgent.Retrieval.anchor_weight_r None", root)
            self.assertIsNone(row["Anchor Weight R"])

    def test_legacy_scheduler_does_not_count_relative_runs_as_fixed_runs(self):
        self.assertEqual(scheduler.row_kind(classifier.classify_run(run())), "retrieval")
        self.assertEqual(scheduler.row_kind(classifier.classify_run(run(enabled=False))), "baseline")
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio):
                self.assertIsNone(scheduler.row_kind(classifier.classify_run(run(ratio))))
                command = ("python train.py -n Pong-17 -seed 17 -env_name ALE/Pong-v5 "
                           f"JointTrainAgent.Retrieval.enable True JointTrainAgent.Retrieval.anchor_weight_r {ratio}")
                self.assertEqual(scheduler.parse_command(command, "3090")["kinds"], [])
        self.assertIsNone(scheduler.row_kind(classifier.classify_run(run(offset=0))))

    def test_classifier_records_requested_and_actual_warmup_for_both_modes(self):
        for enabled in (True, False):
            row = classifier.classify_run(run(enabled=enabled, warmup=20000, offset=-4, actual_step=20590))
            self.assertEqual(row["Warmup Steps"], 20000)
            self.assertEqual(row["Calculated Warmup Steps"], 20590)
            self.assertEqual(row["Anchor Offset"], -4)

    def test_pending_paired_runs_expand_each_warmup_and_offset(self):
        for mode, count in ((True, 3), ("Both", 5), ('["retrieval", "baseline"]', 5)):
            with self.subTest(mode=mode):
                frame = excel.pd.DataFrame([{
                    "Run Name": "Pong_pending_17", "Retrieval Enable": mode, "State": "queued",
                    "Warmup Offset Pairs": "[[20000, -4], [50000, -2], [50000, 0]]",
                    "Warmup Steps": 80000, "Calculated Warmup Steps": 80000,
                }])
                rows = excel.expand_shared_runs(frame).to_dict("records")
                self.assertEqual(len(rows), count)
                retrieval = {(row["Warmup Steps"], row["Anchor Offset"])
                             for row in rows if row["Retrieval Enable"]}
                self.assertEqual(retrieval, {(20000, -4), (50000, -2), (50000, 0)})
                self.assertTrue(all(row["Calculated Warmup Steps"] == row["Warmup Steps"] for row in rows))


if __name__ == "__main__":
    unittest.main()
