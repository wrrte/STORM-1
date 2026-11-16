"""CPU checks for TD-only selection, paired warmup and result separation."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

STORM = Path(__file__).resolve().parent
sys.path[:0] = [str(STORM.parent), str(STORM), str(STORM / "results")]

from retrieval import RetrievalContextManager
from training_branches import (
    configure_storm_retrieval_run, launch_storm_warmup_followup,
    parse_retrieval_mode, parse_storm_training_args,
)
from utils import load_config
import convert_csv_to_excel as exporter


class TDOnlySelectionTests(unittest.TestCase):
    def manager(self, threshold=3.5, statistics_mode="legacy", **overrides):
        config = dict(enable=True, trigger_mode="z_score", score_combination="td_error_only",
                      td_error_z_score_threshold=threshold, hash_bits=2, anchor_offset=-2)
        config.update(overrides)
        manager = RetrievalContextManager(2, config, latent_dim=2, device="cpu",
                                          statistics_mode=statistics_mode)
        for env, base in enumerate((10, 30)):
            for pointer in range(base - 1, base + 2):
                manager.index_to_bucket[(pointer, env)] = 7
        return manager

    def feed(self, manager, errors, values=None, **kwargs):
        values = torch.zeros(2, 5) if values is None else values
        rewards = torch.zeros_like(values)
        # Construct prescribed signed TD residuals independently of value levels.
        rewards[:, 1:4] = torch.tensor(errors) + values[:, 1:4] - 0.5 * values[:, 2:5]
        return manager.add_batch_transitions(
            values, rewards, torch.zeros_like(values), 0.5,
            np.array([10, 30]), np.array([0, 1]), 100, skip_len=1, **kwargs)

    def test_absolute_td_per_env_normalization_and_configured_threshold(self):
        errors = [[-1.0, -5.0, 3.0], [2.0, -4.0, 1.0]]
        for threshold, expected in ((3.0, [(30, 1)]), (1.5, [(10, 0), (30, 1)])):
            with self.subTest(threshold=threshold):
                manager = self.manager(threshold, z_score_threshold=1000.0,
                                       additive_z_score_threshold=1000.0)
                manager.ema_mean[:] = [1.0, 0.0]
                manager.ema_var[:] = [4.0, 1.0]
                self.assertEqual(self.feed(manager, errors), len(expected))
                self.assertEqual([anchor for anchor, _ in manager.active_anchors], expected)

    def test_value_levels_and_value_statistics_do_not_affect_selection(self):
        errors = [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]]
        for signal in ("value_diff", "value"):
            manager = self.manager(value_signal=signal)
            manager.ema_vd_mean[:] = np.nan
            manager.ema_value_mean[:] = np.nan
            values = torch.tensor([[100.0, -80.0, 40.0, -60.0, 200.0],
                                   [-100.0, 80.0, -40.0, 60.0, -200.0]])
            self.assertEqual(self.feed(manager, errors, values), 2)
            self.assertEqual([a for a, _ in manager.active_anchors], [(10, 0), (30, 1)])
            self.assertEqual(manager.statistics_signals, ("td_error",))

    def test_scores_use_previous_ema_before_updating_it(self):
        manager = self.manager(4.5, ema_alpha=1.0)
        self.assertEqual(self.feed(manager, [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]]), 1)
        self.assertEqual(manager.active_anchors[0][0], (10, 0))
        self.assertEqual(manager.ema_mean[0], 3.0)

    def test_transition_mask_excludes_the_largest_error(self):
        manager = self.manager()
        mask = torch.tensor([[True, True, False, True], [True, False, False, False]])
        self.assertEqual(self.feed(manager, [[1.0, 100.0, -4.0], [100.0, 100.0, 100.0]],
                                   transition_mask=mask), 1)
        self.assertEqual(manager.active_anchors[0][0], (11, 0))

    def test_warmup_collects_statistics_without_triggering(self):
        manager = self.manager()
        self.assertEqual(self.feed(manager, [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]], is_warmup=True), 0)
        self.assertFalse(manager.active_anchors)
        self.assertGreater(manager.ema_mean[0], 0)
        np.testing.assert_array_equal(manager.ema_vd_mean, np.zeros(2))
        shared = self.manager(enable=["retrieval", "td_error_only"], score_combination="multiply")
        self.assertEqual(shared.statistics_signals, ("td_error", "value_diff"))
        reusable = self.manager(save_warmup=True)
        self.assertEqual(reusable.statistics_signals, ("td_error", "value_diff", "value"))

    def test_existing_combination_modes_keep_their_thresholds(self):
        for mode, threshold in (("multiply", 3.0), ("add", 5.5)):
            manager = self.manager(1000.0, score_combination=mode,
                                   z_score_threshold=threshold, additive_z_score_threshold=threshold)
            self.assertEqual(self.feed(manager, [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]]), 1)
            self.assertEqual(manager.active_anchors[0][0], (10, 0))

    def test_all_statistics_backends_support_td_only(self):
        errors = [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]]
        reference = None
        for mode in ("legacy", "batched", "vectorized"):
            with self.subTest(mode=mode):
                manager = self.manager(statistics_mode=mode)
                self.feed(manager, errors, is_warmup=True)
                self.assertEqual(self.feed(manager, errors), 2)
                if reference is None:
                    reference = manager
                np.testing.assert_allclose(manager.ema_mean, reference.ema_mean)
                np.testing.assert_allclose(manager.ema_var, reference.ema_var)

    def test_restoring_statistics_keeps_receiving_branch_configuration(self):
        source = self.manager(score_combination="multiply")
        self.feed(source, [[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]], is_warmup=True)
        target = self.manager(2.75)
        target.load_state_dict(source.state_dict())
        self.assertEqual(target.score_combination, "td_error_only")
        self.assertEqual(target.td_error_z_score_threshold, 2.75)
        np.testing.assert_array_equal(target.ema_mean, source.ema_mean)

    def test_invalid_threshold_or_trigger_mode_is_rejected(self):
        for threshold in (-1.0, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                self.manager(threshold)
        with self.assertRaisesRegex(ValueError, "trigger_mode=z_score"):
            self.manager(trigger_mode="absolute")


class TDOnlyIntegrationTests(unittest.TestCase):
    def config(self):
        conf = load_config(STORM / "config_files/STORM.yaml")
        conf.defrost()
        conf.JointTrainAgent.Retrieval.warmup_offset_pairs = []
        conf.freeze()
        return conf

    def test_yaml_can_select_td_only_and_set_its_threshold(self):
        conf = self.config()
        conf.defrost()
        conf.JointTrainAgent.Retrieval.score_combination = "td_error_only"
        conf.JointTrainAgent.Retrieval.td_error_z_score_threshold = 2.75
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "td_only.yaml"
            path.write_text(conf.dump())
            loaded = load_config(path)
        manager = RetrievalContextManager(1, loaded.JointTrainAgent.Retrieval, 2, device="cpu")
        self.assertEqual(manager.score_combination, "td_error_only")
        self.assertEqual(manager.td_error_z_score_threshold, 2.75)

    def test_cli_override_and_shared_child_preset(self):
        self.assertEqual(parse_retrieval_mode("td_error_only", allow_experiments=True), ["td_error_only"])
        args = SimpleNamespace(n="test", seed=7, env_name="ALE/Pong-v5", trajectory_path="unused",
                               resume_warmup=None, branch_experiment=None, branch_mode=None,
                               resume_from=None)
        conf = self.config()
        commands = configure_storm_retrieval_run(conf, args, [
            "JointTrainAgent.Retrieval.enable", "['retrieval', 'td_error_only']",
            "JointTrainAgent.Retrieval.td_error_z_score_threshold", "2.75",
        ], STORM / "train.py")
        self.assertEqual(list(commands), ["retrieval", "td_error_only"])
        self.assertEqual(conf.JointTrainAgent.Retrieval.td_error_z_score_threshold, 2.75)
        self.assertEqual(conf.JointTrainAgent.Retrieval.score_combination, "multiply")
        args.n, args.branch_experiment, args.resume_from = "child", "td_error_only", "unused"
        self.assertIsNone(configure_storm_retrieval_run(conf, args, [], STORM / "train.py"))
        self.assertEqual(args.n, "child_td_error_only")
        self.assertEqual(conf.JointTrainAgent.Retrieval.score_combination, "td_error_only")
        self.assertEqual(conf.JointTrainAgent.Retrieval.td_error_z_score_threshold, 2.75)

    def test_saved_td_statistics_suffice_for_td_only_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text(self.config().dump())
            for name in ("training_state.pt", "replay_buffer.npz"):
                (root / name).touch()
            (root / "warmup_metadata.json").write_text(json.dumps(dict(
                version=1, seed=7, env_name="ALE/Pong-v5", trajectory_path="unused",
                run_name="test", cwd=directory, step=50000, statistics_signals=["td_error"])))
            args, overrides = parse_storm_training_args([
                "--resume_warmup", directory, "JointTrainAgent.Retrieval.enable", "td_error_only",
                "JointTrainAgent.Retrieval.td_error_z_score_threshold", "2.75"])
            with patch("training_branches.launch_training_branches", side_effect=RuntimeError("intercept")) as launch:
                with self.assertRaisesRegex(RuntimeError, "intercept"):
                    launch_storm_warmup_followup(args, overrides, STORM / "train.py", load_config)
            self.assertEqual(list(launch.call_args.kwargs["experiments"]), ["td_error_only"])
            args, overrides = parse_storm_training_args([
                "--resume_warmup", directory, "JointTrainAgent.Retrieval.enable", "retrieval"])
            with self.assertRaisesRegex(ValueError, "lacks the statistics"):
                launch_storm_warmup_followup(args, overrides, STORM / "train.py", load_config)
            (root / "warmups.json").write_text(json.dumps(dict(
                version=1, kind="storm_warmup_sweep", pairs=[[50000, -2]],
                experiments=["retrieval"], run_name="test", seed=7,
                env_name="ALE/Pong-v5", trajectory_path="unused", cwd=directory,
                checkpoints={"50000": {"checkpoint": ".", "actual_step": 50000}})))
            args, overrides = parse_storm_training_args([
                "--resume_warmup", directory, "JointTrainAgent.Retrieval.enable", "td_error_only",
                "JointTrainAgent.Retrieval.td_error_z_score_threshold", "2.75"])
            with patch("training_branches.launch_training_branches", side_effect=RuntimeError("intercept")) as launch:
                with self.assertRaisesRegex(RuntimeError, "intercept"):
                    launch_storm_warmup_followup(args, overrides, STORM / "train.py", load_config)
            self.assertEqual(list(launch.call_args.kwargs["experiments"]), ["w50000_offset_m2_td_error_only"])

    def test_result_export_keeps_thresholds_separate_from_full_flash(self):
        rows = []
        for index, (mode, threshold) in enumerate((("multiply", 3.5), ("td_error_only", 2.0),
                                                   ("td_error_only", 3.0))):
            rows.append({"Run Name": f"Pong_test{index}_7", "Seed": 7, "State": "finished",
                         "Created At": "2026-11-16T00:00:00Z", "Retrieval Enable": True,
                         "Warmup Steps": 50000, "Batch Size Reduction": "retrieved",
                         "Z Score Threshold": 3.5, "Score Combination": mode,
                         "TD Error Z Score Threshold": threshold, "Retrieval Target": 16,
                         "Anchor Weight": 0.0625, "Eval Return": 10 + index, "Hash Bits": 10})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "scores.csv", root / "scores.xlsx"
            pd.DataFrame(rows).to_csv(source, index=False)
            with patch.object(exporter, "read_queued_runs", return_value=[]), \
                    patch.object(exporter, "read_paper_plans", return_value=[]), \
                    patch.object(exporter, "load_excluded_seeds", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                exporter.main(source, output, root)
            frame = pd.read_excel(output, sheet_name="Ablation Results")
            labels = frame["Config"].dropna().tolist()
            self.assertIn("target: 16 (anchor 미설정)", labels)
            self.assertIn("target: 16 (anchor 미설정) [td_error_only, threshold: 2]", labels)
            self.assertIn("target: 16 (anchor 미설정) [td_error_only, threshold: 3]", labels)


if __name__ == "__main__":
    unittest.main()
