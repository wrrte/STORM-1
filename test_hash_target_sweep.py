"""CPU regression checks for shared-prefix hash/target experiments."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

STORM = Path(__file__).resolve().parent
sys.path[:0] = [str(STORM.parent), str(STORM), str(STORM / "results")]

from retrieval import RetrievalContextManager
from storm_warmup_sweep import StormWarmupSweep, parse_hash_target_pairs
from training_branches import (
    capture_rng_state, configure_storm_retrieval_run, launch_storm_warmup_followup,
    parse_storm_training_args, rebuild_retrieval_for_resume, restore_rng_state,
)
from utils import load_config


def config():
    conf = load_config(STORM / "config_files/STORM.yaml")
    conf.defrost()
    conf.JointTrainAgent.Retrieval.warmup_offset_pairs = []
    conf.freeze()
    return conf


def checkpoint(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(config().dump())
    for name in ("training_state.pt", "replay_buffer.npz"):
        (root / name).touch()
    (root / "warmup_metadata.json").write_text(json.dumps(dict(
        version=1, seed=17, env_name="ALE/Pong-v5", trajectory_path="unused",
        run_name="Pong-17", cwd=str(root), step=50010,
        statistics_signals=["td_error", "value_diff"])))


class SweepTests(unittest.TestCase):
    def test_invalid_pairs_fail_before_training(self):
        invalid = [None, "[[10, 8]]", [10, 8], [[10]], [[0, 8]], [[63, 8]],
                   [[True, 8]], [[10.0, 8]], [[10, 0]], [[10, 1.5]], [[10, True]],
                   [[10, 8], [10, 8]]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_hash_target_pairs(value)
        self.assertEqual(parse_hash_target_pairs([[12, 1], [9, 16]]), [[12, 1], [9, 16]])

    def test_hash_grid_alone_collects_one_shared_prefix(self):
        args, overrides = parse_storm_training_args([
            "-n", "hash-grid-unit-test", "-seed", "17", "-config_path", "unused",
            "-env_name", "ALE/Pong-v5", "-trajectory_path", "unused",
            "JointTrainAgent.Retrieval.enable", "Both",
            "JointTrainAgent.Retrieval.hash_target_pairs", "[[9, 4], [12, 16]]"])
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                sweep = configure_storm_retrieval_run(config(), args, overrides, STORM / "train.py")
            finally:
                os.chdir(previous)
        self.assertIsInstance(sweep, StormWarmupSweep)
        self.assertEqual(sweep.targets, [50000])
        self.assertEqual(sweep.hash_target_pairs, [[9, 4], [12, 16]])
        self.assertEqual(sweep.experiments, ["retrieval", "baseline"])

    def test_all_pairs_reuse_snapshot_and_children_cannot_relaunch_sweep(self):
        pairs = [[bits, target] for bits in (9, 10, 12) for target in (4, 8, 16)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root / "snapshot")
            header = dict(run_name="Pong-17", seed=17, env_name="ALE/Pong-v5",
                          trajectory_path="unused", hash_target_pairs=pairs)
            sweep = StormWarmupSweep(root, header, [[50000, -2], [50000, 0]],
                                    ["retrieval", "baseline"], STORM / "train.py", [],
                                    {"50000": {"checkpoint": "snapshot", "actual_step": 50010}})
            commands = sweep.commands()
            self.assertEqual(len(commands), 19)  # Nine pairs per offset, one baseline.
            self.assertEqual(sum(name.endswith("_baseline") for name in commands), 1)
            self.assertEqual(len({cmd[cmd.index("-n") + 1] for cmd in commands.values()}), 19)
            for name, command in commands.items():
                args, extra = parse_storm_training_args(command[2:])
                self.assertEqual(Path(args.resume_from), root / "snapshot")
                self.assertTrue(args.rebuild_retrieval_on_resume)
                conf = config()
                self.assertIsNone(configure_storm_retrieval_run(conf, args, extra, STORM / "train.py"))
                self.assertEqual(conf.JointTrainAgent.Retrieval.hash_target_pairs, [])
                self.assertEqual(conf.JointTrainAgent.Retrieval.warmup_offset_pairs, [])
                if not name.endswith("_baseline"):
                    self.assertIn(f"_h{conf.JointTrainAgent.Retrieval.hash_bits}_t"
                                  f"{conf.JointTrainAgent.Retrieval.target}_", name)

    def test_target_preset_cannot_silently_override_pairs(self):
        with self.assertRaisesRegex(ValueError, "overrides target"):
            StormWarmupSweep(".", {"hash_target_pairs": [[10, 8]]}, [[50000, -2]],
                             ["target1"], STORM / "train.py", [])

    def test_old_collector_can_resume_with_empty_new_default(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                args = SimpleNamespace(n="Pong-17_WarmupSweep", base_run_name="Pong-17",
                                       seed=17, env_name="ALE/Pong-v5", trajectory_path="unused",
                                       resume_from=None)
                conf = config()
                sweep = StormWarmupSweep.collect(conf, args, [[50000, -2]], ["retrieval"],
                                                 STORM / "train.py", [])
                sweep.initialize(0)
                index_path = sweep.root / "warmups.json"
                old = json.loads(index_path.read_text())
                legacy = conf.clone()
                legacy.defrost()
                del legacy.JointTrainAgent.Retrieval["hash_target_pairs"]
                legacy.freeze()
                old["config"] = legacy.dump()
                index_path.write_text(json.dumps(old))
                args.resume_from = "interrupted"
                resumed = StormWarmupSweep.collect(conf, args, [[50000, -2]], ["retrieval"],
                                                   STORM / "train.py", [])
                self.assertEqual(resumed.hash_target_pairs, [])
            finally:
                os.chdir(previous)

    def test_old_shared_checkpoint_can_launch_a_grid_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            original = {p.name: p.read_bytes() for p in root.iterdir()}
            args, extra = parse_storm_training_args([
                "--resume_warmup", directory, "-n", "grid",
                "JointTrainAgent.Retrieval.enable", "['retrieval', 'baseline']",
                "JointTrainAgent.Retrieval.hash_target_pairs", "[[9, 4], [12, 16]]"])
            with patch("training_branches.launch_training_branches", side_effect=RuntimeError("launch")) as launch:
                with self.assertRaisesRegex(RuntimeError, "launch"):
                    launch_storm_warmup_followup(args, extra, STORM / "train.py", load_config)
            commands = launch.call_args.kwargs["experiments"]
            self.assertEqual(list(commands), ["retrieval_h9_t4", "retrieval_h12_t16", "baseline"])
            for command in commands.values():
                child, extra = parse_storm_training_args(command[2:])
                self.assertTrue(child.rebuild_retrieval_on_resume)
                conf = load_config(root / "config.yaml")
                self.assertIsNone(configure_storm_retrieval_run(conf, child, extra, STORM / "train.py"))
            for name, content in original.items():
                self.assertEqual((root / name).read_bytes(), content)

    def test_single_hash_override_also_rebuilds_old_shared_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root)
            args, extra = parse_storm_training_args([
                "--resume_warmup", directory, "JointTrainAgent.Retrieval.enable", "retrieval",
                "JointTrainAgent.Retrieval.hash_bits", "12"])
            with patch("training_branches.launch_training_branches", side_effect=RuntimeError("launch")) as launch:
                with self.assertRaisesRegex(RuntimeError, "launch"):
                    launch_storm_warmup_followup(args, extra, STORM / "train.py", load_config)
            command = launch.call_args.kwargs["experiments"]["retrieval"]
            self.assertTrue(parse_storm_training_args(command[2:])[0].rebuild_retrieval_on_resume)

    def test_saved_sweep_reuses_its_grid_or_accepts_new_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root / "snapshot")
            (root / "warmups.json").write_text(json.dumps(dict(
                version=1, kind="storm_warmup_sweep", run_name="Pong-17", seed=17,
                env_name="ALE/Pong-v5", trajectory_path="unused", cwd=directory,
                pairs=[[50000, -2]], hash_target_pairs=[[9, 4]], experiments=["retrieval"],
                checkpoints={"50000": {"checkpoint": "snapshot", "actual_step": 50010}})))
            for override, expected in (([], "h9_t4"),
                                       (["JointTrainAgent.Retrieval.hash_target_pairs", "[[12, 8]]"], "h12_t8")):
                args, extra = parse_storm_training_args(["--resume_warmup", directory, *override])
                with patch("training_branches.launch_training_branches", side_effect=RuntimeError("launch")) as launch:
                    with self.assertRaisesRegex(RuntimeError, "launch"):
                        launch_storm_warmup_followup(args, extra, STORM / "train.py", load_config)
                command_map = launch.call_args.kwargs["experiments"]
                self.assertEqual(list(command_map), [f"w50000_offset_m2_{expected}_retrieval"])

    def test_snapshot_records_grid_in_index_but_clears_it_in_child_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root / "snapshot")
            conf = config()
            conf.defrost()
            conf.JointTrainAgent.Retrieval.hash_target_pairs = [[9, 4], [12, 16]]
            conf.freeze()
            sweep = StormWarmupSweep(root, dict(hash_target_pairs=[[9, 4], [12, 16]]),
                                    [[50000, -2]], ["retrieval"], STORM / "train.py", [])
            args = SimpleNamespace(base_run_name="Pong-17", seed=17, env_name="ALE/Pong-v5",
                                   trajectory_path="unused")
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.register_snapshot(root / "snapshot", 50010, [50000], conf, args,
                                        ["td_error", "value_diff"])
            metadata = json.loads((root / "snapshot/warmup_metadata.json").read_text())
            self.assertTrue(metadata["rebuild_on_resume"])
            self.assertEqual(json.loads((root / "warmups.json").read_text())["hash_target_pairs"],
                             [[9, 4], [12, 16]])
            loaded = load_config(root / "snapshot/config.yaml")
            self.assertEqual(loaded.JointTrainAgent.Retrieval.hash_target_pairs, [])
            self.assertEqual(loaded.JointTrainAgent.Retrieval.warmup_offset_pairs, [])


class HashRebuildTests(unittest.TestCase):
    def manager(self, bits, **kwargs):
        return RetrievalContextManager(1, dict(enable=True, hash_bits=bits,
                                               context_length=2, **kwargs), 6, device="cpu")

    def test_different_width_requires_rebuild_and_keeps_ema(self):
        source = self.manager(2)
        source.ema_mean[:] = 42
        source.ema_vd_mean[:] = 17
        source._insert_into_bucket(0, 0, 3)
        source.active_anchors.append(((0, 0), 3))
        state = source.state_dict()
        for bits in (1, 2, 4):
            for use_pca in (False, True):
                with self.subTest(bits=bits, use_pca=use_pca):
                    target = self.manager(bits, use_pca=use_pca)
                    if bits != 2:
                        with self.assertRaisesRegex(ValueError, "hash projection"):
                            target.load_state_dict(state)
                    target.load_state_dict(state, restore_hash=False)
                    np.testing.assert_array_equal(target.ema_mean, [42])
                    np.testing.assert_array_equal(target.ema_vd_mean, [17])
                    self.assertEqual(tuple(target.hash_proj.shape), (6, bits))
                    self.assertEqual(target.hash_bit_values.numel(), bits)
                    self.assertFalse(target.hash_memory)
                    self.assertFalse(target.active_anchors)
                    # Rebuild a real CPU index from the same replay, covering
                    # both PCA and random projection after a width change.
                    replay = SimpleNamespace(
                        max_length=24, num_envs=1, length=24, last_pointer=23, store_on_gpu=True,
                        obs_buffer=torch.arange(24 * 6, dtype=torch.uint8).reshape(24, 1, 1, 2, 3),
                        action_buffer=torch.zeros(24, 1, dtype=torch.long),
                        termination_buffer=torch.zeros(24, 1))
                    world = SimpleNamespace(encode_obs=lambda obs, **_: obs.flatten(2))
                    with contextlib.redirect_stdout(io.StringIO()):
                        target.rebuild_all_hash_buckets(replay, world, chunk_size=7)
                    self.assertEqual(tuple(target.hash_proj.shape), (6, bits))
                    self.assertEqual(len(target.index_to_bucket), 24)
                    self.assertTrue(all(0 <= key < 2 ** bits for key in target.hash_memory))
                    self.assertFalse(target.active_anchors)
                    target.active_anchors.append(((1, 0), target.index_to_bucket[(1, 0)]))
                    result = target.retrieve_contexts(replay, world, max_anchors=1,
                                                      multiplier=4, target=4, return_indices=True)
                    self.assertEqual(result[5], 1)
                    self.assertTrue(1 <= len(result[6]) <= 4)
                    self.assertAlmostEqual(sum(result[4]), 1.0)

    def test_rebuild_keeps_training_rng_and_restores_baseline_flag_on_failure(self):
        manager = self.manager(4)
        manager.enabled = False
        replay = SimpleNamespace(ready=lambda: True)
        world = SimpleNamespace(eval=lambda: None)
        initial = capture_rng_state()
        try:
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restore_rng_state(initial)
            def build(*args, **kwargs):
                self.assertTrue(manager.enabled)
                random.random()
                np.random.rand(17)
                torch.rand(29)
            with patch.object(manager, "rebuild_all_hash_buckets", side_effect=build):
                self.assertTrue(rebuild_retrieval_for_resume(manager, replay, world, preserve_rng=True))
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual[:2], expected[:2])
            torch.testing.assert_close(actual[2], expected[2])
            self.assertFalse(manager.enabled)
            restore_rng_state(initial)
            with patch.object(manager, "rebuild_all_hash_buckets", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    rebuild_retrieval_for_resume(manager, replay, world, preserve_rng=True)
            self.assertFalse(manager.enabled)
            torch.testing.assert_close(torch.random.get_rng_state(), initial["torch"])
        finally:
            restore_rng_state(initial)


class ResultExportTests(unittest.TestCase):
    def test_hash_widths_and_small_targets_are_not_dropped_or_merged(self):
        import pandas as pd
        import convert_csv_to_excel as exporter
        rows = []
        for index, (bits, target) in enumerate(((9, 4), (10, 4), (12, 8))):
            rows.append({"Run Name": f"Pong_test{index}_17", "Seed": 17, "State": "finished",
                         "Local Run Name": f"Pong-17_w50000_offset_m2_h{bits}_t{target}_retrieval",
                         "Created At": "2026-11-19T00:00:00Z", "Retrieval Enable": True,
                         "Warmup Steps": 50000, "Batch Size Reduction": "retrieved",
                         "Z Score Threshold": 3.5, "Retrieval Target": target,
                         "Anchor Weight": 0.0625, "Anchor Weight R": 1,
                         "Eval Return": 10 + index, "Hash Bits": bits})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(rows).to_csv(root / "scores.csv", index=False)
            with patch.object(exporter, "read_queued_runs", return_value=[]), \
                    patch.object(exporter, "read_paper_plans", return_value=[]), \
                    patch.object(exporter, "load_excluded_seeds", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                exporter.main(root / "scores.csv", root / "scores.xlsx", root)
            frame = pd.read_excel(root / "scores.xlsx", sheet_name="Ablation Results")
            labels = frame["Config"].dropna().tolist()
            for bits, target in ((9, 4), (10, 4), (12, 8)):
                self.assertIn(f"target: {target} (anchor r: 1) [hash_bits: {bits}]", labels)


if __name__ == "__main__":
    unittest.main()
