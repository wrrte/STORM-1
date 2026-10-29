"""CPU integration checks for fixed/relative retrieval group weights."""
import ast
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from replay_buffer import ReplayBuffer
from retrieval import RetrievalContextManager
from training_branches import configure_storm_retrieval_run
from utils import load_config


class Encoder:
    def encode_obs(self, observations, sample_mode):
        endpoints = (observations[:, 0, 0, 0, 0] * 255).round().long()
        positive = torch.isin(endpoints, torch.tensor([8, 9, 10, 24]))
        return torch.where(positive[:, None, None], 1., -1.).expand(-1, 1, 4)


def make_replay(batched=False):
    if batched:
        replay = ReplayBuffer((1, 1, 1), 1, max_length=32, warmup_length=0,
                              store_on_gpu=False)
        for step in range(32):
            replay.append(np.full((1, 1, 1, 1), step, np.uint8),
                          np.array([step], np.float32), np.zeros(1), np.zeros(1))
        return replay
    return SimpleNamespace(
        max_length=32, num_envs=1, length=32, last_pointer=31, store_on_gpu=False,
        obs_buffer=np.arange(32, dtype=np.uint8).reshape(32, 1, 1, 1, 1),
        action_buffer=np.arange(32, dtype=np.float32).reshape(32, 1),
        termination_buffer=np.zeros((32, 1), np.float32),
    )


def make_manager(**settings):
    manager = RetrievalContextManager(1, {
        "enable": True, "context_length": 3, "hash_bits": 2, "use_pca": False,
        "anchor_weight": .0625, **settings,
    }, latent_dim=4, device="cpu")
    manager.hash_proj.fill_(1)
    for pointer in (8, 9, 10, 11):
        manager._insert_into_bucket(pointer, 0, 3)
    for pointer in (20, 21, 22, 23, 24):
        manager._insert_into_bucket(pointer, 0, 0)
    manager.active_anchors.extend([((8, 0), 3), ((20, 0), 0)])
    return manager


def retrieve(manager, replay, **overrides):
    options = dict(max_anchors=2, multiplier=4, target=16, max_contexts=5,
                   return_indices=True)
    options.update(overrides)
    return manager.retrieve_contexts(replay, Encoder(), **options)


def load_imagine_function():
    tree = ast.parse((ROOT / "train.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "world_model_imagine_data")
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "train.py"), "exec"), namespace)
    return namespace["world_model_imagine_data"]


class AnchorWeightTests(unittest.TestCase):
    def test_relative_weights_use_verified_and_capped_group_sizes(self):
        for batched in (False, True):
            for source in ("retrieval", "random"):
                with self.subTest(batched=batched, source=source):
                    manager = make_manager(anchor_weight_r=2, neighbor_source=source)
                    result = retrieve(manager, make_replay(batched))
                    # Two stale hashes fail verification: groups have 3 and 4
                    # members, then max_contexts trims the second group to 2.
                    self.assertEqual(result[2], 7)
                    self.assertEqual(result[5], 2)
                    self.assertEqual(result[6][0], (8, 0))
                    self.assertEqual(result[6][3], (20, 0))
                    self.assertEqual(result[4], [.5, .25, .25, 2 / 3, 1 / 3])
                    self.assertAlmostEqual(sum(result[4][:3]), 1.)
                    self.assertAlmostEqual(sum(result[4][3:]), 1.)

    def test_ratio_one_is_uniform_and_positive_fraction_can_downweight_anchor(self):
        for ratio, expected in ((1, [1 / 3] * 3), (.5, [.2, .4, .4]),
                                (4, [2 / 3, 1 / 6, 1 / 6])):
            with self.subTest(ratio=ratio):
                result = retrieve(make_manager(anchor_weight_r=ratio), make_replay(),
                                  max_anchors=1)
                self.assertEqual(result[4], expected)

    def test_episode_validity_filtering_precedes_weight_calculation(self):
        replay = make_replay(True)
        replay.termination_buffer[9, 0] = 1  # Invalidates context ending at 10.
        result = retrieve(make_manager(anchor_weight_r=2), replay, max_anchors=1)
        self.assertEqual(result[6], [(8, 0), (9, 0)])
        self.assertEqual(result[4], [2 / 3, 1 / 3])

    def test_fixed_mode_and_explicit_null_keep_legacy_weights(self):
        for weight in (0., .0625, .7, 1.):
            for settings in ({}, {"anchor_weight_r": None}):
                with self.subTest(weight=weight, settings=settings):
                    result = retrieve(make_manager(anchor_weight=weight, **settings),
                                      make_replay())
                    self.assertEqual(result[4], [weight, (1 - weight) / 2,
                                                (1 - weight) / 2, weight, 1 - weight])

    def test_ratio_only_changes_weights_and_survives_checkpoint_loading(self):
        for source in ("retrieval", "random"):
            with self.subTest(source=source):
                fixed = make_manager(neighbor_source=source)
                relative = make_manager(anchor_weight_r=2, neighbor_source=source)
                relative.load_state_dict(fixed.state_dict())
                self.assertEqual(relative.anchor_weight_r, 2.)
                before = random.getstate()
                fixed_result = retrieve(fixed, make_replay(True))
                after = random.getstate()
                random.setstate(before)
                relative_result = retrieve(relative, make_replay(True))
                self.assertEqual(random.getstate(), after)
                torch.testing.assert_close(fixed_result[0], relative_result[0], rtol=0, atol=0)
                torch.testing.assert_close(fixed_result[1], relative_result[1], rtol=0, atol=0)
                self.assertEqual(fixed_result[2:4], relative_result[2:4])
                self.assertEqual(fixed_result[5:], relative_result[5:])
                self.assertNotEqual(fixed_result[4], relative_result[4])
                self.assertEqual(fixed._random_neighbor_rng.getstate(),
                                 relative._random_neighbor_rng.getstate())
                restored_fixed = make_manager()
                restored_fixed.load_state_dict(relative.state_dict())
                self.assertIsNone(restored_fixed.anchor_weight_r)

    def test_single_anchor_and_empty_queue_keep_existing_behavior(self):
        for ratio in (None, .5, 1, 4):
            with self.subTest(ratio=ratio):
                manager = make_manager(anchor_weight_r=ratio)
                result = retrieve(manager, make_replay(), max_anchors=1, target=1)
                self.assertEqual(result[4], [1.])
                manager.active_anchors.clear()
                self.assertIsNone(retrieve(manager, make_replay())[0])

    def test_invalid_ratios_fail_before_hash_projection(self):
        for ratio in (0, -1, float('nan'), float('inf'), float('-inf'),
                      True, False, "invalid", [], {}, 10 ** 1000):
            with self.subTest(ratio=ratio):
                with mock.patch("torch.randn") as projection:
                    with self.assertRaisesRegex(ValueError, "anchor_weight_r"):
                        make_manager(anchor_weight_r=ratio)
                    projection.assert_not_called()

    def test_warmup_keeps_random_contexts_and_unit_weights(self):
        manager = make_manager(anchor_weight_r=4)
        manager.retrieve_contexts = mock.Mock(side_effect=AssertionError("warmup retrieval"))
        replay = make_replay()
        replay.sample = mock.Mock(return_value=(torch.zeros(10, 3, 1, 1, 1),
            torch.zeros(10, 3), None, None, None, None))
        world = SimpleNamespace(eval=mock.Mock(), imagine_data=mock.Mock(
            return_value=tuple(torch.zeros(10, 2) for _ in range(4))))
        result = load_imagine_function()(
            replay, world, mock.Mock(), manager, 10, 0, 3, 2, False, mock.Mock(),
            is_warmup=True)
        manager.retrieve_contexts.assert_not_called()
        replay.sample.assert_called_once_with(10, 0, 3)
        torch.testing.assert_close(result[-1], torch.ones(10))

    def test_imagination_forwards_relative_weights_and_keeps_batch_size(self):
        imagine = load_imagine_function()
        for source in ("retrieval", "random"):
            with self.subTest(source=source):
                manager = make_manager(anchor_weight_r=2, neighbor_source=source)
                manager.config.update(max_anchors=2, target=16, multiplier=4,
                                      max_contexts=5, batch_size_reduction="retrieved")
                replay = make_replay()
                replay.sample = mock.Mock(return_value=(torch.zeros(5, 3, 1, 1, 1),
                    torch.zeros(5, 3), None, None, None, None))
                world = Encoder()
                world.eval = mock.Mock()
                world.imagine_data = mock.Mock(return_value=tuple(torch.zeros(10, 2) for _ in range(4)))
                result = imagine(replay, world, mock.Mock(), manager, 10, 0, 3, 2,
                                 False, mock.Mock())
                replay.sample.assert_called_once_with(5, 0, 3)
                self.assertEqual(world.imagine_data.call_args.kwargs["imagine_batch_size"], 10)
                torch.testing.assert_close(result[-1], torch.tensor(
                    [1.] * 5 + [.5, .25, .25, 2 / 3, 1 / 3]))
                self.assertAlmostEqual(result[-1].sum().item(), 7.)


class AnchorWeightConfigTests(unittest.TestCase):
    def configure(self, config, ratio):
        args = SimpleNamespace(n="weights", seed=0, config_path=str(ROOT / "config_files/STORM.yaml"),
            env_name="ALE/Pong-v5", trajectory_path=str(ROOT / "Pong.pkl"),
            branch_experiment=None, branch_mode=None, resume_from=None)
        configure_storm_retrieval_run(config, args, [
            "JointTrainAgent.Retrieval.enable", "True",
            "JointTrainAgent.Retrieval.anchor_weight_r", repr(ratio),
        ], ROOT / "train.py")

    def test_default_and_older_configs_use_fixed_weights(self):
        config = load_config(ROOT / "config_files/STORM.yaml")
        self.assertIsNone(config.JointTrainAgent.Retrieval.anchor_weight_r)
        self.assertEqual(config.JointTrainAgent.Retrieval.anchor_weight, .0625)
        config.defrost()
        del config.JointTrainAgent.Retrieval["anchor_weight_r"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text(config.dump())
            self.assertIsNone(load_config(path).JointTrainAgent.Retrieval.anchor_weight_r)

    def test_yaml_cli_and_saved_configs_accept_integer_float_and_null(self):
        config = load_config(ROOT / "config_files/STORM.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for ratio in (2, 2., .5, None):
                with self.subTest(ratio=ratio):
                    self.configure(config, ratio)
                    self.assertEqual(config.JointTrainAgent.Retrieval.anchor_weight_r, ratio)
                    path.write_text(config.dump())
                    config = load_config(path)
                    manager = RetrievalContextManager(1, config.JointTrainAgent.Retrieval,
                                                      latent_dim=4, device="cpu")
                    self.assertEqual(manager.anchor_weight_r, ratio)


if __name__ == "__main__":
    unittest.main()
