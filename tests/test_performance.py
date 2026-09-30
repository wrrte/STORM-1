"""Regression checks for opt-in performance paths; no W&B or Atari run is started.

Run on an explicitly selected idle GPU:
    CUDA_VISIBLE_DEVICES=GPU-... python -m unittest discover -s tests -v
"""
import copy
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from replay_buffer import ReplayBuffer
from utils import configure_performance, load_config, seed_np_torch


def legacy_config():
    conf = load_config(str(ROOT / "config_files/STORM.yaml"))
    conf.defrost()
    conf.Performance.VectorizedReplaySampling = False
    conf.Performance.DisableDistributionValidation = False
    conf.Performance.BatchScalarLogging = False
    conf.Performance.RetrievalStatisticsMode = "legacy"
    conf.Performance.ProjectedKVCache = False
    conf.freeze()
    return conf


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3])


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return copy.deepcopy(value)


class ExactCase(unittest.TestCase):
    def assert_exact(self, left, right, path="result"):
        if isinstance(left, torch.Tensor):
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertEqual(left.shape, right.shape, path)
            left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            self.assertTrue(left_bytes == right_bytes, f"{path}: tensor bits differ")
        elif isinstance(left, np.ndarray):
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertEqual(left.shape, right.shape, path)
            self.assertTrue(left.tobytes() == right.tobytes(), f"{path}: array bits differ")
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_exact(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_exact(a, b, f"{path}/{index}")
        elif isinstance(left, float):
            self.assertEqual(struct.pack("!d", left), struct.pack("!d", right), path)
        else:
            self.assertEqual(left, right, path)


class ConfigTests(ExactCase):
    def setUp(self):
        self.previous_validation = torch.distributions.Distribution._validate_args

    def tearDown(self):
        torch.distributions.Distribution.set_default_validate_args(self.previous_validation)

    def test_old_config_without_performance_defaults_to_legacy(self):
        conf = legacy_config()
        self.assertFalse(conf.Performance.VectorizedReplaySampling)
        self.assertFalse(conf.Performance.DisableDistributionValidation)
        conf.defrost()
        del conf["Performance"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text(conf.dump())
            loaded = load_config(str(path))
        self.assertFalse(loaded.Performance.VectorizedReplaySampling)
        self.assertFalse(loaded.Performance.DisableDistributionValidation)
        self.assertFalse(loaded.Performance.BatchScalarLogging)
        self.assertFalse(loaded.Performance.ProjectedKVCache)
        self.assertEqual(loaded.Performance.RetrievalStatisticsMode, "legacy")

    def test_legacy_validation_setting_is_a_noop(self):
        conf = legacy_config()
        for original_default in (True, False):
            torch.distributions.Distribution.set_default_validate_args(original_default)
            with mock.patch.object(torch.distributions.Distribution, "set_default_validate_args") as setter:
                configure_performance(conf)
                setter.assert_not_called()
            self.assertEqual(torch.distributions.Distribution._validate_args, original_default)

    def test_validation_flag_disables_nested_onehot_checks(self):
        conf = legacy_config()
        conf.defrost()
        conf.Performance.DisableDistributionValidation = True
        conf.freeze()
        configure_performance(conf)
        dist = torch.distributions.OneHotCategorical(logits=torch.zeros(2, 4))
        self.assertFalse(dist._validate_args)
        self.assertFalse(dist._categorical._validate_args)
        self.assertFalse(conf.Performance.VectorizedReplaySampling)

    def test_overrides_survive_shared_config_roundtrip(self):
        sys.path.insert(0, str(ROOT.parent))
        from training_branches import configure_storm_retrieval_run, parse_storm_training_args
        args, extras = parse_storm_training_args([
            "-n", "perf-test", "-seed", "3710",
            "-config_path", str(ROOT / "config_files/STORM.yaml"),
            "-env_name", "ALE/Hero-v5", "-trajectory_path", "D_TRAJ/Hero.pkl",
            "Performance.VectorizedReplaySampling", "True",
            "Performance.DisableDistributionValidation", "True",
            "Performance.BatchScalarLogging", "True",
            "Performance.RetrievalStatisticsMode", "vectorized",
            "Performance.ProjectedKVCache", "True",
        ])
        conf = load_config(args.config_path)
        commands = configure_storm_retrieval_run(conf, args, extras, str(ROOT / "train.py"))
        self.assertIsNotNone(commands)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.yaml"
            path.write_text(conf.dump())
            loaded = load_config(str(path))
        self.assertTrue(loaded.Performance.VectorizedReplaySampling)
        self.assertTrue(loaded.Performance.DisableDistributionValidation)
        self.assertTrue(loaded.Performance.BatchScalarLogging)
        self.assertTrue(loaded.Performance.ProjectedKVCache)
        self.assertEqual(loaded.Performance.RetrievalStatisticsMode, "vectorized")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class ReplayParityTests(ExactCase):
    def make_replay(self, num_envs=1, store_on_gpu=True, obs_shape=(4, 4, 3), capacity=64):
        replay = ReplayBuffer(obs_shape, num_envs, max_length=capacity * num_envs,
                              warmup_length=8, store_on_gpu=store_on_gpu)
        generator = np.random.RandomState(14)
        for name in ("obs_buffer", "action_buffer", "reward_buffer", "termination_buffer"):
            target = getattr(replay, name)
            if name == "obs_buffer":
                values = generator.randint(0, 256, size=target.shape, dtype=np.uint8)
            elif name == "action_buffer":
                values = generator.randint(0, 18, size=target.shape).astype(np.float32)
            elif name == "termination_buffer":
                values = generator.randint(0, 2, size=target.shape).astype(np.float32)
            else:
                values = generator.normal(size=target.shape).astype(np.float32)
                values.flat[0] = -0.0
            if store_on_gpu:
                target.copy_(torch.from_numpy(values))
            else:
                target[:] = values
        replay.length = capacity
        replay.last_pointer = 5  # Include a wrapped buffer; preserve legacy index semantics.
        replay.external_buffer = {
            "obs": torch.randint(0, 256, (48, *obs_shape), dtype=torch.uint8, device="cuda"),
            "action": torch.randint(0, 18, (48,), device="cuda").float(),
            "reward": torch.randn(48, device="cuda"),
            "done": torch.randint(0, 2, (48,), device="cuda").float(),
        }
        replay.external_buffer_length = 48
        return replay

    def compare(self, replay, batch_size, external_batch_size, length):
        state = rng_state()
        replay.vectorized_sampling = False
        original = replay.sample(batch_size, external_batch_size, length)
        original_rng = rng_state()
        restore_rng(state)
        replay.vectorized_sampling = True
        optimized = replay.sample(batch_size, external_batch_size, length)
        self.assert_exact(original, optimized)
        self.assert_exact(original_rng, rng_state(), "rng")
        for a, b in zip(original[:4], optimized[:4]):
            self.assertEqual(a.stride(), b.stride())
            self.assertEqual(a.device, b.device)

    def test_gpu_replay_including_demonstrations_and_multiple_envs(self):
        for num_envs in (1, 2, 3):
            replay = self.make_replay(num_envs=num_envs)
            for batch, external, length in ((12, 0, 1), (17, 4, 8), (0, 4, 16), (12, 0, 48)):
                with self.subTest(envs=num_envs, batch=batch, external=external, length=length):
                    self.compare(replay, batch, external, length)

    def test_training_batch_shapes(self):
        replay = self.make_replay(obs_shape=(64, 64, 3), capacity=128)
        self.compare(replay, 16, 0, 64)
        self.compare(replay, 1024, 0, 8)

    def test_cpu_storage_ignores_optimization_and_keeps_existing_path(self):
        replay = self.make_replay(store_on_gpu=False)
        replay.external_buffer_length = None
        with mock.patch.object(replay, "_sample_gpu_vectorized", side_effect=AssertionError("CPU path changed")):
            self.compare(replay, 16, 0, 8)

    def test_disabled_flag_does_not_call_new_helpers(self):
        replay = self.make_replay()
        with mock.patch.object(replay, "_gather_gpu_sequences", side_effect=AssertionError("legacy path changed")):
            replay.sample(16, 4, 8)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class ModelUpdateParityTests(ExactCase):
    def test_original_repeat_and_all_flag_combinations(self):
        from train import build_agent, build_world_model
        previous_validation = torch.distributions.Distribution._validate_args
        previous_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        try:
            seed_np_torch(3710)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            conf = legacy_config()
            world = build_world_model(conf, 18)
            agent = build_agent(conf, 18)
            replay = ReplayParityTests().make_replay(obs_shape=(64, 64, 3), capacity=128)
            replay.external_buffer_length = None
            initial_models = snapshot([world.state_dict(), agent.state_dict()])
            initial_optimizers = snapshot([world.optimizer.state_dict(), agent.optimizer.state_dict()])
            initial_scalers = copy.deepcopy([world.scaler.state_dict(), agent.scaler.state_dict()])
            initial_rng = rng_state()
            gradients = {}

            def save_gradients(name, module):
                def hook(optimizer, args, kwargs):
                    gradients[name] = snapshot({
                        key: parameter.grad for key, parameter in module.named_parameters()
                        if parameter.grad is not None
                    })
                return module.optimizer.register_step_pre_hook(hook)

            handles = [save_gradients("world", world), save_gradients("agent", agent)]
            baseline = None
            for vectorized, disable_validation, batch_logging in (
                    (False, False, False), (False, False, False), (True, False, False),
                    (False, True, False), (True, True, False), (True, True, True)):
                for index, module in enumerate((world, agent)):
                    module.load_state_dict(initial_models[index])
                    module.optimizer.load_state_dict(copy.deepcopy(initial_optimizers[index]))
                    module.scaler.load_state_dict(initial_scalers[index])
                    module.optimizer.zero_grad(set_to_none=True)
                agent.lowerbound_ema.scalar = 0.0
                agent.upperbound_ema.scalar = 0.0
                # Each variant models a fresh process; legacy startup itself is a no-op.
                torch.distributions.Distribution.set_default_validate_args(True)
                conf.defrost()
                conf.Performance.VectorizedReplaySampling = vectorized
                conf.Performance.DisableDistributionValidation = disable_validation
                conf.freeze()
                configure_performance(conf)
                replay.vectorized_sampling = conf.Performance.VectorizedReplaySampling
                world.batch_scalar_logging = batch_logging
                agent.batch_scalar_logging = batch_logging
                restore_rng(initial_rng)

                measurements = []
                for _ in range(2):
                    metrics = {}

                    class Logger:
                        def log(self, tag, value):
                            metrics[tag] = value

                    obs, action, reward, done, *_ = replay.sample(16, 0, 64)
                    latent, features = world.update(obs, action, reward, done, logger=Logger())
                    world.eval()
                    agent.eval()
                    with torch.no_grad():
                        # The environment-action path also constructs distributions.
                        encoded = world.encode_obs(obs[:1, :16])
                        prior, last = world.calc_last_dist_feat(encoded, action[:1, :16])
                        env_action = agent.sample_as_env_action(torch.cat((prior, last), dim=-1))
                        context_obs, context_action, *_ = replay.sample(1024, 0, 8)
                        imagined = world.imagine_data(agent, context_obs, context_action, 1024, 16,
                                                      log_video=False, logger=None)
                    imagined_latent, imagined_action, imagined_reward, imagined_done = imagined
                    agent.update(imagined_latent, imagined_action, None, None,
                                 imagined_reward, imagined_done, logger=Logger(),
                                 weights=torch.ones(1024, device="cuda"))
                    self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                    measurements.append(snapshot({
                        "metrics": metrics, "latent": latent, "features": features,
                        "env_action": env_action, "imagined": imagined,
                        "gradients": gradients,
                        "models": [world.state_dict(), agent.state_dict()],
                        "optimizers": [world.optimizer.state_dict(), agent.optimizer.state_dict()],
                        "scalers": [world.scaler.state_dict(), agent.scaler.state_dict()],
                        "return_ema": [agent.lowerbound_ema.scalar, agent.upperbound_ema.scalar],
                        "rng": rng_state(),
                    }))
                if baseline is None:
                    baseline = measurements
                else:
                    self.assert_exact(baseline, measurements,
                                      f"vectorized={vectorized}, disable_validation={disable_validation}, batch_logging={batch_logging}")
                print(f"Validated flags: vectorized={vectorized}, disable_validation={disable_validation}",
                      flush=True)
            for handle in handles:
                handle.remove()
        finally:
            torch.distributions.Distribution.set_default_validate_args(previous_validation)
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous_cudnn


if __name__ == "__main__":
    unittest.main()
