"""Numerical, RNG, state, and checkpoint contracts for execution optimizations."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from test_performance import ExactCase, legacy_config, rng_state, restore_rng, snapshot
from utils import log_scalar_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from retrieval import RetrievalContextManager


class ScalarLoggingTests(ExactCase):
    def test_mixed_dtypes_order_signed_zero_and_rng(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            metrics = {
                "half": torch.tensor(-0., device=device, dtype=torch.float16),
                "single": torch.tensor(1.234567, device=device),
                "double": torch.tensor(1.2345678912345, device=device, dtype=torch.float64),
                "brain": torch.tensor([3.14], device=device, dtype=torch.bfloat16),
            }
            results = []
            before = rng_state()
            for batched in (False, True):
                logs = []
                class Logger:
                    def log(self, tag, value):
                        logs.append((tag, value))
                log_scalar_metrics(Logger(), metrics, batched=batched)
                results.append(logs)
            self.assert_exact(results[0], results[1])
            self.assert_exact(before, rng_state())


class RetrievalStatisticsTests(ExactCase):
    def manager(self, device, mode, num_envs=4):
        return RetrievalContextManager(num_envs, {
            "enable": ["value", "add"], "trigger_mode": "z_score",
            "hash_bits": 2, "ema_alpha": .01, "use_pca": False,
        }, 4, device=device, statistics_mode=mode)

    def test_masked_reductions_empty_singleton_and_checkpoint(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                managers = [self.manager(device, mode) for mode in ("legacy", "batched", "vectorized")]
                initial = managers[0].state_dict()
                for manager in managers[1:]:
                    manager.load_state_dict(initial)
                generator = torch.Generator().manual_seed(17)
                signals = {name: torch.randn(6, 9, generator=generator, dtype=dtype).to(device)
                           for name in managers[0].statistics_signals}
                envs = torch.tensor([0, 1, 0, 2, -1, 0], device=device)
                mask = (torch.rand(6, 9, generator=generator) > .3).to(device)
                mask[4] = False
                for value in signals.values():
                    value[4] = float("nan")  # Excluded demonstrations cannot poison stats.
                empty = torch.zeros_like(mask)
                singleton = empty.clone()
                singleton[1, 3] = True
                before = rng_state()
                for selected in (empty, singleton, mask, mask, mask):
                    for manager in managers:
                        manager._update_ema_stats(signals, selected, envs)
                    for name in managers[0].statistics_signals:
                        self.assert_exact(managers[0].get_ema_stats(name), managers[1].get_ema_stats(name))
                        for expected, actual in zip(managers[0].get_ema_stats(name), managers[2].get_ema_stats(name)):
                            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
                self.assert_exact(before, rng_state())
                # New modes keep the old checkpoint schema and CPU float64 arrays.
                restored = self.manager(device, "legacy")
                restored.load_state_dict(managers[2].state_dict())
                self.assert_exact(restored.state_dict(), managers[2].state_dict())
                for name in restored.statistics_signals:
                    self.assertEqual(restored.get_ema_stats(name)[0].dtype, np.float64)

    def test_scoring_uses_previous_ema_and_preserves_anchors_away_from_threshold(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            managers = [self.manager(device, mode, 1) for mode in ("legacy", "batched", "vectorized")]
            for manager in managers:
                for pointer in range(100):
                    manager._insert_into_bucket(pointer, 0, 1)
            v = torch.tensor([[0., 2., 4., 32., 64.], [0., 4., 8., 64., 128.]], device=device)
            rewards = torch.zeros_like(v)
            ends = torch.ones_like(v)
            for warmup in (True, False, False):
                counts = [m.add_batch_transitions(v, rewards, ends, .985,
                          np.array([10, 30]), np.array([0, 0]), 100,
                          skip_len=1, is_warmup=warmup) for m in managers]
                self.assertEqual(counts, [counts[0]] * 3)
                self.assertEqual(list(managers[0].active_anchors), list(managers[1].active_anchors))
                self.assertEqual(list(managers[0].active_anchors), list(managers[2].active_anchors))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.manager("cpu", "unknown")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class ProjectedCacheTests(ExactCase):
    def model(self, dtype=torch.float32, amp_dtype=None):
        from sub_models.transformer_model import StochasticTransformerKVCache
        with mock.patch("sub_models.attention_blocks.get_amp_dtype", return_value=amp_dtype or dtype):
            return StochasticTransformerKVCache(32, 18, 64, 2, 8, 32, .1).cuda().to(dtype).eval()

    @torch.no_grad()
    def run_cache(self, model, inputs, actions, projected, amp_dtype=None):
        model.projected_kv_cache = projected
        model.reset_kv_cache_list(inputs.shape[0], amp_dtype or inputs.dtype)
        with torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
            return torch.cat([model.forward_with_kv_cache(inputs[:, t:t+1], actions[:, t:t+1])
                              for t in range(inputs.shape[1])], dim=1)

    def test_teacher_forced_float64_fp32_amp_and_cache_reset(self):
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for dtype, amp_dtype, atol, rtol in (
                    (torch.float64, None, 1e-10, 1e-10),
                    (torch.float32, None, 3e-6, 3e-6),
                    (torch.float32, torch.bfloat16, .05, .03),
                    (torch.float32, torch.float16, .008, .008)):
                torch.manual_seed(3710)
                model = self.model(dtype, amp_dtype)
                state = snapshot(model.state_dict())
                for batch, length in ((3, 24), (1, 1), (5, 8)):
                    inputs = torch.randn(batch, length, 32, device="cuda", dtype=dtype)
                    actions = torch.randint(18, (batch, length), device="cuda")
                    before = rng_state()
                    expected = self.run_cache(model, inputs, actions, False, amp_dtype)
                    actual = self.run_cache(model, inputs, actions, True, amp_dtype)
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    self.assert_exact(before, rng_state())
                    self.assert_exact(state, snapshot(model.state_dict()))
                # Loading a new checkpoint followed by reset must discard every old K/V.
                replacement = self.model(dtype, amp_dtype)
                model.load_state_dict(replacement.state_dict())
                torch.testing.assert_close(self.run_cache(model, inputs, actions, True, amp_dtype),
                                           self.run_cache(model, inputs, actions, False, amp_dtype),
                                           atol=atol, rtol=rtol)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    def test_only_new_tokens_are_projected_and_training_falls_back(self):
        model = self.model()
        inputs = torch.randn(2, 24, 32, device="cuda")
        actions = torch.zeros(2, 24, device="cuda")
        counts = []
        handle = model.layer_stack[0].slf_attn.w_ks.register_forward_pre_hook(
            lambda module, args: counts.append(args[0].shape[1]))
        try:
            self.run_cache(model, inputs, actions, False)
            self.assertEqual(sum(counts), 300)
            counts.clear()
            self.run_cache(model, inputs, actions, True)
            self.assertEqual(counts, [1] * 24)
            model.train()
            model.reset_kv_cache_list(2, torch.float32)
            self.assertFalse(model._projected_cache_active)
            model.forward_with_kv_cache(inputs[:, :1], actions[:, :1]).sum().backward()
            self.assertIsNotNone(model.stem[0].weight.grad)
        finally:
            handle.remove()

    def test_optimizer_updates_cannot_reuse_previous_rollout_kv(self):
        from train import build_world_model, build_agent
        torch.manual_seed(3710)
        conf = legacy_config()
        world, agent = build_world_model(conf, 18), build_agent(conf, 18)
        transformer = world.storm_transformer
        transformer.projected_kv_cache = True
        obs = torch.rand(4, 8, 3, 64, 64, device="cuda")
        actions = torch.randint(18, (4, 8), device="cuda")
        rewards = torch.zeros(4, 8, device="cuda")
        ends = torch.zeros_like(rewards)
        validation = torch.distributions.Distribution._validate_args
        try:
            torch.distributions.Distribution.set_default_validate_args(False)
            world.eval()
            agent.eval()
            with torch.no_grad():
                world.imagine_data(agent, obs, actions, 4, 4, False, None)

            for _ in range(2):
                # Keep references so storage cannot be recycled; poison the entire
                # previous rollout to detect even partial reuse after an update.
                previous_cache = list(transformer._projected_cache)
                with torch.no_grad():
                    for key, value in previous_cache:
                        key.fill_(float("nan"))
                        value.fill_(float("nan"))
                key_weight = transformer.layer_stack[0].slf_attn.w_ks.weight
                weight_before = key_weight.detach().clone()
                world.update(obs, actions, rewards, ends)
                self.assertFalse(torch.equal(weight_before, key_weight))

                world.eval()
                agent.eval()
                state_after_update = snapshot([world.state_dict(), agent.state_dict()])
                with mock.patch.object(transformer, "reset_kv_cache_list",
                                       wraps=transformer.reset_kv_cache_list) as reset:
                    with torch.no_grad():
                        outputs = world.imagine_data(agent, obs, actions, 4, 4, False, None)
                    reset.assert_called_once_with(4, dtype=world.tensor_dtype)
                self.assertTrue(transformer._projected_cache_active)
                self.assertEqual(transformer._cache_position, 12)
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
                for old, new in zip(previous_cache, transformer._projected_cache):
                    for old_tensor, new_tensor in zip(old, new):
                        self.assertIsNot(old_tensor, new_tensor)
                        self.assertNotEqual(old_tensor.data_ptr(), new_tensor.data_ptr())
                        self.assertTrue(torch.isfinite(new_tensor[:, :12]).all())
                # No parameters or mutable buffers (including BatchNorm) change
                # while a rollout is consuming its projected K/V.
                self.assert_exact(state_after_update,
                                  snapshot([world.state_dict(), agent.state_dict()]))
        finally:
            torch.distributions.Distribution.set_default_validate_args(validation)

    def test_imagination_rng_call_sequence_and_optimized_updates(self):
        from train import build_world_model, build_agent
        torch.manual_seed(3710)
        conf = legacy_config()
        world, agent = build_world_model(conf, 18), build_agent(conf, 18)
        obs = torch.rand(16, 8, 3, 64, 64, device="cuda")
        actions = torch.randint(18, (16, 8), device="cuda")
        initial_rng = rng_state()
        validation = torch.distributions.Distribution._validate_args
        try:
            torch.distributions.Distribution.set_default_validate_args(False)
            after = []
            for projected in (False, True):
                world.storm_transformer.projected_kv_cache = projected
                world.eval()
                agent.eval()
                restore_rng(initial_rng)
                with torch.no_grad():
                    outputs = world.imagine_data(agent, obs, actions, 16, 16, False, None)
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
                after.append(rng_state())
            self.assert_exact(after[0], after[1])
            world.batch_scalar_logging = agent.batch_scalar_logging = True
            metrics = {}
            class Logger:
                def log(self, tag, value):
                    metrics[tag] = value
            world.update(obs, actions, torch.zeros_like(actions, dtype=torch.float32),
                         torch.zeros_like(actions, dtype=torch.float32), logger=Logger())
            agent.update(outputs[0], outputs[1], None, None, outputs[2], outputs[3], logger=Logger())
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            for model in (world, agent):
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        finally:
            torch.distributions.Distribution.set_default_validate_args(validation)


if __name__ == "__main__":
    unittest.main()
