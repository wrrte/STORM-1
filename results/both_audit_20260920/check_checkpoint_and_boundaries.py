"""Read-only CPU diagnostics for STORM's Both path; no training fixes are applied.

Run with the storm Python environment. CUDA allocation sites are redirected to
CPU and AMP is disabled; this checks serialization, not CUDA numerical behavior.
The real STORM models, optimizers, checkpoint helpers and replay sampler are used.
"""

import copy
import json
import random
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

STORM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STORM))
sys.path.insert(1, str(STORM.parent))
import train
from retrieval import RetrievalContextManager
from training_branches import capture_rng_state, restore_rng_state


@contextmanager
def cpu_allocations():
    zeros = torch.zeros

    def cpu_zeros(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return zeros(*args, **kwargs)

    with mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self), \
         mock.patch.object(torch, "zeros", cpu_zeros):
        yield


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b, (a, b)


def models():
    world = train.WorldModel(3, 4, 16, 32, 1, 2)
    agent = train.agents.ActorCriticAgent(1024 + 32, 1, 32, 4, .985, .95, 3e-4)
    for model in (world, agent):
        model.use_amp = False
        model.tensor_dtype = torch.float32
    return world, agent


def model_state(world, agent):
    return copy.deepcopy({
        "world": world.state_dict(), "agent": agent.state_dict(),
        "world_optimizer": world.optimizer.state_dict(),
        "agent_optimizer": agent.optimizer.state_dict(),
        "world_scaler": world.scaler.state_dict(),
        "agent_scaler": agent.scaler.state_dict(),
        "lower_ema": agent.lowerbound_ema.scalar,
        "upper_ema": agent.upperbound_ema.scalar,
    })


def buffer():
    return train.ReplayBuffer((2, 2, 1), 1, max_length=16, warmup_length=2)


def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    random.seed(710)
    np.random.seed(710)
    torch.manual_seed(710)
    results = {"device": "cpu", "amp": False, "torch": torch.__version__}
    world, agent = models()
    replay = buffer()
    for i in range(12):
        replay.append(np.full((1, 2, 2, 1), i, np.uint8), np.array([i % 4]),
                      np.array([i / 10]), np.array([i == 6]))
    manager = RetrievalContextManager(1, {"enable": "Both", "hash_bits": 2}, 4, device="cpu")
    manager.ema_mean[0] = 1.23
    manager._insert_into_bucket(5, 0, 1)
    obs = torch.rand(2, 4, 3, 64, 64)
    action = torch.randint(0, 4, (2, 4)).float()
    reward = torch.rand(2, 4)
    termination = torch.zeros(2, 4)
    latent = torch.randn(4, 9, 1056)
    actor_action = torch.randint(0, 4, (4, 8))
    actor_reward = torch.rand(4, 8)
    actor_done = torch.zeros(4, 8)

    def update(w, a):
        w.update(obs, action, reward, termination)
        a.update(latent, actor_action, None, None, actor_reward, actor_done)

    with cpu_allocations(), tempfile.TemporaryDirectory(prefix="storm-both-audit-") as directory:
        update(world, agent)
        update(world, agent)
        before = model_state(world, agent)
        train.args = SimpleNamespace(n="cpu_audit_Both")
        logger = SimpleNamespace(tag_step={"WorldModel/total_loss": 1})
        train.save_full_checkpoint(directory, world, agent, replay, 12, logger,
                                   11, True, [1., 2.], retrieval_manager=manager,
                                   shared_warmup=True)
        checkpoint_rng = capture_rng_state()
        update(world, agent)
        expected_next = model_state(world, agent)
        expected_rng_after = capture_rng_state()
        for mode in (True, False):
            restored_world, restored_agent = models()
            restored_replay = buffer()
            loaded = train.load_full_checkpoint(directory, restored_world, restored_agent, restored_replay)
            assert_equal(before, model_state(restored_world, restored_agent))
            assert loaded[0] == 12
            assert_equal(checkpoint_rng, capture_rng_state())
            for name in ("obs_buffer", "action_buffer", "reward_buffer", "termination_buffer"):
                assert_equal(getattr(replay, name)[:12], getattr(restored_replay, name)[:12])
            assert replay.last_pointer == restored_replay.last_pointer
            restored_manager = RetrievalContextManager(1, {"enable": mode, "hash_bits": 2}, 4, device="cpu")
            restored_manager.load_state_dict(loaded[6])
            assert restored_manager.enabled is mode
            assert_equal(manager.ema_mean, restored_manager.ema_mean)
            assert_equal(manager.hash_proj, restored_manager.hash_proj)
            assert_equal(manager.index_to_bucket, restored_manager.index_to_bucket)
            restore_rng_state(loaded[7])
            update(restored_world, restored_agent)
            assert_equal(expected_next, model_state(restored_world, restored_agent))
            assert_equal(expected_rng_after, capture_rng_state())
        results["checkpoint_roundtrip"] = "exact: real weights, BN buffers, slow critic, Adam, EMA, replay, CPU RNG"
        results["next_update_with_identical_inputs"] = "bitwise identical for original and both restored branches"
        results["receiver_enable_flag"] = "True and False preserved despite shared Both retrieval state"

        boundary = buffer()
        for i in (10, 11, 12, 13, 0, 1, 2, 3):
            boundary.append(np.full((1, 2, 2, 1), i, np.uint8), np.array([0]),
                            np.array([0.]), np.array([False]))
        boundary.termination_buffer[3] = 1
        with mock.patch.object(np.random, "randint", return_value=np.array([2])):
            sampled_obs, _, _, sampled_term, _, _ = boundary.sample(1, 0, 4)
        values = sampled_obs[0, :, 0, 0, 0].mul(255).round().int().tolist()
        assert values == [12, 13, 0, 1]
        results["sample_crosses_forced_terminal"] = {"observations": values, "terminals": sampled_term[0].tolist()}

        # Use the actual global rebuild to check its effect on training RNG.
        pca = RetrievalContextManager(1, {"enable": "Both", "hash_bits": 2, "use_pca": True}, 4, device="cpu")
        fake_encoder = SimpleNamespace(encode_obs=lambda x, sample_mode: x.flatten(2))
        before_pca = torch.random.get_rng_state().clone()
        pca.rebuild_all_hash_buckets(boundary, fake_encoder, chunk_size=4)
        results["global_rebuild_advances_torch_rng"] = not torch.equal(before_pca, torch.random.get_rng_state())
        assert results["global_rebuild_advances_torch_rng"]

        # train.py logs its open-loop video after world.update() has called
        # world.train(), and before world_model_imagine_data() calls eval().
        world.train()
        batch_norms = {name: module for name, module in world.named_modules()
                       if isinstance(module, torch.nn.BatchNorm2d)}
        counts = {name: int(module.num_batches_tracked) for name, module in batch_norms.items()}
        video_obs = torch.rand(2, 7, 3, 64, 64)
        video_actions = torch.randint(0, 4, (2, 7))
        video_rng = torch.random.get_rng_state().clone()
        world.log_openloop_video(video_obs, video_actions, 3, 4, SimpleNamespace(log=lambda *args: None))
        changes = {name: int(module.num_batches_tracked) - counts[name] for name, module in batch_norms.items()}
        assert all(delta > 0 for delta in changes.values())
        results["video_logging_changes_batchnorm_counts"] = changes
        results["video_logging_advances_torch_rng"] = not torch.equal(video_rng, torch.random.get_rng_state())

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
