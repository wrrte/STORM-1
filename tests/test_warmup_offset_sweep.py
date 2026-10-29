"""CPU tests of real prefix checkpoints and paired post-warmup commands."""
import ast
import contextlib
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from collections import deque

import numpy as np
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from replay_buffer import ReplayBuffer
from retrieval import RetrievalContextManager
from storm_warmup_sweep import parse_warmup_offset_pairs, read_sweep_index
from training_branches import (capture_rng_state, restore_rng_state, configure_storm_retrieval_run,
    parse_storm_training_args, launch_storm_warmup_followup)
from utils import load_config


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def extract_function(name, namespace):
    tree = ast.parse((ROOT / "train.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    for argument in function.args.args:
        argument.annotation = None
    function.returns = None
    namespace = {"torch": torch, "np": np, **namespace}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "train.py"), "exec"), namespace)
    return namespace[name]


def make_args(**overrides):
    return SimpleNamespace(**{
        "n": "Pong-17", "seed": 17, "config_path": str(ROOT / "config_files/STORM.yaml"),
        "env_name": "ALE/Pong-v5", "trajectory_path": str(ROOT / "Pong.pkl"),
        "resume_from": None, "branch_mode": None, "branch_experiment": None, **overrides,
    })


def make_config(pairs, mode=True):
    conf = load_config(ROOT / "config_files/STORM.yaml")
    conf.defrost()
    conf.JointTrainAgent.SampleMaxSteps = 12
    conf.JointTrainAgent.FinalEvalStep = None
    conf.JointTrainAgent.EvalMode = "inactive"
    conf.JointTrainAgent.Retrieval.enable = mode
    conf.JointTrainAgent.Retrieval.hash_bits = 2
    conf.JointTrainAgent.Retrieval.use_pca = False
    conf.JointTrainAgent.Retrieval.warmup_offset_pairs = pairs
    return conf


def consume_randomness():
    return random.random() + float(np.random.rand()) + torch.rand(()).item()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=.01)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.lowerbound_ema = SimpleNamespace(scalar=-1.)
        self.upperbound_ema = SimpleNamespace(scalar=1.)

    def learn(self):
        self.optimizer.zero_grad()
        (self.weight - consume_randomness()).square().backward()
        self.optimizer.step()

    def encode_obs(self, obs, **kwargs):
        return torch.zeros(obs.shape[0], obs.shape[1], 1024)

    def calc_last_dist_feat(self, latent, action):
        return torch.zeros(1, 1, 1024), torch.zeros(1, 1, 4)

    def sample_as_env_action(self, *args, **kwargs):
        return np.array([1])

    def update(self, **kwargs):
        self.learn()


class SupervisorStarted(Exception):
    pass


class WarmupOffsetSweepTests(unittest.TestCase):
    def scenario(self, pairs, *, mode=True, noise=False, ends=(5, 9), max_steps=12,
                 rebuild=False, resume_step=0, resumed_enabled=True):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with working_directory(directory.name):
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            conf, args = make_config(pairs, mode), make_args()
            sweep = configure_storm_retrieval_run(conf, args, [], ROOT / "train.py") if pairs else None
            conf.defrost()
            conf.JointTrainAgent.Retrieval.enable = conf.JointTrainAgent.Retrieval.enable if pairs else resumed_enabled
            conf.JointTrainAgent.Retrieval.warmup_steps = max([p[0] for p in pairs], default=0)
            conf.freeze()
            world, agent = TinyModel(), TinyModel()
            logger = SimpleNamespace(tag_step={}, log=mock.Mock(), flush_wandb=mock.Mock(), writer=mock.Mock())
            env = SimpleNamespace(action_space=mock.Mock(), reset=mock.Mock(return_value=(
                np.zeros((1, 2, 2, 1), np.uint8), {"episode_frame_number": np.array([0])})), close=mock.Mock())
            env.action_space.sample.return_value = np.array([0])
            events = []
            def step(action):
                count = env.step.call_count
                return (np.full((1, 2, 2, 1), count, np.uint8), np.array([1.]),
                    np.array([count in ends]), np.array([False]),
                    {"life_loss": np.array([False]), "episode_frame_number": np.array([count * 4])})
            env.step = mock.Mock(side_effect=step)
            managers = []
            def manager_factory(**kwargs):
                manager = RetrievalContextManager(**kwargs, device="cpu")
                manager.rebuild_all_hash_buckets = mock.Mock(side_effect=lambda *a, **kw:
                    (events.append("rebuild"), consume_randomness()))
                managers.append(manager)
                return manager
            colorama = SimpleNamespace(Fore=SimpleNamespace(YELLOW="", GREEN="", CYAN=""),
                                       Style=SimpleNamespace(RESET_ALL=""))
            save = extract_function("save_full_checkpoint", {
                "os": os, "copy": copy, "args": args, "capture_rng_state": capture_rng_state,
                "colorama": colorama,
            })
            def save_with_optional_noise(*a, **kw):
                if noise:
                    consume_randomness()
                save(*a, **kw)
            def world_step(**kwargs):
                self.assertTrue(kwargs["is_warmup"])
                world.learn()
                values = torch.arange(6, dtype=torch.float32).unsqueeze(0) * env.step.call_count
                kwargs["retrieval_manager"].add_batch_transitions(
                    values, torch.ones_like(values) * env.step.call_count, torch.zeros_like(values),
                    .985, np.array([0]), np.array([0]), 32, skip_len=1, is_warmup=True)
                return 0
            imagine = mock.Mock(return_value=(torch.zeros(1), torch.zeros(1), None, None,
                                              torch.zeros(1), torch.zeros(1), 0., torch.ones(1)))
            train = extract_function("joint_train_world_model_agent", {
                "os": os, "args": args, "conf": conf, "build_vec_env": lambda *a, **kw: env,
                "colorama": colorama, "deque": deque, "tqdm": lambda items: items,
                "RetrievalContextManager": manager_factory, "capture_rng_state": capture_rng_state,
                "restore_rng_state": restore_rng_state, "save_full_checkpoint": save_with_optional_noise,
                "save_final_models": mock.Mock(), "train_world_model_step": world_step,
                "world_model_imagine_data": imagine, "rearrange": rearrange,
            })
            replay = ReplayBuffer((2, 2, 1), 1, max_length=32, warmup_length=1, store_on_gpu=False)
            if resume_step:
                for count in range(resume_step):
                    replay.append(np.zeros((1, 2, 2, 1), np.uint8), np.array([0]), np.array([0.]), np.array([False]))
            with mock.patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self), \
                    mock.patch("torch.save", wraps=torch.save), \
                    mock.patch("training_branches.os.execv", side_effect=SupervisorStarted) as execute, \
                    mock.patch.dict(sys.modules, {"wandb": SimpleNamespace(finish=lambda: None)}), \
                    contextlib.redirect_stdout(__import__("io").StringIO()):
                try:
                    train(env_name="test", max_steps=max_steps, num_envs=1, image_size=(2, 2),
                        replay_buffer=replay, world_model=world, agent=agent,
                        train_dynamics_every_steps=1, train_agent_every_steps=1,
                        batch_size=1, demonstration_batch_size=0, batch_length=3,
                        imagine_batch_size=1, imagine_demonstration_batch_size=0,
                        imagine_context_length=3, imagine_batch_length=2,
                        save_every_steps=100, seed=17, logger=logger, branch_commands=sweep,
                        resume_step=resume_step, warmup_finished_resume=bool(resume_step),
                        rebuild_retrieval_on_resume=rebuild)
                except SupervisorStarted:
                    pass
            snapshots = {}
            index = read_sweep_index(sweep.root) if sweep else None
            for record in (index["checkpoints"].values() if index else []):
                path = sweep.root / record["checkpoint"]
                snapshots[record["actual_step"]] = torch.load(path / "training_state.pt", weights_only=False)
            return SimpleNamespace(directory=Path(directory.name), sweep=sweep, conf=conf, args=args,
                index=index, snapshots=snapshots, world=world, agent=agent, execute=execute,
                managers=managers, events=events, env=env, imagine=imagine, rng=capture_rng_state())

    def assert_rng_equal(self, left, right):
        self.assertEqual(left["python"], right["python"])
        self.assertEqual(left["numpy"][0], right["numpy"][0])
        np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
        self.assertEqual(left["numpy"][2:], right["numpy"][2:])
        torch.testing.assert_close(left["torch"], right["torch"], rtol=0, atol=0)

    def test_pairs_share_one_prefix_and_same_boundary_uses_one_snapshot(self):
        result = self.scenario([[3, -4], [4, -2], [8, 0], [8, -2]])
        self.assertEqual(result.env.step.call_count, 9)
        self.assertTrue(result.index["complete"])
        self.assertEqual(set(result.snapshots), {5, 9})
        self.assertEqual(result.index["checkpoints"]["3"], result.index["checkpoints"]["4"])
        self.assertEqual(result.snapshots[5]["total_steps"], 5)
        self.assertEqual(result.snapshots[9]["total_steps"], 9)
        self.assertTrue(result.snapshots[5]["warmup_finished"])
        self.assertFalse(torch.equal(result.snapshots[5]["world_model_state_dict"]["weight"],
                                     result.snapshots[9]["world_model_state_dict"]["weight"]))
        for step, updates in ((5, 4), (9, 8)):
            optimizer_state = result.snapshots[step]["world_model_optimizer_state_dict"]["state"]
            self.assertEqual(next(iter(optimizer_state.values()))["step"].item(), updates)
        self.assertNotEqual(result.snapshots[5]["retrieval_state"]["ema_vd_mean"].tolist(),
                            result.snapshots[9]["retrieval_state"]["ema_vd_mean"].tolist())
        for step in (5, 9):
            path = result.sweep.root / f"shared_warmup_{step}"
            with np.load(path / "replay_buffer.npz") as data:
                self.assertEqual(int(data["length"]), step)
                self.assertEqual(data["obs"].shape[0], step)
            metadata = json.loads((path / "warmup_metadata.json").read_text())
            self.assertTrue(metadata["rebuild_on_resume"])
        result.managers[0].rebuild_all_hash_buckets.assert_not_called()
        self.assertTrue(all(call.kwargs["is_warmup"] for call in result.imagine.call_args_list))

    def test_saving_earlier_prefixes_does_not_change_training_or_rng(self):
        many = self.scenario([[3, -4], [8, -2]], noise=True)
        one = self.scenario([[8, -2]])
        torch.testing.assert_close(many.world.weight, one.world.weight, rtol=0, atol=0)
        torch.testing.assert_close(many.agent.weight, one.agent.weight, rtol=0, atol=0)
        self.assert_rng_equal(many.rng, one.rng)
        self.assert_rng_equal(many.snapshots[9]["random_states"], one.snapshots[9]["random_states"])

    def test_commands_restore_the_right_prefix_and_apply_each_offset(self):
        result = self.scenario([[3, -4], [8, -2], [8, 0]], mode="Both")
        commands = result.sweep.commands()
        self.assertEqual(len(commands), 5)  # Three retrievals, two baselines.
        for name, command in commands.items():
            args, overrides = parse_storm_training_args(command[2:])
            conf = load_config(args.config_path)
            configure_storm_retrieval_run(conf, args, overrides, ROOT / "train.py")
            self.assertEqual(conf.JointTrainAgent.Retrieval.warmup_offset_pairs, [])
            self.assertFalse(conf.JointTrainAgent.Retrieval.save_warmup)
            expected_step = 5 if name.startswith("w3_") else 9
            self.assertTrue(args.resume_from.endswith(f"shared_warmup_{expected_step}"))
            expected_offset = -4 if name.startswith("w3_") else (0 if "offset_0" in name else -2)
            self.assertEqual(conf.JointTrainAgent.Retrieval.anchor_offset, expected_offset)
            self.assertEqual(conf.JointTrainAgent.Retrieval.warmup_steps, 3 if expected_step == 5 else 8)

    def test_zero_warmup_and_missing_episode_boundary(self):
        zero = self.scenario([[0, -2]], ends=())
        self.assertEqual(zero.env.step.call_count, 0)
        self.assertEqual(set(zero.snapshots), {0})
        incomplete = self.scenario([[3, -2]], ends=(), max_steps=6)
        self.assertFalse(incomplete.index["complete"])
        self.assertEqual(incomplete.snapshots, {})
        incomplete.execute.assert_not_called()

    def test_rebuild_runs_after_rng_restore_even_in_baseline(self):
        enabled = self.scenario([], rebuild=True, resume_step=3, max_steps=3)
        baseline = self.scenario([], rebuild=True, resume_step=3, max_steps=3, resumed_enabled=False)
        enabled.managers[0].rebuild_all_hash_buckets.assert_called_once()
        baseline.managers[0].rebuild_all_hash_buckets.assert_called_once()
        self.assertFalse(baseline.managers[0].enabled)
        self.assert_rng_equal(enabled.rng, baseline.rng)

    def test_followups_select_saved_targets_without_overwriting_collection(self):
        result = self.scenario([[3, -4], [8, -2]])
        original = (result.sweep.root / "branches.json").read_bytes()
        args, overrides = parse_storm_training_args([
            "--resume_warmup", str(result.sweep.root),
            "JointTrainAgent.Retrieval.enable", "True",
            "JointTrainAgent.Retrieval.warmup_offset_pairs", "[[3, 1], [8, -1]]",
        ])
        with mock.patch("training_branches.os.execv", side_effect=SupervisorStarted) as execute:
            with self.assertRaises(SupervisorStarted):
                launch_storm_warmup_followup(args, overrides, ROOT / "train.py", load_config)
        manifest = json.loads(Path(execute.call_args.args[1][-1]).read_text())
        self.assertEqual(len(manifest["commands"]), 2)
        self.assertEqual((result.sweep.root / "branches.json").read_bytes(), original)
        args, overrides = parse_storm_training_args([
            "--resume_warmup", str(result.sweep.root),
            "JointTrainAgent.Retrieval.warmup_offset_pairs", "[[4, -2]]",
        ])
        with self.assertRaisesRegex(ValueError, "No saved snapshot"):
            launch_storm_warmup_followup(args, overrides, ROOT / "train.py", load_config)

    def test_invalid_pairs_and_evaluation_budget_fail_before_training(self):
        for value in (None, {}, "invalid", [[-1, -2]], [[12, -2]], [[1.5, -2]],
                      [[3, -2.5]], [[3, True]], [[True, -2]], [[3]], [[3, -2], [3, -2]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_warmup_offset_pairs(value, 12)
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            conf = make_config([[8, -2]])
            conf.JointTrainAgent.EvalMode = "final_only"
            conf.JointTrainAgent.FinalEvalStep = 8
            with self.assertRaisesRegex(ValueError, "FinalEvalStep"):
                configure_storm_retrieval_run(conf, make_args(), [], ROOT / "train.py")

    def test_collection_registry_prevents_stale_or_missing_prefix_resume(self):
        result = self.scenario([[3, -4], [8, -2]])
        with working_directory(result.directory):
            with self.assertRaises(FileExistsError):
                configure_storm_retrieval_run(make_config([[3, -4], [8, -2]]),
                                             make_args(), [], ROOT / "train.py")
            restored = configure_storm_retrieval_run(make_config([[3, -4], [8, -2]]),
                make_args(resume_from=str(result.sweep.root / "shared_warmup_9")), [], ROOT / "train.py")
            restored.initialize(9)
            with self.assertRaisesRegex(ValueError, "precedes"):
                restored.initialize(5)
            with self.assertRaisesRegex(ValueError, "same name"):
                configure_storm_retrieval_run(make_config([[3, -3], [8, -2]]),
                    make_args(resume_from="checkpoint"), [], ROOT / "train.py")
        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            sweep = configure_storm_retrieval_run(make_config([[3, -2], [8, -2]]),
                make_args(resume_from="checkpoint"), [], ROOT / "train.py")
            with self.assertRaisesRegex(ValueError, "earlier warmup"):
                sweep.initialize(6)


if __name__ == "__main__":
    unittest.main()
