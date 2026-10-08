"""CPU regressions for choosing the weights used by final evaluation."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train
import eval as evaluation
from utils import load_config


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.world = torch.nn.Linear(2, 1)
        self.agent = torch.nn.Linear(2, 1)

    def fill_models(self, world_value, agent_value):
        with torch.no_grad():
            for model, value in ((self.world, world_value), (self.agent, agent_value)):
                for parameter in model.parameters():
                    parameter.fill_(value)

    def assert_model_values(self, world_value, agent_value):
        for model, value in ((self.world, world_value), (self.agent, agent_value)):
            for parameter in model.parameters():
                torch.testing.assert_close(parameter, torch.full_like(parameter, value))

    def save_models(self, directory):
        self.fill_models(1, 2)
        torch.save(self.world.state_dict(), Path(directory) / "world_model_100000.pth")
        torch.save(self.agent.state_dict(), Path(directory) / "agent_100000.pth")
        self.fill_models(3, 4)
        train.save_final_models(directory, self.world, self.agent, 102000)

    def test_loads_saved_pair_and_preserves_final_training_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.save_models(directory)
            paths = [Path(directory) / name for name in
                     ("world_model_final.pth", "agent_final.pth", "training_complete.json")]
            final_contents = [path.read_bytes() for path in paths]
            train.load_final_eval_models(directory, self.world, self.agent, 100000)
            self.assert_model_values(1, 2)
            self.assertEqual([path.read_bytes() for path in paths], final_contents)

    def test_missing_checkpoint_fails_before_replacing_either_model(self):
        for missing in ("world_model_100000.pth", "agent_100000.pth"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                self.save_models(directory)
                (Path(directory) / missing).unlink()
                with self.assertRaisesRegex(FileNotFoundError, missing):
                    train.load_final_eval_models(directory, self.world, self.agent, 100000)
                self.assert_model_values(3, 4)

    def test_none_uses_current_models_without_reading_checkpoints(self):
        self.fill_models(3, 4)
        with mock.patch.object(torch, "load") as load:
            train.load_final_eval_models("unused", self.world, self.agent, None)
        load.assert_not_called()
        self.assert_model_values(3, 4)

    def test_config_and_legacy_defaults_allow_cli_overrides(self):
        config = load_config(str(ROOT / "config_files/STORM.yaml"))
        self.assertEqual(config.JointTrainAgent.FinalEvalStep, 100000)
        self.assertEqual(config.JointTrainAgent.SampleMaxSteps, 102000)
        config.defrost()
        config.merge_from_list(["JointTrainAgent.FinalEvalStep", "50000"])
        self.assertEqual(config.JointTrainAgent.FinalEvalStep, 50000)
        config.merge_from_list(["JointTrainAgent.FinalEvalStep", "None"])
        self.assertIsNone(config.JointTrainAgent.FinalEvalStep)
        del config.JointTrainAgent["FinalEvalStep"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(config.dump())
            self.assertIsNone(load_config(str(path)).JointTrainAgent.FinalEvalStep)

    def test_final_evaluation_entrypoint_uses_selected_weights(self):
        # Run the real final-evaluation block without starting Atari training.
        tree = ast.parse((ROOT / "train.py").read_text())
        block = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Attribute)
                     and node.test.left.attr == "EvalMode")
        code = compile(ast.Module(body=[block], type_ignores=[]), str(ROOT / "train.py"), "exec")
        config = load_config(str(ROOT / "config_files/STORM.yaml"))
        config.defrost()
        for mode, step in (("final_only", 100000), ("active", 100000),
                           ("final_only", None), ("inactive", 100000)):
            with self.subTest(mode=mode, step=step), tempfile.TemporaryDirectory() as directory:
                self.save_models(directory)
                config.JointTrainAgent.EvalMode = mode
                config.JointTrainAgent.FinalEvalStep = step
                expected = (1, 2) if step is not None else (3, 4)

                def evaluate(**kwargs):
                    self.assertIs(kwargs["world_model"], self.world)
                    self.assertIs(kwargs["agent"], self.agent)
                    self.assert_model_values(*expected)
                    return 10., [10.] * 20

                evaluator = mock.Mock(side_effect=evaluate)
                loader = mock.Mock(side_effect=lambda ckpt_dir, world, agent, checkpoint_step:
                                   train.load_final_eval_models(directory, world, agent, checkpoint_step))
                logger = mock.Mock()
                namespace = {"conf": config, "args": SimpleNamespace(n="unit_run", env_name="ALE/Krull-v5"),
                             "world_model": self.world, "agent": self.agent,
                             "load_final_eval_models": loader, "logger": logger,
                             "colorama": train.colorama, "np": np, "json": json}
                with mock.patch.object(evaluation, "eval_episodes", evaluator), \
                     mock.patch("builtins.print"):
                    exec(code, namespace)
                if mode == "inactive":
                    loader.assert_not_called()
                    evaluator.assert_not_called()
                else:
                    loader.assert_called_once_with("ckpt/unit_run", self.world, self.agent, step)
                    evaluator.assert_called_once()
                    logger.log.assert_any_call("eval/checkpoint_step", 100000 if step is not None else 102000)


if __name__ == "__main__":
    unittest.main()
