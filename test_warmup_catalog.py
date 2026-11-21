"""CPU integration coverage for seed selection and existing warmup reuse."""
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from queue_train import ROOT, PAIR_PREFIX, TrainingQueue, command_for, read_seed_files
from warmup_catalog import discover, inspect_warmup, plan_resume_jobs, resolve_prepared
from training_branches import parse_storm_training_args, prepare_storm_warmup_followup
from utils import load_config


class WarmupCatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="storm-warmup-catalog-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.queue_dir = self.root / "training_queue"
        self.seed_dir = self.root / "config_files/game_seeds"
        self.queue = TrainingQueue(self.queue_dir, self.seed_dir)
        self.overrides = {PAIR_PREFIX + "warmup_offset_pairs": [[50000, -2]],
                          PAIR_PREFIX + "hash_target_pairs": [[9, 4], [10, 8]]}

    def checkpoint(self, game="Alien", seed=3000, *, legacy=False, suffix="", signals=None):
        name = f"{game}-{seed}{suffix}_Shared"
        source = self.root / "ckpt" / name / "shared_warmup_50010"
        source.mkdir(parents=True)
        config = load_config(ROOT / "config_files/STORM.yaml")
        config.defrost()
        config.JointTrainAgent.Retrieval.enable = ["retrieval"]
        config.JointTrainAgent.Retrieval.warmup_offset_pairs = []
        config.JointTrainAgent.Retrieval.hash_target_pairs = []
        config.JointTrainAgent.Retrieval.save_warmup = not legacy
        config.freeze()
        metadata = dict(version=1, run_name=name, seed=seed, env_name=f"ALE/{game}-v5",
                        trajectory_path=str(self.root / f"{game}.pkl"), cwd=str(self.root),
                        step=50010, statistics_signals=signals or ["td_error", "value_diff", "value"])
        fields = {"td_error": ("ema_mean", "ema_var"), "value_diff": ("ema_vd_mean", "ema_vd_var"),
                  "value": ("ema_value_mean", "ema_value_var")}
        state = {key: {} for key in ("world_model_state_dict", "agent_state_dict",
                 "world_model_optimizer_state_dict", "agent_optimizer_state_dict",
                 "world_model_scaler_state_dict", "agent_scaler_state_dict")}
        state.update(total_steps=50010, shared_warmup=True, warmup_finished=True,
                     agent_lowerbound_ema_scalar=0., agent_upperbound_ema_scalar=1.,
                     random_states=dict(python=(), numpy=(), torch=torch.get_rng_state()),
                     retrieval_state={field: np.array([1.]) for name in metadata["statistics_signals"]
                                      for field in fields[name]})
        torch.save(state, source / "training_state.pt")
        np.savez_compressed(source / "replay_buffer.npz", obs=np.zeros((64, 1, 64, 64, 3), dtype=np.uint8),
                            action=np.zeros((64, 1)), reward=np.zeros((64, 1)), termination=np.zeros((64, 1)),
                            length=64, num_envs=1, max_length=100000, last_pointer=63)
        if legacy:
            config_path = self.root / "runs" / name / "config.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(config.dump())
            command = [sys.executable, str(ROOT / "train.py"), "-n", name, "-seed", str(seed),
                       "-env_name", metadata["env_name"], "-trajectory_path", metadata["trajectory_path"],
                       "-config_path", str(config_path)]
            (source / "branches.json").write_text(json.dumps(dict(commands={"retrieval": command})))
        else:
            (source / "config.yaml").write_text(config.dump())
            (source / "warmup_metadata.json").write_text(json.dumps(metadata))
        return source

    def scan(self):
        return discover(["0", "1"], root=self.root, queue_dir=self.queue_dir, seed_dir=self.seed_dir)

    def plan(self, overrides=None, mode=True):
        entries = [entry for entry in read_seed_files(self.seed_dir).values() if entry.get("enabled", True)]
        return plan_resume_jobs(entries, overrides or self.overrides, mode,
                                root=self.root, queue_dir=self.queue_dir)

    def test_scan_preserves_all_sources_and_groups_same_seed_on_one_gpu(self):
        self.checkpoint(legacy=True)
        self.checkpoint(suffix="-second")
        self.checkpoint("Pong", 7010)
        report = self.scan()
        self.assertEqual((len(report["available"]), report["excluded"]), (3, []))
        entries = read_seed_files(self.seed_dir)
        self.assertEqual(set(entries), {"alien:3000", "pong:7010"})
        self.assertEqual(len(entries["alien:3000"]["warmups"]), 2)
        self.assertNotEqual(entries["alien:3000"]["gpu"], entries["pong:7010"]["gpu"])
        self.assertFalse(list(self.queue_dir.glob("gpu_*.jsonl")))

    def test_legacy_import_does_not_modify_original_and_is_usable_by_trainer(self):
        source = self.checkpoint(legacy=True)
        originals = {path.name: path.read_bytes() for path in source.iterdir()}
        self.scan()
        checkpoint, metadata = resolve_prepared(source, root=self.root, queue_dir=self.queue_dir)
        self.assertNotEqual(checkpoint, source)
        self.assertTrue((checkpoint / "training_state.pt").is_symlink())
        self.assertEqual((checkpoint / "training_state.pt").resolve(), source / "training_state.pt")
        self.assertEqual(metadata["statistics_signals"], ["td_error", "value_diff"])
        self.assertEqual(originals, {path.name: path.read_bytes() for path in source.iterdir()})
        specs, skipped = self.plan(mode="Both")
        self.assertEqual(skipped, [])
        added, _ = self.queue.enqueue_jobs(specs, ["0", "1"])
        command = command_for(added[0], self.root)
        self.assertNotIn("-config_path", command)
        self.assertIn("--resume_warmup", command)
        args, extra = parse_storm_training_args(command[3:])
        saved, _, branches = prepare_storm_warmup_followup(args, extra, ROOT / "train.py", load_config)
        self.assertEqual(saved, checkpoint)
        self.assertEqual(len(branches), 3)  # Two retrieval variants and one baseline.
        for branch in branches.values():
            self.assertEqual(branch[branch.index("--resume_from") + 1], str(checkpoint))
        selected = (self.seed_dir / "gpu_0.json").read_bytes()
        self.queue.claim("0")
        self.assertEqual((self.seed_dir / "gpu_0.json").read_bytes(), selected)

    def test_different_requested_warmup_is_skipped_without_retraining(self):
        self.checkpoint()
        self.scan()
        overrides = {**self.overrides, PAIR_PREFIX + "warmup_offset_pairs": [[30000, -2]]}
        specs, skipped = self.plan(overrides)
        self.assertEqual(specs, [])
        self.assertIn("requested warmup 30000", skipped[0])
        self.assertFalse(list(self.queue_dir.glob("gpu_*.jsonl")))

    def test_missing_statistics_excludes_only_incompatible_warmup(self):
        self.checkpoint(legacy=True, signals=["td_error", "value_diff"])
        self.checkpoint("Pong", 7010)
        self.scan()
        specs, skipped = self.plan(mode=["value"])
        self.assertEqual([job["game"] for job in specs], ["Pong"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("lacks the statistics", skipped[0])

    def test_disabled_and_removed_selections_survive_rediscovery(self):
        self.checkpoint()
        self.checkpoint(suffix="-second")
        self.scan()
        path = self.seed_dir / "gpu_0.json"
        selected = json.loads(path.read_text())
        selected["game_seeds"][0]["enabled"] = False
        selected["game_seeds"][0]["warmups"].pop()
        path.write_text(json.dumps(selected))
        self.scan()
        self.assertEqual(json.loads(path.read_text()), selected)
        self.assertEqual(self.plan(), ([], []))

    def test_duplicate_seed_files_and_later_migration_are_rejected(self):
        self.checkpoint()
        self.scan()
        original = json.loads((self.seed_dir / "gpu_0.json").read_text())
        (self.seed_dir / "gpu_1.json").write_text(json.dumps({**original, "gpu": "1"}))
        with self.assertRaisesRegex(ValueError, "Duplicate game/seed"):
            read_seed_files(self.seed_dir)
        (self.seed_dir / "gpu_1.json").write_text(json.dumps(dict(gpu="1", game_seeds=[])))
        specs, _ = self.plan()
        self.queue.enqueue_jobs(specs, ["0", "1"])
        self.queue.claim("0")
        (self.seed_dir / "gpu_0.json").write_text(json.dumps(dict(gpu="0", game_seeds=[])))
        (self.seed_dir / "gpu_1.json").write_text(json.dumps({**original, "gpu": "1"}))
        specs, _ = self.plan()
        with self.assertRaisesRegex(ValueError, "already belongs to GPU 0"):
            self.queue.enqueue_jobs(specs, ["0", "1"])

    def test_incomplete_checkpoint_is_reported_and_not_selected(self):
        source = self.checkpoint()
        (source / "replay_buffer.npz").unlink()
        report = self.scan()
        self.assertEqual(report["available"], [])
        self.assertIn("replay_buffer.npz", report["excluded"][0]["reason"])
        self.assertEqual(read_seed_files(self.seed_dir), {})

    def test_model_shape_override_is_rejected_before_queueing(self):
        self.checkpoint()
        self.scan()
        with self.assertRaisesRegex(ValueError, "Cannot change Models"):
            self.plan({**self.overrides, "Models.WorldModel.TransformerHiddenDim": 256})

    def test_default_cli_and_worker_resume_selected_seed_without_consuming_selection(self):
        self.checkpoint()
        self.checkpoint("Pong", 7010)
        self.scan()
        for name in ("enqueue_train.py", "queue_train.py", "warmup_catalog.py", "queue_train.sh", "0_train.sh"):
            shutil.copy2(ROOT / name, self.root / name)
        shutil.copy2(ROOT / "config_files/STORM.yaml", self.root / "config_files/STORM.yaml")
        bindir = self.root / "bin"
        bindir.mkdir()
        query = bindir / "nvidia-smi"
        query.write_text('#!/bin/bash\nprintf "GPU-test-0\\n"\n')
        query.chmod(0o755)
        environment = {**os.environ, "PYTHONPATH": f"{ROOT.parent}:{ROOT}",
                       "PATH": f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}"}
        args = [sys.executable, str(self.root / "enqueue_train.py"), "--gpus", "0",
                "--hash-target-pairs", "[[9,4],[10,8]]", "--warmup-offset-pairs", "[[50000,-2]]"]
        preview = subprocess.run([*args, "--dry-run"], env=environment, cwd="/tmp", capture_output=True, text=True, timeout=30)
        self.assertEqual(preview.returncode, 0, preview.stdout + preview.stderr)
        self.assertIn("Planned 1 job(s)", preview.stdout)
        self.assertFalse(list(self.queue_dir.glob("gpu_*.jsonl")))
        queued = subprocess.run(args, env=environment, cwd="/tmp", capture_output=True, text=True, timeout=30)
        self.assertEqual(queued.returncode, 0, queued.stdout + queued.stderr)
        self.assertEqual(len(self.queue.jobs("0")), 1)
        self.assertEqual(self.queue.jobs("1"), [])
        (self.root / "train.py").write_text('''import json
from pathlib import Path
from training_branches import parse_storm_training_args, prepare_storm_warmup_followup
from utils import load_config
args, overrides = parse_storm_training_args()
_, _, commands = prepare_storm_warmup_followup(args, overrides, __file__, load_config)
Path("executed_commands.json").write_text(json.dumps(commands))
''')
        before = {path: path.read_bytes() for path in self.seed_dir.glob("*.json")}
        worker = subprocess.run(["bash", str(self.root / "0_train.sh"), "--once"],
                                cwd="/tmp", env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(worker.returncode, 0, worker.stdout + worker.stderr)
        commands = json.loads((self.root / "executed_commands.json").read_text())
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--resume_from" in command for command in commands.values()))
        self.assertEqual(before, {path: path.read_bytes() for path in self.seed_dir.glob("*.json")})
        self.assertEqual(self.queue.jobs("0"), [])

    def test_empty_warmup_selection_does_not_opt_into_fresh_training(self):
        self.checkpoint()
        self.scan()
        path = self.seed_dir / "gpu_0.json"
        selected = json.loads(path.read_text())
        selected["game_seeds"][0]["warmups"] = []
        path.write_text(json.dumps(selected))
        self.assertEqual(self.plan(), ([], []))


if __name__ == "__main__":
    unittest.main()
