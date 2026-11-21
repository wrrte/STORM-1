"""CPU-only integration tests for overrides, FIFO consumption and GPU affinity."""
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from enqueue_train import validate_overrides
from queue_train import PAIR_PREFIX, ROOT, TrainingQueue, command_for


OVERRIDES = {PAIR_PREFIX + "warmup_offset_pairs": [],
             PAIR_PREFIX + "hash_target_pairs": [[9, 4], [10, 8]]}


def concurrent_add(directory, gpu, barrier):
    barrier.wait()
    TrainingQueue(directory).enqueue([["Alien", 3000]], [gpu], ROOT / "config_files/STORM.yaml", OVERRIDES)


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="storm-paired-queue-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.queue = TrainingQueue(self.root / "training_queue")
        for name in ["queue_train.py", "enqueue_train.py", "queue_train.sh",
                     *(f"{gpu}_train.sh" for gpu in range(8))]:
            shutil.copy2(ROOT / name, self.root / name)
        (self.root / "config_files").mkdir()
        self.config = self.root / "config_files/STORM.yaml"
        shutil.copy2(ROOT / "config_files/STORM.yaml", self.config)
        bindir = self.root / "bin"
        bindir.mkdir()
        query = bindir / "nvidia-smi"
        query.write_text('#!/bin/bash\nprintf "GPU-%s-%s\\n" "$4" "${QUEUE_TEST_DEVICE:-original}"\n')
        query.chmod(0o755)
        self.environment = {**os.environ,
                            "PATH": f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                            "PYTHONPATH": f"{ROOT.parent}:{ROOT}"}
        (self.root / "train.py").write_text('''import json, os, signal, sys, time
with open("executed.jsonl", "a") as handle:
    handle.write(json.dumps(dict(argv=sys.argv[1:], gpu=os.environ["CUDA_VISIBLE_DEVICES"])) + "\\n")
if os.environ.get("QUEUE_TEST_BLOCK"):
    def stop(*_):
        open("child_stopped", "w").close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    open("child_ready", "w").close()
    while True:
        time.sleep(0.1)
raise SystemExit(int(os.environ.get("QUEUE_TEST_EXIT", 0)))
''')

    def add(self, pairs=None, gpus=None, overrides=None, **kwargs):
        return self.queue.enqueue(pairs or [["Alien", 3000]], gpus or ["0", "1"],
                                  self.config, overrides or OVERRIDES, **kwargs)

    def worker(self, gpu, *options):
        return subprocess.run(["bash", str(self.root / f"{gpu}_train.sh"), *options],
                              cwd="/tmp", env=self.environment, text=True,
                              capture_output=True, timeout=20)

    def start_worker(self, gpu, *options):
        process = subprocess.Popen(["bash", str(self.root / f"{gpu}_train.sh"), *options],
                                   cwd="/tmp", env=self.environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=40)
        self.addCleanup(cleanup)
        return process

    def test_assignment_persists_after_consumption_and_new_gpu_list(self):
        added, _ = self.add([["alien", 3000], ["ALE/Pong-v5", 3000]])
        self.assertEqual([job["gpu"] for job in added], ["0", "1"])
        self.assertEqual(self.queue.claim("0")["game"], "Alien")
        self.assertEqual(self.queue.path("0").read_text(), "")
        followup, _ = self.add([["ALIEN", 3000]], ["1"])
        self.assertEqual(followup[0]["gpu"], "0")
        self.assertEqual(len(self.queue.jobs("1")), 1)

    def test_pending_duplicates_skipped_and_additional_settings_append(self):
        first, _ = self.add()
        duplicate, skipped = self.add()
        self.assertEqual((duplicate, skipped), ([], 1))
        changed = {**OVERRIDES, PAIR_PREFIX + "warmup_offset_pairs": [[30000, -2], [50000, 0]]}
        second, _ = self.add(overrides=changed)
        self.assertNotEqual(first[0]["name"], second[0]["name"])
        self.assertEqual([self.queue.claim("0")["id"] for _ in range(2)],
                         [first[0]["id"], second[0]["id"]])

    def test_concurrent_producers_assign_same_pair_to_only_one_gpu(self):
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        processes = [context.Process(target=concurrent_add,
                                     args=(self.queue.directory, gpu, barrier)) for gpu in ("0", "1")]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join()
            self.assertEqual(process.exitcode, 0)
        jobs = self.queue.jobs("0") + self.queue.jobs("1")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(self.queue.registry()["owners"], {"alien:3000": jobs[0]["gpu"]})

    def test_moved_or_malformed_head_is_not_consumed(self):
        self.add(gpus=["0"])
        original = self.queue.path("0").read_text()
        self.queue.path("1").write_text(original)
        with self.assertRaisesRegex(ValueError, "Ownership mismatch"):
            self.queue.claim("1")
        self.assertEqual(self.queue.path("1").read_text(), original)
        self.queue.path("0").write_text("invalid json\n" + original)
        with self.assertRaises(ValueError):
            self.queue.claim("0")
        self.assertEqual(self.queue.path("0").read_text(), "invalid json\n" + original)

    def test_missing_registry_cannot_reassign_existing_jobs(self):
        self.add()
        (self.queue.directory / "assignments.json").unlink()
        with self.assertRaisesRegex(ValueError, "assignments.json is missing"):
            self.add(gpus=["1"])

    def test_default_wrappers_consume_own_heads_once_with_correct_overrides(self):
        self.add([["Alien", seed] for seed in range(3000, 3008)], [str(gpu) for gpu in range(8)])
        processes = [self.start_worker(gpu, "--once") for gpu in range(8)]
        for process in processes:
            out, err = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, out + err)
        executions = [json.loads(line) for line in (self.root / "executed.jsonl").read_text().splitlines()]
        self.assertEqual(len(executions), 8)
        for execution in executions:
            argv = execution["argv"]
            gpu = str(int(argv[argv.index("-seed") + 1]) - 3000)
            self.assertEqual(execution["gpu"], f"GPU-{gpu}-original")
            self.assertEqual(argv[argv.index(PAIR_PREFIX + "warmup_offset_pairs") + 1], "[]")
            self.assertEqual(argv[argv.index(PAIR_PREFIX + "hash_target_pairs") + 1], "[[9, 4], [10, 8]]")
            self.assertEqual(self.queue.path(gpu).read_text(), "")

    def test_failures_are_recorded_and_next_job_runs(self):
        self.add([["Alien", 3000], ["Pong", 3000]], ["0"])
        self.environment["QUEUE_TEST_EXIT"] = "7"
        result = self.worker(0, "--max-jobs", "2")
        self.assertEqual(result.returncode, 7, result.stderr)
        events = [json.loads(line) for line in (self.queue.directory / "history.jsonl").read_text().splitlines()]
        self.assertEqual([event["returncode"] for event in events if event["event"] == "finished"], [7, 7])
        self.assertEqual(self.worker(0, "--once").returncode, 0)

    def test_physical_gpu_change_is_rejected_before_consuming(self):
        self.add()
        self.assertEqual(self.worker(0, "--once").returncode, 0)
        self.add()
        self.environment["QUEUE_TEST_DEVICE"] = "replacement"
        result = self.worker(0, "--once")
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing to move", result.stderr)
        self.assertEqual(len(self.queue.jobs("0")), 1)

    def test_duplicate_worker_and_sigterm_leave_next_job_queued(self):
        self.add([["Alien", 3000], ["Pong", 3000]], ["0"])
        self.environment["QUEUE_TEST_BLOCK"] = "1"
        process = self.start_worker(0)
        deadline = time.monotonic() + 10
        while not (self.root / "child_ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((self.root / "child_ready").exists())
        self.assertEqual(len(self.queue.jobs("0")), 1)
        duplicate = self.worker(0, "--once")
        self.assertEqual(duplicate.returncode, 1)
        self.assertIn("already running", duplicate.stderr)
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        self.assertEqual(process.returncode, 143)
        self.assertTrue((self.root / "child_stopped").exists())
        self.assertEqual(len(self.queue.jobs("0")), 1)
        events = [json.loads(line) for line in (self.queue.directory / "history.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "interrupted")

    def test_enqueue_cli_overrides_yaml_and_preserves_source(self):
        before = self.config.read_bytes()
        result = subprocess.run([
            sys.executable, str(self.root / "enqueue_train.py"), "--gpus", "0", "1",
            "--game-seeds", '[["alien",3000],["Pong",3010]]',
            "--hash-target-pairs", "[[9,4],[10,8]]",
            "--set", PAIR_PREFIX + "enable", "Both",
            "--set", PAIR_PREFIX + "anchor_weight_r", "2"],
            env=self.environment, cwd="/tmp", text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse((self.root / "executed.jsonl").exists())
        job = self.queue.jobs("0")[0]
        self.assertEqual(job["overrides"][PAIR_PREFIX + "warmup_offset_pairs"], [])
        self.assertEqual(job["overrides"][PAIR_PREFIX + "anchor_weight_r"], 2)
        from training_branches import configure_storm_retrieval_run, parse_storm_training_args
        from utils import load_config
        args, overrides = parse_storm_training_args(command_for(job)[3:])
        conf = load_config(self.config)
        sweep = configure_storm_retrieval_run(conf, args, overrides, ROOT / "train.py")
        self.assertEqual(sweep.pairs, [[50000, -2]])
        self.assertEqual(sweep.hash_target_pairs, [[9, 4], [10, 8]])
        self.assertEqual(sweep.experiments, ["retrieval", "baseline"])
        self.assertEqual(conf.JointTrainAgent.Retrieval.anchor_weight_r, 2)

    def test_invalid_config_pairs_fail_before_enqueue(self):
        for overrides in [
                {**OVERRIDES, PAIR_PREFIX + "hash_target_pairs": [[63, 4]]},
                {**OVERRIDES, PAIR_PREFIX + "warmup_offset_pairs": [[100000, -2]]},
                {**OVERRIDES, PAIR_PREFIX + "warmup_offset_pairs": [[30000, 0.5]]},
                {**OVERRIDES, "JointTrainAgent.Typo": 10},
                {**OVERRIDES, PAIR_PREFIX + "enable": "target1"}]:
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, KeyError, AssertionError)):
                validate_overrides(self.config, overrides)
        self.assertFalse(self.queue.directory.exists())


if __name__ == "__main__":
    unittest.main()
