"""Exercise real shell queue workers using a fake GPU query and temporary jobs."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class QueueWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="storm-queue-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ["queue_train.sh", *(f"{gpu}_train.sh" for gpu in range(8))]:
            shutil.copy2(ROOT / name, self.root / name)
        bindir = self.root / "bin"
        bindir.mkdir()
        query = bindir / "nvidia-smi"
        query.write_text('#!/bin/bash\nprintf "%s\\n" "$QUEUE_TEST_GPU_NAME"\n')
        query.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}",
                    "QUEUE_TEST_GPU_NAME": "NVIDIA GeForce RTX 3090"}

    def worker(self, *args, gpu=0, **kwargs):
        return subprocess.run(["bash", str(self.root / f"{gpu}_train.sh"), *args],
                              cwd="/tmp", env=self.env, capture_output=True,
                              text=True, timeout=15, **kwargs)

    def test_default_gpu_queues_and_all_wrapper_device_ids(self):
        names = ["NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090", "NVIDIA TITAN RTX",
                 "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "Other GPU"]
        suffixes = ["A6000", "3090", "titan", "pro6k", "default"]
        for gpu in range(8):
            with self.subTest(gpu=gpu):
                self.env["QUEUE_TEST_GPU_NAME"] = names[gpu % len(names)]
                queue = self.root / f"job_queue_{suffixes[gpu % len(names)]}.txt"
                queue.write_text('printf "%s:%s\\n" "$CUDA_VISIBLE_DEVICES" "$PWD" > result\n')
                result = self.worker("--once", gpu=gpu)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.root / "result").read_text(), f"{gpu}:{self.root}\n")
                self.assertEqual(queue.read_text(), "")

    def test_quotes_comments_environment_prefix_and_one_job_limit(self):
        queue = self.root / "custom queue.txt"
        queue.write_text('\n  # ignored\nSTORM_AMP_DTYPE=fp16 bash -c '
                         "'printf \"%s:%s\\n\" \"quoted argument\" \"$STORM_AMP_DTYPE\" > result'\n"
                         'printf "second\\n" >> result\n')
        result = self.worker(str(queue), "--once")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "result").read_text(), "quoted argument:fp16\n")
        self.assertEqual(queue.read_text(), 'printf "second\\n" >> result\n')

    def test_failed_exit_and_changed_directory_do_not_stop_or_leak_into_next_job(self):
        queue = self.root / "jobs.txt"
        queue.write_text('export QUEUE_TEST_LEAK=changed; cd /; exit 7\n'
                         'printf "%s:%s\\n" "${QUEUE_TEST_LEAK-unset}" "$PWD" > result\n')
        self.env.pop("QUEUE_TEST_LEAK", None)
        result = self.worker(str(queue), "--max-jobs", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Job failed (exit 7)", result.stderr)
        self.assertEqual((self.root / "result").read_text(), f"unset:{self.root}\n")

    def test_once_reports_job_failure_and_exits_when_empty(self):
        queue = self.root / "jobs.txt"
        queue.write_text("exit 23\n")
        self.assertEqual(self.worker(str(queue), "--once").returncode, 23)
        self.assertEqual(self.worker(str(queue), "--once").returncode, 0)

    def test_two_workers_claim_each_job_once(self):
        queue = self.root / "jobs.txt"
        output = shlex.quote(str(self.root / "claimed jobs.txt"))
        queue.write_text("".join(f"printf '%s\\n' {job} >> {output}\n" for job in range(8)))
        workers = [subprocess.Popen(
            ["bash", str(self.root / f"{gpu}_train.sh"), str(queue), "--max-jobs", "4"],
            cwd="/tmp", env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True) for gpu in (0, 1)]
        try:
            for worker in workers:
                stdout, stderr = worker.communicate(timeout=15)
                self.assertEqual(worker.returncode, 0, stdout + stderr)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                    worker.communicate()
        self.assertEqual(sorted((self.root / "claimed jobs.txt").read_text().splitlines()),
                         [str(job) for job in range(8)])
        self.assertEqual(queue.read_text(), "")

    def test_invalid_limits_fail_before_consuming_jobs(self):
        queue = self.root / "jobs.txt"
        queue.write_text("exit 0\n")
        for options in [("--max-jobs", "0"), ("--max-jobs", "bad"),
                        ("--once", "--max-jobs", "1")]:
            with self.subTest(options=options):
                result = self.worker(str(queue), *options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(queue.read_text(), "exit 0\n")


if __name__ == "__main__":
    unittest.main()
