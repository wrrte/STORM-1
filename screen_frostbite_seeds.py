#!/usr/bin/env python3
"""Train retrieval-free Frostbite seeds concurrently, rejecting weak prefixes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import uuid

from seed_screening import SCREEN_REJECTED_EXIT_CODE


ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"], help="Physical GPU IDs or UUIDs")
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--successes-per-gpu", type=int, default=1,
                        help="Fully train this many accepted seeds per GPU; 0 keeps searching")
    parser.add_argument("--max-attempts-per-gpu", type=int, default=0, help="0 means no attempt limit")
    parser.add_argument("--screen-steps", type=int, default=20000)
    parser.add_argument("--min-return", type=float, default=500.0, help="Exclusive score threshold")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--python", default=sys.executable, help="Python executable with STORM dependencies")
    parser.add_argument("--log-dir", type=Path, help="New directory for worker logs and outcomes")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without files or GPU access")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or any(
            not gpu or not all(c.isalnum() or c in "-_" for c in gpu) for gpu in args.gpus):
        parser.error("--gpus must contain distinct GPU indices or UUIDs")
    if not 0 <= args.start_seed <= 2**32 - len(args.gpus):
        parser.error("Starting seeds must fit in the NumPy seed range 0..2**32-1")
    if args.successes_per_gpu < 0 or args.max_attempts_per_gpu < 0:
        parser.error("Success and attempt limits must be nonnegative")
    if args.screen_steps <= 0 or not math.isfinite(args.min_return) or args.eval_episodes <= 0:
        parser.error("Screen steps and evaluation episodes must be positive; the threshold must be finite")
    return args


def run_name(campaign, seed):
    return f"Frostbite-seed-screen-{campaign}-seed{seed}"


def train_command(args, campaign, seed):
    return [
        args.python, "-u", str(ROOT / "train.py"),
        "-n", run_name(campaign, seed), "-seed", str(seed),
        "-config_path", str(ROOT / "config_files/STORM.yaml"),
        "-env_name", "ALE/Frostbite-v5", "-trajectory_path", "D_TRAJ/Frostbite.pkl",
        "JointTrainAgent.Retrieval.enable", "False",
        "JointTrainAgent.Retrieval.save_warmup", "False",
        "JointTrainAgent.Retrieval.warmup_offset_pairs", "[]",
        "JointTrainAgent.NumEnvs", "1",
        "JointTrainAgent.SeedScreenSteps", str(args.screen_steps),
        "JointTrainAgent.SeedScreenMinReturn", str(args.min_return),
        "JointTrainAgent.EvalMode", "final_only",
        "JointTrainAgent.EvalEpisodes", str(args.eval_episodes),
    ]


def gpu_preflight(args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(args.gpus)}
    check = subprocess.run([
        args.python, "-c",
        "import sys, torch; "
        "n = torch.cuda.device_count(); "
        "print('Visible CUDA devices:', n); "
        "sys.exit(0 if torch.cuda.is_available() and n == int(sys.argv[1]) else 1)",
        str(len(args.gpus)),
    ], cwd=ROOT, env=env, capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError("Selected GPUs are unavailable; no training was started.\n" +
                           check.stdout + check.stderr)


def stop_process(process):
    # train.py also owns AsyncVectorEnv children; stop its whole process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def worker(args, campaign, log_dir, index, stop):
    gpu = args.gpus[index]
    seed = args.start_seed + index
    stride = len(args.gpus)
    completed = attempts = 0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    outcomes = log_dir / f"gpu_{gpu}.jsonl"
    while not stop.is_set() and (not args.successes_per_gpu or completed < args.successes_per_gpu):
        if args.max_attempts_per_gpu and attempts >= args.max_attempts_per_gpu:
            raise RuntimeError(f"GPU {gpu}: attempt limit reached after {completed} completed models")
        if seed >= 2**32:
            raise RuntimeError(f"GPU {gpu}: exhausted the valid seed range")
        command = train_command(args, campaign, seed)
        name = run_name(campaign, seed) + "_X"
        log_path = log_dir / f"gpu_{gpu}_seed_{seed}.log"
        report_path = ROOT / "runs" / name / "seed_screen.json"
        print(f"[GPU {gpu}] Training seed {seed}; log: {log_path}", flush=True)
        attempts += 1
        with log_path.open("x") as output:
            output.write(shlex.join(command) + "\n")
            output.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if stop.wait(0.5):
                        stop_process(process)
                        return completed
            finally:
                if process.poll() is None:
                    stop_process(process)
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        code = process.returncode
        status = "completed" if code == 0 else "rejected" if code == SCREEN_REJECTED_EXIT_CODE else "error"
        # Only the screening-specific return code permits trying another seed.
        expected_report = "accepted" if status == "completed" else "rejected"
        if status != "error" and (report.get("status") != expected_report or report.get("seed") != seed):
            status = "error"
        with outcomes.open("a") as handle:
            handle.write(json.dumps({
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": gpu, "seed": seed, "status": status, "exit_code": code,
                "run_name": name, "log_path": str(log_path), "screen": report,
            }) + "\n")
        if status == "error":
            raise RuntimeError(f"GPU {gpu}, seed {seed}: training failed (exit {code}); see {log_path}")
        if status == "completed":
            completed += 1
            print(f"[GPU {gpu}] Seed {seed} fully trained and evaluated ({completed} completed).", flush=True)
        else:
            print(f"[GPU {gpu}] Seed {seed} rejected at {args.screen_steps} steps; trying the next seed.", flush=True)
        seed += stride
    return completed


def main(argv=None):
    args = parse_args(argv)
    campaign = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    if args.dry_run:
        for index, gpu in enumerate(args.gpus):
            seed = args.start_seed + index
            print(f"GPU {gpu}: seeds {seed}, {seed + len(args.gpus)}, ...")
            print("CUDA_VISIBLE_DEVICES=" + shlex.quote(gpu) + " " +
                  shlex.join(train_command(args, campaign, seed)))
        print(f"Stop after {args.successes_per_gpu or 'unlimited'} completed models per GPU. "
              f"Screen: score > {args.min_return:g} within {args.screen_steps} environment steps.")
        return 0
    try:
        gpu_preflight(args)
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    log_dir = (args.log_dir or ROOT / "diagnostics" / f"frostbite_seed_screen_{campaign}").resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "campaign.json").write_text(json.dumps({
        **vars(args), "python": args.python, "log_dir": str(log_dir),
        "campaign": campaign, "seed_stride": len(args.gpus),
    }, indent=2) + "\n")
    stop = threading.Event()
    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
    failed = False
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, args, campaign, log_dir, index, stop)
                       for index in range(len(args.gpus))]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failed = True
                    stop.set()
                    print(str(error), file=sys.stderr, flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(f"Outcomes and logs: {log_dir}", flush=True)
    return 1 if failed else 130 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
