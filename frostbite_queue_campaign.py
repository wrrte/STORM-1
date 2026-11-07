#!/usr/bin/env python3
"""A shared Frostbite seed search, followed by the existing GPU job queue.

Queue workers reserve at most three candidates in total. Rejections release a
reservation; an accepted candidate keeps it until both paired branches finish.
Unexpected failures pause the campaign rather than silently trying new seeds.
Stopping a worker instead schedules the same seed for a separate retry.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from screen_frostbite_seeds import stop_process
from seed_screening import SCREEN_REJECTED_EXIT_CODE


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("randomneighor", "retrieval")


class TrainingInterrupted(Exception):
    """The child was stopped by SIGINT/SIGTERM, rather than failing training."""


def now():
    return datetime.now(timezone.utc).isoformat()


def interrupted(*_):
    raise KeyboardInterrupt


def write_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def locked(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def train_command(plan, seed):
    return [plan["python"], "-u", str(ROOT / "train.py"),
            "-n", f"Frostbite-score-screen-{plan['campaign']}-seed{seed}",
            "-seed", str(seed), "-config_path", plan["config"],
            "-env_name", "ALE/Frostbite-v5", "-trajectory_path", "D_TRAJ/Frostbite.pkl"]


def publish(directory, state):
    state["updated_at"] = now()
    write_json(directory / "state.json", state)
    output = io.StringIO()
    columns = ["seed", "status", "gpu", "best_episode_return", "decision_step",
               "random_neighbor", "full_flash", "flash_minus_random", "log_path", "error"]
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for attempt in state["attempts"]:
        row = {**attempt, **attempt.get("screen", {}), **attempt.get("scores", {})}
        writer.writerow(row)
    temporary = directory / f"results.{os.getpid()}.tmp"
    temporary.write_text(output.getvalue())
    temporary.replace(directory / "results.csv")


def claim(directory, plan, gpu):
    with locked(directory / "state.lock"):
        state = json.loads((directory / "state.json").read_text())
        if state["status"] != "running":
            return state["status"], None
        occupied = sum(a["status"] in ("running", "completed") for a in state["attempts"])
        if occupied >= plan["target_seeds"]:
            return "wait", None
        retries = state.setdefault("retry_seeds", [])
        seed = retries.pop(0) if retries else state["next_seed"]
        if seed >= 2**32:
            state.update(status="error", error="Exhausted the valid seed range")
            publish(directory, state)
            return "error", None
        if seed == state["next_seed"]:
            state["next_seed"] += 1
        command = train_command(plan, seed)
        attempt_number = 1 + sum(a["seed"] == seed for a in state["attempts"])
        suffix = f"-retry{attempt_number}" if attempt_number > 1 else ""
        command[command.index("-n") + 1] += suffix
        log_suffix = f"_retry{attempt_number}" if attempt_number > 1 else ""
        attempt = {"seed": seed, "status": "running", "gpu": gpu,
                   "attempt_number": attempt_number,
                   "started_at": now(), "command": command,
                   "run_base": command[command.index("-n") + 1],
                   "log_path": str(directory / f"seed_{seed}{log_suffix}.log")}
        state["attempts"].append(attempt)
        publish(directory, state)
        return "run", attempt


def validate_outcome(plan, attempt, exit_code):
    if exit_code in (-signal.SIGINT, -signal.SIGTERM, 130, 143):
        raise TrainingInterrupted(f"Training stopped with exit code {exit_code}")
    if exit_code not in (0, SCREEN_REJECTED_EXIT_CODE):
        raise RuntimeError(f"Training failed with exit code {exit_code}")
    base = attempt["run_base"]
    report_path = ROOT / "runs" / f"{base}_Shared" / "seed_screen.json"
    report = json.loads(report_path.read_text())
    if (report["seed"] != attempt["seed"] or report["run_name"] != f"{base}_Shared"
            or report["screen_steps"] != plan["screen_steps"]
            or report["min_return_exclusive"] != plan["min_return_exclusive"]):
        raise ValueError("Screen report does not match this candidate and campaign")
    best = report["best_episode_return"]
    step = report["decision_step"]
    if not math.isfinite(best) or type(step) is not int or not 0 < step <= plan["screen_steps"]:
        raise ValueError("Invalid screening score or decision step")
    if exit_code == SCREEN_REJECTED_EXIT_CODE:
        if report["status"] != "rejected" or best > plan["min_return_exclusive"] or step != plan["screen_steps"]:
            raise ValueError("Invalid rejection report")
        return "rejected", report, {}
    if report["status"] != "accepted" or best <= plan["min_return_exclusive"]:
        raise ValueError("Training completed without passing the screening criterion")
    scores = {}
    checkpoints = []
    # Check both local final results, not just the supervisor's return code.
    import yaml
    for experiment, column in zip(EXPERIMENTS, ("random_neighbor", "full_flash")):
        run_dir = ROOT / "runs" / f"{base}_{experiment}"
        result = json.loads((run_dir / "final_evaluation.json").read_text())
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        joint = config["JointTrainAgent"]
        retrieval = joint["Retrieval"]
        source = "random" if experiment == "randomneighor" else "retrieval"
        if (result["seed"] != attempt["seed"] or result["env_name"] != "ALE/Frostbite-v5"
                or result["run_name"] != run_dir.name or result["checkpoint_step"] != 100000
                or result["precision"] != "bf16" or len(result["episode_returns"]) != 20
                or joint["SeedScreenSteps"] != 0 or retrieval["enable"] is not True
                or retrieval["neighbor_source"] != source
                or retrieval["warmup_steps"] != 50000 or retrieval["anchor_offset"] != -2
                or retrieval["target"] != 16 or retrieval["anchor_weight"] != 0.0625
                or retrieval["anchor_weight_r"] is not None):
            raise ValueError(f"Invalid final result or settings: {run_dir}")
        score = float(result["episode_avg_return"])
        if (not math.isfinite(score) or not all(math.isfinite(x) for x in result["episode_returns"])
                or not math.isclose(score, sum(result["episode_returns"]) / 20, abs_tol=1e-5)):
            raise ValueError(f"Invalid evaluation mean: {run_dir}")
        scores[column] = score
        checkpoints.append(json.loads((run_dir / "shared_warmup.json").read_text())["checkpoint_dir"])
    if checkpoints[0] != checkpoints[1]:
        raise ValueError("The two branches did not use the same shared checkpoint")
    scores["flash_minus_random"] = scores["full_flash"] - scores["random_neighbor"]
    return "completed", report, scores


def finish(directory, plan, attempt, status, report=None, scores=None, error=None):
    with locked(directory / "state.lock"):
        state = json.loads((directory / "state.json").read_text())
        row = next(a for a in state["attempts"] if a["run_base"] == attempt["run_base"])
        row.update(status=status, finished_at=now(), screen=report or {}, scores=scores or {})
        if error is not None:
            message = str(error) or type(error).__name__
            row.update(error=message, error_type=type(error).__name__)
            if status == "interrupted":
                retries = state.setdefault("retry_seeds", [])
                if row["seed"] not in retries:
                    retries.append(row["seed"])
            else:
                state.update(status="error", error=f"Seed {row['seed']}: {message}")
        completed = sum(a["status"] == "completed" for a in state["attempts"])
        if state["status"] != "error" and completed == plan["target_seeds"]:
            state.update(status="completed", completed_at=now())
        publish(directory, state)
        done = state["status"] == "completed"
    if done:
        queue = Path(plan["queue"])
        with locked(queue.with_suffix(".lock")):
            # Remove only our remaining barrier entries; preserve the ordinary queue.
            lines = queue.read_text().splitlines(keepends=True)
            temporary = queue.with_name(queue.name + ".frostbite.tmp")
            temporary.write_text("".join(line for line in lines if line.strip() != plan["guard_command"]))
            temporary.replace(queue)


def refresh_barrier(directory, plan):
    """Keep priority entries available when queue workers are restarted."""
    queue = Path(plan["queue"])
    with locked(queue.with_suffix(".lock")):
        if json.loads((directory / "state.json").read_text())["status"] == "completed":
            return
        lines = queue.read_text().splitlines(keepends=True)
        ordinary = [line for line in lines if line.strip() != plan["guard_command"]]
        temporary = queue.with_name(queue.name + ".frostbite.tmp")
        temporary.write_text((plan["guard_command"] + "\n") * plan["barrier_entries"] + "".join(ordinary))
        temporary.replace(queue)


def recover_interrupted(directory, seed):
    """Repair an inspected legacy interruption, preserving every prior artifact.

    The stopped worker must no longer own a training process. Genuine failures
    with a diagnostic message cannot be cleared with this operation.
    """
    directory = directory.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    with locked(directory / "state.lock"):
        state = json.loads((directory / "state.json").read_text())
        row = next(a for a in reversed(state["attempts"]) if a["seed"] == seed)
        legacy = row["status"] == "error" and row.get("error") == ""
        known = row.get("error_type") in ("KeyboardInterrupt", "SystemExit", "TrainingInterrupted")
        if not (legacy or known) or row["status"] not in ("error", "interrupted"):
            raise ValueError("Only a confirmed worker interruption can be retried here")
        snapshot = directory / ("state_before_recovery_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
        write_json(snapshot, state)
        event = {"time_utc": now(), "seed": seed, "previous_error": row.get("error"),
                 "action": "retry same seed under a new run name; preserve prior logs and checkpoints"}
        row.update(status="interrupted", error="Worker interrupted before completing the candidate",
                   error_type="TrainingInterrupted")
        retries = state.setdefault("retry_seeds", [])
        if seed not in retries:
            retries.insert(0, seed)
        state.setdefault("recoveries", []).append(event)
        if not any(a["status"] == "error" for a in state["attempts"]):
            state["status"] = "running"
            state.pop("error", None)
        publish(directory, state)
    refresh_barrier(directory, plan)
    print(f"Seed {seed} queued for retry; previous artifacts retained.", flush=True)
    return 0


def worker(directory):
    directory = directory.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("Start this command through one N_train.sh worker per GPU")
    print(f"[GPU {gpu}] Frostbite priority campaign: {directory}", flush=True)
    refresh_barrier(directory, plan)
    last_message = 0
    while True:
        status, attempt = claim(directory, plan, gpu)
        if status == "completed":
            print("Three paired seeds completed; continuing the ordinary job queue.", flush=True)
            return 0
        if status in ("wait", "error"):
            if time.monotonic() - last_message > 60:
                message = ("Campaign paused after an error; inspect state.json and seed logs."
                           if status == "error" else "Waiting for the three reserved candidates to finish.")
                print(f"[GPU {gpu}] {message}", flush=True)
                last_message = time.monotonic()
            time.sleep(2)
            continue
        process = None
        try:
            digest = hashlib.sha256(Path(plan["config"]).read_bytes()).hexdigest()
            if digest != plan["config_sha256"]:
                raise ValueError("The campaign config changed after the queue was prepared")
            print(f"[GPU {gpu}] Seed {attempt['seed']}; log: {attempt['log_path']}", flush=True)
            with Path(attempt["log_path"]).open("x") as output:
                output.write(shlex.join(attempt["command"]) + "\n")
                output.flush()
                process = subprocess.Popen(attempt["command"], cwd=ROOT, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                code = process.wait()
            outcome, report, scores = validate_outcome(plan, attempt, code)
            finish(directory, plan, attempt, outcome, report, scores)
            print(f"[GPU {gpu}] Seed {attempt['seed']}: {outcome}", flush=True)
        except (KeyboardInterrupt, SystemExit, TrainingInterrupted) as error:
            if process is not None and process.poll() is None:
                stop_process(process)
            finish(directory, plan, attempt, "interrupted", error=error)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            print(f"[GPU {gpu}] {error}; retrying the same seed.", flush=True)
        except Exception as error:
            if process is not None and process.poll() is None:
                stop_process(process)
            finish(directory, plan, attempt, "error", error=error)
            print(f"[GPU {gpu}] Paused: {error}", file=sys.stderr, flush=True)


def prepare(config, directory, start_seed, target=3):
    """Stage one barrier per physical GPU before the untouched ordinary queue."""
    import yaml
    config = config.resolve()
    conf = yaml.safe_load(config.read_text())
    joint = conf["JointTrainAgent"]
    retrieval = joint["Retrieval"]
    if (retrieval["enable"] != list(EXPERIMENTS) or retrieval["warmup_steps"] != 50000
            or retrieval["warmup_offset_pairs"] != [] or not retrieval["save_warmup"]
            or joint["SeedScreenSteps"] != 20000 or joint["SeedScreenMinReturn"] != 500.0
            or joint["EvalEpisodes"] != 20 or joint["FinalEvalStep"] != 100000):
        raise ValueError("Unexpected campaign configuration")
    if not 0 <= start_seed < 2**32 or target != 3:
        raise ValueError("This campaign requires three seeds and a valid starting seed")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    queue = ROOT / "job_queue_3090.txt"
    guard = shlex.join([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(directory)])
    plan = {"version": 1, "campaign": directory.name, "created_at": now(),
            "python": sys.executable, "config": str(config),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "queue": str(queue), "guard_command": guard, "barrier_entries": 8,
            "target_seeds": target, "start_seed": start_seed,
            "screen_steps": 20000, "min_return_exclusive": 500.0,
            "selection": "first three accepted candidates, never selected by final score",
            "includes_unfinished_episodes": True, "experiments": list(EXPERIMENTS),
            "reporting": "additional score-screened experiment; frozen paper/CI selection unchanged"}
    write_json(directory / "plan.json", plan)
    publish(directory, {"status": "running", "next_seed": start_seed, "attempts": []})
    with locked(queue.with_suffix(".lock")):
        before = queue.read_text()
        if "frostbite_queue_campaign.py" in before:
            raise ValueError("A Frostbite campaign is already queued")
        (directory / "queue_before.txt").write_text(before)
        after = (guard + "\n") * 8 + before
        temporary = queue.with_name(queue.name + ".frostbite.tmp")
        temporary.write_text(after)
        temporary.replace(queue)
        (directory / "queue_after.txt").write_text(after)
    print(directory)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--worker", type=Path)
    action.add_argument("--status", type=Path)
    action.add_argument("--prepare", type=Path, metavar="CONFIG")
    action.add_argument("--recover-interrupted", type=int, metavar="SEED")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--start-seed", type=int, default=11000)
    args = parser.parse_args()
    if args.recover_interrupted is not None:
        if args.directory is None:
            parser.error("--recover-interrupted requires --directory")
        return recover_interrupted(args.directory, args.recover_interrupted)
    if args.prepare:
        directory = args.directory or ROOT / "diagnostics" / (
            "frostbite_neighbor_screen_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_" + uuid.uuid4().hex[:6])
        return prepare(args.prepare, directory, args.start_seed)
    if args.status:
        state = json.loads((args.status / "state.json").read_text())
        counts = {name: sum(a["status"] == name for a in state["attempts"])
                  for name in ("running", "rejected", "completed", "interrupted", "error")}
        print(json.dumps({"status": state["status"], "counts": counts,
                          "completed_seeds": [a["seed"] for a in state["attempts"] if a["status"] == "completed"],
                          "results": str(args.status / "results.csv"), "error": state.get("error")}, indent=2))
        return 0
    signal.signal(signal.SIGTERM, interrupted)
    return worker(args.worker)


if __name__ == "__main__":
    sys.exit(main())
