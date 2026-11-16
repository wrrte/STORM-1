"""Prepare one TD-only result per Atari game for the existing N_train.sh workers.

Preparation never starts training. All GPUs must consume the same queue file.
Failed jobs retry at its tail; successful final evaluations are never rerun.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config_files/STORM.td_error_only.yaml"
DEFAULT_QUEUE = ROOT / "job_queue_td_error_only.txt"
DEFAULT_SCORES = ROOT / "results/main_performance_std/paired_scores.csv"


@contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def game_seeds(scores_path, fixed_seed=None):
    seeds = {}
    with scores_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["Backbone"] == "STORM":
                seeds.setdefault(row["Game"], set()).add(int(row["Seed"]))
    if len(seeds) != 26:
        raise ValueError(f"Expected 26 STORM games in {scores_path}, found {len(seeds)}")
    # Choose by seed number, never by evaluation return.
    return {game: min(values) if fixed_seed is None else fixed_seed
            for game, values in sorted(seeds.items())}


def validate_config(config):
    agent = config["JointTrainAgent"]
    retrieval = agent["Retrieval"]
    if not (retrieval["enable"] is True and retrieval["score_combination"] == "td_error_only"
            and retrieval["trigger_mode"] == "z_score"
            and retrieval.get("warmup_offset_pairs", []) == []
            and not retrieval.get("save_warmup", False)):
        raise ValueError("Use enable: True, score_combination: td_error_only, trigger_mode: z_score, "
                         "warmup_offset_pairs: [], save_warmup: False")
    threshold = float(retrieval["td_error_z_score_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("td_error_z_score_threshold must be finite and non-negative")
    if not (agent["NumEnvs"] == 1 and agent["FinalEvalStep"] == 100000
            and agent["SampleMaxSteps"] >= 100000 and agent["EvalMode"] == "final_only"
            and 0 <= retrieval["warmup_steps"] < 100000):
        raise ValueError("Use NumEnvs: 1, final_only evaluation at 100000, and a warmup below 100000")


def job_paths(manifest_path, game):
    return manifest_path.parent / f"{game}.json", manifest_path.parent / f"{game}.lock"


def queue_command(manifest, manifest_path, game):
    return shlex.join([manifest["python"], "-u", str(Path(manifest["root"]) / Path(__file__).name),
                       "run-job", "--manifest", str(manifest_path), "--game", game])


def enqueue(manifest, manifest_path, games):
    queue = Path(manifest["queue"])
    # This is the exact lock convention used by queue_train.sh.
    with locked(queue.with_suffix(".lock")):
        text = queue.read_text() if queue.is_file() else ""
        existing = set(text.splitlines())
        additions = [queue_command(manifest, manifest_path, game) for game in games]
        additions = [command for command in additions if command not in existing]
        if additions:
            with queue.open("a") as handle:
                if text and not text.endswith("\n"):
                    handle.write("\n")
                handle.write("\n".join(additions) + "\n")
    return len(additions)


def completed(manifest, game, state):
    config = yaml.safe_load(Path(manifest["config"]).read_text())["JointTrainAgent"]
    seed = manifest["seeds"][game]
    for attempt in range(1, state.get("attempts", 0) + 1):
        name = f"{game}-td_only-{manifest['id']}-seed{seed}-attempt{attempt}_O"
        path = Path(manifest["root"]) / "runs" / name / "final_evaluation.json"
        result = read_json(path)
        if result is None:
            continue
        returns = result.get("episode_returns", [])
        expected_precision = "bf16" if config["EvalPrecision"] == "both" else config["EvalPrecision"]
        if (result.get("run_name") == name and result.get("seed") == seed
                and result.get("env_name") == f"ALE/{game}-v5"
                and result.get("checkpoint_step") == config["FinalEvalStep"]
                and result.get("sample_max_steps") == config["SampleMaxSteps"]
                and result.get("precision") == expected_precision
                and len(returns) == config["EvalEpisodes"]
                and all(math.isfinite(float(value)) for value in returns)
                and math.isfinite(float(result.get("episode_avg_return", "nan")))):
            return result, path
    return None


def prepare(args):
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    seeds = game_seeds(args.scores, args.seed)
    fingerprint = json.dumps({"config": config, "seeds": seeds}, sort_keys=True)
    campaign_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    directory = args.output.resolve() / campaign_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    with locked(directory / "prepare.lock"):
        manifest = read_json(manifest_path)
        if manifest is None:
            snapshot = yaml.safe_dump(config, sort_keys=False)
            config_path = directory / "config.yaml"
            config_path.write_text(snapshot)
            manifest = dict(id=campaign_id, root=str(ROOT), python=str(Path(sys.executable).absolute()),
                            queue=str(args.queue.resolve()), config=str(config_path), seeds=seeds,
                            config_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
                            max_attempts=args.max_attempts)
            write_json(manifest_path, manifest)
        elif manifest["queue"] != str(args.queue.resolve()) or manifest["max_attempts"] != args.max_attempts:
            raise ValueError("This campaign already has a different queue or max_attempts; use its original settings")
        pending = []
        for game in seeds:
            state_path, lock_path = job_paths(manifest_path, game)
            with locked(lock_path, blocking=False) as acquired:
                if not acquired:
                    continue
                state = read_json(state_path, {})
                if not completed(manifest, game, state) and state.get("attempts", 0) < manifest["max_attempts"]:
                    pending.append(game)
        added = enqueue(manifest, manifest_path, pending)
    print(f"Prepared {added} new jobs; one selected seed per game, no second-seed jobs.")
    print(f"Manifest: {manifest_path}")
    print(f"Queue: {manifest['queue']}")
    print(f"TD threshold: {config['JointTrainAgent']['Retrieval']['td_error_z_score_threshold']}")
    for game, seed in seeds.items():
        print(f"  {game}: seed {seed}")
    return 0


def run_job(args):
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    game = args.game
    seed = manifest["seeds"][game]
    state_path, lock_path = job_paths(manifest_path, game)
    with locked(lock_path, blocking=False) as acquired:
        if not acquired:
            print(f"{game}: another worker already owns this job; skipping duplicate")
            return 0
        state = read_json(state_path, {})
        if completed(manifest, game, state):
            print(f"{game}: final evaluation already exists; skipping")
            return 0
        if state.get("attempts", 0) >= manifest["max_attempts"]:
            print(f"{game}: attempt limit reached; inspect {state_path}")
            return 1
        snapshot = Path(manifest["config"]).read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != manifest["config_sha256"]:
            raise ValueError("Campaign config snapshot was modified; edit the source YAML and prepare a new campaign")
        attempt = state.get("attempts", 0) + 1
        name = f"{game}-td_only-{manifest['id']}-seed{seed}-attempt{attempt}"
        trajectory = "Freeway.pkl" if game == "Freeway" else f"D_TRAJ/{game}.pkl"
        command = [manifest["python"], "-u", "train.py", "-n", name, "-seed", str(seed),
                   "-config_path", manifest["config"], "-env_name", f"ALE/{game}-v5",
                   "-trajectory_path", trajectory,
                   "JointTrainAgent.UseDemonstration", str(game == "Freeway")]
        log = manifest_path.parent / f"{game}-attempt{attempt}.log"
        state.update(attempts=attempt, status="running", command=command, log=str(log))
        write_json(state_path, state)
        print(f"{game}: starting attempt {attempt}; log: {log}", flush=True)
        with log.open("w") as output:
            try:
                result = subprocess.run(command, cwd=manifest["root"], stdout=output, stderr=subprocess.STDOUT)
                returncode = result.returncode
            except OSError as error:
                output.write(str(error) + "\n")
                returncode = 127
        outcome = completed(manifest, game, state)
        state.update(status="done" if outcome else "failed", returncode=returncode)
        if outcome:
            evaluation, path = outcome
            state.update(score=evaluation["episode_avg_return"], evaluation=str(path))
        write_json(state_path, state)
        if outcome:
            print(f"{game}: completed, score={state['score']}")
            return 0
        if attempt < manifest["max_attempts"]:
            enqueue(manifest, manifest_path, [game])
            print(f"{game}: failed; retry appended after the remaining games")
        else:
            print(f"{game}: failed after {attempt} attempts; inspect {log}")
        return returncode or 1


def status(args):
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    done = 0
    for game, seed in manifest["seeds"].items():
        state_path, _ = job_paths(manifest_path, game)
        state = read_json(state_path, {})
        outcome = completed(manifest, game, state)
        done += outcome is not None
        label = f"done score={outcome[0]['episode_avg_return']}" if outcome else state.get("status", "queued")
        print(f"{game:16} seed={seed:<5} {label}; attempts={state.get('attempts', 0)}")
    print(f"Games with a final result: {done}/{len(manifest['seeds'])}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    setup = commands.add_parser("prepare", help="Write a frozen config and append one job per missing game")
    setup.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    setup.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    setup.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    setup.add_argument("--output", type=Path, default=ROOT / "results/td_error_only_queue")
    setup.add_argument("--seed", type=int, help="Override the default lowest existing Full FLASH seed for every game")
    setup.add_argument("--max-attempts", type=int, default=3)
    for name in ("run-job", "status"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--manifest", type=Path, required=True)
        if name == "run-job":
            subparser.add_argument("--game", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare" and args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    return {"prepare": prepare, "run-job": run_job, "status": status}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
