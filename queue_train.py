"""Persistent game/seed ownership and consumable, per-GPU training queues."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parent
QUEUE_DIR = ROOT / "training_queue"
SEED_DIR = ROOT / "config_files/game_seeds"
PAIR_PREFIX = "JointTrainAgent.Retrieval."
GAMES = """Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout
ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond
Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest
UpNDown""".split()


def canonical_game(value):
    value = re.sub(r"^ALE/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-v\d+$", "", value, flags=re.IGNORECASE)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", value):
        raise ValueError(f"Invalid Atari game: {value!r}")
    return {game.lower(): game for game in GAMES}.get(value.lower(), value)


def pair_key(game, seed):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seeds must be integers between 0 and 2**32 - 1")
    return f"{canonical_game(game).lower()}:{seed}"


def gpu_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("Use a single numeric GPU index, e.g. 0 or 1")
    return value


@contextmanager
def locked(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise ValueError(f"A worker is already running for this GPU: {path}") from error
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_atomic(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_line(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def override_args(overrides):
    return [part for key, value in overrides.items()
            for part in (key, value if isinstance(value, str) else repr(value))]


def read_seed_files(directory=SEED_DIR):
    """Read permanent, user-editable seed selections; reject duplicates globally."""
    selections = {}
    for path in sorted(Path(directory).glob("gpu_*.json")):
        data = json.loads(path.read_text())
        gpu = gpu_id(data["gpu"])
        if path.name != f"gpu_{gpu}.json" or not isinstance(data["game_seeds"], list):
            raise ValueError(f"Invalid GPU seed file: {path}")
        for entry in data["game_seeds"]:
            key = pair_key(entry["game"], entry["seed"])
            if key in selections:
                raise ValueError(f"Duplicate game/seed {key} in GPU seed files (including disabled entries)")
            if type(entry.get("enabled", True)) is not bool:
                raise ValueError(f"{path}: enabled must be true or false")
            if type(entry.get("fresh", False)) is not bool:
                raise ValueError(f"{path}: fresh must be true or false")
            if not isinstance(entry.get("warmups", []), list) or any(
                    not isinstance(source, str) for source in entry.get("warmups", [])):
                raise ValueError(f"{path}: warmups must be a list of checkpoint paths")
            if entry.get("fresh") and entry.get("warmups"):
                raise ValueError(f"{path}: choose warmups or fresh training, not both")
            selections[key] = {**entry, "game": canonical_game(entry["game"]), "gpu": gpu}
    return selections


def command_for(job, root=ROOT):
    pair_key(job["game"], job["seed"])
    gpu_id(job["gpu"])
    if not isinstance(job["overrides"], dict):
        raise ValueError("Job overrides must be a mapping")
    if "BasicSettings.Seed" in job["overrides"]:
        raise ValueError("Set the seed with --seeds / --game-seeds")
    for key in ("warmup_offset_pairs", "hash_target_pairs"):
        if not isinstance(job["overrides"].get(PAIR_PREFIX + key), list):
            raise ValueError(f"Job must explicitly contain a {key} list")
    if job.get("resume_warmup"):
        # The saved checkpoint owns its config, environment, seed and trajectory.
        # Supplying a fresh -config_path here would be rejected by the trainer.
        return [job["python"], "-u", str(root / "train.py"), "-n", job["name"],
                "-seed", str(job["seed"]), "-env_name", f"ALE/{job['game']}-v5",
                "--resume_warmup", job["resume_warmup"], *override_args(job["overrides"])]
    return [job["python"], "-u", str(root / "train.py"),
            "-n", job["name"], "-seed", str(job["seed"]),
            "-config_path", job["config"], "-env_name", f"ALE/{job['game']}-v5",
            "-trajectory_path", str(root / "D_TRAJ" / f"{job['game']}.pkl"),
            *override_args(job["overrides"])]


class TrainingQueue:
    def __init__(self, directory=QUEUE_DIR, seed_directory=None):
        self.directory = Path(directory)
        self.seed_directory = Path(seed_directory) if seed_directory is not None else self.directory.parent / "config_files/game_seeds"

    def path(self, gpu):
        return self.directory / f"gpu_{gpu_id(gpu)}.jsonl"

    def registry(self):
        path = self.directory / "assignments.json"
        if path.exists():
            registry = json.loads(path.read_text())
            if registry.get("version") != 1:
                raise ValueError("Unsupported training queue version")
            return registry
        # Losing the registry must never silently assign an existing job elsewhere.
        if any(self.directory.glob("gpu_*.jsonl")) or (self.directory / "history.jsonl").exists():
            raise ValueError("assignments.json is missing; restore it before using this queue")
        return {"version": 1, "owners": {}, "devices": {}}

    def save_registry(self, registry):
        write_atomic(self.directory / "assignments.json", json_line(registry))

    def jobs(self, gpu):
        path = self.path(gpu)
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def enqueue(self, game_seeds, gpus, config, overrides, python=sys.executable):
        specs = [dict(game=game, seed=seed, config=str(Path(config).absolute()), overrides=overrides)
                 for game, seed in game_seeds]
        return self.enqueue_jobs(specs, gpus, python=python)

    def enqueue_jobs(self, specs, gpus, python=sys.executable):
        gpus = list(dict.fromkeys(gpu_id(gpu) for gpu in gpus))
        if not gpus:
            raise ValueError("At least one GPU is required")
        added, skipped = [], 0
        with locked(self.directory / "queue.lock"):
            registry = self.registry()
            selections = read_seed_files(self.seed_directory)
            for key, entry in selections.items():
                previous = registry["owners"].get(key)
                if previous is not None and previous != entry["gpu"]:
                    raise ValueError(f"{key} already belongs to GPU {previous}; restore its seed file assignment")
            queues = {}
            new_selections = False
            for spec in specs:
                game, seed = canonical_game(spec["game"]), spec["seed"]
                key = pair_key(game, seed)
                selected_gpu = selections.get(key, {}).get("gpu", spec.get("gpu"))
                if selected_gpu is not None:
                    gpu_id(selected_gpu)
                    if spec.get("gpu", selected_gpu) != selected_gpu:
                        raise ValueError(f"GPU selection disagrees with seed file for {key}")
                    if registry["owners"].get(key, selected_gpu) != selected_gpu:
                        raise ValueError(f"Cannot move {key} from GPU {registry['owners'][key]} to {selected_gpu}")
                    registry["owners"][key] = selected_gpu
                if key not in registry["owners"]:
                    # Stable tie-breaking follows --gpus order. Historical owners count too.
                    registry["owners"][key] = min(
                        gpus, key=lambda gpu: list(registry["owners"].values()).count(gpu))
                gpu = registry["owners"][key]
                if key not in selections:
                    selections[key] = dict(game=game, seed=seed, gpu=gpu, enabled=True,
                                           warmups=[spec["warmup_source"]] if spec.get("warmup_source") else [],
                                           fresh=not bool(spec.get("resume_warmup")))
                    new_selections = True
                if gpu not in queues:
                    queues[gpu] = self.jobs(gpu)
                job_id = uuid.uuid4().hex
                job = dict(spec, id=job_id, game=game, seed=seed, gpu=gpu,
                           name=f"{game}-seed{seed}-queue-{job_id}",
                           python=str(Path(python).absolute()))
                command_for(job)
                json_line(job)  # Reject nonserializable/nonfinite values before any writes.
                if any(all(previous.get(field) == job.get(field)
                           for field in ("game", "seed", "config", "python", "overrides", "resume_warmup"))
                       for previous in queues[gpu]):
                    skipped += 1
                    continue
                queues[gpu].append(job)
                added.append(job)
            # Persist ownership first: an interrupted enqueue cannot cause migration.
            self.save_registry(registry)
            if new_selections:
                self.seed_directory.mkdir(parents=True, exist_ok=True)
                for gpu in sorted({entry["gpu"] for entry in selections.values()}, key=int):
                    entries = [{key: value for key, value in entry.items() if key != "gpu"}
                               for entry in selections.values() if entry["gpu"] == gpu]
                    write_atomic(self.seed_directory / f"gpu_{gpu}.json",
                                 json.dumps(dict(gpu=gpu, game_seeds=entries), indent=2) + "\n")
            for gpu, jobs in queues.items():
                write_atomic(self.path(gpu), "".join(map(json_line, jobs)))
        return added, skipped

    def bind_device(self, gpu, device):
        with locked(self.directory / "queue.lock"):
            registry = self.registry()
            previous = registry["devices"].get(gpu)
            if previous is not None and previous != device:
                raise ValueError(f"GPU {gpu} was bound to {previous}, but is now {device}; "
                                 "refusing to move game/seed pairs to another device")
            registry["devices"][gpu] = device
            self.save_registry(registry)

    def claim(self, gpu):
        with locked(self.directory / "queue.lock"):
            registry = self.registry()
            path = self.path(gpu)
            if not path.exists():
                return None
            with path.open() as handle:
                head = handle.readline()
                tail = handle.read()
            if not head:
                return None
            job = json.loads(head)
            key = pair_key(job["game"], job["seed"])
            if job["gpu"] != gpu or registry["owners"].get(key) != gpu:
                raise ValueError(f"Ownership mismatch for {key}; queued job was left in {path}")
            selection = read_seed_files(self.seed_directory).get(key)
            if selection is not None and selection["gpu"] != gpu:
                raise ValueError(f"Seed file moved {key} away from its permanent GPU {gpu}")
            command_for(job)  # Validate the head before removing it.
            write_atomic(path, tail)
            self.record("started", job)
            return job

    def record(self, event, job, **details):
        # Called under queue.lock for claims; independent workers serialize finishes here.
        with locked(self.directory / "history.lock"):
            with (self.directory / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json_line(dict(event=event, time=time.time(), job=job, **details)))
                handle.flush()
                os.fsync(handle.fileno())

    def status(self):
        with locked(self.directory / "queue.lock"):
            registry = self.registry()
            for gpu in sorted(set(registry["owners"].values()), key=int):
                jobs = self.jobs(gpu)
                owners = sum(owner == gpu for owner in registry["owners"].values())
                print(f"GPU {gpu}: {len(jobs)} pending job(s), {owners} assigned game/seed pair(s)")
                for job in jobs:
                    print(f"  {job['game']} / {job['seed']}  {json.dumps(job['overrides'])}")


def run_worker(gpu, *, once=False, max_jobs=0):
    gpu_id(gpu)
    queue = TrainingQueue()
    # This lock lasts for the entire worker, not just a single queue pop.
    with locked(queue.directory / f"worker_gpu_{gpu}.lock", blocking=False) as guard:
        device_uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", gpu],
            text=True).strip()
        if not device_uuid.startswith("GPU-") or "\n" in device_uuid:
            raise ValueError(f"Cannot identify physical GPU {gpu}: {device_uuid!r}")
        queue.bind_device(gpu, dict(host=socket.gethostname(), uuid=device_uuid))
        print(f"GPU {gpu} ({device_uuid}): consuming {queue.path(gpu)}", flush=True)
        # Select the physical device directly; CUDA's numeric ordering can differ
        # from nvidia-smi (and can change with CUDA_DEVICE_ORDER).
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device_uuid}
        completed, child, current_job = 0, None, None

        def interrupted(signum, _frame):
            raise KeyboardInterrupt(signum)

        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while True:
                current_job = queue.claim(gpu)
                if current_job is None:
                    if once:
                        return 0
                    time.sleep(10)
                    continue
                command = command_for(current_job)
                print(f"[GPU {gpu}] {shlex.join(command)}", flush=True)
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             start_new_session=True, pass_fds=(guard.fileno(),))
                    status = child.wait()
                except OSError as error:
                    print(f"[GPU {gpu}] Could not launch job: {error}", file=sys.stderr, flush=True)
                    status = 127
                child = None
                queue.record("finished", current_job, returncode=status)
                current_job = None
                completed += 1
                print(f"[GPU {gpu}] Job finished (exit {status}).", flush=True)
                if once or (max_jobs and completed >= max_jobs):
                    return status if status >= 0 else 128 - status
        except KeyboardInterrupt as error:
            signum = error.args[0] if error.args else signal.SIGINT
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if current_job is not None:
                queue.record("interrupted", current_job, signal=signum)
            return 128 + signum
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("status")
    worker = subparsers.add_parser("worker")
    worker.add_argument("gpu")
    limits = worker.add_mutually_exclusive_group()
    limits.add_argument("--once", action="store_true")

    def positive_integer(value):
        try:
            number = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a positive integer") from error
        if number < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return number

    limits.add_argument("--max-jobs", type=positive_integer, default=0)
    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            TrainingQueue().status()
            return 0
        return run_worker(args.gpu, once=args.once, max_jobs=args.max_jobs)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Queue error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
