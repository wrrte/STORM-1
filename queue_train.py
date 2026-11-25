"""Durable priority queues shared by workers of the same GPU model."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import hashlib
import shutil
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
# Private runner exit code for a completed job recovered entirely from saved progress.
NO_BRANCH_NEEDED = 20
# New seeds encode the model in million-sized ranges. Legacy seeds keep their values.
SEED_RANGES = {name: ((i + 1) * 1000000, (i + 2) * 1000000 - 1)
               for i, name in enumerate(("A6000", "3090", "TITAN_RTX", "PRO6000_BLACKWELL",
                                          "4090", "5090", "RTX6000_ADA"))}
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


def experiment_name(value):
    """A stable queue cohort name, also usable as a report directory name."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value):
        raise ValueError('Experiment names must be 1–80 letters/digits, dots, underscores or hyphens, starting with a letter/digit')
    return value


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


def gpu_type(value):
    """Normalize model aliases without combining physically different GPU models."""
    value = re.sub(r"^NVIDIA\s+", "", value.strip(), flags=re.I)
    value = re.sub(r"^GeForce\s+", "", value, flags=re.I)
    upper = value.upper()
    if upper in {"PRO6K", "PRO6000"} or ("PRO" in upper and "6000" in upper and "BLACKWELL" in upper):
        return "PRO6000_BLACKWELL"
    if "6000" in upper and "ADA" in upper:
        return "RTX6000_ADA"
    if upper == "TITAN":
        return "TITAN_RTX"
    value = re.sub(r"^RTX[\s_-]*(?=A?\d)", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    if not value or (value.isdigit() and int(value) < 1000):
        raise ValueError("Supply a GPU model, e.g. A6000 or RTX 3090")
    return value


def detect_device(index):
    gpu_id(index)
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader", "-i", index], text=True).strip()
    name, device_uuid = [part.strip() for part in output.split(",")]
    if not device_uuid.startswith("GPU-") or "\n" in device_uuid:
        raise ValueError(f"Cannot identify physical GPU {index}")
    return dict(gpu_type=gpu_type(name), host=socket.gethostname(), uuid=device_uuid, index=index)


def read_seed_files(directory=SEED_DIR):
    selections, owners = {}, {}
    for path in sorted(Path(directory).glob("type_*.json")):
        data = json.loads(path.read_text())
        model = gpu_type(data["gpu_type"])
        if path.name != f"type_{model}.json":
            raise ValueError(f"Invalid model seed file: {path}")
        for entry in data["game_seeds"]:
            key = pair_key(entry["game"], entry["seed"])
            if key in selections:
                raise ValueError(f"Duplicate game/seed {key}")
            if owners.setdefault(entry["seed"], model) != model:
                raise ValueError(f"Seed {entry['seed']} belongs to multiple GPU types")
            if type(entry.get("enabled", True)) is not bool:
                raise ValueError(f"Invalid enabled flag in {path}")
            if not isinstance(entry.get("warmups", []), list):
                raise ValueError(f"Invalid warmups in {path}")
            selections[key] = {**entry, "game": canonical_game(entry["game"]), "gpu_type": model}
    return selections


def save_seed_files(selections, directory=SEED_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for model in sorted({entry["gpu_type"] for entry in selections.values()}):
        entries = [{k: v for k, v in entry.items() if k != "gpu_type"}
                   for entry in selections.values() if entry["gpu_type"] == model]
        write_atomic(directory / f"type_{model}.json", json.dumps(
            dict(gpu_type=model, game_seeds=entries), ensure_ascii=False, indent=2) + "\n")


class TrainingQueue:
    """SQLite transactions make claims, recovery and reprioritization atomic."""
    def __init__(self, directory=QUEUE_DIR, seed_directory=None):
        self.directory = Path(directory)
        self.seed_directory = Path(seed_directory) if seed_directory else self.directory.parent / "config_files/game_seeds"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "queue.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS seeds(seed INTEGER PRIMARY KEY, gpu_type TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS seed_ranges(gpu_type TEXT PRIMARY KEY, start INTEGER UNIQUE, end INTEGER);
                CREATE TABLE IF NOT EXISTS jobs(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    gpu_type TEXT NOT NULL, priority INTEGER NOT NULL, status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL, worker TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, event TEXT, details TEXT, time REAL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=60)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def event(db, job_id, event, **details):
        db.execute("INSERT INTO events(job_id,event,details,time) VALUES(?,?,?,?)",
                   (job_id, event, json.dumps(details), time.time()))

    @staticmethod
    def check_seed(db, seed, model):
        pair_key("Alien", seed)
        for owner, (start, end) in SEED_RANGES.items():
            if start <= seed <= end and owner != model:
                raise ValueError(f"Seed {seed} is in the reserved {owner} range")
        reserved = db.execute("SELECT gpu_type FROM seed_ranges WHERE start<=? AND end>=?", (seed, seed)).fetchone()
        if reserved and reserved[0] != model:
            raise ValueError(f"Seed {seed} is in the reserved {reserved[0]} range")
        previous = db.execute("SELECT gpu_type FROM seeds WHERE seed=?", (seed,)).fetchone()
        if previous and previous[0] != model:
            raise ValueError(f"Seed {seed} belongs to {previous[0]}, not {model}; choose another seed")

    @staticmethod
    def own_seed(db, seed, model):
        TrainingQueue.check_seed(db, seed, model)
        db.execute("INSERT OR IGNORE INTO seeds VALUES(?,?)", (seed, model))

    def register_selections(self, selections):
        with self.connect() as db:
            for entry in selections.values():
                self.own_seed(db, entry["seed"], gpu_type(entry["gpu_type"]))

    def allocate_seeds(self, model, count, *, reserve=True):
        model = gpu_type(model)
        if count < 1:
            raise ValueError("Seed count must be positive")
        with self.connect() as db:
            row = db.execute("SELECT start,end FROM seed_ranges WHERE gpu_type=?", (model,)).fetchone()
            if model in SEED_RANGES:
                start, end = SEED_RANGES[model]
            elif row:
                start, end = row
            else:
                start = max(8000000, db.execute("SELECT COALESCE(MAX(end),0)+1 FROM seed_ranges").fetchone()[0])
                while db.execute("SELECT 1 FROM seeds WHERE seed BETWEEN ? AND ? AND gpu_type!=?",
                                 (start, start + 999999, model)).fetchone():
                    start += 1000000
                end = start + 999999
            last = db.execute("SELECT MAX(seed) FROM seeds WHERE seed BETWEEN ? AND ?", (start, end)).fetchone()[0]
            next_seed = start if last is None else last + 1
            if next_seed + count - 1 > min(end, 2**32 - 1):
                raise ValueError(f"Seed range for {model} exhausted")
            result = list(range(next_seed, next_seed + count))
            if reserve:
                db.execute("INSERT OR IGNORE INTO seed_ranges VALUES(?,?,?)", (model, start, end))
                for seed in result:
                    self.own_seed(db, seed, model)
            return result

    def enqueue_jobs(self, specs, model=None, python=sys.executable, priority=0):
        added, skipped = [], 0
        with self.connect() as db:
            prepared, experiments = [], {}
            for spec in specs:
                kind = gpu_type(spec.get("gpu_type", model or ""))
                job = dict(spec, game=canonical_game(spec["game"]), gpu_type=kind,
                           python=str(Path(python).absolute()))
                # Runtime progress never changes the identity of a requested experiment.
                identity = {k: job.get(k) for k in ("game", "seed", "gpu_type", "config_text", "overrides", "python")}
                experiment = job.get('experiment_id')
                if experiment is not None:
                    experiment_name(experiment)
                    identity['experiment_id'] = experiment
                fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                prepared.append((job, kind, fingerprint, experiment))
                if experiment:
                    experiments.setdefault(experiment, set()).add(fingerprint)
            registered = {}
            for experiment, fingerprints in experiments.items():
                previous = db.execute('SELECT value FROM metadata WHERE key=?', ('experiment:' + experiment,)).fetchone()
                if previous:
                    record = json.loads(previous[0])
                    if set(record['fingerprints']) != fingerprints:
                        raise ValueError(f'Experiment {experiment!r} already names another job set; use a new --experiment name')
                    current = {row[0] for row in db.execute('SELECT id FROM jobs')}
                    if not set(record['job_ids']) <= current:
                        raise ValueError(f'Experiment {experiment!r} has removed queue jobs; use a new name for a new queue')
                    registered[experiment] = record
                elif (self.directory.parent / 'results' / 'experiments' / experiment / 'manifest.json').exists():
                    raise ValueError(f'Experiment {experiment!r} already has a saved report; use a new --experiment name')
            for job, kind, fingerprint, experiment in prepared:
                self.own_seed(db, job['seed'], kind)
                sql = 'SELECT id FROM jobs WHERE fingerprint=?'
                # Named cohorts remain one fixed set even after completion/failure.
                # Use queue_train.py retry to resume a failed member.
                if not experiment:
                    sql += " AND status IN ('pending','running')"
                if db.execute(sql, (fingerprint,)).fetchone():
                    skipped += 1
                    continue
                job.setdefault("id", uuid.uuid4().hex)
                job.setdefault("name", f"{job['game']}-seed{job['seed']}-queue-{job['id']}")
                job.setdefault("completed_branches", [])
                level = job.get("priority", priority)
                if type(level) is not int:
                    raise ValueError("Priority must be an integer")
                now = time.time()
                db.execute("INSERT INTO jobs(id,gpu_type,priority,status,fingerprint,payload,created,updated) "
                           "VALUES(?,?,?,'pending',?,?,?,?)",
                           (job["id"], kind, level, fingerprint, json_line(job), now, now))
                self.event(db, job["id"], "enqueued", gpu_type=kind, priority=level)
                added.append(job)
            for experiment, fingerprints in experiments.items():
                if experiment not in registered:
                    job_ids = [row[0] for row in db.execute('SELECT id,payload FROM jobs')
                               if json.loads(row[1]).get('experiment_id') == experiment]
                    record = dict(experiment_id=experiment, fingerprints=sorted(fingerprints),
                                  job_ids=job_ids, created=time.time())
                    db.execute('INSERT INTO metadata(key,value) VALUES(?,?)',
                               ('experiment:' + experiment, json.dumps(record)))
        return added, skipped

    def jobs(self, model=None, statuses=("pending", "running")):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY priority DESC,seq").fetchall()
        return [{**json.loads(row["payload"]), "status": row["status"], "priority": row["priority"],
                 "attempts": row["attempts"], "worker": json.loads(row["worker"]) if row["worker"] else None}
                for row in rows if (model is None or row["gpu_type"] == gpu_type(model))
                and (statuses is None or row["status"] in statuses)]

    def lease_path(self, job_id):
        return self.directory / "leases" / f"{job_id}.lock"

    def recover(self):
        recovered = []
        # A live worker OR its surviving trainer holds this lock. Never time out a long job.
        with self.connect() as db:
            for row in db.execute("SELECT id FROM jobs WHERE status='running'").fetchall():
                try:
                    with locked(self.lease_path(row["id"]), blocking=False):
                        db.execute("UPDATE jobs SET status='pending',worker=NULL,updated=? WHERE id=?",
                                   (time.time(), row["id"]))
                        self.event(db, row["id"], "recovered")
                        recovered.append(row["id"])
                except ValueError:
                    continue
        return recovered

    def claim(self, device):
        self.recover()
        model = gpu_type(device["gpu_type"])
        with self.connect() as db:
            # Serialize discovery/collection for a pair; published warmups can be used by many cards.
            busy = {pair_key(j["game"], j["seed"]) for row in db.execute(
                "SELECT payload FROM jobs WHERE status='running'") for j in [json.loads(row[0])] if not j.get("plan")}
            for row in db.execute("SELECT * FROM jobs WHERE status='pending' AND gpu_type=? "
                                  "ORDER BY priority DESC,seq", (model,)).fetchall():
                job = json.loads(row["payload"])
                if pair_key(job["game"], job["seed"]) in busy:
                    continue
                path = self.lease_path(job["id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                lease = path.open("a")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                try:
                    db.execute("UPDATE jobs SET status='running',worker=?,attempts=attempts+1,updated=? WHERE id=?",
                               (json.dumps(device), time.time(), job["id"]))
                    self.event(db, job["id"], "started", **device)
                except BaseException:
                    lease.close()
                    raise
                return job, lease
        return None, None

    def update_job(self, job):
        with self.connect() as db:
            db.execute("UPDATE jobs SET payload=?,updated=? WHERE id=?", (json_line(job), time.time(), job["id"]))

    def finish(self, job_id, status, returncode=None):
        if status not in {"completed", "failed", "pending"}:
            raise ValueError(status)
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,worker=NULL,updated=? WHERE id=?", (status, time.time(), job_id))
            event = status
            if status == "pending":
                event = "yielded" if returncode == 0 else "interrupted"
            self.event(db, job_id, event, returncode=returncode)

    def reprioritize(self, job_id, priority):
        with self.connect() as db:
            count = db.execute("UPDATE jobs SET priority=? WHERE id=? AND status='pending'", (priority, job_id)).rowcount
            if not count:
                raise ValueError("No pending job with that ID")
            self.event(db, job_id, "priority", priority=priority)

    def retry(self, job_id):
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET status='pending',worker=NULL,updated=? WHERE id=? AND status='failed'",
                                 (time.time(), job_id)).rowcount
            if not changed:
                raise ValueError("No failed job with that ID")
            self.event(db, job_id, "retried")

    def migrate(self, model):
        """Idempotent import; keep legacy inputs and a backup, never rewrite checkpoints/seeds."""
        model = gpu_type(model)
        with locked(self.directory / "queue.lock"):
            with self.connect() as db:
                if db.execute("SELECT 1 FROM metadata WHERE key='legacy_migrated'").fetchone():
                    return 0
            # Old workers retain these locks for their lifetime.
            from contextlib import ExitStack
            with ExitStack() as stack:
                for path in self.directory.glob("worker_gpu_*.lock"):
                    stack.enter_context(locked(path, blocking=False))
                old_files = list(self.seed_directory.glob("gpu_*.json")) + list(self.directory.glob("gpu_*.jsonl"))
                if not old_files:
                    return 0
                backup = self.directory / "legacy_backup"
                backup.mkdir(exist_ok=True)
                for path in old_files + [self.directory / "assignments.json", self.directory / "history.jsonl"]:
                    if path.exists() and not (backup / path.name).exists():
                        shutil.copy2(path, backup / path.name)
                selections = read_seed_files(self.seed_directory)
                for path in sorted(self.seed_directory.glob("gpu_*.json")):
                    for entry in json.loads(path.read_text())["game_seeds"]:
                        key = pair_key(entry["game"], entry["seed"])
                        selections.setdefault(key, {k: v for k, v in dict(entry, gpu_type=model).items() if k != "fresh"})
                self.register_selections(selections)
                save_seed_files(selections, self.seed_directory)
                specs = []
                for path in sorted(self.directory.glob("gpu_*.jsonl")):
                    for line in path.read_text().splitlines():
                        if line.strip():
                            job = json.loads(line)
                            job.pop("gpu", None)
                            job["gpu_type"] = model
                            job["config_text"] = Path(job["config"]).read_text()
                            job.pop("resume_warmup", None)  # Re-resolve against the shared catalog.
                            specs.append(job)
                # Recover jobs removed by a stopped legacy worker, too.
                history = self.directory / "history.jsonl"
                latest = {}
                if history.exists():
                    for line in history.read_text().splitlines():
                        if line.strip():
                            event = json.loads(line)
                            latest[event["job"]["id"]] = event
                    for event in latest.values():
                        if event["event"] in {"started", "interrupted"}:
                            job = dict(event["job"], gpu_type=model)
                            job.pop("gpu", None)
                            job.pop("resume_warmup", None)
                            job["config_text"] = Path(job["config"]).read_text()
                            specs.append(job)
                # The new runner handles one warmup/offset target per job.
                # Expand legacy sweeps so migration does not drop any requested target.
                from enqueue_train import build_specs
                expanded = []
                for old in specs:
                    variants = build_specs([[old["game"], old["seed"]]], model, Path(old["config"]),
                                           old["overrides"], old.get("priority", 0))
                    for i, variant in enumerate(variants):
                        identity = old["id"] if len(variants) == 1 else f"{old['id']}-{i}"
                        with self.connect() as db:
                            if db.execute("SELECT 1 FROM jobs WHERE id=?", (identity,)).fetchone():
                                continue
                        variant.update(id=identity, legacy_id=old["id"])
                        if old.get("name"):
                            variant["name"] = old["name"] + (f"-target{i}" if len(variants) > 1 else "")
                        # Preserve the original interpreter for legacy queued experiments.
                        variant["python"] = old.get("python", sys.executable)
                        expanded.append(variant)
                added = []
                for variant in expanded:
                    imported, _ = self.enqueue_jobs([variant], python=variant["python"])
                    added.extend(imported)
                with self.connect() as db:
                    db.execute("INSERT OR REPLACE INTO metadata VALUES('legacy_migrated',?)", (model,))
                return len(added)


def run_worker(index, *, once=False, max_jobs=0):
    device = detect_device(index)
    queue = TrainingQueue()
    if list(queue.seed_directory.glob("gpu_*.json")) or (queue.directory / "legacy_backup").exists():
        with queue.connect() as db:
            if not db.execute("SELECT 1 FROM metadata WHERE key='legacy_migrated'").fetchone():
                raise ValueError("Complete legacy migration before starting workers: queue_train.py migrate --gpu-type MODEL")
    worker_key = hashlib.sha256(f"{device['host']}:{device['uuid']}".encode()).hexdigest()[:24]
    with locked(queue.directory / f"worker_{worker_key}.lock", blocking=False) as guard:
        print(f"GPU {index} ({device['gpu_type']}): shared priority queue {queue.database}", flush=True)
        completed, child, job, lease = 0, None, None, None
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(signum)
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while True:
                job, lease = queue.claim(device)
                if job is None:
                    if once:
                        return 0
                    time.sleep(5)
                    continue
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device["uuid"],
                       "STORM_GPU_TYPE": device["gpu_type"],
                       "STORM_QUEUE_FDS": f"{guard.fileno()},{lease.fileno()}"}
                # The runner keeps the lease through collection and every branch, even if this worker dies.
                command = [job["python"], "-u", str(ROOT / "queue_job.py"),
                           "--queue-dir", str(queue.directory), "--job-id", job["id"]]
                if once:
                    command.append("--once")
                child = subprocess.Popen(command,
                                         cwd=ROOT, env=env, start_new_session=True,
                                         pass_fds=(guard.fileno(), lease.fileno()))
                result = child.wait()
                child = None
                # Runner normally records completion; a hard exit still returns the job to the queue.
                with queue.connect() as db:
                    status = db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
                if status == "running":
                    queue.finish(job["id"], "pending" if result < 0 else "failed", result)
                lease.close()
                lease, job = None, None
                if once and status == "completed" and result == NO_BRANCH_NEEDED:
                    continue
                completed += 1
                if once or (max_jobs and completed >= max_jobs):
                    return result if result >= 0 else 128 - result
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
                    child.wait(timeout=max(0, float(os.environ.get("STORM_QUEUE_STOP_GRACE", "120"))) + 10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if job is not None:
                queue.finish(job["id"], "pending", 128 + signum)
            return 128 + signum
        finally:
            if lease is not None:
                lease.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status")
    status.add_argument("--gpu-type")
    status.add_argument("--all", action="store_true")
    worker = sub.add_parser("worker")
    worker.add_argument("gpu")
    limits = worker.add_mutually_exclusive_group()
    limits.add_argument("--once", action="store_true", help="Run one unfinished training configuration and exit")
    limits.add_argument("--max-jobs", type=int, default=0, help="Process up to N whole queue jobs", metavar="N")
    migrate = sub.add_parser("migrate")
    migrate.add_argument("--gpu-type", required=True)
    priority = sub.add_parser("priority")
    priority.add_argument("job_id")
    priority.add_argument("value", type=int)
    retry = sub.add_parser("retry")
    retry.add_argument("job_id")
    seeds = sub.add_parser("seeds")
    seeds.add_argument("--gpu-type")
    args = parser.parse_args(argv)
    try:
        if args.action == "worker":
            if args.max_jobs < 0:
                raise ValueError("--max-jobs must be positive")
            return run_worker(args.gpu, once=args.once, max_jobs=args.max_jobs)
        queue = TrainingQueue()
        if args.action == "migrate":
            print(f"Imported {queue.migrate(args.gpu_type)} pending jobs; legacy files backed up.")
        elif args.action == "priority":
            queue.reprioritize(args.job_id, args.value)
        elif args.action == "retry":
            queue.retry(args.job_id)
        elif args.action == "seeds":
            with queue.connect() as db:
                for row in db.execute("SELECT seed,gpu_type FROM seeds ORDER BY seed"):
                    if not args.gpu_type or row[1] == gpu_type(args.gpu_type):
                        print(f"{row[0]} -> {row[1]}")
        else:
            recovered = queue.recover()
            if recovered:
                print(f"Recovered {len(recovered)} abandoned job(s).")
            jobs = queue.jobs(args.gpu_type, None if args.all else ("pending", "running"))
            for job in jobs:
                print(f"{job['id']} {job['gpu_type']} priority={job['priority']} {job['status']} "
                      f"{job['game']}/{job['seed']} attempts={job['attempts']}")
            print(f"{len(jobs)} job(s).")
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Queue error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
