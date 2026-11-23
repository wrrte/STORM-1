"""Inspect shared warmups by game/model/seed and exchange metadata on a Git branch.

The inventory branch contains JSON only. Local queues, checkpoint payloads, and
the source checkout's index/branch are never part of the synchronization.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import zipfile

from queue_train import ROOT, QUEUE_DIR, PAIR_PREFIX, SEED_RANGES, gpu_type, locked, write_atomic

DEFAULT_CACHE = QUEUE_DIR / "shared_warmups"
DEFAULT_MODELS = ["A6000", "3090", "TITAN_RTX"]
DEFAULT_BRANCH = "warmup-inventory"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid round/machine identifier: {value!r}")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def round_dir(cache, round_id):
    return Path(cache) / "rounds" / identifier(round_id)


def load_manifest(cache, round_id):
    path = round_dir(cache, round_id) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Round {round_id!r} not found. Run start here, or sync/pull from Git first.")
    data = json.loads(path.read_text())
    if (data["version"] != 1 or data["round"] != round_id or
            data["profile_sha256"] != digest(data["profile"]) or
            not data["gpu_types"] or data["gpu_types"] != sorted(set(map(gpu_type, data["gpu_types"])))):
        raise ValueError(f"Invalid round manifest: {path}")
    return data


def start_round(cache, round_id, models, config, overrides):
    from enqueue_train import build_specs, validate_overrides
    validate_overrides(config, overrides)
    specs = build_specs([["Alien", 0]], "A6000", config, overrides, 0)
    profile = dict(config_text=specs[0]["config_text"], requests=[s["overrides"] for s in specs])
    manifest = dict(version=1, round=identifier(round_id), created_at=timestamp(),
                    gpu_types=sorted(set(map(gpu_type, models))), profile=profile,
                    profile_sha256=digest(profile))
    path = round_dir(cache, round_id) / "manifest.json"
    if path.exists():
        existing = load_manifest(cache, round_id)
        if existing["profile"] != profile or existing["gpu_types"] != manifest["gpu_types"]:
            raise ValueError("Round already exists with a different profile/model list. Use a new round name.")
        return existing
    save(path, manifest)
    return manifest


def scan(cache, round_id, model, machine=None, *, root=ROOT):
    """Inspect local payloads and assign them to this machine's supplied GPU model."""
    from warmup_catalog import warmup_sources, inspect_warmup, source_stamp, compatibility
    model, machine = gpu_type(model), identifier(machine or socket.gethostname())
    manifest = load_manifest(cache, round_id)
    if model not in manifest["gpu_types"]:
        raise ValueError(f"{model} is not a participant in round {round_id}")
    games, excluded = {}, []
    for source in warmup_sources(root, include_unpublished=True):
        name = str(source.relative_to(root))
        try:
            before = source_stamp(source)
            record = inspect_warmup(source, root=root, queue_dir=root / "training_queue")
            # Checkpoints stay on the machine that generated them. Its supplied
            # GPU model is authoritative; historical hardware logs are optional.
            record["gpu_type"] = model
            checks = []
            for i, overrides in enumerate(manifest["profile"]["requests"]):
                job = dict(game=record["game"], seed=record["seed"], gpu_type=model,
                           config_text=manifest["profile"]["config_text"], overrides=overrides)
                reason = compatibility(record, job)
                checks.append(dict(request=i, compatible=reason is None, reason=reason))
            if source_stamp(source) != before:
                raise ValueError("Checkpoint changed during inspection; scan again after publication")
            item = dict(source=name, step=record["step"], warmup_steps=record["warmup_steps"],
                        statistics_signals=record["statistics_signals"],
                        saved_config_sha256=hashlib.sha256(record["config"].encode()).hexdigest(),
                        provenance=dict(kind="machine_location", machine=machine, gpu_type=model), checks=checks)
            seeds = games.setdefault(record["game"], {}).setdefault(model, {})
            seeds.setdefault(str(record["seed"]), []).append(item)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, AssertionError,
                zipfile.BadZipFile) as error:
            excluded.append(dict(source=name, reason=f"{type(error).__name__}: {error}"))
    commit = git(root, "rev-parse", "HEAD", check=False)
    report = dict(version=1, round=round_id, profile_sha256=manifest["profile_sha256"],
                  gpu_type=model, machine=machine, gpu_assignment="machine_location", scanned_at=timestamp(),
                  code_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                  games=games, excluded=excluded)
    path = round_dir(cache, round_id) / "machines" / model / f"{machine}.json"
    save(path, report)
    return path, report


def aggregate(cache, round_id):
    manifest = load_manifest(cache, round_id)
    publication_path = Path(cache) / "published" / f"{identifier(round_id)}.json"
    published = json.loads(publication_path.read_text()) if publication_path.exists() else {}
    games, owners, scans, excluded = {}, {}, [], []
    for path in sorted((round_dir(cache, round_id) / "machines").glob("*/*.json")):
        report = json.loads(path.read_text())
        model = gpu_type(report["gpu_type"])
        if (report["version"] != 1 or report["round"] != round_id or
                report["profile_sha256"] != manifest["profile_sha256"] or
                model not in manifest["gpu_types"] or path.parent.name != model or
                path.stem != identifier(report["machine"])):
            raise ValueError(f"Report does not match this round/profile: {path}")
        report_key = str(path.relative_to(round_dir(cache, round_id)))
        scans.append(dict({k: report[k] for k in ("machine", "gpu_type", "scanned_at")},
                          published=published.get(report_key) == digest(report)))
        excluded.extend(dict(item, machine=report["machine"], gpu_type=model) for item in report["excluded"])
        for game, models in report["games"].items():
            if set(models) != {model}:
                raise ValueError(f"Report has inconsistent model ownership: {path}")
            for seed, sources in models[model].items():
                from queue_train import pair_key
                pair_key(game, int(seed))
                owners.setdefault(seed, set()).add(model)
                entry = games.setdefault(game, {}).setdefault(model, {}).setdefault(seed, dict(snapshots=[]))
                entry["snapshots"].extend(dict(source, machine=report["machine"], scanned_at=report["scanned_at"])
                                          for source in sources)
    conflicts = []
    for seed, models in sorted(owners.items(), key=lambda pair: int(pair[0])):
        reserved = next((kind for kind, (start, end) in SEED_RANGES.items() if start <= int(seed) <= end), None)
        if len(models) > 1 or (reserved and models != {reserved}):
            conflicts.append(dict(seed=int(seed), gpu_types=sorted(models), reserved_for=reserved))
    conflicted = {str(item["seed"]) for item in conflicts}
    counts = {model: dict(valid_pairs=0, usable_pairs=0, snapshots=0) for model in manifest["gpu_types"]}
    for models in games.values():
        for model, seeds in models.items():
            for seed, entry in seeds.items():
                entry["seed_conflict"] = seed in conflicted
                entry["compatible_requests"] = sorted({c["request"] for s in entry["snapshots"]
                                                        for c in s["checks"] if c["compatible"]})
                entry["usable"] = bool(entry["compatible_requests"]) and not entry["seed_conflict"]
                counts[model]["valid_pairs"] += 1
                counts[model]["usable_pairs"] += int(entry["usable"])
                counts[model]["snapshots"] += len(entry["snapshots"])
    pending = sorted(set(manifest["gpu_types"]) - {s["gpu_type"] for s in scans})
    return dict(round=round_id, profile_sha256=manifest["profile_sha256"],
                collection_complete=not pending, git_complete=not pending and all(s["published"] for s in scans),
                pending_gpu_types=pending, scans=scans,
                counts=counts, seed_conflicts=conflicts, games=games, excluded=excluded)


def render(report):
    state = "complete" if report["collection_complete"] else "pending: " + ", ".join(report["pending_gpu_types"])
    lines = [f"Round {report['round']} — collection {state}",
             "Git round: " + ("complete" if report["git_complete"] else "incomplete (missing scans or unpublished results)"),
             "Availability is as of each scan; usable means compatible with at least one frozen request."]
    for scan_info in report["scans"]:
        publication = "published" if scan_info["published"] else "local only"
        lines.append(f"  {scan_info['gpu_type']} @ {scan_info['machine']}: {scan_info['scanned_at']} ({publication})")
    for model, count in report["counts"].items():
        if model in report["pending_gpu_types"]:
            lines.append(f"{model}: not scanned in this round")
            continue
        lines.append(f"{model}: {count['usable_pairs']} usable / {count['valid_pairs']} valid pairs, "
                     f"{count['snapshots']} snapshots")
    lines.append("\nGame | GPU type | Usable seeds | Other valid seeds")
    for game, models in sorted(report["games"].items()):
        for model, seeds in sorted(models.items()):
            usable = [seed for seed in sorted(seeds, key=int) if seeds[seed]["usable"]]
            other = [seed for seed in sorted(seeds, key=int) if not seeds[seed]["usable"]]
            lines.append(f"{game} | {model} | {', '.join(usable) or '-'} | {', '.join(other) or '-'}")
            for seed in other:
                entry = seeds[seed]
                reasons = {c["reason"] for s in entry["snapshots"] for c in s["checks"] if c["reason"]}
                if entry["seed_conflict"]:
                    reasons.add("seed belongs to multiple GPU types or another type's reserved range")
                lines.append(f"  {seed}: {'; '.join(sorted(reasons))}")
    for conflict in report["seed_conflicts"]:
        lines.append(f"SEED CONFLICT {conflict['seed']}: {', '.join(conflict['gpu_types'])}"
                     + (f"; reserved for {conflict['reserved_for']}" if conflict["reserved_for"] else ""))
    for item in report["excluded"]:
        lines.append(f"EXCLUDED {item['machine']} {item['source']}: {item['reason']}")
    return "\n".join(lines)


def git(directory, *args, check=True):
    result = subprocess.run(["git", "-C", str(directory), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        # Git may echo a credential-bearing URL; do not include its stderr in errors.
        raise ValueError(f"Git {args[0]} failed (exit {result.returncode}); check remote access, branch, and Git identity.")
    return result


@contextmanager
def remote_checkout(root, remote, branch):
    """A disposable repository keeps source edits, index, and HEAD untouched."""
    if git(root, "check-ref-format", "--branch", branch, check=False).returncode:
        raise ValueError("Invalid inventory branch")
    if branch in {"main", "master"}:
        raise ValueError("Use a dedicated inventory branch, not main/master")
    url = git(root, "remote", "get-url", remote).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="storm-inventory-") as temporary:
        repo = Path(temporary)
        git(repo, "init", "--quiet")
        git(repo, "remote", "add", "origin", url)
        for key in ("user.name", "user.email"):
            value = git(root, "config", "--get", key, check=False).stdout.strip()
            if value:
                git(repo, "config", key, value)
        git(repo, "config", "core.hooksPath", "/dev/null")
        found = git(repo, "ls-remote", "--heads", "origin", f"refs/heads/{branch}").stdout.strip()
        if found:
            git(repo, "fetch", "--quiet", "--depth=1", "origin", f"refs/heads/{branch}")
            git(repo, "checkout", "--quiet", "-B", branch, "FETCH_HEAD")
            if not (repo / "inventory.json").is_file() or json.loads((repo / "inventory.json").read_text()) != {"version": 1, "kind": "storm-shared-warmups"}:
                raise ValueError("Remote branch is not a STORM inventory branch")
        else:
            git(repo, "checkout", "--quiet", "--orphan", branch)
            save(repo / "inventory.json", dict(version=1, kind="storm-shared-warmups"))
        yield repo, bool(found)


def import_round(remote_repo, cache, round_id):
    directory = round_dir(remote_repo, round_id)
    if not directory.is_dir():
        return False
    incoming = load_manifest(remote_repo, round_id)
    local = round_dir(cache, round_id)
    if (local / "manifest.json").exists() and load_manifest(cache, round_id) != incoming:
        raise ValueError("Local and remote round manifests differ; choose a new round name")
    aggregate(remote_repo, round_id)  # Validate all incoming reports before writing any.
    save(local / "manifest.json", incoming)
    published = {}
    for path in sorted((directory / "machines").glob("*/*.json")):
        destination = local / path.relative_to(directory)
        incoming_report = json.loads(path.read_text())
        published[str(path.relative_to(directory))] = digest(incoming_report)
        if destination.exists():
            current = json.loads(destination.read_text())
            if current["scanned_at"] > incoming_report["scanned_at"]:
                continue  # Preserve a newer local scan after an interrupted/offline push.
        save(destination, incoming_report)
    save(Path(cache) / "published" / f"{round_id}.json", published)
    return True


def synchronize(cache, round_id, model=None, machine=None, *, root=ROOT,
                remote="origin", branch=DEFAULT_BRANCH, publish=True):
    """Fetch -> inspect -> commit only this machine -> push; retry concurrent writers."""
    with locked(Path(cache) / "sync.lock"):
        own_path = own_report = manifest = None
        for attempt in range(3):
            with remote_checkout(root, remote, branch) as (repo, exists):
                imported = import_round(repo, cache, round_id) if exists else False
                if not publish:
                    if not imported:
                        raise ValueError("This round has not been published on the inventory branch")
                    return aggregate(cache, round_id)
                manifest = load_manifest(cache, round_id)
                if own_report is None:
                    own_path, own_report = scan(cache, round_id, model, machine, root=root)
                # An identical round name cannot silently merge incompatible profiles.
                remote_manifest = round_dir(repo, round_id) / "manifest.json"
                save(remote_manifest, manifest)
                destination = repo / own_path.relative_to(cache)
                if destination.exists() and json.loads(destination.read_text())["scanned_at"] > own_report["scanned_at"]:
                    raise ValueError("A newer scan for this machine was published concurrently; scan/sync again")
                save(destination, own_report)
                git(repo, "add", "--", "inventory.json", str(remote_manifest.relative_to(repo)),
                    str(destination.relative_to(repo)))
                changed = git(repo, "diff", "--cached", "--quiet", check=False).returncode
                if changed:
                    git(repo, "commit", "--quiet", "-m", f"Inventory {round_id}: {own_report['gpu_type']} / {own_report['machine']}")
                result = git(repo, "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}", check=False)
                if result.returncode == 0:
                    import_round(repo, cache, round_id)
                    return aggregate(cache, round_id)
        raise ValueError("Inventory push failed after 3 attempts. Local scan is saved; check Git access and retry sync.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="Freeze a new collection round and compatibility profile")
    start.add_argument("--gpu-types", nargs="+", default=DEFAULT_MODELS)
    start.add_argument("--config", type=Path, default=ROOT / "config_files/STORM.yaml")
    start.add_argument("--warmup-offset-pairs", type=json.loads, default=[])
    start.add_argument("--hash-target-pairs", type=json.loads, default=[])
    start.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    for command in (start, commands.add_parser("scan"), commands.add_parser("sync"),
                    commands.add_parser("pull"), commands.add_parser("report")):
        command.add_argument("--round", required=True)
        if command.prog.split()[-1] in {"scan", "sync"}:
            command.add_argument("--gpu-type", required=True,
                                 help="GPU model of this machine; assigns all local checkpoints to that model")
            command.add_argument("--machine", help="Stable machine ID; defaults to hostname")
        if command.prog.split()[-1] in {"sync", "pull"}:
            command.add_argument("--remote", default="origin")
            command.add_argument("--branch", default=DEFAULT_BRANCH)
        command.add_argument("--json", action="store_true", help="Print full game/model/seed details")
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            import yaml
            overrides = {PAIR_PREFIX + "warmup_offset_pairs": args.warmup_offset_pairs,
                         PAIR_PREFIX + "hash_target_pairs": args.hash_target_pairs}
            for key, value in args.set:
                if key in overrides or key == "BasicSettings.Seed":
                    raise ValueError(f"Use the dedicated option for {key}")
                overrides[key] = yaml.safe_load(value)
            start_round(args.cache_dir, args.round, args.gpu_types, args.config, overrides)
        elif args.command == "scan":
            scan(args.cache_dir, args.round, args.gpu_type, args.machine)
        elif args.command in {"sync", "pull"}:
            report = synchronize(args.cache_dir, args.round, getattr(args, "gpu_type", None),
                                 getattr(args, "machine", None), remote=args.remote, branch=args.branch,
                                 publish=args.command == "sync")
            print(json.dumps(report, indent=2) if args.json else render(report))
            return 0
        report = aggregate(args.cache_dir, args.round)
        print(json.dumps(report, indent=2) if args.json else render(report))
        return 0
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
