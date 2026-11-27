"""Preview or apply one machine's portion of a shared warmup prune plan."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3

from cleanup_warmups import queue_guard, remove_references, safe_source
from collect_warmup_scores import TARGET_STEP, inventory, path_key
from queue_train import ROOT, gpu_type, locked, pair_key
from warmup_catalog import imported_path, inspect_warmup, source_stamp
from warmup_registry import digest, identifier, save, timestamp


ADAPTER_FILES = ("config.yaml", "training_state.pt", "replay_buffer.npz", "warmup_metadata.json")


def load_plan(path, model, machine):
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan_id = plan.pop("plan_id")
    if (plan.get("kind"), plan.get("version"), plan.get("target_step")) != ("storm-warmup-prune", 1, TARGET_STEP):
        raise ValueError("Unsupported warmup prune plan")
    if digest(plan) != plan_id:
        raise ValueError("Plan checksum does not match")
    if machine not in plan["machines"].get(model, {}):
        raise ValueError(f"Plan contains no entry for {model} @ {machine}")
    batch = plan["machines"][model][machine]
    names = set()
    for item in batch["keep"] + batch["delete"]:
        path = Path(item["source"])
        if (len(path.parts) != 3 or path.parts[0] != "ckpt" or ".." in path.parts or
                str(path) != item["source"] or path.name != f"shared_warmup_{item['step']}" or
                type(item["step"]) is not int or item["step"] < TARGET_STEP or item["targets"] != [TARGET_STEP]):
            raise ValueError(f"Invalid or out-of-scope snapshot in plan: {path}")
        pair_key(item["game"], item["seed"])
        if item["source"] in names:
            raise ValueError(f"Duplicate snapshot in plan: {path}")
        names.add(item["source"])
    return plan_id, batch


def checked_stamp(root, item):
    source = safe_source(root, item["source"])
    before = source_stamp(source)
    scanned = datetime.fromisoformat(item["scanned_at"])
    if scanned.utcoffset() is None:
        raise ValueError("Inventory scan time must specify its timezone")
    elapsed = scanned - datetime(1970, 1, 1, tzinfo=timezone.utc)
    scan_ns = ((elapsed.days * 86_400 + elapsed.seconds) * 10**6 + elapsed.microseconds) * 1000
    if any(mtime > scan_ns for _, _, mtime in before):
        raise ValueError(f"Snapshot changed since the inventory scan: {item['source']}")
    record = inspect_warmup(source, root=root, queue_dir=root / "training_queue")
    if ((record["game"], record["seed"], record["step"], record["warmup_steps"]) !=
            (item["game"], item["seed"], item["step"], item["targets"]) or
            hashlib.sha256(record["config"].encode()).hexdigest() != item["config_sha256"]):
        raise ValueError(f"Snapshot identity or saved configuration differs from the plan: {item['source']}")
    if source_stamp(source) != before:
        raise ValueError(f"Snapshot changed during inspection: {item['source']}")
    return before


def pinned_jobs(root, item, jobs):
    source = root / item["source"]
    alias = imported_path(source, root / "training_queue")
    keys = {path_key(str(path), "ckpt", "training_queue") for path in (source, source.parent, alias)}
    matched = []
    for job in jobs:
        if (job.get("game"), job.get("seed")) == (item["game"], item["seed"]):
            matched.append(job["id"])
            continue
        references = [job.get("warmup_source"), job.get("resume_warmup")]
        for command in (job.get("plan") or {}).values():
            for i, token in enumerate(command):
                flag, separator, value = token.partition("=")
                if flag in ("--resume_from", "--resume_warmup"):
                    references.append(value if separator else command[i + 1])
        if any(path_key(path, "ckpt", "training_queue") in keys for path in references if path):
            matched.append(job["id"])
    return matched


def locator_updates(root, source):
    """Prepare updates to the parent locator and sweep index for this exact snapshot."""
    parent = source.parent
    updates = {}
    locator = parent / "warmup.json"
    if locator.is_symlink():
        raise ValueError(f"Symlinked warmup locator: {locator}")
    if locator.exists():
        target = json.loads(locator.read_text(encoding="utf-8"))["checkpoint"]
        if (parent / target).resolve() == source:
            updates[locator] = None
    index = parent / "warmups.json"
    if index.is_symlink():
        raise ValueError(f"Symlinked sweep index: {index}")
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        removed = {key for key, record in data["checkpoints"].items()
                   if (parent / record["checkpoint"]).resolve() == source}
        if removed:
            data["checkpoints"] = {key: record for key, record in data["checkpoints"].items() if key not in removed}
            data["pairs"] = [pair for pair in data["pairs"] if str(pair[0]) not in removed]
            data["complete"] = bool(data["pairs"]) and all(str(pair[0]) in data["checkpoints"] for pair in data["pairs"])
            updates[index] = data
    alias = imported_path(source, root / "training_queue")
    if alias.resolve() != alias:
        raise ValueError(f"Symlinked imported warmup directory: {alias}")
    if alias.exists():
        files = {file.name for file in alias.iterdir()}
        metadata = alias / "warmup_metadata.json"
        if files:
            value = json.loads(metadata.read_text(encoding="utf-8"))["imported_from"]
            if (files - set(ADAPTER_FILES) or (root / value).resolve() != source or any(
                    (alias / name).exists() and not (alias / name).is_symlink()
                    for name in ("training_state.pt", "replay_buffer.npz"))):
                raise ValueError(f"Imported warmup is not an adapter to {source}")
        updates[alias] = None
    return updates


def apply_locator_updates(updates):
    for path, data in updates.items():
        if data is not None:
            save(path, data)
        elif path.is_dir():
            for name in ADAPTER_FILES:
                (path / name).unlink(missing_ok=True)
            path.rmdir()
        else:
            path.unlink(missing_ok=True)


def preflight(root, batch, plan_id, audit, jobs):
    expected = {item["source"] for item in batch["keep"] + batch["delete"]}
    observed, _, _ = inventory(root)
    if extra := set(observed) - expected:
        raise ValueError(f"New local 50k warmups outside this plan: {', '.join(sorted(extra))}")
    for item in batch["keep"]:
        checked_stamp(root, item)

    operations, issues = [], []
    for item in batch["delete"]:
        name = item["source"]
        source = root / name
        retired = source.with_name(f"{source.name}.{plan_id[:12]}.retired")
        if source.resolve() != source or retired.resolve() != retired or (source.exists() and retired.exists()):
            raise ValueError(f"Conflicting or symlinked checkpoint paths: {name}")
        if name in audit["deleted"]:
            if source.exists() or retired.exists():
                raise ValueError(f"A deleted checkpoint reappeared: {name}")
            continue
        if protected := pinned_jobs(root, item, jobs):
            issues.append(f"{name} is used by queue jobs {', '.join(protected)}")
            continue
        if source.exists():
            stamp = checked_stamp(root, item)
        elif name not in audit["started"]:
            raise ValueError(f"Planned deletion missing before the first attempt: {name}")
        else:
            stamp = None
        locator_updates(root, source)
        directory = source if source.exists() else retired
        size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
        operations.append((name, source, retired, stamp, size))
    return operations, issues


def apply_deletions(root, operations, audit_path, audit, *, seed_file=None):
    save(audit_path, audit)
    for name, source, retired, stamp, _ in operations:
        if name not in audit["started"]:
            audit["started"].append(name)
            save(audit_path, audit)
        if source.exists():
            if source_stamp(safe_source(root, name)) != stamp:
                raise ValueError(f"Snapshot changed before deletion: {name}")
            os.replace(source, retired)
        if retired.exists():
            shutil.rmtree(retired)
        apply_locator_updates(locator_updates(root, source))
        remove_references(root, [name], seed_file=seed_file)
        audit["deleted"].append(name)
        save(audit_path, audit)


def execute(plan_path, model, machine, *, apply=False, root=ROOT):
    root = Path(root).resolve()
    model, machine = gpu_type(model), identifier(machine)
    plan_id, batch = load_plan(plan_path, model, machine)
    queue = root / "training_queue"
    audit_path = queue / "warmup_prune_audit" / f"{plan_id}-{model}-{machine}.json"
    with locked(queue / "shared_warmups/sync.lock"), locked(queue / "catalog.lock"), queue_guard(root) as jobs:
        audit = dict(plan_id=plan_id, gpu_type=model, machine=machine, created_at=timestamp(), started=[], deleted=[])
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected = {item["source"] for item in batch["delete"]}
            if ((audit["plan_id"], audit["gpu_type"], audit["machine"]) != (plan_id, model, machine) or
                    not set(audit["deleted"]) <= set(audit["started"]) <= expected):
                raise ValueError("Deletion journal does not match the plan")
        operations, issues = preflight(root, batch, plan_id, audit, jobs)
        print(f"{model} @ {machine} · keep={len(batch['keep'])} · pending={len(operations)} · "
              f"already deleted={len(audit['deleted'])} · {sum(op[-1] for op in operations) / 2**30:.2f} GiB")
        for name, *_ in operations:
            print(f"DELETE {name}")
        if issues:
            raise ValueError("\n".join(issues))
        if not apply:
            return audit
        apply_deletions(root, operations, audit_path, audit)
    print(f"Deleted {len(audit['deleted'])} planned snapshots in total. Audit: {audit_path}")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--gpu-type", required=True)
    parser.add_argument("--machine", default=socket.gethostname(), help="Machine ID in the plan (default: hostname)")
    parser.add_argument("--apply", action="store_true", help="Permanently delete the planned snapshots")
    args = parser.parse_args(argv)
    try:
        execute(args.plan, args.gpu_type, args.machine, apply=args.apply)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
