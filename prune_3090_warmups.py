"""Preview or delete local RTX 3090 warmups outside the final GPU allocation."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3

from apply_warmup_prune import apply_deletions, locator_updates
from cleanup_warmups import protected, queue_guard, safe_source
from queue_train import ROOT, locked, pair_key
from warmup_catalog import source_stamp, warmup_sources


ALLOCATION = ROOT / "results/warmup_gpu_assignment_kangaroo_3090_20261126.xlsx"
MODEL = "3090"


def read_allocation(path):
    from openpyxl import load_workbook

    data = path.read_bytes()
    workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
    keep, fresh, pairs = {}, set(), set()
    try:
        rows = workbook["시드와 체크포인트"].iter_rows(max_col=5, values_only=True)
        if next(rows)[:4] != ("GPU", "게임", "시드", "처리"):
            raise ValueError("Unexpected headers in the allocation workbook")
        for model, game, seed, action, source in rows:
            if str(model) != MODEL:
                continue
            pair = pair_key(game, seed)
            if pair in pairs:
                raise ValueError(f"Duplicate RTX 3090 allocation: {pair}")
            pairs.add(pair)
            if action == "재사용" and isinstance(source, str) and source and source not in keep:
                keep[source] = pair
            elif action == "신규 수집" and source is None:
                fresh.add(pair)
            else:
                raise ValueError(f"Invalid RTX 3090 allocation: {pair}")
    finally:
        workbook.close()
    if not keep:
        raise ValueError("The workbook has no RTX 3090 warmups to retain")
    return keep, fresh, hashlib.sha256(data).hexdigest()


def preflight(root, keep, fresh, jobs, plan_id, audit):
    inventory = json.loads((root / "training_queue/warmup_inventory.json").read_text())
    records = {record["source"]: record for record in inventory["available"] if record["gpu_type"] == MODEL}
    for name, pair in keep.items():
        path = safe_source(root, name)
        record = records.get(name)
        if (not record or pair_key(record["game"], record["seed"]) != pair
                or record.get("stamp") != [list(item) for item in source_stamp(path)]):
            raise ValueError(f"Retained warmup differs from the inventory: {name}")

    published = {path.relative_to(root).as_posix() for path in warmup_sources(root)}
    started, deleted = set(audit["started"]), set(audit["deleted"])
    collector_dirs = {f"{job['name']}_collector_Shared" for job in jobs}
    operations, skipped = [], []
    for name in sorted((published | started) - keep.keys()):
        source = root / name
        retired = source.with_name(f"{source.name}.{plan_id[:12]}.retired")
        if (source.parent.parent != root / "ckpt" or not re.fullmatch(r"shared_warmup_\d+", source.name)
                or source.resolve() != source or retired.resolve() != retired):
            if name in started:
                raise ValueError(f"Invalid snapshot path in the deletion journal: {name}")
            skipped.append((name, "outside the local snapshot layout"))
            continue
        if name in deleted:
            if source.exists() or retired.exists():
                raise ValueError(f"A deleted checkpoint reappeared: {name}")
            continue
        if retired.exists() and (name not in started or source.exists() or not retired.is_dir()):
            raise ValueError(f"Conflicting retired checkpoint: {retired}")
        record = records.get(name)
        stamp = None
        if source.exists():
            safe_source(root, name)
            if source.parent.name in collector_dirs:
                skipped.append((name, "belongs to a queued collector"))
                continue
            if not record:
                skipped.append((name, "no verified RTX 3090 inventory record"))
                continue
            if pair_key(record["game"], record["seed"]) in fresh:
                skipped.append((name, "new warmup in the final allocation"))
                continue
            stamp = source_stamp(source)
            if record.get("stamp") != [list(item) for item in stamp]:
                skipped.append((name, "snapshot changed since inventory inspection"))
                continue
        if reason := protected(root, name, record, jobs):
            if not source.exists():
                raise ValueError(f"Retired warmup is referenced by a queue job: {name} ({reason})")
            skipped.append((name, reason))
            continue
        locator_updates(root, source)
        directory = source if source.exists() else retired
        size = sum(path.stat().st_size for path in directory.rglob("*")
                   if not path.is_symlink() and path.is_file())
        operations.append((name, source, retired, stamp, size))
    return operations, skipped


def prune(allocation, *, apply=False, root=ROOT):
    root = Path(root).resolve()
    keep, fresh, plan_id = read_allocation(allocation)
    queue = root / "training_queue"
    if not (queue / "queue.sqlite3").is_file():
        raise ValueError("The local training queue is required")
    audit_path = queue / "cleanup" / f"3090-allocation-{plan_id}.json"
    with locked(queue / "shared_warmups/sync.lock"), locked(queue / "catalog.lock"), queue_guard(root) as jobs:
        audit = dict(plan_id=plan_id, started=[], deleted=[])
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            started, deleted = set(audit["started"]), set(audit["deleted"])
            if audit["plan_id"] != plan_id or not deleted <= started or started & keep.keys():
                raise ValueError("Deletion journal does not match the allocation")
        operations, skipped = preflight(root, keep, fresh, jobs, plan_id, audit)
        for name, *_ in operations:
            print(f"DELETE {name}")
        for name, reason in skipped:
            print(f"KEEP {name} ({reason})")
        print(f"{len(keep)} allocated warmups retained, {len(skipped)} other warmups retained, "
              f"{len(operations)} deletions, {sum(op[-1] for op in operations) / 2**30:.2f} GiB")
        if apply and operations:
            apply_deletions(root, operations, audit_path, audit)
            print(f"Deleted {len(audit['deleted'])} snapshots in total. Journal: {audit_path}")
        elif not apply:
            print("Preview only. Use --apply to delete after rechecking.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, default=ALLOCATION)
    parser.add_argument("--apply", action="store_true", help="Permanently delete the eligible snapshots")
    args = parser.parse_args()
    try:
        prune(args.allocation, apply=args.apply)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
