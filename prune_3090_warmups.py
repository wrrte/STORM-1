"""Preview or delete local RTX 3090 warmups using the seed registry or allocation workbooks."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sqlite3

from apply_warmup_prune import apply_deletions, locator_updates
from cleanup_warmups import protected, queue_guard, safe_source
from queue_train import ROOT, locked, pair_key
from warmup_catalog import source_stamp, warmup_sources


ALLOCATION = ROOT / "results/warmup_gpu_assignment_kangaroo_3090_20261126.xlsx"
COMPARISON = ROOT / "results/shared_warmup_selection_3090_check.xlsx"
MODEL = "3090"
REGISTRY = Path("config_files/game_seeds/type_3090.json")


def read_registry(root):
    data = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    if data["gpu_type"] != MODEL or not isinstance(data["game_seeds"], list):
        raise ValueError(f"Invalid RTX 3090 seed registry: {REGISTRY}")
    keep = {}
    for entry in data["game_seeds"]:
        pair = pair_key(entry["game"], entry["seed"])
        warmups = entry.get("warmups", [])
        if not isinstance(warmups, list):
            raise ValueError(f"Invalid warmups for {pair} in {REGISTRY}")
        for name in warmups:
            path = Path(name)
            if (len(path.parts) != 3 or path.parts[0] != "ckpt" or ".." in path.parts
                    or path.as_posix() != name or not re.fullmatch(r"shared_warmup_\d+", path.name)
                    or (root / path).resolve() != root / path):
                raise ValueError(f"Invalid local warmup path in {REGISTRY}: {name!r}")
            if keep.setdefault(name, pair) != pair:
                raise ValueError(f"Warmup belongs to multiple game/seed entries: {name}")
    if not keep:
        raise ValueError(f"No registered warmup paths in {REGISTRY}")
    return keep, hashlib.sha256(json.dumps(sorted(keep)).encode()).hexdigest()


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


def read_comparison(path, required):
    from openpyxl import load_workbook
    from select_shared_warmups import read_warmup_combinations

    with path.open("rb") as stream:
        data = stream.read()
        report_mtime = os.fstat(stream.fileno()).st_mtime_ns
    workbook = load_workbook(BytesIO(data), read_only=True, data_only=True)
    try:
        tolerance, combinations = read_warmup_combinations(workbook)
    finally:
        workbook.close()
    if {game: combo.required for game, combo in combinations.items()} != required:
        raise ValueError("The comparison games or required seed counts differ from the allocation")
    keep, unconfirmed = {}, set()
    for game, combo in combinations.items():
        if not combo.passes(tolerance):
            unconfirmed.add(game)
            continue
        for _, seed, source, _, _ in combo.checkpoints:
            if source in keep:
                raise ValueError(f"Repeated checkpoint in confirmed combinations: {source}")
            keep[source] = pair_key(game, seed)
    if not keep:
        raise ValueError("The comparison has no confirmed RTX 3090 warmup combinations")
    return keep, unconfirmed, hashlib.sha256(data).hexdigest(), report_mtime


def preflight(root, keep, fresh, jobs, plan_id, audit, *, unconfirmed=(), comparison_mtime=None, registry=False):
    inventory_path = root / "training_queue/warmup_inventory.json"
    inventory = json.loads(inventory_path.read_text()) if not registry or inventory_path.exists() else {"available": []}
    records = {record["source"]: record for record in inventory["available"] if record["gpu_type"] == MODEL}
    if not registry:
        for name, pair in keep.items():
            path = safe_source(root, name)
            record = records.get(name)
            if (not record or pair_key(record["game"], record["seed"]) != pair
                    or record.get("stamp") != [list(item) for item in source_stamp(path)]):
                raise ValueError(f"Retained warmup differs from the inventory: {name}")

    snapshots = {path.relative_to(root).as_posix()
                 for path in warmup_sources(root, include_unpublished=registry)}
    started, deleted = set(audit["started"]), set(audit["deleted"])
    collector_dirs = {f"{job['name']}_collector_Shared" for job in jobs}
    operations, skipped = [], []
    for name in sorted((snapshots | started) - keep.keys()):
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
            stamp = source_stamp(source)
            if not registry:
                if not record:
                    skipped.append((name, "no verified RTX 3090 inventory record"))
                    continue
                if record["game"] in unconfirmed:
                    skipped.append((name, "game has no confirmed combination"))
                    continue
                if pair_key(record["game"], record["seed"]) in fresh:
                    skipped.append((name, "new warmup in the final allocation"))
                    continue
                if record.get("stamp") != [list(item) for item in stamp]:
                    skipped.append((name, "snapshot changed since inventory inspection"))
                    continue
                if comparison_mtime is not None and any(mtime > comparison_mtime for _, _, mtime in stamp):
                    skipped.append((name, "snapshot changed after the comparison workbook"))
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


def prune(allocation=ALLOCATION, *, comparison=None, registry=False, apply=False, root=ROOT):
    root = Path(root).resolve()
    queue = root / "training_queue"
    if not (queue / "queue.sqlite3").is_file():
        raise ValueError("The local training queue is required")
    with locked(queue / "shared_warmups/sync.lock"), locked(queue / "catalog.lock"), queue_guard(root) as jobs:
        fresh, unconfirmed, comparison_mtime = set(), set(), None
        if registry:
            keep, plan_id = read_registry(root)
            audit_kind, retained = "registry", "registered"
        else:
            keep, fresh, plan_id = read_allocation(allocation)
            audit_kind, retained = "allocation", "allocated"
            if comparison is not None:
                from check_shared_warmup_results import required_seeds

                keep, unconfirmed, comparison_id, comparison_mtime = read_comparison(comparison, required_seeds(allocation))
                fresh, retained = set(), "confirmed"
                plan_id = hashlib.sha256(f"{plan_id}:{comparison_id}".encode()).hexdigest()
        audit_path = queue / "cleanup" / f"3090-{audit_kind}-{plan_id}.json"
        audit = dict(plan_id=plan_id, started=[], deleted=[])
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            started, deleted = set(audit["started"]), set(audit["deleted"])
            if audit["plan_id"] != plan_id or not deleted <= started or started & keep.keys():
                raise ValueError("Deletion journal does not match the retained paths")
        operations, skipped = preflight(root, keep, fresh, jobs, plan_id, audit,
                                        unconfirmed=unconfirmed, comparison_mtime=comparison_mtime,
                                        registry=registry)
        if unconfirmed:
            print(f"Keeping warmups for unconfirmed games: {', '.join(sorted(unconfirmed))}")
        for name, *_ in operations:
            print(f"DELETE {name}")
        for name, reason in skipped:
            print(f"KEEP {name} ({reason})")
        print(f"{len(keep)} {retained} warmup paths, "
              f"{len(skipped)} other warmups retained, "
              f"{len(operations)} deletions, {sum(op[-1] for op in operations) / 2**30:.2f} GiB")
        if apply and operations:
            apply_deletions(root, operations, audit_path, audit)
            print(f"Deleted {len(audit['deleted'])} snapshots in total. Journal: {audit_path}")
        elif not apply:
            print("Preview only. Use --apply to delete after rechecking.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="store_true", help=f"Keep the warmup paths listed in {REGISTRY}")
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--comparison", type=Path, nargs="?", const=COMPARISON,
                        help="Keep confirmed combinations from this workbook and every warmup in unconfirmed games")
    parser.add_argument("--apply", action="store_true", help="Permanently delete the eligible snapshots")
    args = parser.parse_args()
    if args.registry and (args.allocation is not None or args.comparison is not None):
        parser.error("--registry cannot be combined with --allocation or --comparison")
    try:
        prune(args.allocation or ALLOCATION, comparison=args.comparison, registry=args.registry, apply=args.apply)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
