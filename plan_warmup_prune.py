"""Write a cross-GPU keep/delete plan for the 50k shared warmups."""

import argparse
from collections import defaultdict
from math import isclose, isfinite, sqrt
from pathlib import Path
from statistics import fmean

from openpyxl import load_workbook

from collect_warmup_scores import TARGET_STEP
from queue_train import ROOT, SEED_DIR, read_seed_files
from select_shared_warmups import (BASELINE, OPTIONS, RETRIEVAL, Snapshot, SEED_COUNTS, SUMMARY, SUMMARY_HEADER,
                                  candidate_pools, owner_notes, paper_references, read_reports, tolerance)
from warmup_registry import DEFAULT_CACHE, aggregate, digest, save, timestamp


PLAN_KIND = "storm-warmup-prune"
MAX_GPU_LOSS = 0.05


def inventory_entries(cache, round_id):
    report = aggregate(cache, round_id)
    if not report["collection_complete"]:
        raise ValueError(f"Missing inventory scans for {report['pending_gpu_types']}")
    entries = {}
    for game, models in report["games"].items():
        for model, seeds in models.items():
            for seed, group in seeds.items():
                for item in group["snapshots"]:
                    if TARGET_STEP not in item["warmup_steps"]:
                        continue
                    snap = Snapshot(game, int(seed), model, item["machine"], item["source"])
                    if snap in entries:
                        raise ValueError(f"Duplicate snapshot in inventory: {snap}")
                    entries[snap] = dict(game=game, seed=int(seed), source=item["source"], step=item["step"],
                                         scanned_at=item["scanned_at"], config_sha256=item["saved_config_sha256"],
                                         targets=item["warmup_steps"])
    return entries


def workbook_selection(path, references):
    """Use the workbook controls and the static combination rows, even without cached Excel formulas."""
    result = {}
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        summary = workbook[SUMMARY]
        stability, paper_tolerance = (summary[cell].value for cell in ("B1", "E1"))
        if any(type(value) not in (int, float) or not isfinite(value) or value < 0
               for value in (stability, paper_tolerance)):
            raise ValueError("Expected numeric HNS tolerances in B1 and E1")
        strict = next((text.endswith("아니요") for name, text in workbook["기준"].values if name == "평가 조건"), False)
        option_rows = {(row[1], row[2]): (row[5], row[6])
                       for row in workbook[OPTIONS].iter_rows(min_row=2, values_only=True)}
        groups = defaultdict(list)
        for row in workbook["조합 체크포인트"].iter_rows(min_row=2, values_only=True):
            game, size, method, gpu, seed, _, hns, _, _, _, machine, source, ids, *_ = row
            groups[game, size].append((Snapshot(game, seed, gpu, machine, source), hns, method, ids))
        for row in summary.iter_rows(min_row=SUMMARY_HEADER + 1, values_only=True):
            game = row[0]
            if game not in references:
                continue
            sd, paper, available, reproduced = (row[i] for i in (4, 9, 19, 20))
            reference = references[game]
            if (game in result or not isclose(paper, reference.normalize(reference.paper), abs_tol=1e-12) or
                    (reference.reproduced is not None and
                     not isclose(reproduced, reference.normalize(reference.reproduced), abs_tol=1e-12))):
                raise ValueError(f"Workbook references are stale or duplicated for {game}")
            if sd is None or available < 2:
                raise ValueError(f"Workbook has no recommendation for {game}")
            count = next((k for k in SEED_COUNTS if sd / sqrt(k) <= stability), SEED_COUNTS[-1])
            actual = min(count, available)
            group = groups[game, actual]
            mean, delta = option_rows[game, actual]
            if (len(group) != actual or len({record[0].seed for record in group}) != actual or
                    not isclose(fmean(record[1] for record in group), mean, rel_tol=0, abs_tol=1e-12) or
                    not isclose(mean - paper, delta, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"Incomplete workbook combination for {game}")
            warned = -delta > paper_tolerance and (reproduced is None or mean < reproduced)
            result[game] = dict(count=count, group=group, hns=mean, warned=warned,
                                preferred=BASELINE if row[2] == BASELINE else RETRIEVAL)
    finally:
        workbook.close()
    if set(result) != set(references):
        raise ValueError(f"Workbook does not cover all reference games: {sorted(set(references) - set(result))}")
    return result, strict, dict(stability_tolerance=stability, paper_tolerance=paper_tolerance)


def rank_scores(profiles, preferred):
    """Keep one comparable score per physical snapshot before ranking seed performance."""
    scores, by_run = {}, {}
    methods = [preferred, RETRIEVAL if preferred == BASELINE else BASELINE]
    methods.extend(sorted(set(profiles) - set(methods)))
    for method in methods:
        for candidate in sorted((c for group in profiles.get(method, {}).values() for c in group),
                                key=lambda c: (c.eval_step is None, -len(c.run_ids), c.profile, c.run_ids)):
            scores.setdefault(candidate.snapshot, candidate)
            by_run[candidate.snapshot, ", ".join(candidate.run_ids)] = candidate
    return scores, by_run


def top_seeds(scores, count, initial=()):
    selected = {candidate.snapshot.seed: candidate for candidate in initial}
    for candidate in sorted(scores, key=lambda c: (-c.hns, bool(c.owner_note), c.evidence_order())):
        if len(selected) >= count:
            break
        selected.setdefault(candidate.snapshot.seed, candidate)
    return list(selected.values())


def choose_game(available, workbook_row, profiles, warning_gpu_loss):
    count, warned = workbook_row["count"], workbook_row["warned"]
    scores, by_run = rank_scores(profiles, workbook_row["preferred"])
    workbook_group = []
    for snap, hns, method, ids in workbook_row["group"]:
        candidate = by_run.get((snap, ids))
        if snap not in available or candidate is None or candidate.method != method or not isclose(hns, candidate.hns, abs_tol=1e-12):
            raise ValueError(f"Workbook score does not match the score reports: {snap}")
        workbook_group.append(candidate)

    candidates = {snap: score for snap, score in scores.items() if snap in available}
    if not warned:
        candidates.update((c.snapshot, c) for c in workbook_group)
    starting = top_seeds(candidates.values(), count, () if warned else workbook_group)
    selected = starting
    starting_mean = fmean(c.hns for c in starting)
    if len(starting) == count and len({c.snapshot.gpu for c in starting}) > 1:
        by_gpu = defaultdict(list)
        for candidate in candidates.values():
            by_gpu[candidate.snapshot.gpu].append(candidate)
        alternatives = [top_seeds(group, count) for group in by_gpu.values()]
        # A non-warning game also retains its workbook mean to within the stated margin.
        comparison = starting_mean if warned else max(starting_mean, workbook_row["hns"])
        allowance = warning_gpu_loss if warned else MAX_GPU_LOSS
        qualified = []
        for group in alternatives:
            if len(group) == count:
                mean = fmean(c.hns for c in group)
                loss = comparison - mean
                if loss <= 0 or (loss < allowance and not isclose(loss, allowance, rel_tol=0, abs_tol=1e-12)):
                    qualified.append((mean, group))
        if qualified:
            selected = min(qualified, key=lambda item: (-item[0], tuple(c.snapshot for c in item[1])))[1]

    kept = {c.snapshot: c for c in selected}
    # Reserve a physical warmup if the workbook's comparable scores cover fewer than k seeds.
    seed_ids = {snap.seed for snap in kept}
    models = {snap.gpu for snap in kept}
    for reserve in sorted(available, key=lambda snap: (snap.gpu not in models, snap)):
        if len(kept) == count:
            break
        if reserve.seed not in seed_ids:
            kept[reserve] = None
            seed_ids.add(reserve.seed)
    if len(kept) != count:
        raise ValueError(f"Need {count} distinct warmups but can retain only {len(kept)}")
    return kept, starting_mean, fmean(c.hns for c in selected)


def build_plan(args):
    entries = inventory_entries(args.cache, args.round)
    refs = paper_references(args.reference)
    _, scored_snapshots, runs = read_reports(args.reports_dir)
    if set(entries) != scored_snapshots:
        raise ValueError(f"Inventory and score reports differ: {len(set(entries) - scored_snapshots)} inventory-only, "
                         f"{len(scored_snapshots - set(entries))} scores-only")
    if any(item["targets"] != [TARGET_STEP] for item in entries.values()):
        raise ValueError("This inventory includes snapshots shared by multiple warmup targets")
    selections, strict, controls = workbook_selection(args.workbook, refs)
    notes = owner_notes(scored_snapshots)
    pools = candidate_pools(runs, refs, notes, strict)
    available = defaultdict(set)
    for snap in entries:
        available[snap.game].add(snap)
    if set(available) != set(selections):
        raise ValueError("Inventory and workbook cover different games")

    machines = {}
    for snap in sorted(entries):
        machines.setdefault(snap.gpu, {}).setdefault(snap.machine, {"keep": [], "delete": []})
    games, kept_models = {}, defaultdict(set)
    owners = {entry["seed"]: entry["gpu_type"] for entry in read_seed_files(SEED_DIR).values()}
    for game, row in selections.items():
        kept, initial_mean, final_mean = choose_game(available[game], row, pools[game], args.warning_gpu_loss)
        before_gpu = sorted({snap.gpu for snap, *_ in row["group"]})
        games[game] = dict(keep_count=row["count"], paper_warning=row["warned"], workbook_hns=row["hns"],
                           workbook_gpu_types=before_gpu, before_consolidation_hns=initial_mean, kept_hns=final_mean,
                           scored_count=sum(c is not None for c in kept.values()), keep=[])
        for snap in sorted(available[game]):
            item = machines[snap.gpu][snap.machine]
            if snap not in kept:
                item["delete"].append(entries[snap])
                continue
            c = kept[snap]
            item["keep"].append(entries[snap])
            kept_models[snap.seed].add(snap.gpu)
            games[game]["keep"].append(dict(gpu_type=snap.gpu, machine=snap.machine, seed=snap.seed, source=snap.path,
                                              hns=c.hns if c else None, score_method=c.method if c else None,
                                              run_ids=c.run_ids if c else (), owner_note=notes[snap]))
    conflicts = [dict(seed=seed, planned_gpu_types=sorted(models), existing_queue_owner=owners.get(seed))
                 for seed, models in sorted(kept_models.items())
                 if len(models) > 1 or (seed in owners and models != {owners[seed]})]
    plan = dict(kind=PLAN_KIND, version=1, created_at=timestamp(), round=args.round, target_step=TARGET_STEP,
                controls=controls, warning_gpu_loss=args.warning_gpu_loss, other_gpu_loss=MAX_GPU_LOSS,
                strict_100k=strict, machines=machines, games=games, seed_owner_conflicts=conflicts)
    return dict(plan, plan_id=digest(plan))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", default="warmups-20261123")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workbook", type=Path, default=ROOT / "results/shared_warmup_selection.xlsx")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/converted_results.xlsx")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "training_queue/warmup_scores")
    parser.add_argument("--output", type=Path, default=ROOT / "training_queue/warmup_prune_plan.json")
    parser.add_argument("--warning-gpu-loss", type=tolerance, default=MAX_GPU_LOSS,
                        help="Maximum HNS loss when consolidating a warning game (default: 0.05, exclusive)")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".json" or args.output.resolve().is_relative_to(args.reports_dir.resolve()):
        parser.error("--output must be a JSON plan outside the score reports directory")
    try:
        plan = build_plan(args)
        save(args.output, plan)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    for model, machines in sorted(plan["machines"].items()):
        for machine, item in sorted(machines.items()):
            print(f"{model} @ {machine}  keep={len(item['keep'])}  delete={len(item['delete'])}")
    print(f"{len(plan['games'])} games · {len(plan['seed_owner_conflicts'])} existing/planned seed ownership conflicts")
    for conflict in plan["seed_owner_conflicts"]:
        print(f"Owner conflict for seed {conflict['seed']}: existing={conflict['existing_queue_owner']}, "
              f"plan={','.join(conflict['planned_gpu_types'])}")
    print(f"Plan {plan['plan_id']} → {args.output}")


if __name__ == "__main__":
    main()
