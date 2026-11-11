"""Compare logged anchor insertion frequency: Full FLASH (3.5) vs add (4.0).

Run with the storm conda environment; see anchor_frequency_README.md.
Only reads W&B. Caches raw histories and metadata for offline reproduction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
METRIC = "Retrieval/triggered_anchors_step"
QUERY = """query AnchorHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $name) { sampledHistory(specs: $specs) }
  }
}"""
# Defaults for historical runs predating the corresponding optional settings.
RET_DEFAULTS = {
    "score_combination": "multiply", "value_signal": "value_diff",
    "neighbor_source": "retrieval", "anchor_weight_r": None,
    # An unrecorded weight is not sufficient evidence for a controlled pair.
    "anchor_weight": None,
}
RET_MATCH_KEYS = (
    "trigger_mode", "value_signal", "neighbor_source", "target", "hash_bits",
    "hash_sample_mode", "use_pca", "max_pca_samples", "ema_alpha",
    "anchor_offset", "anchor_weight", "anchor_weight_r", "batch_size_reduction",
    "warmup_steps", "max_anchors", "multiplier", "max_contexts", "max_bucket_size",
    "global_rebuild_enable", "global_rebuild_threshold", "global_rebuild_cooldown",
)
TRAIN_MATCH_KEYS = (
    "NumEnvs", "BatchSize", "BatchLength", "BufferWarmUp", "BufferMaxLength",
    "UseDemonstration", "ImagineBatchSize", "ImagineBatchLength",
    "ImagineContextLength", "TrainDynamicsEverySteps", "TrainAgentEverySteps",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retrieval(config):
    return {**RET_DEFAULTS, **config.get("JointTrainAgent", {}).get("Retrieval", {})}


def identity(run):
    """The CLI seed appears in the run name; BasicSettings.Seed is often 0."""
    match = re.fullmatch(r"(.+)_" + re.escape(run["id"]) + r"_(\d+)(?:_[OX])?", run["name"])
    meta = run["config"].get("RunMetadata", {})
    game = meta.get("game") or (match.group(1) if match else None)
    seed = meta.get("seed")
    if seed is None and match:
        seed = int(match.group(2))
    if game is None or seed is None:
        raise ValueError(f"Cannot identify game/training seed for {run['name']}")
    return game, int(seed)


def method(run):
    ret = retrieval(run["config"])
    train = run["config"].get("JointTrainAgent", {})
    if (run["state"] != "finished" or ret.get("enable") is not True
            or ret.get("trigger_mode") != "z_score"
            or ret["value_signal"] != "value_diff" or ret["neighbor_source"] != "retrieval"
            or ret.get("target") != 16 or ret.get("warmup_steps") != 50000
            or ret.get("batch_size_reduction") != "retrieved"
            or ret["anchor_weight_r"] is not None or ret["anchor_weight"] != 0.0625
            or train.get("UseDemonstration") is not False
            or train.get("NumEnvs") != 1 or train.get("TrainDynamicsEverySteps") != 1
            or train.get("TrainAgentEverySteps") != 1):
        return None
    combo = ret["score_combination"]
    if combo == "multiply" and ret.get("z_score_threshold") == 3.5:
        return "full"
    if combo == "add" and ret.get("additive_z_score_threshold") == 4.0:
        return "add"
    return None


def signature(run):
    """Match training and retrieval settings; do not collapse offsets -2/-3."""
    cfg, ret = run["config"], retrieval(run["config"])
    train = cfg["JointTrainAgent"]
    return json.dumps({
        "retrieval": {key: ret.get(key) for key in RET_MATCH_KEYS},
        "training": {key: train.get(key) for key in TRAIN_MATCH_KEYS},
        "models": cfg.get("Models"),
        "image_size": cfg.get("BasicSettings", {}).get("ImageSize"),
    }, sort_keys=True)


def select_pairs(runs, games=None, seeds=None):
    groups, audit = defaultdict(lambda: defaultdict(list)), []
    for run in runs:
        kind = method(run)
        if kind is None:
            continue
        try:
            game, seed = identity(run)
        except ValueError as error:
            audit.append({"run_id": run["id"], "reason": str(error)})
            continue
        if (games and game not in games) or (seeds and seed not in seeds):
            continue
        groups[game, seed][kind].append(run)
    pairs = []
    for (game, seed), methods in sorted(groups.items()):
        if not methods["add"]:
            continue
        candidates = [(full, add) for full in methods["full"] for add in methods["add"]
                      if signature(full) == signature(add)]
        if not candidates:
            audit.append({"game": game, "seed": seed,
                          "add_run_ids": [run["id"] for run in methods["add"]],
                          "reason": "No Full FLASH run with matching training/retrieval settings",
                          "full_run_ids": [run["id"] for run in methods["full"]]})
            continue
        # Deterministic choice independent of rewards and anchor frequencies.
        full, add = max(candidates, key=lambda pair: (
            pair[1]["created_at"], pair[0]["created_at"], pair[1]["id"], pair[0]["id"]))
        pairs.append({"game": game, "seed": seed, "full": full, "add": add})
        for kind in ("full", "add"):
            unused = [run["id"] for run in methods[kind] if run["id"] != (full if kind == "full" else add)["id"]]
            if unused:
                audit.append({"game": game, "seed": seed, "method": kind,
                              "reason": "Other candidate runs; newest compatible pair retained",
                              "run_ids": unused})
    return pairs, audit


def cli_value(args, key):
    result = None
    for index, arg in enumerate(args):
        if arg == key and index + 1 < len(args):
            result = args[index + 1]
        elif arg.startswith(key + "="):
            result = arg.split("=", 1)[1]
    return result


def expected_updates(run):
    """Environment steps are zero-based train.py loop indices (one env)."""
    args = run["metadata"].get("args", [])
    _, seed = identity(run)
    cli_seed = cli_value(args, "-seed")
    if cli_seed is not None and int(cli_seed) != seed:
        raise ValueError("CLI seed differs from run identity")
    checkpoint = cli_value(args, "--resume_from")
    train = run["config"]["JointTrainAgent"]
    if checkpoint:
        match = re.search(r"/(?:shared_warmup|resume_ckpt)_(\d+)/?$", checkpoint)
        if match is None:
            raise ValueError(f"Cannot resolve resume step from {checkpoint}")
        start = int(match.group(1))
    else:
        start = int(train["BufferWarmUp"])
    stop = int(train["SampleMaxSteps"])
    if not 0 <= start < stop:
        raise ValueError("Invalid training interval")
    return start, stop, checkpoint


def validate_history(rows, start, stop, batch_size):
    """Reject sampling, duplicate steps, missing zeros, and partial run history."""
    rows = sorted(rows, key=lambda row: row["_step"])
    expected = stop - start
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} update records, received {len(rows)}; refusing sampled/partial totals")
    for index, row in enumerate(rows):
        step, count = row.get("_step"), row.get(METRIC)
        if not isinstance(step, (int, float)) or not math.isfinite(step) or int(step) != step:
            raise ValueError("Invalid W&B _step")
        if index and step != rows[index - 1]["_step"] + 1:
            raise ValueError("Non-contiguous W&B update records; cannot reconstruct environment steps")
        if (not isinstance(count, (int, float)) or isinstance(count, bool)
                or not math.isfinite(count) or int(count) != count or not 0 <= count <= batch_size):
            raise ValueError(f"Invalid anchor count: {count!r}")
    return rows


def anchor_rows(rows):
    # Some SDK/backend versions include evaluation-only rows despite keys=.
    # Exclude absent/null values; a recorded zero is a real update and stays.
    return [{"_step": row["_step"], METRIC: row[METRIC]} for row in rows
            if row.get(METRIC) is not None]


def fetch_history(run, raw_run, backend, window):
    start, stop, _ = expected_updates(raw_run)
    batch_size = raw_run["config"]["JointTrainAgent"]["BatchSize"]
    scan_error = None
    if backend in ("auto", "scan"):
        try:
            rows = anchor_rows(run.scan_history(keys=["_step", METRIC], page_size=window))
            return validate_history(rows, start, stop, batch_size), {"backend": "scan_history"}
        except Exception as error:
            if backend == "scan":
                raise
            scan_error = f"{type(error).__name__}: {error}"
    # sampledHistory has a server-side cap even if samples=1_000_000. Query
    # bounded ranges below that cap. maxStep is inclusive on this server;
    # apply half-open bounds locally so either server convention is safe.
    rows = []
    history_stop = raw_run["last_history_step"] + 1
    for left in range(0, history_stop, window):
        right = min(left + window, history_stop)
        spec = {"keys": ["_step", METRIC], "minStep": left,
                "maxStep": right, "samples": right - left + 1}
        response = run._exec(QUERY, specs=[json.dumps(spec)])
        chunk = response["project"]["run"]["sampledHistory"][0]
        rows.extend({"_step": row["_step"], METRIC: row[METRIC]}
                    for row in chunk if METRIC in row and left <= row["_step"] < right)
    return validate_history(rows, start, stop, batch_size), {
        "backend": "ranged_sampledHistory_validated", "window_size": window,
        "scan_error": scan_error,
    }


def make_api(output):
    import wandb
    os.environ["WANDB_CACHE_DIR"] = str(output / ".wandb_cache")
    key = os.environ.get("WANDB_API_KEY")
    if key is None and (ROOT / ".wandb_api_key").is_file():
        key = (ROOT / ".wandb_api_key").read_text().strip()
    return wandb.Api(api_key=key, timeout=60)


def discover(api, project):
    result = []
    for run in api.runs(project, filters={"state": "finished", "config.JointTrainAgent.Retrieval.enable": True}, per_page=100):
        result.append({"id": run.id, "name": run.name, "state": run.state,
                       "created_at": run.created_at, "config": run.config})
    return result


def load_run(candidate, args):
    directory = args.output / "raw" / candidate["id"]
    meta_path, history_path = directory / "run.json", directory / "anchors.csv.gz"
    if meta_path.exists() and history_path.exists() and not args.refresh:
        raw = json.loads(meta_path.read_text())
        if raw["project"] != args.project or raw["id"] != candidate["id"]:
            raise ValueError("Cache run identity mismatch")
        if signature(raw) != signature(candidate) or method(raw) != method(candidate):
            raise ValueError("Cached config differs from discovery; rerun with --refresh")
        with gzip.open(history_path, "rt") as stream:
            rows = [{"_step": int(row["_step"]), METRIC: int(row[METRIC])}
                    for row in csv.DictReader(stream)]
        start, stop, _ = expected_updates(raw)
        validate_history(rows, start, stop, raw["config"]["JointTrainAgent"]["BatchSize"])
        return raw, rows
    if args.offline:
        raise ValueError(f"Missing offline cache: {directory}")
    api = make_api(args.output)
    run = api.run(f"{args.project}/{candidate['id']}")
    directory.mkdir(parents=True, exist_ok=True)
    with run.file("wandb-metadata.json").download(root=str(directory), replace=True) as stream:
        metadata = json.load(stream)
    raw = {**candidate, "project": args.project, "config": run.config,
           "metadata": {key: metadata.get(key) for key in ("args", "git")},
           "last_history_step": run.lastHistoryStep}
    if signature(raw) != signature(candidate) or method(raw) != method(candidate):
        raise ValueError("Run config changed since discovery; rerun with --refresh")
    rows, provenance = fetch_history(run, raw, args.history_backend, args.window_size)
    raw["history_provenance"] = provenance
    with gzip.open(history_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["_step", METRIC])
        writer.writeheader()
        writer.writerows(rows)
    write_json(meta_path, raw)
    return raw, rows


def verify_scan(args):
    """Cross-check previously cached ranged histories using scan_history()."""
    api = make_api(args.output)
    checks = []
    for run_id in args.verify_scan:
        directory = args.output / "raw" / run_id
        raw = json.loads((directory / "run.json").read_text())
        with gzip.open(directory / "anchors.csv.gz", "rt") as stream:
            cached = [{"_step": int(row["_step"]), METRIC: int(row[METRIC])}
                      for row in csv.DictReader(stream)]
        run = api.run(f"{args.project}/{run_id}")
        rows = anchor_rows(run.scan_history(keys=["_step", METRIC], page_size=args.window_size, use_cache=False))
        start, stop, _ = expected_updates(raw)
        rows = validate_history(rows, start, stop, raw["config"]["JointTrainAgent"]["BatchSize"])
        if rows != cached:
            raise ValueError(f"scan_history and cached ranged history differ: {run_id}")
        check = {"run_id": run_id, "project": args.project, "rows": len(rows),
                 "anchor_insertions_entire_training_run": sum(row[METRIC] for row in rows),
                 "scan_history_matches_cached_values": True, "scan_use_cache": False}
        checks.append(check)
        print(json.dumps(check), flush=True)
    write_json(args.output / "scan_verification.json", checks)


def measure(raw, rows, lo, hi):
    start, stop, checkpoint = expected_updates(raw)
    if lo < start or hi > stop or hi <= lo:
        raise ValueError("Comparison interval is outside available training updates")
    counts = [int(row[METRIC]) for row in rows[lo - start:hi - start]]
    total = sum(counts)
    return {
        "run_id": raw["id"], "url": f"https://wandb.ai/{raw['project']}/runs/{raw['id']}",
        "method": method(raw), "env_step_start": lo, "env_step_stop_exclusive": hi,
        "updates": len(counts), "anchor_insertions": total,
        "anchors_per_update": total / len(counts),
        "anchors_per_1000_env_steps": 1000 * total / (hi - lo),
        "selected_sequence_fraction": total / (len(counts) * raw["config"]["JointTrainAgent"]["BatchSize"]),
        "updates_with_any_anchor_fraction": sum(count > 0 for count in counts) / len(counts),
        "anchor_offset": retrieval(raw["config"])["anchor_offset"],
        "training_start_env_step": start, "resume_checkpoint": checkpoint or "",
        "commit": (raw["metadata"].get("git") or {}).get("commit", ""),
        "history_backend": raw["history_provenance"]["backend"],
    }


def aggregate(pairs, margin, bootstrap_samples):
    import numpy as np
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["game"]].append(pair)
    games = []
    for game, records in sorted(groups.items()):
        full = statistics.mean(row["full_anchors_per_update"] for row in records)
        add = statistics.mean(row["add_anchors_per_update"] for row in records)
        games.append({"game": game, "paired_seeds": len(records),
                      "full_anchors_per_update": full, "add_anchors_per_update": add,
                      "add_minus_full": add - full, "add_over_full": add / full if full else None})
    full = np.array([row["full_anchors_per_update"] for row in games])
    add = np.array([row["add_anchors_per_update"] for row in games])
    point = float(add.mean() / full.mean()) if full.mean() else None
    ci = None
    if len(games) > 1 and full.mean():
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(games), size=(bootstrap_samples, len(games)))
        denominator = full[indices].mean(axis=1)
        if np.all(denominator > 0):
            ci = np.quantile(add[indices].mean(axis=1) / denominator, [0.025, 0.975]).tolist()
    return games, {
        "games": len(games), "pairs": len(pairs),
        "full_game_balanced_anchors_per_update": float(full.mean()),
        "add_game_balanced_anchors_per_update": float(add.mean()),
        "add_over_full_game_balanced": point,
        "ratio_95pct_game_bootstrap_ci": ci,
        "bootstrap_unit": "paired game means (seeds averaged within each game)",
        "bootstrap_samples": bootstrap_samples,
        "descriptive_relative_margin": margin,
        "point_ratio_within_margin": None if point is None else 1 - margin <= point <= 1 + margin,
        "ratio_ci_within_margin": None if ci is None else ci[0] >= 1 - margin and ci[1] <= 1 + margin,
    }


def plot_results(output, pairs, games, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, max(5, len(games) * .32)))
    y = np.arange(len(games))
    left.barh(y - .18, [row["full_anchors_per_update"] for row in games], .35, label="Full FLASH (3.5)")
    left.barh(y + .18, [row["add_anchors_per_update"] for row in games], .35, label="Additive (4.0)")
    left.set(yticks=y, yticklabels=[row["game"] for row in games], xlabel="Anchor insertions / world-model update")
    left.invert_yaxis()
    left.legend()
    right.scatter([row["full_anchors_per_update"] for row in pairs],
                  [row["add_anchors_per_update"] for row in pairs], alpha=.65)
    limit = max([row[f"{kind}_anchors_per_update"] for row in pairs for kind in ("full", "add")]) * 1.05
    right.plot([0, limit], [0, limit], "k--", linewidth=1, label="Equal frequency")
    right.set(xlabel="Full FLASH anchors / update", ylabel="Additive anchors / update",
              xlim=(0, limit), ylim=(0, limit))
    right.set_aspect("equal", adjustable="box")
    right.legend()
    fig.tight_layout()
    fig.savefig(output / "frequency_comparison.png", dpi=180)
    fig.savefig(output / "frequency_comparison.pdf")
    plt.close(fig)
    columns = min(4, len(games))
    rows = math.ceil(len(games) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.7 * rows), squeeze=False)
    for ax, game in zip(axes.flat, games):
        for kind, label in (("full", "Full FLASH (3.5)"), ("add", "Additive (4.0)")):
            bins = defaultdict(list)
            for record in curves:
                if record["game"] == game["game"] and record["method"] == kind:
                    bins[record["env_step_bin_start"]].append(record["anchors_per_update"])
            ax.plot(sorted(bins), [statistics.mean(bins[key]) for key in sorted(bins)], label=label)
        ax.set(title=game["game"], xlabel="Environment step (bin start)", ylabel="Anchors / update", ylim=(0, 16))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    for ax in list(axes.flat)[len(games):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "frequency_curves.png", dpi=180)
    plt.close(fig)


def report(args, pairs, games, summary, audit, failures):
    ratio = summary["add_over_full_game_balanced"]
    ratio_text = "undefined (zero Full FLASH mean)" if ratio is None else f"{ratio:.4f}"
    ci = summary["ratio_95pct_game_bootstrap_ci"]
    ci_text = "unavailable" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    lines = ["# Anchor insertion frequency: Full FLASH vs additive", "",
             f"Compared **{len(pairs)} paired training seeds across {len(games)} games**.", "",
             f"- Full FLASH: **{summary['full_game_balanced_anchors_per_update']:.4f} anchors/update**.",
             f"- Additive: **{summary['add_game_balanced_anchors_per_update']:.4f} anchors/update**.",
             f"- Additive / Full FLASH, with equal game weights: **{ratio_text}**; 95% game-bootstrap CI: {ci_text}.",
             f"- Descriptive ±{100*args.relative_margin:g}% band: point estimate inside = {summary['point_ratio_within_margin']}; entire CI inside = {summary['ratio_ci_within_margin']}.", "",
             "Each pair uses the same game, CLI training seed, training settings, retrieval settings, and raw anchor offset. "
             "The newest compatible pair is chosen without consulting rewards or anchor counts. "
             "Missing historical score_combination/value_signal/neighbor_source/anchor_weight_r use the legacy defaults listed in the script. "
             "Runs without an explicitly recorded anchor weight are excluded. "
             "Commits and resume checkpoints are retained in per_run.csv. Equal seeds/configs do not imply equal trajectories or a shared checkpoint.", "",
             f"Only updates in [{args.start_env_step}, {args.end_env_step}) are requested; "
             "each pair uses their common available interval after both warmups. No missing values are replaced by zero. "
             "All histories must contain exactly the expected number of updates with consecutive W&B steps before any totals are reported.", "",
             f"`{METRIC}` counts queue insertions after a sequence maximum passes threshold AND its anchor has a hash key. "
             "There is at most one insertion per sampled sequence per update. Repeated physical replay states can be counted again; "
             "these are not unique anchors, raw threshold exceedances, or anchors ultimately consumed by the actor. "
             "The sequence fraction divides by BatchSize; the any-anchor fraction counts updates with at least one insertion.", "",
             "The bootstrap resamples paired game means, not correlated individual training updates. "
             "The margin is an analysis convention and must be justified independently for a formal equivalence claim. "
             "Observed frequency similarity is empirical evidence, not a proof that threshold 4.0 must match 3.5. "
             "For x=ReLU(z_TD), y=softplus(z_value), x+y and xy have no global ordering. "
             "Logged EMA means/variances do not identify their joint score distribution or permit retrospective threshold recalibration.", "",
             f"Selection audit: {len(audit)} entries; failed histories: {len(failures)}. See selection_audit.json and failures.json.", "",
             "| Game | Seed pairs | Full anchors/update | Add anchors/update | Add/Full |",
             "|---|---:|---:|---:|---:|"]
    for game in games:
        value = game["add_over_full"]
        lines.append(f"| {game['game']} | {game['paired_seeds']} | {game['full_anchors_per_update']:.4f} | "
                     f"{game['add_anchors_per_update']:.4f} | {'undefined' if value is None else f'{value:.4f}'} |")
    lines.extend(["", "W&B source runs and exact comparison windows are listed in pairs.csv and per_run.csv.", "",
                  "The full-history API contract is described in the [W&B SDK source](https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py). "
                  "The fallback uses the SDK's sampledHistory GraphQL path with bounded ranges and explicit completeness checks.", ""])
    (args.output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="choemj-kaist/STORM")
    parser.add_argument("--output", type=Path, default=ROOT / "results/anchor_frequency")
    parser.add_argument("--games", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--history-backend", choices=("auto", "scan", "ranged"), default="auto")
    parser.add_argument("--window-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-env-step", type=int, default=50000)
    parser.add_argument("--end-env-step", type=int, default=100000)
    parser.add_argument("--bin-size", type=int, default=5000)
    parser.add_argument("--relative-margin", type=float, default=.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verify-scan", nargs="+", metavar="RUN_ID",
                        help="Compare full scan_history records with cached histories, then exit")
    args = parser.parse_args(argv)
    if (not 1 <= args.window_size <= 9000 or args.workers < 1 or args.bin_size < 1
            or args.bootstrap_samples < 1 or not 0 < args.relative_margin < 1
            or args.start_env_step < 50000 or args.end_env_step <= args.start_env_step
            or args.offline and args.refresh):
        parser.error("Invalid range, window size, workers, margin, or offline/refresh combination")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_scan:
        if args.offline:
            parser.error("--verify-scan requires network access")
        verify_scan(args)
        return
    manifest = args.output / "discovery.json"
    if manifest.exists() and not args.refresh:
        discovery = json.loads(manifest.read_text())
        if discovery["project"] != args.project:
            parser.error("Discovery cache belongs to another project")
        runs = discovery["runs"]
    else:
        if args.offline:
            parser.error("--offline requires a previous discovery.json")
        print(f"Discovering finished runs in {args.project} ...", flush=True)
        runs = discover(make_api(args.output), args.project)
        write_json(manifest, {"project": args.project, "runs": runs})
    pairs, audit = select_pairs(runs, args.games, args.seeds)
    write_json(args.output / "selection_audit.json", audit)
    print(f"Selected {len(pairs)} compatible game/seed pairs; {len(audit)} audit entries.", flush=True)
    candidates = {pair[kind]["id"]: pair[kind] for pair in pairs for kind in ("full", "add")}
    loaded, failures = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(load_run, run, args): run for run in candidates.values()}
        for index, future in enumerate(as_completed(pending), 1):
            run = pending[future]
            try:
                loaded[run["id"]] = future.result()
                print(f"[{index}/{len(pending)}] {run['name']}: complete", flush=True)
            except Exception as error:
                failures.append({"run_id": run["id"], "name": run["name"], "error": f"{type(error).__name__}: {error}"})
                print(f"[{index}/{len(pending)}] {run['name']}: {error}", file=sys.stderr, flush=True)
    records, comparisons, curves = [], [], []
    for pair in pairs:
        if any(pair[kind]["id"] not in loaded for kind in ("full", "add")):
            continue
        data = {kind: loaded[pair[kind]["id"]] for kind in ("full", "add")}
        lo = max(args.start_env_step, *(max(expected_updates(raw)[0], retrieval(raw["config"])["warmup_steps"]) for raw, _ in data.values()))
        hi = min(args.end_env_step, *(expected_updates(raw)[1] for raw, _ in data.values()))
        if hi <= lo:
            failures.append({"game": pair["game"], "seed": pair["seed"], "error": "Empty common post-warmup interval"})
            continue
        comparison = {"game": pair["game"], "seed": pair["seed"],
                      "env_step_start": lo, "env_step_stop_exclusive": hi}
        for kind, (raw, rows) in data.items():
            record = {"game": pair["game"], "seed": pair["seed"], **measure(raw, rows, lo, hi)}
            records.append(record)
            comparison.update({f"{kind}_{key}": record[key] for key in ("run_id", "anchors_per_update", "anchor_insertions", "updates")})
            start = expected_updates(raw)[0]
            for left in range(args.start_env_step, hi, args.bin_size):
                right = min(left + args.bin_size, hi)
                a, b = max(left, lo), right
                if b > a:
                    counts = [row[METRIC] for row in rows[a-start:b-start]]
                    curves.append({"game": pair["game"], "seed": pair["seed"], "method": kind,
                                   "run_id": raw["id"], "env_step_bin_start": left,
                                   "observed_env_step_start": a, "env_step_stop_exclusive": b,
                                   "updates": len(counts), "anchors_per_update": sum(counts) / len(counts)})
        full, add = comparison["full_anchors_per_update"], comparison["add_anchors_per_update"]
        comparison.update(add_minus_full=add-full, add_over_full=add/full if full else None)
        comparisons.append(comparison)
    write_json(args.output / "failures.json", failures)
    write_csv(args.output / "per_run.csv", records)
    write_csv(args.output / "pairs.csv", comparisons)
    write_csv(args.output / "timeseries.csv", curves)
    if not comparisons:
        raise SystemExit("No complete compatible pairs; inspect selection_audit.json and failures.json")
    games, summary = aggregate(comparisons, args.relative_margin, args.bootstrap_samples)
    summary.update(project=args.project, metric=METRIC, requested_env_steps=[args.start_env_step, args.end_env_step],
                   selected_pairs=len(pairs), failed_runs_or_intervals=len(failures))
    write_csv(args.output / "per_game.csv", games)
    write_json(args.output / "summary.json", summary)
    report(args, comparisons, games, summary, audit, failures)
    if not args.no_plots:
        plot_results(args.output, comparisons, games, curves)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output / 'report.md'}", flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
