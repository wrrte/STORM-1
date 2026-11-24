"""Collect W&B scores for runs descended from local 50,000-step shared warmups."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from io import StringIO
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import socket
import sqlite3
import sys

from queue_job import stage_run_name
from queue_train import ROOT, canonical_game, gpu_type, pair_key, write_atomic
from warmup_catalog import imported_path, saved_identity, warmup_sources
from warmup_registry import identifier, timestamp


TARGET_STEP = 50_000
DEFAULT_PROJECT = "choemj-kaist/STORM"
EVAL_RETURN = "eval/episode_avg_return"
BF16_RETURN = "eval/bf16/episode_avg_return"
FP32_RETURN = "eval/fp32/episode_avg_return"
RETRIEVAL_TARGET = "JointTrainAgent.Retrieval.warmup_steps"
WARMUP_COLUMNS = ("gpu_type", "machine", "game", "seed", "warmup_source",
                  "warmup_target_step", "warmup_saved_step", "run_count", "final_score_count")
RUN_COLUMNS = WARMUP_COLUMNS[:-2] + (
    "wandb_project", "run_id", "run_name", "created_at", "state", "run_gpu", "branch",
    "retrieval_enable", "sample_max_steps", "eval_mode", "eval_precision", "eval_seed",
    "eval_episodes", "eval_checkpoint_step", "final_score", "score_status", "last_eval_score",
    "score_metric", "last_bf16_score", "last_fp32_score", "run_url", "resume_path", "link_source",
    "metadata_source",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arguments(args, flags):
    """Accept both argparse's separated and joined option forms."""
    values = {}
    for index, token in enumerate(args):
        key, separator, value = token.partition("=")
        if key in flags and (separator or index + 1 < len(args)):
            values[key] = value if separator else args[index + 1]
    return values


def path_key(value, *anchors):
    """Compare paths from different checkouts using the full repository-relative suffix."""
    if not isinstance(value, str) or not value:
        return None
    parts = PurePosixPath(posixpath.normpath(value)).parts
    start = next((i for i, part in enumerate(parts) if part in anchors), None)
    if start is not None:
        return str(PurePosixPath(*parts[start:]))
    return None


def saved_warmup(root, source):
    import yaml

    item = source.relative_to(root).as_posix()
    metadata_path = source / "warmup_metadata.json"
    branch_path = source / "branches.json"
    branches = read_json(branch_path) if branch_path.is_file() else {}
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata["version"] != 1:
            raise ValueError("Unsupported warmup metadata version")
        step = metadata["step"]
        config_path = source / "config.yaml"
    else:
        metadata = saved_identity(source)
        step = int(source.name.removeprefix("shared_warmup_"))
        config_relative = path_key(metadata["config"], "runs", "ckpt", "config_files")
        config_path = root / (config_relative or metadata["config"])

    game, seed = canonical_game(metadata["env_name"]), metadata["seed"]
    pair_key(game, seed)
    targets = metadata.get("requested_warmup_steps")
    if targets is None:
        commands = branches.get("commands", {}).values() if not metadata_path.is_file() else ()
        overrides = {arguments(command, {RETRIEVAL_TARGET}).get(RETRIEVAL_TARGET) for command in commands}
        if overrides - {None}:
            if len(overrides) != 1:
                raise ValueError("Saved branches disagree about the warmup target")
            targets = [int(overrides.pop())]
        else:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            targets = [config["JointTrainAgent"]["Retrieval"]["warmup_steps"]]
    if TARGET_STEP not in targets:
        return None, None
    if type(step) is not int or step < TARGET_STEP:
        raise ValueError("Invalid saved step for a 50,000-step warmup")
    if any(not (source / name).is_file() for name in ("training_state.pt", "replay_buffer.npz")):
        raise ValueError("Missing checkpoint or replay payload")
    return dict(game=game, seed=seed, warmup_source=item, warmup_target_step=TARGET_STEP,
                warmup_saved_step=step), branches.get("checkpoint_dir")


def inventory(root):
    import yaml

    snapshots, aliases, issues = {}, defaultdict(set), []
    for source in warmup_sources(root, include_unpublished=True):
        try:
            record, original = saved_warmup(root, source)
        except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
            issues.append(dict(kind="warmup_unreadable", warmup_source=str(source.relative_to(root)),
                               error=str(error)))
            continue
        if record is None:
            continue
        name = record["warmup_source"]
        snapshots[name] = record
        for path in (str(source), original):
            if path_key(path, "ckpt") == name:
                aliases[name].add(name)
                if Path(path).is_absolute():
                    imported = imported_path(path, root / "training_queue")
                    aliases[path_key(str(imported), "training_queue")].add(name)

    for path in sorted((root / "training_queue/imported_warmups").glob("*/snapshot/warmup_metadata.json")):
        try:
            name = path_key(read_json(path).get("imported_from"), "ckpt")
            if name in snapshots:
                aliases[path_key(str(path.parent), "training_queue")].add(name)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            issues.append(dict(kind="import_unreadable", path=str(path.relative_to(root)), error=str(error)))
    return snapshots, aliases, issues


def queue_retries(root, aliases, issues):
    """Map temporary resume directories through the queue's saved per-stage plans."""
    database = root / "training_queue/queue.sqlite3"
    stages = defaultdict(set)
    if database.is_file():
        try:
            with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
                for (payload,) in connection.execute("SELECT payload FROM jobs"):
                    for command in (json.loads(payload).get("plan") or {}).values():
                        flags = arguments(command, {"--resume_from"})
                        checkpoint = path_key(flags.get("--resume_from"), "ckpt", "training_queue")
                        candidates = aliases.get(checkpoint)
                        if candidates:
                            stages[stage_run_name(command)].update(candidates)
        except (sqlite3.Error, ValueError, TypeError, KeyError, IndexError) as error:
            issues.append(dict(kind="queue_plans_unreadable", error=type(error).__name__))
    return stages


def local_metadata(root):
    return {path.parent.parent.name.rsplit("-", 1)[-1]: path
            for path in sorted((root / "wandb").glob("run-*/files/wandb-metadata.json"))}


def run_metadata(run, local_path):
    if local_path is not None:
        try:
            metadata = read_json(local_path)
            if isinstance(metadata.get("args"), list):
                return metadata, "local_wandb_metadata"
        except (OSError, ValueError, AttributeError):
            pass
    try:
        metadata = run.metadata
        if isinstance(metadata, dict) and isinstance(metadata.get("args"), list):
            return metadata, "wandb_api_metadata"
    except Exception as error:
        return None, type(error).__name__
    return None, "No readable W&B command arguments"


def find_link(flags, aliases, stages):
    resume = flags.get("--resume_from") or flags.get("--resume_warmup")
    checkpoint = path_key(resume, "ckpt", "training_queue")
    if checkpoint in aliases:
        link_source = "imported_checkpoint" if checkpoint.startswith("training_queue/") else "checkpoint"
        return resume, aliases[checkpoint], link_source
    if checkpoint and re.fullmatch(r"ckpt/[^/]+/queue_resume_[0-9a-f]{32}", checkpoint):
        stage = PurePosixPath(checkpoint).parent.name
        name, branch = flags.get("-n"), flags.get("--branch_experiment")
        if name and branch and stage == f"{name}_{branch}":
            return resume, stages.get(stage, set()), "queue_stage_plan"
    return resume, set(), None


def scalar(value):
    return value if type(value) in (float, int) and isfinite(value) else None


def evaluation(summary, state, config):
    values = {key: scalar(summary.get(key)) for key in (EVAL_RETURN, BF16_RETURN, FP32_RETURN)}
    metric, score = next(((key, value) for key, value in values.items() if value is not None), (None, None))
    step = scalar(summary.get("eval/checkpoint_step"))
    final_score, status = None, "no_eval_score"
    if score is not None:
        if step is not None:
            final_score, status = score, "final_checkpoint_logged"
        elif state == "finished" and config.get("EvalMode") == "final_only":
            final_score, status = score, "final_only_finished"
        else:
            status = "last_eval_only"
    return dict(eval_checkpoint_step=step, final_score=final_score, score_status=status,
                score_metric=metric, last_eval_score=score, last_bf16_score=values[BF16_RETURN],
                last_fp32_score=values[FP32_RETURN])


def result_row(run, config, metadata, metadata_source, snapshot, project, flags, resume, link, issues):
    training = config.get("JointTrainAgent") or {}
    retrieval = training.get("Retrieval") or {}
    try:
        score = evaluation(run.summary, run.state, training)
    except Exception as error:
        issues.append(dict(kind="wandb_summary_unavailable", run_id=run.id, error=type(error).__name__))
        score = dict(score_status="summary_unavailable")
    return dict(**snapshot, wandb_project=project, run_id=run.id, run_name=run.name,
                created_at=run.created_at, state=run.state, run_gpu=metadata.get("gpu"),
                branch=flags.get("--branch_experiment") or {"true": "retrieval", "false": "baseline"}.get(
                    flags.get("--branch_mode")), retrieval_enable=retrieval.get("enable"),
                sample_max_steps=training.get("SampleMaxSteps"), eval_mode=training.get("EvalMode"),
                eval_precision=training.get("EvalPrecision"), eval_seed=training.get("EvalSeed"),
                eval_episodes=training.get("EvalEpisodes"), run_url=run.url, resume_path=resume,
                link_source=link, metadata_source=metadata_source, wandb_config=config, **score)


def collect(root, model, machine, project, api):
    snapshots, aliases, issues = inventory(root)
    if not snapshots:
        return dict(project=project, gpu_type=model, machine=machine, collected_at=timestamp(),
                    warmups=[], runs=[], issues=issues, wandb_scanned=0)
    stages, cached = queue_retries(root, aliases, issues), local_metadata(root)
    pairs = {(snapshot["game"], snapshot["seed"]) for snapshot in snapshots.values()}
    project_runs = list(iter(api.runs(project, per_page=100)))
    print(f"Scanning {len(project_runs)} W&B runs for {len(snapshots)} local warmups", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as workers:
        metadata_rows = workers.map(lambda run: run_metadata(run, cached.get(run.id)), project_runs)
        for index, (run, (metadata, source)) in enumerate(zip(project_runs, metadata_rows), 1):
            if index % 250 == 0:
                print(f"Scanned {index}/{len(project_runs)} W&B runs", flush=True)
            if metadata is None:
                issues.append(dict(kind="wandb_metadata_unavailable", run_id=run.id, error=source))
                continue
            flags = arguments(metadata["args"], {"--resume_from", "--resume_warmup", "--branch_experiment",
                                                 "--branch_mode", "-n", "-env_name", "-seed"})
            resume, candidates, link = find_link(flags, aliases, stages)
            if not resume:
                continue
            try:
                identity = canonical_game(flags["-env_name"]), int(flags["-seed"])
            except (ValueError, KeyError):
                identity = None
            matching = [snapshots[name] for name in candidates
                        if (snapshots[name]["game"], snapshots[name]["seed"]) == identity]
            if len(matching) != 1:
                if candidates or identity in pairs:
                    issues.append(dict(kind="unresolved_resume", run_id=run.id, resume_path=resume,
                                       game_seed=identity, candidate_warmups=sorted(candidates)))
                continue
            snapshot = dict(**matching[0], gpu_type=model, machine=machine)
            try:
                config = run.config
            except Exception as error:
                issues.append(dict(kind="wandb_config_unavailable", run_id=run.id, error=type(error).__name__))
                config = {}
            rows.append(result_row(run, config, metadata, source, snapshot, project, flags, resume, link, issues))

    counts = Counter(row["warmup_source"] for row in rows)
    scored = Counter(row["warmup_source"] for row in rows if row.get("final_score") is not None)
    warmups = [dict(**snapshot, gpu_type=model, machine=machine, run_count=counts[name],
                    final_score_count=scored[name]) for name, snapshot in snapshots.items()]
    rows.sort(key=lambda row: (row["game"], row["seed"], row["warmup_saved_step"], row["warmup_source"], row["run_id"]))
    return dict(project=project, gpu_type=model, machine=machine, collected_at=timestamp(),
                warmups=sorted(warmups, key=lambda row: (row["game"], row["seed"], row["warmup_saved_step"])),
                runs=rows, issues=issues, wandb_scanned=len(project_runs))


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def write_reports(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    for name, columns in (("runs", RUN_COLUMNS), ("warmups", WARMUP_COLUMNS)):
        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in report[name]:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items() if key in columns})
        write_atomic(directory / f"{name}.csv", buffer.getvalue())
    write_atomic(directory / "report.json",
                 json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def wandb_api(root):
    import wandb

    if not hasattr(wandb, "Api"):
        raise ImportError("The W&B SDK is required")
    key_path = root / ".wandb_api_key"
    if not os.environ.get("WANDB_API_KEY") and key_path.is_file():
        return wandb.Api(api_key=key_path.read_text(encoding="utf-8").strip())
    return wandb.Api()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", required=True, help="GPU model of this machine, e.g. A6000, RTX3090 or titan")
    parser.add_argument("--machine", default=socket.gethostname(), help="Machine ID for the report")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B entity/project")
    parser.add_argument("--output-dir", type=Path, help="Default: training_queue/warmup_scores/GPU_TYPE/MACHINE")
    args = parser.parse_args(argv)
    try:
        model, machine = gpu_type(args.gpu_type), identifier(args.machine)
    except ValueError as error:
        parser.error(str(error))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.project):
        parser.error("--project must be entity/project")
    output = args.output_dir or ROOT / "training_queue/warmup_scores" / model / machine
    try:
        report = collect(ROOT, model, machine, args.project, wandb_api(ROOT))
    except ImportError:
        parser.exit(1, "The W&B SDK and PyYAML are required. Use the STORM conda environment.\n")
    except Exception as error:
        parser.exit(1, f"Collection failed ({type(error).__name__}). Check W&B credentials, project access and connectivity.\n")
    write_reports(output, report)
    scored = sum(item["final_score_count"] for item in report["warmups"])
    issue_counts = Counter(issue["kind"] for issue in report["issues"])
    print(f"{model} @ {machine}: {len(report['warmups'])} warmups, {len(report['runs'])} runs, "
          f"{scored} final scores -> {output}")
    if issue_counts:
        print(", ".join(f"{kind}={count}" for kind, count in sorted(issue_counts.items())) + " (see report.json)")
    return int(any(kind.endswith("_unavailable") or kind.endswith("_unreadable") for kind in issue_counts))


if __name__ == "__main__":
    sys.exit(main())
