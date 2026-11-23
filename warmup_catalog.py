"""Discover reusable warmups and maintain game/seed selections for each GPU model."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import zipfile

from queue_train import (ROOT, QUEUE_DIR, SEED_DIR, PAIR_PREFIX, TrainingQueue,
                         canonical_game, gpu_type, json_line, locked, pair_key, save_seed_files,
                         read_seed_files, write_atomic, override_args)

sys.path.insert(0, str(ROOT.parent))


def relative(path, root=ROOT):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def source_path(source, root=ROOT):
    path = (root / source).resolve()
    if (path / "warmup.json").exists():
        path = (path / json.loads((path / "warmup.json").read_text())["checkpoint"]).resolve()
    return path


def imported_path(source, queue_dir=QUEUE_DIR):
    digest = hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:20]
    return Path(queue_dir) / "imported_warmups" / digest / "snapshot"


def saved_identity(source):
    """Recover identity from saved command arguments, never from a folder-name guess."""
    path = source / "branches.json"
    if not path.exists():
        raise ValueError("Missing warmup_metadata.json and branches.json")
    manifest = json.loads(path.read_text())
    identities = []
    for command in manifest["commands"].values():
        def arg(key):
            return command[command.index(key) + 1]
        identities.append(dict(run_name=arg("-n"), seed=int(arg("-seed")),
                               env_name=arg("-env_name"), trajectory_path=arg("-trajectory_path"),
                               config=arg("-config_path")))
    if not identities or any(item != identities[0] for item in identities[1:]):
        raise ValueError("Saved branch commands disagree about checkpoint identity/config")
    return identities[0]


def inspect_warmup(source, *, root=ROOT, queue_dir=QUEUE_DIR):
    """CPU inspection of state, replay headers and saved configuration; no GPU work."""
    import numpy as np
    import torch
    from utils import load_config
    from training_branches import (parse_retrieval_mode, RETRIEVAL_EXPERIMENTS,
                                   retrieval_statistics_signals)
    from storm_warmup_sweep import experiment_names, validate_branch_step

    source = source_path(source, root)
    for name in ("training_state.pt", "replay_buffer.npz"):
        if not (source / name).is_file() or (source / name).stat().st_size == 0:
            raise ValueError(f"Missing/empty {name}")
    meta_path = source / "warmup_metadata.json"
    modern = meta_path.is_file()
    if modern:
        metadata = json.loads(meta_path.read_text())
        if metadata.get("version") != 1:
            raise ValueError("Unsupported warmup metadata version")
        config_path = source / "config.yaml"
    else:
        identity = saved_identity(source)
        config_path = Path(identity.pop("config"))
        metadata = dict(identity, version=1, cwd=str(root))
    pair_key(canonical_game(metadata["env_name"]), metadata["seed"])
    conf = load_config(config_path)
    if conf.JointTrainAgent.NumEnvs != 1:
        raise ValueError("Shared warmup requires NumEnvs=1")
    state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
    required = {"total_steps", "world_model_state_dict", "agent_state_dict",
                "world_model_optimizer_state_dict", "agent_optimizer_state_dict",
                "world_model_scaler_state_dict", "agent_scaler_state_dict",
                "agent_lowerbound_ema_scalar", "agent_upperbound_ema_scalar", "random_states"}
    if required - state.keys():
        raise ValueError(f"Incomplete training state: {sorted(required - state.keys())}")
    if not state.get("shared_warmup") or not state.get("warmup_finished"):
        raise ValueError("Checkpoint is not a finished shared warmup")
    if not {"python", "numpy", "torch"} <= state["random_states"].keys():
        raise ValueError("Incomplete RNG state")
    step = state["total_steps"]
    if type(step) is not int or step <= 0 or (modern and metadata["step"] != step):
        raise ValueError("Invalid/inconsistent saved warmup step")
    validate_branch_step(conf, step)
    with np.load(source / "replay_buffer.npz", allow_pickle=False) as replay:
        required_replay = {"obs", "action", "reward", "termination", "length", "last_pointer", "num_envs", "max_length"}
        if required_replay - set(replay.files):
            raise ValueError("Incomplete replay buffer")
        length, envs, capacity = (int(replay[key]) for key in ("length", "num_envs", "max_length"))
        if envs != 1 or not conf.JointTrainAgent.BatchLength <= length <= capacity:
            raise ValueError("Invalid replay buffer dimensions/length")
        if not 0 <= int(replay["last_pointer"]) < capacity:
            raise ValueError("Invalid replay buffer pointer")
    # Read array headers without inflating gigabytes of observations into memory.
    with zipfile.ZipFile(source / "replay_buffer.npz") as archive:
        for name in ("obs", "action", "reward", "termination"):
            with archive.open(name + ".npy") as handle:
                version = np.lib.format.read_magic(handle)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(handle)
                if shape[:2] != (length, envs) or dtype.hasobject:
                    raise ValueError(f"Invalid replay {name} header")
                if archive.getinfo(name + ".npy").file_size != handle.tell() + int(np.prod(shape)) * dtype.itemsize:
                    raise ValueError(f"Truncated replay {name}")
    retrieval = conf.JointTrainAgent.Retrieval
    if retrieval.save_warmup:
        expected = {"td_error", "value_diff", "value"}
    else:
        names = experiment_names(parse_retrieval_mode(retrieval.enable, allow_experiments=True))
        expected = {"td_error"}
        for name in names:
            if RETRIEVAL_EXPERIMENTS[name]["enable"]:
                expected.update(retrieval_statistics_signals({**dict(retrieval), **RETRIEVAL_EXPERIMENTS[name]}))
    fields = {"td_error": ("ema_mean", "ema_var"), "value_diff": ("ema_vd_mean", "ema_vd_var"),
              "value": ("ema_value_mean", "ema_value_var")}
    stored = state.get("retrieval_state") or {}
    available = {name for name, keys in fields.items() if all(key in stored for key in keys)}
    signals = set(metadata["statistics_signals"]) if modern else expected & available
    if not signals <= available:
        raise ValueError("Metadata claims statistics missing from the checkpoint")
    for name in signals:
        mean, variance = (np.asarray(stored[key]) for key in fields[name])
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
            raise ValueError(f"Invalid {name} statistics")
    metadata = dict(metadata, step=step, statistics_signals=sorted(signals))
    if not modern:
        metadata.update(imported_from=str(source))
    resume = source if modern else imported_path(source, queue_dir)
    return dict(source=relative(source, root), game=canonical_game(metadata["env_name"]),
                seed=metadata["seed"], step=step, warmup_steps=metadata.get("requested_warmup_steps", [retrieval.warmup_steps]),
                statistics_signals=sorted(signals), replay_length=length, legacy=not modern,
                resume_warmup=str(resume), metadata=metadata, config=conf.dump())


def materialize(record):
    """Add only metadata/config and file references; keep original checkpoints intact."""
    if not record["legacy"]:
        return
    destination = Path(record["resume_warmup"])
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(record["metadata"]["imported_from"])
    for name in ("training_state.pt", "replay_buffer.npz"):
        target = destination / name
        if target.is_symlink():
            if target.resolve() != source / name:
                raise ValueError(f"Unexpected checkpoint link: {target}")
        elif target.exists():
            raise ValueError(f"Refusing to replace {target}")
        else:
            target.symlink_to(source / name)
    write_atomic(destination / "config.yaml", record["config"])
    write_atomic(destination / "warmup_metadata.json", json_line(record["metadata"]))


def source_stamp(source):
    return [(name, (source / name).stat().st_size, (source / name).stat().st_mtime_ns)
            for name in ("training_state.pt", "replay_buffer.npz", "warmup_metadata.json", "config.yaml", "branches.json", "queue_origin.json")
            if (source / name).exists()]


def warmup_sources(root=ROOT, *, include_unpublished=False):
    """Shared source enumeration for the local queue and cross-machine inventory."""
    candidates = {path for path in (root / "ckpt").glob("*/shared_warmup_*")
                  if path.is_dir() and not path.name.endswith(".retired") and
                  (include_unpublished or (path / "warmup_metadata.json").is_file() or
                   (path / "branches.json").is_file())}
    candidates.update(path.parent for path in (root / "ckpt").rglob("warmup_metadata.json")
                      if not path.parent.name.endswith(".retired"))
    return sorted(candidates)


def discover(model, *, root=ROOT, queue_dir=QUEUE_DIR, seed_dir=SEED_DIR, refresh=False, persist=True):
    """Incremental discovery on enqueue and execution; metadata is the publication marker."""
    model = gpu_type(model)
    queue = TrainingQueue(queue_dir, seed_dir)
    with locked(queue.directory / "catalog.lock"):
        report_path = queue.directory / "warmup_inventory.json"
        previous = json.loads(report_path.read_text()) if report_path.exists() else {}
        cached = {item["source"]: item for item in previous.get("available", [])}
        candidates = warmup_sources(root)
        selections = read_seed_files(seed_dir)
        records, excluded = [], []
        for source in sorted(candidates):
            name, stamp = relative(source, root), source_stamp(source)
            try:
                record = cached.get(name)
                if refresh or not record or record.get("stamp") != json.loads(json.dumps(stamp)) or "config" not in record:
                    record = inspect_warmup(source, root=root, queue_dir=queue_dir)
                    record["stamp"] = stamp
                origin = source / "queue_origin.json"
                kind = json.loads(origin.read_text())["gpu_type"] if origin.exists() else selections.get(
                    pair_key(record["game"], record["seed"]), {}).get("gpu_type", model)
                record["gpu_type"] = gpu_type(kind)
                if persist:
                    materialize(record)
                records.append(record)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError, EOFError, zipfile.BadZipFile) as error:
                excluded.append(dict(source=name, reason=f"{type(error).__name__}: {error}"))
        # The same lock protects catalog -> selection updates; never override user disabled choices.
        for record in records:
            key = pair_key(record["game"], record["seed"])
            entry = selections.setdefault(key, dict(game=record["game"], seed=record["seed"],
                                                     gpu_type=record["gpu_type"], enabled=True, warmups=[]))
            if entry["gpu_type"] != record["gpu_type"]:
                raise ValueError(f"Warmup GPU type conflicts with seed ownership: {record['source']}")
            if record["source"] not in entry.setdefault("warmups", []):
                entry["warmups"].append(record["source"])
            entry.pop("fresh", None)
        if persist:
            queue.register_selections(selections)
            save_seed_files(selections, seed_dir)
        report = dict(available=records, excluded=excluded,
                      note="Legacy sources use the migrated seed type; original hardware is not inferred.")
        if persist:
            write_atomic(report_path, json.dumps(report, indent=2) + "\n")
    return report


def resolve_prepared(source, *, root=ROOT, queue_dir=QUEUE_DIR):
    from training_branches import load_storm_warmup_metadata
    source = source_path(source, root)
    resume = source if (source / "warmup_metadata.json").exists() else imported_path(source, queue_dir)
    checkpoint, metadata = load_storm_warmup_metadata(resume)
    if resume != source and metadata.get("imported_from") != str(source):
        raise ValueError("Imported checkpoint source does not match; refresh the catalog")
    return checkpoint, metadata


def request_config(job):
    from yacs.config import CfgNode
    from training_branches import parse_retrieval_mode
    conf = CfgNode.load_cfg(job["config_text"])
    overrides = dict(job["overrides"])
    mode = overrides.pop(PAIR_PREFIX + "enable", conf.JointTrainAgent.Retrieval.enable)
    if PAIR_PREFIX + "anchor_weight_r" in overrides:
        conf.JointTrainAgent.Retrieval.anchor_weight_r = None
    conf.merge_from_list(override_args(overrides))
    conf.JointTrainAgent.Retrieval.enable = parse_retrieval_mode(mode, allow_experiments=True)
    return conf


def compatibility(record, job):
    """Only post-warmup retrieval/evaluation choices may differ from the saved prefix."""
    from yacs.config import CfgNode
    from training_branches import parse_retrieval_mode, RETRIEVAL_EXPERIMENTS, retrieval_statistics_signals
    from storm_warmup_sweep import experiment_names
    saved = CfgNode.load_cfg(record["config"])
    requested = request_config(job)
    if record["gpu_type"] != job["gpu_type"]:
        return "GPU type mismatch"
    target = requested.JointTrainAgent.Retrieval.warmup_steps
    if target not in record["warmup_steps"]:
        return f"warmup target {target} not in {record['warmup_steps']}"
    if requested.JointTrainAgent.SampleMaxSteps <= record["step"]:
        return "warmup is past the requested training end"
    # Model/optimizer/data-collection settings determine the reusable prefix.
    for key in ("Models",):
        if dict(saved[key]) != dict(requested[key]):
            return f"changed {key}"
    for section, keys in {
        "BasicSettings": ["ImageSize", "ReplayBufferOnGPU"],
        "JointTrainAgent": ["NumEnvs", "BufferMaxLength", "BufferWarmUp", "BatchSize", "BatchLength",
                           "DemonstrationBatchSize", "ImagineBatchSize", "ImagineDemonstrationBatchSize",
                           "ImagineContextLength", "ImagineBatchLength", "TrainDynamicsEverySteps",
                           "TrainAgentEverySteps", "UseDemonstration"],
    }.items():
        for key in keys:
            if saved[section].get(key) != requested[section].get(key):
                return f"changed {section}.{key}"
    before, after = saved.JointTrainAgent.Retrieval, requested.JointTrainAgent.Retrieval
    if before.ema_alpha != after.ema_alpha:
        return "changed statistics ema_alpha"
    if target == -1:
        from storm_dynamic_warmup import DynamicWarmup
        if DynamicWarmup(before).settings != DynamicWarmup(after).settings:
            return "changed dynamic warmup settings"
    for name in experiment_names(parse_retrieval_mode(after.enable, allow_experiments=True)):
        effective = {**dict(after), **RETRIEVAL_EXPERIMENTS[name]}
        if effective["enable"] and effective.get("trigger_mode", "absolute") == "z_score":
            if not set(retrieval_statistics_signals(effective)) <= set(record["statistics_signals"]):
                return f"missing statistics for {name}"
    return None


def choose_warmup(job, report):
    candidates, incompatible = [], []
    for record in report["available"]:
        if pair_key(record["game"], record["seed"]) != pair_key(job["game"], job["seed"]):
            continue
        reason = compatibility(record, job)
        if reason:
            incompatible.append(dict(source=record["source"], reason=reason))
        else:
            candidates.append(record)
    # Stable choice across retries; favor snapshots usable by more experiment presets.
    candidates.sort(key=lambda record: (-len(record["statistics_signals"]), record["step"], record["source"]))
    return (candidates[0] if candidates else None), incompatible


def branch_commands(job, record, *, root=ROOT):
    from training_branches import parse_storm_training_args, prepare_storm_warmup_followup
    from storm_warmup_sweep import experiment_names
    conf = request_config(job)
    # Apply the whole requested configuration, not unrelated defaults from a saved experiment.
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = prefix + key
            if isinstance(item, dict):
                result.update(flatten(item, path + "."))
            else:
                result[path] = item
        return result
    effective = flatten(conf)
    effective[PAIR_PREFIX + "enable"] = experiment_names(conf.JointTrainAgent.Retrieval.enable)
    effective[PAIR_PREFIX + "save_warmup"] = False
    effective[PAIR_PREFIX + "warmup_offset_pairs"] = []
    args, extra = parse_storm_training_args([
        "--resume_warmup", record["resume_warmup"], "-n", job["name"],
        "-seed", str(job["seed"]), "-env_name", f"ALE/{job['game']}-v5", *override_args(effective)])
    _, _, commands = prepare_storm_warmup_followup(args, extra, root / "train.py", __import__("utils").load_config)
    for command in commands.values():
        command[0] = job["python"]
    return commands


def prune_replaced(job, replacement, incompatible, queue, *, root=ROOT):
    """Delete incompatible sources only after a verified replacement, with live uses pinned."""
    deleted = []
    with locked(queue.directory / "catalog.lock"):
        report_path = queue.directory / "warmup_inventory.json"
        report = json.loads(report_path.read_text())
        records = {item["source"]: item for item in report["available"]}
        # Active jobs retain their exact source. Pending experiments retain any compatible source.
        others = [item for item in queue.jobs(statuses=("pending", "running", "failed")) if item["id"] != job["id"]]
        for item in incompatible:
            record = records.get(item["source"])
            if not record or record["source"] == replacement["source"]:
                continue
            # Different warmup lengths are useful independent assets, not obsolete snapshots.
            if not set(record["warmup_steps"]) <= set(replacement["warmup_steps"]):
                continue
            if any(other.get("warmup_source") == record["source"] or (
                pair_key(other["game"], other["seed"]) == pair_key(record["game"], record["seed"])
                and compatibility(record, other) is None) for other in others):
                continue
            source = source_path(record["source"], root)
            if not source.is_relative_to((root / "ckpt").resolve()) or not source.name.startswith("shared_warmup_"):
                continue
            # Atomic rename makes the source unavailable before deleting its large files.
            tombstone = source.with_name(source.name + ".retired")
            if tombstone.exists():
                continue
            os.replace(source, tombstone)
            shutil.rmtree(tombstone)
            adapter = imported_path(source, queue.directory)
            if adapter.exists():
                shutil.rmtree(adapter.parent)
            deleted.append(record["source"])
        if deleted:
            selections = read_seed_files(queue.seed_directory)
            for entry in selections.values():
                entry["warmups"] = [source for source in entry.get("warmups", []) if source not in deleted]
            save_seed_files(selections, queue.seed_directory)
            report["available"] = [record for record in report["available"] if record["source"] not in deleted]
            write_atomic(report_path, json.dumps(report, indent=2) + "\n")
            with queue.connect() as db:
                queue.event(db, job["id"], "pruned", sources=deleted, replacement=replacement["source"])
    return deleted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", default="A6000", help="Type assigned to previously unregistered legacy sources")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    report = discover(args.gpu_type, refresh=args.refresh)
    print(f"Available: {len(report['available'])} warmup(s); excluded: {len(report['excluded'])}")
    for item in report["excluded"]:
        print(f"Excluded {item['source']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
