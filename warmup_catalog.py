"""Discover reusable warmups and maintain editable game/seed files for each GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from queue_train import (ROOT, QUEUE_DIR, SEED_DIR, PAIR_PREFIX, TrainingQueue,
                         canonical_game, gpu_id, json_line, locked, pair_key,
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


def discover(gpus, *, root=ROOT, queue_dir=QUEUE_DIR, seed_dir=SEED_DIR):
    gpus = list(dict.fromkeys(gpu_id(gpu) for gpu in gpus))
    if not gpus:
        raise ValueError("At least one GPU is required")
    candidates = {path for path in (root / "ckpt").glob("*/shared_warmup_*") if path.is_dir()}
    candidates.update(path.parent for path in (root / "ckpt").rglob("warmup_metadata.json"))
    records, excluded = [], []
    for source in sorted(candidates):
        try:
            records.append(inspect_warmup(source, root=root, queue_dir=queue_dir))
        except Exception as error:
            excluded.append(dict(source=relative(source, root), reason=f"{type(error).__name__}: {error}"))
    queue = TrainingQueue(queue_dir, seed_dir)
    with locked(queue.directory / "queue.lock"):
        registry = queue.registry()
        selections = read_seed_files(seed_dir)
        report_path = queue.directory / "warmup_inventory.json"
        previous = json.loads(report_path.read_text()) if report_path.exists() else {}
        known = {item["source"] for item in previous.get("available", [])}
        for record in records:
            materialize(record)
            key = pair_key(record["game"], record["seed"])
            if key not in selections:
                # A deliberately removed seed/source stays removed on rediscovery.
                if record["source"] in known:
                    continue
                counts = {gpu: sum(entry["gpu"] == gpu for entry in selections.values()) for gpu in gpus}
                gpu = registry["owners"].get(key, min(gpus, key=counts.get))
                selections[key] = dict(game=record["game"], seed=record["seed"], gpu=gpu, enabled=True, warmups=[])
            entry = selections[key]
            if registry["owners"].get(key, entry["gpu"]) != entry["gpu"]:
                raise ValueError(f"{key}: seed file disagrees with permanent GPU ownership")
            if entry.get("fresh", False):
                continue
            if record["source"] not in known and record["source"] not in entry.get("warmups", []):
                entry.setdefault("warmups", []).append(record["source"])
        seed_dir.mkdir(parents=True, exist_ok=True)
        for gpu in sorted(set(gpus) | {entry["gpu"] for entry in selections.values()}, key=int):
            entries = [{key: value for key, value in entry.items() if key != "gpu"}
                       for entry in selections.values() if entry["gpu"] == gpu]
            write_atomic(seed_dir / f"gpu_{gpu}.json", json.dumps(dict(gpu=gpu, game_seeds=entries), indent=2) + "\n")
        report = dict(available=[{key: value for key, value in item.items() if key not in ("config", "metadata")}
                                 for item in records], excluded=excluded,
                      note="Assignments are for future queue jobs; historical physical GPU IDs are unknown.")
        write_atomic(report_path, json.dumps(report, indent=2) + "\n")
    return report


def resolve_prepared(source, *, root=ROOT, queue_dir=QUEUE_DIR):
    from training_branches import load_storm_warmup_metadata
    source = source_path(source, root)
    resume = source if (source / "warmup_metadata.json").exists() else imported_path(source, queue_dir)
    checkpoint, metadata = load_storm_warmup_metadata(resume)
    if resume != source and metadata.get("imported_from") != str(source):
        raise ValueError("Imported checkpoint source does not match; rerun warmup_catalog.py")
    return checkpoint, metadata


def plan_resume_jobs(selections, overrides, mode, *, root=ROOT, queue_dir=QUEUE_DIR):
    from utils import load_config
    from training_branches import (parse_retrieval_mode, parse_storm_training_args,
                                   prepare_storm_warmup_followup)
    from storm_warmup_sweep import experiment_names, parse_warmup_offset_pairs
    from storm_dynamic_warmup import DynamicWarmup

    names = experiment_names(parse_retrieval_mode(overrides.get(PAIR_PREFIX + "enable", mode), allow_experiments=True))
    specs, skipped = [], []
    for entry in selections:
        for source in dict.fromkeys(entry.get("warmups", [])):
            checkpoint, metadata = resolve_prepared(source, root=root, queue_dir=queue_dir)
            if pair_key(entry["game"], entry["seed"]) != pair_key(canonical_game(metadata["env_name"]), metadata["seed"]):
                raise ValueError(f"Seed file identity does not match {source}")
            conf = load_config(checkpoint / "config.yaml")
            saved = conf.JointTrainAgent.Retrieval
            targets = metadata.get("requested_warmup_steps", [saved.warmup_steps])
            requested = overrides[PAIR_PREFIX + "warmup_offset_pairs"] or [[
                overrides.get(PAIR_PREFIX + "warmup_steps", targets[0]),
                overrides.get(PAIR_PREFIX + "anchor_offset", saved.anchor_offset)]]
            pairs = parse_warmup_offset_pairs(requested, conf.JointTrainAgent.SampleMaxSteps)
            for target, offset in pairs:
                if target not in targets:
                    skipped.append(f"{source}: requested warmup {target}; saved targets {targets}")
                    continue
                effective = {**overrides, PAIR_PREFIX + "enable": names,
                             PAIR_PREFIX + "warmup_offset_pairs": [],
                             PAIR_PREFIX + "warmup_steps": target, PAIR_PREFIX + "anchor_offset": offset}
                # Architecture/replay dimensions must still match the saved weights.
                for key, value in effective.items():
                    if key.startswith("Models.") or key in ("BasicSettings.ImageSize", "JointTrainAgent.NumEnvs",
                                                            "JointTrainAgent.BufferMaxLength"):
                        original = conf
                        for part in key.split("."):
                            original = original[part]
                        if original != value:
                            raise ValueError(f"Cannot change {key} when reusing {source}")
                if target == -1:
                    changed = conf.clone()
                    changed.defrost()
                    for key, value in effective.items():
                        if key.startswith(PAIR_PREFIX + "dynamic_") or key in (PAIR_PREFIX + "min_warmup_steps", PAIR_PREFIX + "max_warmup_steps"):
                            changed.JointTrainAgent.Retrieval[key.removeprefix(PAIR_PREFIX)] = value
                    if DynamicWarmup(changed.JointTrainAgent.Retrieval).settings != DynamicWarmup(saved).settings:
                        raise ValueError("Changing dynamic warmup settings requires a new collector")
                args, extra = parse_storm_training_args([
                    "--resume_warmup", str(checkpoint), "-n", "queue-validation",
                    "-seed", str(entry["seed"]), "-env_name", metadata["env_name"], *override_args(effective)])
                try:
                    prepare_storm_warmup_followup(args, extra, root / "train.py", load_config)
                except ValueError as error:
                    if "lacks the statistics" not in str(error):
                        raise
                    skipped.append(f"{source}: {error}")
                    continue
                specs.append(dict(game=entry["game"], seed=entry["seed"], gpu=entry["gpu"],
                                  config=str(checkpoint / "config.yaml"), resume_warmup=str(checkpoint),
                                  warmup_source=source, overrides=effective))
    return specs, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=[str(gpu) for gpu in range(8)])
    args = parser.parse_args(argv)
    report = discover(args.gpus)
    print(f"Available: {len(report['available'])} warmup(s); excluded: {len(report['excluded'])}")
    print(f"Edit {SEED_DIR}/gpu_N.json to select seeds/warmups. No training jobs were queued.")
    for item in report["excluded"]:
        print(f"Excluded {item['source']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
