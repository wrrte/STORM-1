"""Collect retrieval-free STORM prefixes and launch paired retrieval sweeps."""
import json
import os
from pathlib import Path
import sys
import tempfile


PAIR_KEY = "JointTrainAgent.Retrieval.warmup_offset_pairs"
HASH_TARGET_KEY = "JointTrainAgent.Retrieval.hash_target_pairs"


def parse_hash_target_pairs(value):
    if not isinstance(value, list):
        raise ValueError("Retrieval.hash_target_pairs must be a list of [hash_bits, target] pairs")
    result = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each hash_target_pairs entry must be [hash_bits, target]")
        bits, target = pair
        if type(bits) is not int or not 1 <= bits <= 62:
            raise ValueError("Paired hash_bits must be an integer in [1, 62]")
        if type(target) is not int or target < 1:
            raise ValueError("Paired target must be a positive integer (including the anchor)")
        item = [bits, target]
        if item in result:
            raise ValueError(f"Duplicate hash/target pair: {item}")
        result.append(item)
    return result


def hash_target_variants(experiments, pairs):
    """Expand retrieval runs, retaining one baseline and explicit pair order."""
    from training_branches import RETRIEVAL_EXPERIMENTS
    pairs = parse_hash_target_pairs(pairs)
    variants = []
    for experiment in experiments:
        if pairs and "target" in RETRIEVAL_EXPERIMENTS[experiment]:
            raise ValueError(f"{experiment} overrides target; use target=1 in hash_target_pairs instead")
        selected = pairs if experiment != "baseline" and pairs else [None]
        for pair in selected:
            suffix, overrides = "", []
            if pairs:
                # All arms rebuild from their own configuration, even old shared
                # snapshots that predate the rebuild_on_resume metadata flag.
                overrides = [HASH_TARGET_KEY, "[]", "--rebuild_retrieval_on_resume"]
            if pair is not None:
                bits, target = pair
                suffix = f"_h{bits}_t{target}"
                overrides += ["JointTrainAgent.Retrieval.hash_bits", str(bits),
                              "JointTrainAgent.Retrieval.target", str(target)]
            variants.append((experiment, suffix, overrides))
    return variants


def parse_warmup_offset_pairs(value, max_steps):
    if not isinstance(value, list):
        raise ValueError("Retrieval.warmup_offset_pairs must be a list of [warmup_steps, anchor_offset] pairs")
    result = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each warmup_offset_pairs entry must be [warmup_steps, anchor_offset]")
        warmup, offset = pair
        if isinstance(warmup, str) and warmup.strip().lower() == "dynamic_warmup":
            warmup = -1
        if type(warmup) is not int or not -1 <= warmup < max_steps:
            raise ValueError("Paired warmup_steps must be 'dynamic_warmup', -1, or an integer with 0 <= warmup_steps < SampleMaxSteps")
        if type(offset) is not int:
            raise ValueError("Paired anchor_offset must be an integer (.5 offsets are unsupported)")
        item = [warmup, offset]
        if item in result:
            raise ValueError(f"Duplicate warmup/offset pair: {item}")
        result.append(item)
    return result


def experiment_names(mode):
    if isinstance(mode, list):
        return list(mode)
    return ["retrieval", "baseline"] if mode == "Both" else ["retrieval" if mode else "baseline"]


class StormWarmupSweep:
    is_warmup_sweep = True

    def __init__(self, root, header, pairs, experiments, train_path, extra_args, checkpoints=None,
                 hash_target_pairs=None):
        self.root = Path(root).resolve()
        self.header = header
        self.pairs = pairs
        self.targets = sorted({pair[0] for pair in pairs})
        self.has_dynamic = -1 in self.targets
        self.experiments = experiments
        self.train_path = str(Path(train_path).resolve())
        self.extra_args = list(extra_args)
        self.checkpoints = checkpoints if checkpoints is not None else {}
        self.baseline_first = header.get("baseline_first", False)
        self.hash_target_pairs = parse_hash_target_pairs(
            header.get("hash_target_pairs", []) if hash_target_pairs is None else hash_target_pairs)
        hash_target_variants(experiments, self.hash_target_pairs)

    @classmethod
    def collect(cls, conf, args, pairs, experiments, train_path, extra_args):
        root = Path(f"ckpt/{args.n}").resolve()
        header = {
            "version": 1, "kind": "storm_warmup_sweep", "pairs": pairs,
            "experiments": experiments, "run_name": args.base_run_name,
            "seed": args.seed, "env_name": args.env_name,
            "trajectory_path": os.path.abspath(args.trajectory_path),
            "cwd": os.getcwd(), "config": conf.dump(),
        }
        if getattr(args, "baseline_first", False):
            header["baseline_first"] = True
        if conf.JointTrainAgent.Retrieval.get("hash_target_pairs", []):
            header["hash_target_pairs"] = parse_hash_target_pairs(
                conf.JointTrainAgent.Retrieval.hash_target_pairs)
        checkpoints = {}
        index = root / "warmups.json"
        if index.exists():
            if not args.resume_from:
                raise FileExistsError(f"Warmup sweep already exists: {index}; use a new -n or --resume_from")
            saved = read_sweep_index(root)
            # The newly introduced empty default must not invalidate an older
            # collector's otherwise identical saved configuration.
            if not conf.JointTrainAgent.Retrieval.get("hash_target_pairs", []):
                legacy_conf = conf.clone()
                legacy_conf.defrost()
                if "hash_target_pairs" in legacy_conf.JointTrainAgent.Retrieval:
                    del legacy_conf.JointTrainAgent.Retrieval["hash_target_pairs"]
                legacy_conf.freeze()
                if saved.get("config") == legacy_conf.dump():
                    saved["config"] = header["config"]
            if any(saved.get(key) != item for key, item in header.items()):
                raise ValueError("Resuming a warmup collector requires the same name, seed, environment, pairs and config")
            checkpoints = saved["checkpoints"]
        return cls(root, header, pairs, experiments, train_path, extra_args, checkpoints)

    def initialize(self, resume_step):
        for target, record in self.checkpoints.items():
            if record["actual_step"] > resume_step:
                raise ValueError("Resume checkpoint precedes an already collected warmup snapshot")
            self.checkpoint_for(int(target))
        missing = [step for step in self.targets
                   if 0 <= step < resume_step and str(step) not in self.checkpoints]
        if missing:
            raise ValueError(f"Cannot reconstruct earlier warmup snapshots from resume step {resume_step}: {missing}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_index()

    def due(self, step, *, dynamic_ready=False):
        return [target for target in self.targets
                if (dynamic_ready if target == -1 else target <= step)
                and str(target) not in self.checkpoints]

    @property
    def complete(self):
        return all(str(target) in self.checkpoints for target in self.targets)

    def write_index(self):
        from training_branches import _write_json
        _write_json(self.root / "warmups.json", {
            **self.header, "checkpoints": self.checkpoints, "complete": self.complete,
        })

    def register_snapshot(self, checkpoint_dir, step, targets, conf, args, statistics_signals,
                          *, dynamic_state=None):
        from training_branches import save_storm_warmup_metadata
        snapshot_conf = conf.clone()
        snapshot_conf.defrost()
        snapshot_conf.JointTrainAgent.Retrieval.warmup_steps = targets[0]
        snapshot_conf.JointTrainAgent.Retrieval.warmup_offset_pairs = []
        snapshot_conf.JointTrainAgent.Retrieval.hash_target_pairs = []
        snapshot_conf.JointTrainAgent.Retrieval.save_warmup = False
        snapshot_conf.freeze()
        save_storm_warmup_metadata(checkpoint_dir, snapshot_conf, args, step,
                                  statistics_signals, rebuild_on_resume=True,
                                  requested_warmup_steps=targets,
                                  dynamic_warmup=dynamic_state if -1 in targets else None)
        relative_path = str(Path(checkpoint_dir).resolve().relative_to(self.root))
        for target in targets:
            self.checkpoints[str(target)] = {"checkpoint": relative_path, "actual_step": step}
            if target == -1:
                self.checkpoints[str(target)]["dynamic_warmup"] = dynamic_state
        self.write_index()

    def checkpoint_for(self, warmup):
        from training_branches import load_storm_warmup_metadata
        if str(warmup) not in self.checkpoints:
            raise ValueError(f"No saved snapshot for warmup_steps={warmup}; available targets: {sorted(map(int, self.checkpoints))}")
        path = (self.root / self.checkpoints[str(warmup)]["checkpoint"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Warmup snapshot path escapes the sweep directory")
        path, metadata = load_storm_warmup_metadata(path)
        if metadata["step"] != self.checkpoints[str(warmup)]["actual_step"]:
            raise ValueError("Warmup snapshot step does not match the sweep index")
        return path, metadata

    def commands(self, base_name=None):
        commands = {}
        pairs = self.pairs
        if self.baseline_first and self.experiments == ["baseline"]:
            # All targets are saved; only one baseline continuation is needed.
            pairs = [max(pairs, key=lambda pair: self.checkpoints[str(pair[0])]["actual_step"])]
        for warmup, offset in pairs:
            checkpoint, _ = self.checkpoint_for(warmup)
            label = "wdynamic" if warmup == -1 else f"w{warmup}"
            suffix = f"{label}_offset_{'m' + str(-offset) if offset < 0 else str(offset)}"
            for experiment, hash_suffix, hash_overrides in hash_target_variants(
                    self.experiments, self.hash_target_pairs):
                # Offset does not affect baseline: run it once per requested warmup.
                variant = f"{label}_baseline" if experiment == "baseline" else f"{suffix}{hash_suffix}_{experiment}"
                if variant in commands:
                    continue
                run_suffix = label if experiment == "baseline" else suffix + hash_suffix
                commands[variant] = [
                    sys.executable, self.train_path,
                    "-n", f"{base_name or self.header['run_name']}_{run_suffix}",
                    "-seed", str(self.header["seed"]),
                    "-config_path", str(checkpoint / "config.yaml"),
                    "-env_name", self.header["env_name"],
                    "-trajectory_path", self.header["trajectory_path"],
                    *self.extra_args,
                    PAIR_KEY, "[]",
                    "JointTrainAgent.Retrieval.warmup_steps", str(warmup),
                    "JointTrainAgent.Retrieval.anchor_offset", str(offset),
                    *hash_overrides,
                    "--branch_experiment", experiment, "--resume_from", str(checkpoint),
                ]
        return commands

    def launch(self, *, base_name=None, manifest_dir=None):
        from training_branches import launch_training_branches
        launch_training_branches(self.root, experiments=self.commands(base_name),
                                 manifest_dir=manifest_dir, working_directory=self.header["cwd"])
        raise RuntimeError("The warmup sweep supervisor unexpectedly returned")


def read_sweep_index(root):
    path = Path(root) / "warmups.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("kind") != "storm_warmup_sweep":
        raise ValueError(f"Unsupported warmup sweep index: {path}")
    return data


def launch_sweep_followup(args, extra_args, train_path, load_config):
    from training_branches import (split_retrieval_override, parse_retrieval_mode,
                                   RETRIEVAL_EXPERIMENTS, retrieval_statistics_signals)
    from storm_dynamic_warmup import DynamicWarmup
    root = Path(args.resume_warmup).resolve()
    index = read_sweep_index(root)
    remaining, mode = split_retrieval_override(extra_args, "JointTrainAgent.Retrieval.enable", allow_experiments=True)
    if not index["checkpoints"]:
        raise ValueError("This sweep has not saved any warmup snapshots yet")
    for key in ("seed", "env_name"):
        if getattr(args, key, None) is not None and getattr(args, key) != index[key]:
            raise ValueError(f"-{key} must match the saved warmup sweep")
    if args.trajectory_path is not None and os.path.abspath(args.trajectory_path) != index["trajectory_path"]:
        raise ValueError("-trajectory_path must match the saved warmup sweep")
    if args.config_path is not None:
        raise ValueError("--resume_warmup uses saved sweep configs; omit -config_path")
    record = next(iter(index["checkpoints"].values()))
    conf = load_config(root / record["checkpoint"] / "config.yaml")
    saved_dynamic_settings = DynamicWarmup(conf.JointTrainAgent.Retrieval).settings
    conf.defrost()
    if "JointTrainAgent.Retrieval.anchor_weight_r" in remaining[::2]:
        conf.JointTrainAgent.Retrieval.anchor_weight_r = None
    if remaining:
        conf.merge_from_list(remaining)
    if conf.JointTrainAgent.NumEnvs != 1:
        raise ValueError("Paired shared warmup requires NumEnvs=1")
    pairs = (conf.JointTrainAgent.Retrieval.warmup_offset_pairs
             if PAIR_KEY in remaining[::2] else index["pairs"])
    pairs = parse_warmup_offset_pairs(pairs, conf.JointTrainAgent.SampleMaxSteps)
    if not pairs:
        raise ValueError("A sweep followup requires nonempty warmup_offset_pairs")
    if any(warmup == -1 for warmup, _ in pairs):
        if DynamicWarmup(conf.JointTrainAgent.Retrieval).settings != saved_dynamic_settings:
            raise ValueError("Changing dynamic warmup settings requires a new collector; "
                             "saved dynamic checkpoints can be reused with new offsets")
    experiments = experiment_names(mode if mode is not None else parse_retrieval_mode(index["experiments"], allow_experiments=True))
    hash_pairs = (conf.JointTrainAgent.Retrieval.hash_target_pairs
                  if HASH_TARGET_KEY in remaining[::2] else index.get("hash_target_pairs", []))
    sweep = StormWarmupSweep(root, index, pairs, experiments, train_path, remaining,
                             index["checkpoints"], hash_target_pairs=hash_pairs)
    for warmup, _ in pairs:
        _, metadata = sweep.checkpoint_for(warmup)
        validate_branch_step(conf, metadata["step"])
        for experiment in experiments:
            effective = dict(conf.JointTrainAgent.Retrieval)
            effective.update(RETRIEVAL_EXPERIMENTS[experiment])
            if effective["enable"] and effective.get("trigger_mode") == "z_score":
                needed = set(retrieval_statistics_signals(effective))
                if not needed.issubset(metadata["statistics_signals"]):
                    raise ValueError(f"Saved snapshot lacks statistics for {experiment}")
    jobs = root / "followups"
    jobs.mkdir(exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="job_", dir=jobs))
    sweep.launch(base_name=args.n or f"{index['run_name']}_{job.name}", manifest_dir=job)


def validate_branch_step(conf, step):
    if step >= conf.JointTrainAgent.SampleMaxSteps:
        raise ValueError("Warmup snapshot must precede SampleMaxSteps")
    evaluation_step = conf.JointTrainAgent.get("FinalEvalStep")
    if (conf.JointTrainAgent.get("EvalMode", "inactive") in ("active", "final_only")
            and evaluation_step is not None and step >= evaluation_step):
        raise ValueError("Warmup snapshot must precede FinalEvalStep to leave training time before evaluation")
