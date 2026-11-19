#!/usr/bin/env python3
r"""Resume STORM and measure anchors for additive, multiplicative or TD-only scores.

Run in the storm conda environment, from /home/choemj, for example:
    python measure_additive_anchors.py \
        --checkpoint STORM-1/ckpt/Qbert-6010_Shared

Requires a full checkpoint (training_state.pt + replay_buffer.npz), not just
model weights. Config, environment and seed are read from warmup metadata when
available. Older checkpoints can use --config, --env and --seed explicitly.

SD is the sample SD across the new updates. The 95% CI is a circular moving-
block bootstrap interval for this short run's mean, not a game/seed-level CI.
SE is the standard deviation of the bootstrap means.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
STORM = ROOT / "STORM-1"


def summarize_counts(counts, block_size=10, bootstrap_samples=10000, seed=0):
    """Bootstrap consecutive blocks without consuming the training RNG stream."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two post-warmup updates are required")
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Anchor counts must be finite, nonnegative integers")
    if not 1 <= block_size <= values.size // 2:
        raise ValueError("block_size must be between 1 and half the number of updates")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")

    n = values.size
    # Circular blocks give each observed update the same resampling weight.
    prefix = np.concatenate(([0.0], np.cumsum(np.tile(values, 2))))
    lengths = np.minimum(block_size, n - np.arange(0, n, block_size))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_samples)
    for offset in range(0, bootstrap_samples, 256):
        size = min(256, bootstrap_samples - offset)
        starts = rng.integers(0, n, size=(size, lengths.size))
        sums = prefix[starts + lengths] - prefix[starts]
        bootstrap_means[offset:offset + size] = sums.sum(axis=1) / n

    return {
        "updates": int(n),
        "total_anchor_insertions": int(values.sum()),
        "mean_anchors_per_update": float(values.mean()),
        "within_run_sd_across_updates": float(values.std(ddof=1)),
        "mean_se": float(bootstrap_means.std(ddof=1)),
        "mean_ci95": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
        "ci_method": "circular moving-block percentile bootstrap",
        "block_size_updates": block_size,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


class AnchorRecorder:
    """Record every new world-model update, including zero-anchor updates."""

    def __init__(self, stream, start_step, steps, batch_size):
        self.writer = csv.DictWriter(stream, fieldnames=["env_step", "anchors", "is_warmup"])
        self.writer.writeheader()
        self.stream = stream
        self.start_step = start_step
        self.steps = steps
        self.batch_size = batch_size
        self.counts = []

    def __call__(self, env_step, anchors, is_warmup):
        expected_step = self.start_step + len(self.counts)
        if env_step != expected_step or len(self.counts) >= self.steps:
            raise ValueError(f"Expected environment step {expected_step}; got {env_step}")
        if is_warmup:
            raise ValueError("Checkpoint has not finished retrieval warmup")
        if int(anchors) != anchors or not 0 <= anchors <= self.batch_size:
            raise ValueError(f"Invalid anchor count: {anchors}")
        self.writer.writerow({"env_step": env_step, "anchors": int(anchors), "is_warmup": False})
        self.stream.flush()
        self.counts.append(int(anchors))

    def finish(self):
        if len(self.counts) != self.steps:
            raise ValueError(f"Expected {self.steps} updates; recorded {len(self.counts)}")
        return self.counts


class LocalLogger:
    """The probe writes its own CSV; do not initialize W&B or record videos."""

    def __init__(self):
        self.tag_step = {}
        self.writer = SimpleNamespace(close=lambda: None)

    def log(self, tag, value):
        pass

    def flush_wandb(self):
        pass


def resolve_checkpoint(path):
    checkpoint = path.expanduser().resolve()
    locator = checkpoint / "warmup.json"
    if locator.is_file():
        checkpoint = (checkpoint / json.loads(locator.read_text())["checkpoint"]).resolve()
    for name in ("training_state.pt", "replay_buffer.npz"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Full checkpoint required: missing {checkpoint / name}")
    metadata_path = checkpoint / "warmup_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    return checkpoint, metadata


def resolve_settings(args, checkpoint, metadata):
    config = args.config
    if config is None:
        config = checkpoint / "config.yaml"
        if not config.is_file():
            config = STORM / "runs" / checkpoint.parent.name / "config.yaml"
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError("No saved config.yaml found; provide --config")
    env = args.env if args.env is not None else metadata.get("env_name")
    seed = args.seed if args.seed is not None else metadata.get("seed")
    if not env or seed is None:
        raise ValueError("Checkpoint has no environment/seed metadata; provide --env and --seed")
    for name, value in (("env_name", env), ("seed", seed)):
        if name in metadata and metadata[name] != value:
            raise ValueError(f"{name} differs from the saved checkpoint metadata")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    needed = {"td_error"} if getattr(args, "score", None) == "td_error_only" else {"td_error", "value_diff"}
    if metadata and not needed.issubset(metadata.get("statistics_signals", [])):
        raise ValueError(f"Saved warmup lacks EMA statistics: {sorted(needed)}")
    return config, env, seed


def configure_probe(conf, steps, score, threshold=3.5):
    if score not in {"add", "multiply", "td_error_only"} or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Require a supported score and a finite, nonnegative threshold")
    conf.defrost()
    joint = conf.JointTrainAgent
    if conf.Task != "JointTrainAgent" or joint.NumEnvs != 1 or joint.UseDemonstration:
        raise ValueError("This probe requires single-environment STORM without demonstrations")
    if joint.TrainDynamicsEverySteps != 1 or joint.TrainAgentEverySteps != 1:
        raise ValueError("This probe requires one world-model and actor update per environment step")
    if steps < 2:
        raise ValueError("At least two new steps are required for SD/CI")
    retrieval = joint.Retrieval
    retrieval.enable = True
    retrieval.save_warmup = False
    retrieval.warmup_offset_pairs = []
    retrieval.trigger_mode = "z_score"
    retrieval.value_signal = "value_diff"
    retrieval.score_combination = score
    retrieval.additive_z_score_threshold = 3.5
    retrieval.z_score_threshold = 3.5
    field = {"add": "additive_z_score_threshold", "multiply": "z_score_threshold",
             "td_error_only": "td_error_z_score_threshold"}[score]
    setattr(retrieval, field, threshold)
    joint.EvalMode = "inactive"
    joint.FinalEvalStep = None
    joint.SeedScreenSteps = 0


def run_probe(args):
    threshold = getattr(args, "threshold", 3.5)
    burn_in = getattr(args, "burn_in", 0)
    if burn_in < 0:
        raise ValueError("burn_in must be nonnegative")
    checkpoint, metadata = resolve_checkpoint(args.checkpoint)
    config_path, env, seed = resolve_settings(args, checkpoint, metadata)
    # Reuse the production models, replay restoration, scoring, and training loop.
    sys.path.insert(0, str(STORM))
    train = importlib.import_module("train")
    torch = train.torch
    conf = train.load_config(str(config_path))
    configure_probe(conf, args.steps, args.score, threshold)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for STORM; run this command on a GPU host")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train.configure_performance(conf)
    train.seed_np_torch(seed)

    dummy_env = train.build_single_env(env, conf.BasicSettings.ImageSize, seed=0)
    try:
        action_dim = dummy_env.action_space.n
    finally:
        dummy_env.close()
    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    joint = conf.JointTrainAgent
    replay = train.ReplayBuffer(
        obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
        num_envs=1, max_length=joint.BufferMaxLength, warmup_length=joint.BufferWarmUp,
        store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        vectorized_sampling=conf.Performance.VectorizedReplaySampling,
    )
    (start, _, last_rebuild, warmup_finished, rewards, dynamic_step,
     retrieval_state, rng_state, _) = train.load_full_checkpoint(str(checkpoint), world_model, agent, replay)
    required_fields = ["ema_mean", "ema_var"]
    if args.score != "td_error_only":
        required_fields += ["ema_vd_mean", "ema_vd_var"]
    if retrieval_state is None or not all(key in retrieval_state for key in required_fields):
        raise ValueError(f"Full checkpoint must contain EMA statistics: {required_fields}")
    fixed_warmup = joint.Retrieval.warmup_steps
    if not warmup_finished and not (fixed_warmup >= 0 and start >= fixed_warmup):
        raise ValueError("Choose a checkpoint at or after the end of retrieval warmup")
    if not replay.ready():
        raise ValueError("Restored replay buffer is not ready for training")

    stop = start + burn_in + args.steps
    joint.SampleMaxSteps = stop
    joint.SaveEverySteps = stop + 1  # No intermediate saves or videos in [start, stop).
    rebuild = bool(metadata.get("rebuild_on_resume", False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or STORM / "results" / f"anchor_probe_{args.score}_{stamp}").resolve()
    # A fresh directory prevents mixing old or partial traces with this result.
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(conf.dump(), encoding="utf-8")
    train.conf = conf
    train.args = SimpleNamespace(n="anchor_probe")
    previous_cwd = Path.cwd()
    try:
        os.chdir(output)  # All incidental training artifacts stay outside the source checkpoint.
        with (output / "anchors.csv").open("w", newline="") as stream:
            recorder = AnchorRecorder(stream, start, burn_in + args.steps, joint.BatchSize)
            completed = train.joint_train_world_model_agent(
                env_name=env, max_steps=stop, num_envs=1, image_size=conf.BasicSettings.ImageSize,
                replay_buffer=replay, world_model=world_model, agent=agent,
                train_dynamics_every_steps=1, train_agent_every_steps=1,
                batch_size=joint.BatchSize, demonstration_batch_size=0, batch_length=joint.BatchLength,
                imagine_batch_size=joint.ImagineBatchSize, imagine_demonstration_batch_size=0,
                imagine_context_length=joint.ImagineContextLength,
                imagine_batch_length=joint.ImagineBatchLength,
                save_every_steps=joint.SaveEverySteps, seed=seed, logger=LocalLogger(),
                eval_mode="inactive", resume_step=start, resume_last_rebuild_step=last_rebuild,
                warmup_finished_resume=True, episode_rewards_resume=rewards,
                dynamic_warmup_met_step_resume=dynamic_step,
                retrieval_state_resume=retrieval_state, rng_state_resume=rng_state,
                rebuild_retrieval_on_resume=rebuild,
                on_world_model_update=recorder, save_final_checkpoint=False,
            )
            if not completed:
                raise RuntimeError("Training stopped before the requested probe completed")
            counts = recorder.finish()[burn_in:]
    finally:
        os.chdir(previous_cwd)

    summary = summarize_counts(counts, args.block_size, args.bootstrap_samples, args.bootstrap_seed)
    summary.update({
        "checkpoint": str(checkpoint), "source_config": str(config_path),
        "env": env, "seed": seed, "score_combination": args.score, "threshold": threshold,
        "env_step_start": start + burn_in, "env_step_stop_exclusive": stop,
        "probe_env_step_start": start, "burn_in_updates": burn_in,
        "environment_reset_at_resume": True, "rebuild_retrieval_on_resume": rebuild,
        "scope": "Within this short resumed run; not uncertainty across games or training seeds.",
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lo, hi = summary["mean_ci95"]
    label = {"add": "Additive", "multiply": "Multiplicative", "td_error_only": "TD-error only"}[args.score]
    row = (f"| {label} (threshold {threshold:g}) | {summary['mean_anchors_per_update']:.3f} | "
           f"{summary['within_run_sd_across_updates']:.3f} | {summary['mean_se']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    report = (
        "| Setting | Mean anchors per update | Within-run SD across updates | SE | 95% CI of the mean |\n"
        "|---|---:|---:|---:|---|\n" + row + "\n\n"
        f"{env}, seed {seed}, {args.steps} measured updates at environment steps [{start + burn_in}, {stop}); "
        f"{burn_in} preceding updates excluded. "
        f"Sample SD (ddof=1); circular block bootstrap, block size {args.block_size}, "
        f"{args.bootstrap_samples} resamples. Zero-anchor updates are included.\n\n"
        "Counts are new anchor queue insertions, including repeated replay positions. "
        "The environment resets on resume, as in train.py. The interval describes only "
        "this short continuation under a locally stationary block-bootstrap approximation; "
        "it does not estimate variation across games or training seeds. For the rebuttal, "
        "compare both score rules using the same checkpoint and probe window.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nResults: {output}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full checkpoint or reusable warmup run directory")
    parser.add_argument("--steps", type=int, default=100, help="Additional environment steps (default: 100)")
    parser.add_argument("--burn-in", type=int, default=0, help="Updates before the measured window (default: 0)")
    parser.add_argument("--config", type=Path, help="Saved YAML config, if automatic discovery is unavailable")
    parser.add_argument("--env", help="ALE game, e.g. ALE/Pong-v5 (normally read from metadata)")
    parser.add_argument("--seed", type=int, help="Original training seed (normally read from metadata)")
    parser.add_argument("--score", choices=("add", "multiply", "td_error_only"), default="add")
    parser.add_argument("--threshold", type=float, default=3.5, help="Threshold for the selected score (default: 3.5)")
    parser.add_argument("--output", type=Path, help="New output directory; default: timestamped STORM-1/results directory")
    parser.add_argument("--block-size", type=int, default=10, help="Consecutive updates per bootstrap block (default: 10)")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.steps < 2 or not 1 <= args.block_size <= args.steps // 2 or args.bootstrap_samples < 100:
        parser.error("Require steps >= 2, 1 <= block-size <= steps/2, and bootstrap-samples >= 100")
    if args.burn_in < 0 or not math.isfinite(args.threshold) or args.threshold < 0:
        parser.error("Require burn-in >= 0 and a finite, nonnegative threshold")
    try:
        run_probe(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
