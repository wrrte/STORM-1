"""Append hyperparameter experiments without launching training or editing STORM.yaml."""
import argparse
import json
from pathlib import Path
import sys

from queue_train import PAIR_PREFIX, ROOT, SEED_DIR, TrainingQueue, override_args, read_seed_files, gpu_id


def validate_overrides(config_path, overrides):
    # Use the trainer's own config merge and sweep validation; no training or writes.
    sys.path.insert(0, str(ROOT.parent))
    from utils import load_config
    from training_branches import configure_storm_retrieval_run, parse_storm_training_args
    import uuid

    args, extra = parse_storm_training_args([
        "-n", f"queue-validation-{uuid.uuid4().hex}", "-seed", "0",
        "-config_path", str(config_path), "-env_name", "ALE/Alien-v5",
        "-trajectory_path", "unused", *override_args(overrides)])
    configure_storm_retrieval_run(load_config(config_path), args, extra, ROOT / "train.py")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+",
                        help="Filter GPU seed files; for explicit new pairs, choose their initial GPUs")
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--games", nargs="+", help="Games to combine with every --seeds value")
    games.add_argument("--game-seeds", type=json.loads, help='Exact pairs, e.g. [["Alien",3000],["Pong",3010]]')
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "config_files/STORM.yaml")
    parser.add_argument("--warmup-offset-pairs", type=json.loads, default=[],
                        help='JSON pairs, e.g. [[30000,-2],[50000,0],["dynamic_warmup",-2]]')
    parser.add_argument("--hash-target-pairs", type=json.loads, default=[],
                        help="JSON pairs, e.g. [[9,4],[10,8]]")
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="Additional config override (repeatable); VALUE is a YAML scalar/list")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without enqueueing")
    args = parser.parse_args(argv)
    if bool(args.games) != bool(args.seeds):
        parser.error("Use --games with --seeds, or --game-seeds without --seeds")
    try:
        import yaml
        overrides = {PAIR_PREFIX + "warmup_offset_pairs": args.warmup_offset_pairs,
                     PAIR_PREFIX + "hash_target_pairs": args.hash_target_pairs}
        for key, text in args.set:
            if key in overrides:
                raise ValueError(f"Duplicate override {key}; use its dedicated pair option")
            if key == "BasicSettings.Seed":
                raise ValueError("Set the seed with --seeds / --game-seeds")
            overrides[key] = yaml.safe_load(text)
        if args.gpus is not None:
            args.gpus = [gpu_id(gpu) for gpu in args.gpus]
        validate_overrides(args.config, overrides)
        if args.games or args.game_seeds is not None:
            if not args.gpus:
                raise ValueError("--gpus is required for explicit new game/seed pairs")
            pairs = args.game_seeds if args.game_seeds is not None else [
                [game, seed] for game in args.games for seed in args.seeds]
            if not isinstance(pairs, list) or not pairs or any(
                    not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
                    for pair in pairs):
                raise ValueError("--game-seeds must be a nonempty list of [game, seed] pairs")
            if args.dry_run:
                print(f"Fresh training plan: {pairs}; GPUs for new pairs: {args.gpus}; overrides: {overrides}")
                return 0
            added, skipped = TrainingQueue().enqueue(pairs, args.gpus, args.config, overrides)
        else:
            from warmup_catalog import plan_resume_jobs
            from utils import load_config
            entries = [entry for entry in read_seed_files().values() if entry.get("enabled", True)
                       and (args.gpus is None or entry["gpu"] in args.gpus)]
            if not entries:
                raise ValueError(f"No enabled game/seed selections in {SEED_DIR}; run warmup_catalog.py first")
            mode = load_config(args.config).JointTrainAgent.Retrieval.enable
            specs, incompatible = plan_resume_jobs(entries, overrides, mode)
            for reason in incompatible:
                print(f"Skipped: {reason}", file=sys.stderr)
            # Fresh collection must be explicit; deleting a saved source must not
            # accidentally start expensive training from scratch.
            specs.extend(dict(game=entry["game"], seed=entry["seed"], gpu=entry["gpu"],
                              config=str(args.config.absolute()), overrides=overrides)
                         for entry in entries if entry.get("fresh", False))
            if not specs:
                raise ValueError("No compatible warmups for this experiment; no jobs were queued")
            if args.dry_run:
                for spec in specs:
                    print(f"GPU {spec['gpu']}: {spec['game']} / {spec['seed']}  "
                          f"resume={spec.get('resume_warmup', 'fresh')}  overrides={json.dumps(spec['overrides'])}")
                print(f"Planned {len(specs)} job(s); {len(incompatible)} incompatible selection(s); no jobs queued.")
                return 0
            added, skipped = TrainingQueue().enqueue_jobs(specs, sorted({entry["gpu"] for entry in entries}, key=int))
    except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
        parser.error(str(error))
    for job in added:
        print(f"GPU {job['gpu']}: {job['game']} / {job['seed']} -> training_queue/gpu_{job['gpu']}.jsonl")
    print(f"Added {len(added)} job(s); skipped {skipped} identical pending job(s). Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
