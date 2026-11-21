"""Append hyperparameter experiments without launching training or editing STORM.yaml."""
import argparse
import json
from pathlib import Path
import sys

from queue_train import PAIR_PREFIX, ROOT, TrainingQueue, override_args


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
    parser.add_argument("--gpus", nargs="+", required=True,
                        help="GPU indices for NEW game/seed pairs; existing owners never change")
    games = parser.add_mutually_exclusive_group(required=True)
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
        pairs = args.game_seeds if args.game_seeds is not None else [
            [game, seed] for game in args.games for seed in args.seeds]
        if not isinstance(pairs, list) or not pairs or any(
                not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
                for pair in pairs):
            raise ValueError("--game-seeds must be a nonempty list of [game, seed] pairs")
        validate_overrides(args.config, overrides)
        added, skipped = TrainingQueue().enqueue(pairs, args.gpus, args.config, overrides)
    except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
        parser.error(str(error))
    for job in added:
        print(f"GPU {job['gpu']}: {job['game']} / {job['seed']} -> training_queue/gpu_{job['gpu']}.jsonl")
    print(f"Added {len(added)} job(s); skipped {skipped} identical pending job(s). Training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
