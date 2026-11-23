"""Add priority jobs to a GPU-model pool; reuse compatible warmups before collecting."""
import argparse
import json
from pathlib import Path
import sys

from queue_train import (ROOT, SEED_DIR, PAIR_PREFIX, TrainingQueue, canonical_game,
                         gpu_type, pair_key, override_args, read_seed_files, save_seed_files, locked)


def validate_overrides(config_path, overrides):
    sys.path.insert(0, str(ROOT.parent))
    from utils import load_config
    from training_branches import configure_storm_retrieval_run, parse_storm_training_args
    import uuid
    args, extra = parse_storm_training_args([
        "-n", f"queue-validation-{uuid.uuid4().hex}", "-seed", "0",
        "-config_path", str(config_path), "-env_name", "ALE/Alien-v5",
        "-trajectory_path", "unused", *override_args(overrides)])
    configure_storm_retrieval_run(load_config(config_path), args, extra, ROOT / "train.py")


def build_specs(pairs, model, config, overrides, priority):
    from utils import load_config
    from warmup_catalog import request_config, game_overrides
    from storm_warmup_sweep import parse_warmup_offset_pairs
    text = load_config(config).dump()
    template = dict(gpu_type=model, config=str(config.absolute()), config_text=text, overrides=overrides)
    conf = request_config(template)
    requested = conf.JointTrainAgent.Retrieval
    targets = parse_warmup_offset_pairs(requested.warmup_offset_pairs or [
        [requested.warmup_steps, requested.anchor_offset]], conf.JointTrainAgent.SampleMaxSteps)
    specs = []
    for game, seed in pairs:
        pair_key(game, seed)
        for target, offset in targets:
            effective = {**game_overrides(game, overrides), PAIR_PREFIX + "warmup_offset_pairs": [],
                         PAIR_PREFIX + "warmup_steps": target, PAIR_PREFIX + "anchor_offset": offset,
                         PAIR_PREFIX + "save_warmup": True}
            specs.append(dict(template, game=canonical_game(game), seed=seed,
                              overrides=effective, priority=priority))
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-type", default="A6000", help="Model pool, e.g. A6000 or RTX 3090")
    games = parser.add_mutually_exclusive_group()
    games.add_argument("--games", nargs="+")
    games.add_argument("--game-seeds", type=json.loads)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", nargs="+", type=int)
    seeds.add_argument("--new-seeds", type=int, help="Allocate globally unused seeds for this model")
    parser.add_argument("--config", type=Path, default=ROOT / "config_files/STORM.yaml")
    parser.add_argument("--warmup-offset-pairs", type=json.loads, default=[])
    parser.add_argument("--hash-target-pairs", type=json.loads, default=[])
    parser.add_argument("--set", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--priority", type=int, default=0, help="Higher numbers run first; FIFO for ties")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        import yaml
        model = gpu_type(args.gpu_type)
        if bool(args.games) != bool(args.seeds is not None or args.new_seeds is not None):
            raise ValueError("Use --games with --seeds or --new-seeds; --game-seeds supplies exact pairs")
        if args.new_seeds is not None and args.new_seeds < 1:
            raise ValueError("--new-seeds must be positive")
        overrides = {PAIR_PREFIX + "warmup_offset_pairs": args.warmup_offset_pairs,
                     PAIR_PREFIX + "hash_target_pairs": args.hash_target_pairs}
        for key, value in args.set:
            if key in overrides or key == "BasicSettings.Seed":
                raise ValueError(f"Use the dedicated option for {key}")
            overrides[key] = yaml.safe_load(value)
        overrides[PAIR_PREFIX + "save_warmup"] = True
        validate_overrides(args.config, overrides)
        queue = TrainingQueue()
        if list(SEED_DIR.glob("gpu_*.json")) and not list(SEED_DIR.glob("type_*.json")):
            raise ValueError("Migrate the old selections first: python queue_train.py migrate --gpu-type A6000")
        from warmup_catalog import discover, choose_warmup
        report = discover(model, persist=not args.dry_run)
        selections = read_seed_files()
        if args.new_seeds is not None:
            args.seeds = queue.allocate_seeds(model, args.new_seeds, reserve=not args.dry_run)
        if args.games:
            pairs = [[game, seed] for game in args.games for seed in args.seeds]
        elif args.game_seeds is not None:
            pairs = args.game_seeds
        else:
            pairs = [[entry["game"], entry["seed"]] for entry in selections.values()
                     if entry["gpu_type"] == model and entry.get("enabled", True)]
        if not isinstance(pairs, list) or not pairs or any(
            not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str) for pair in pairs):
            raise ValueError("No valid game/seed selections; supply --games and --seeds, or --game-seeds")
        with queue.connect() as db:
            for _, seed in pairs:
                queue.check_seed(db, seed, model)
        specs = build_specs(pairs, model, args.config, overrides, args.priority)
        for spec in specs:
            record, incompatible = choose_warmup(spec, report)
            source = record["source"] if record else "collect new shared warmup"
            print(f"{model} {spec['game']}/{spec['seed']} priority={args.priority}: {source}")
            if not record:
                for item in incompatible:
                    print(f"  incompatible: {item['source']}: {item['reason']}")
        if args.dry_run:
            print(f"Planned {len(specs)} job(s); no jobs enqueued or checkpoints removed.")
            return 0
        # Catalog lock also protects user selection merges from concurrent enqueuers/discovery.
        with locked(queue.directory / "catalog.lock"):
            selections = read_seed_files()
            for game, seed in pairs:
                key = pair_key(game, seed)
                entry = selections.setdefault(key, dict(game=canonical_game(game), seed=seed,
                                                         gpu_type=model, enabled=True, warmups=[]))
                if entry["gpu_type"] != model:
                    raise ValueError(f"{key} already belongs to {entry['gpu_type']}")
            queue.register_selections(selections)
            save_seed_files(selections)
            added, skipped = queue.enqueue_jobs(specs, priority=args.priority)
        print(f"Added {len(added)} job(s); skipped {skipped} duplicate pending/running job(s).")
        return 0
    except (ValueError, KeyError, TypeError, AssertionError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
