"""Combine independent completed rollout seeds, preserving episode clusters.

python probing/ram_retrieval_eval/aggregate.py --inputs output/pong/seed_* \
    --output output/pong/combined_summary.json
"""
import argparse
import json
from pathlib import Path

try:
    from .metrics import summarize_queries
except ImportError:
    from metrics import summarize_queries


def aggregate(inputs, bootstrap_samples=5000, seed=0, allow_incomplete=False):
    queries, identities, rollout_seeds, metadata_records = [], set(), set(), []
    for directory in inputs:
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["status"] not in ("complete", "complete_no_valid_pairs") and not allow_incomplete:
            raise ValueError(f"{directory}: run is not complete; use --allow-incomplete explicitly for partial data")
        identity = (metadata["game"], metadata["model_hash_before"], metadata["metric_sha256"],
                    metadata["retrieval_sha256"], json.dumps(metadata["config"], sort_keys=True),
                    metadata.get("effective_model_dtype"), metadata.get("replay_capacity"),
                    json.dumps({k: v for k, v in metadata.get("analysis_settings", {}).items()
                                if k != "seed"}, sort_keys=True))
        identities.add(identity)
        rollout_seed = metadata["evaluation_seed"]
        if rollout_seed in rollout_seeds:
            raise ValueError("Repeated rollout seed: it is not a new independent replicate")
        rollout_seeds.add(rollout_seed)
        file = directory / "queries.jsonl"
        for line in file.read_text().splitlines():
            if line.strip():
                query = json.loads(line)
                if query["evaluation_seed"] != rollout_seed:
                    raise ValueError("Query seed does not match metadata")
                queries.append(query)
        metadata_records.append(dict(path=str(directory), seed=rollout_seed, status=metadata["status"]))
    if len(identities) != 1:
        raise ValueError("Only the same game, model parameters, configuration and metric can be combined")
    summary = summarize_queries(queries, seed=seed, bootstrap_samples=bootstrap_samples)
    summary.update(game=next(iter(identities))[0], input_runs=metadata_records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("bootstrap-samples must be nonnegative")
    summary = aggregate(args.inputs, args.bootstrap_samples, args.seed, args.allow_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
