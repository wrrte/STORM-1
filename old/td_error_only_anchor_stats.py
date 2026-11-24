"""Mean anchor insertions per step, bootstrap CI and SE from local TD-only logs.

Reads frozen campaign manifests and TensorBoard events; never starts training or
contacts W&B. Run with the storm environment. See td_error_only_anchor_stats.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
METRIC = 'Retrieval/triggered_anchors_step'
CACHE_VERSION = 1


def mean_statistics(values, replicates, confidence, **extra):
    alpha = (1 - confidence) / 2
    return {
        'mean_anchors_per_step': float(np.mean(values)),
        'se': float(np.std(replicates, ddof=1)) if replicates is not None else None,
        'ci_low': float(np.quantile(replicates, alpha)) if replicates is not None else None,
        'ci_high': float(np.quantile(replicates, 1 - alpha)) if replicates is not None else None,
        'confidence': confidence,
        **extra,
    }


def block_statistics(counts, block_size=1000, bootstrap_samples=10000,
                     confidence=0.95, seed=0):
    """Circular moving-block percentile CI; SE is the SD of bootstrap means."""
    values = np.asarray(counts, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()):
        raise ValueError('At least two finite nonnegative integer anchor counts are required')
    if block_size < 1 or bootstrap_samples < 100 or not 0 < confidence < 1:
        raise ValueError('Require block_size >= 1, bootstrap_samples >= 100 and 0 < confidence < 1')
    n = len(values)
    block_size = min(block_size, n // 2)
    # Prefix sums avoid materializing B resamples of n individual updates.
    prefix = np.concatenate(([0.0], np.cumsum(np.tile(values, 2))))
    lengths = np.minimum(block_size, n - np.arange(0, n, block_size))
    rng = np.random.default_rng(seed)
    replicates = np.empty(bootstrap_samples)
    for offset in range(0, bootstrap_samples, 256):
        size = min(256, bootstrap_samples - offset)
        starts = rng.integers(0, n, size=(size, len(lengths)))
        totals = (prefix[starts + lengths] - prefix[starts]).sum(axis=1)
        replicates[offset:offset + size] = totals / n
    return mean_statistics(
        values, replicates, confidence,
        updates=n, total_anchor_insertions=int(values.sum()),
        sd_across_updates=float(values.std(ddof=1)),
        block_size_updates=block_size,
        ci_method='circular moving-block percentile bootstrap',
        uncertainty_unit='consecutive updates within this observed training run',
    )


def independent_statistics(values, bootstrap_samples=10000, confidence=0.95, seed=0):
    """Bootstrap independent run/game means, never individual training updates."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('At least one finite mean is required')
    if bootstrap_samples < 100 or not 0 < confidence < 1:
        raise ValueError('Require bootstrap_samples >= 100 and 0 < confidence < 1')
    replicates = None
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        replicates = np.empty(bootstrap_samples)
        for offset in range(0, bootstrap_samples, 256):
            size = min(256, bootstrap_samples - offset)
            indices = rng.integers(0, len(values), size=(size, len(values)))
            replicates[offset:offset + size] = values[indices].mean(axis=1)
    return mean_statistics(values, replicates, confidence, independent_units=len(values))


def discover_runs(root, manifests, include_running=False, games=None, threshold=None):
    """Latest completed attempt per campaign/game; selection never uses scores."""
    selected, skipped = [], []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for game, seed in manifest['seeds'].items():
            if games and game not in games:
                continue
            state_path = manifest_path.with_name(f'{game}.json')
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
            choice = None
            for attempt in range(state.get('attempts', 0), 0, -1):
                name = f"{game}-td_only-{manifest['id']}-seed{seed}-attempt{attempt}_O"
                directory = root / 'runs' / name
                final = directory / 'final_evaluation.json'
                if final.is_file():
                    result = json.loads(final.read_text())
                    if (result.get('run_name') != name or result.get('seed') != seed
                            or result.get('env_name') != f'ALE/{game}-v5'):
                        raise ValueError(f'{final}: final evaluation identity mismatch')
                    status = 'finished'
                elif include_running and attempt == state['attempts'] and state.get('status') == 'running':
                    status = 'running'
                else:
                    continue
                config_path = directory / 'config.yaml'
                config = yaml.safe_load(config_path.read_text())
                agent = config['JointTrainAgent']
                ret = agent['Retrieval']
                if ret.get('enable') is not True or ret.get('score_combination') != 'td_error_only':
                    raise ValueError(f'{config_path}: not an enabled TD-error-only run')
                if threshold is not None and float(ret['td_error_z_score_threshold']) != threshold:
                    continue
                if (agent['NumEnvs'] != 1 or agent['TrainDynamicsEverySteps'] != 1
                        or ret.get('warmup_offset_pairs') or ret['warmup_steps'] < 0):
                    raise ValueError(f'{name}: requires fresh single-env, one-update-per-step, fixed-warmup campaign')
                if '--resume_from' in state.get('command', []):
                    raise ValueError(f'{name}: resumed event-step mapping is not supported')
                choice = {
                    'campaign': manifest['id'], 'game': game, 'seed': seed,
                    'run_name': name, 'status': status, 'directory': directory,
                    'config': config, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                }
                break
            if choice is None:
                skipped.append({'campaign': manifest['id'], 'game': game, 'seed': seed,
                                'reason': 'no selected completed run' if not include_running else 'not started or failed'})
            else:
                selected.append(choice)
    return selected, skipped


def read_event_counts(paths):
    """Stream every event, parsing protobuf only for the requested scalar tag."""
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    from tensorboard.util.tensor_util import make_ndarray

    steps, counts = [], []
    marker = METRIC.encode()
    for path in paths:
        for raw in RawEventFileLoader(str(path)).Load():
            if marker not in raw:
                continue
            event = event_pb2.Event.FromString(raw)
            for value in event.summary.value:
                if value.tag != METRIC:
                    continue
                if value.HasField('simple_value'):
                    count = value.simple_value
                elif value.HasField('tensor'):
                    count = make_ndarray(value.tensor).item()
                else:
                    raise ValueError(f'{path}: anchor metric is not a scalar')
                steps.append(event.step)
                counts.append(count)
    order = np.argsort(steps)
    return np.asarray(steps, dtype=np.int64)[order], np.asarray(counts, dtype=np.float64)[order]


def load_counts(run, output, refresh=False):
    paths = sorted(run['directory'].glob('events.out.tfevents.*'))
    if not paths:
        raise ValueError(f"{run['run_name']}: no TensorBoard event files")
    fingerprint = {
        'version': CACHE_VERSION, 'config_sha256': run['config_sha256'],
        'files': [{'path': str(path.resolve()), 'size': path.stat().st_size,
                   'mtime_ns': path.stat().st_mtime_ns} for path in paths],
    }
    cache = output / 'cache' / f"{run['run_name']}.npz"
    steps = counts = None
    cache_hit = False
    if cache.is_file() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(saved['metadata'].item()) == fingerprint:
                steps, counts = saved['steps'], saved['counts']
                cache_hit = True
    if counts is None:
        steps, counts = read_event_counts(paths)
    # Logger.tag_step is a zero-based counter specific to this metric. In fresh
    # runs update 0 occurs at BufferWarmUp, including demonstration-enabled runs.
    agent = run['config']['JointTrainAgent']
    expected = agent['SampleMaxSteps'] - agent['BufferWarmUp']
    if not len(counts) or not np.array_equal(steps, np.arange(len(steps))):
        raise ValueError(f"{run['run_name']}: missing/duplicate/reset metric steps; cannot reconstruct env steps")
    if len(counts) > expected or (run['status'] == 'finished' and len(counts) != expected):
        raise ValueError(f"{run['run_name']}: expected {expected} complete update records, found {len(counts)}")
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any() or (counts > agent['BatchSize']).any()):
        raise ValueError(f"{run['run_name']}: invalid anchor counts")
    if not cache_hit:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, steps=steps, counts=counts.astype(np.int16),
                            metadata=json.dumps(fingerprint, sort_keys=True))
    return counts, fingerprint


def analyze_run(run, counts, args):
    agent = run['config']['JointTrainAgent']
    retrieval = agent['Retrieval']
    first_env_step = agent['BufferWarmUp']
    available_stop = first_env_step + len(counts)
    start = max(args.start if args.start is not None else retrieval['warmup_steps'], first_env_step)
    requested_stop = args.stop if args.stop is not None else (agent.get('FinalEvalStep') or agent['SampleMaxSteps'])
    if run['status'] == 'finished' and requested_stop > available_stop:
        raise ValueError(f"{run['run_name']}: requested window extends beyond the completed history")
    stop = min(requested_stop, available_stop)
    if stop - start < 2:
        raise ValueError(f"{run['run_name']}: fewer than two observed updates in the requested window")
    values = counts[start - first_env_step:stop - first_env_step]
    stats = block_statistics(values, args.block_size, args.bootstrap_samples, args.confidence,
                             args.bootstrap_seed)
    return {
        'campaign': run['campaign'], 'game': run['game'], 'seed': run['seed'],
        'run_name': run['run_name'], 'status': run['status'],
        'threshold': retrieval['td_error_z_score_threshold'],
        'anchor_weight_r': retrieval.get('anchor_weight_r'),
        'env_step_start': start, 'env_step_stop_exclusive': stop,
        'requested_stop_exclusive': requested_stop, 'complete_window': stop == requested_stop,
        'batch_size': agent['BatchSize'], 'use_demonstration': agent.get('UseDemonstration', False),
        **stats,
    }


def aggregate(records, args):
    groups = defaultdict(list)
    for row in records:
        groups[row['campaign'], row['game']].append(row)
    per_game = []
    for (campaign, game), runs in sorted(groups.items()):
        stats = independent_statistics([row['mean_anchors_per_step'] for row in runs],
                                       args.bootstrap_samples, args.confidence, args.bootstrap_seed)
        per_game.append({'campaign': campaign, 'game': game, 'runs': len(runs), **stats,
                         'uncertainty_unit': 'independent training runs within this game'})
    summaries = []
    for campaign in sorted({row['campaign'] for row in per_game}):
        games = [row for row in per_game if row['campaign'] == campaign]
        runs = [row for row in records if row['campaign'] == campaign]
        stats = independent_statistics([row['mean_anchors_per_step'] for row in games],
                                       args.bootstrap_samples, args.confidence, args.bootstrap_seed)
        summaries.append({
            'campaign': campaign, 'games': len(games), 'runs': len(runs),
            'thresholds': sorted({row['threshold'] for row in runs}),
            'aggregation': 'equal game weights; equal run weights within each game',
            'ci_method': 'percentile bootstrap of game means',
            'uncertainty_unit': 'games; not independent training-seed uncertainty',
            'total_updates': sum(row['updates'] for row in runs),
            'total_anchor_insertions': sum(row['total_anchor_insertions'] for row in runs),
            **stats,
        })
    return per_game, summaries


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_outputs(output, records, per_game, summaries, skipped, failures, provenance, args):
    write_csv(output / 'per_run.csv', records)
    write_csv(output / 'per_game.csv', per_game)
    payload = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'metric': METRIC,
        'step_unit': 'one world-model update = one environment step in the selected configurations',
        'confidence': args.confidence, 'bootstrap_samples': args.bootstrap_samples,
        'bootstrap_seed': args.bootstrap_seed, 'campaigns': summaries,
        'skipped': skipped, 'failures': failures, 'sources': provenance,
    }
    (output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n')
    report = ['# TD-error-only anchor insertions per step', '',
              f'Metric: `{METRIC}`. Counts include zero-anchor updates and repeated insertions of the same replay state.',
              'One step is one world-model update (also one environment step for these runs).', '',
              f'Confidence level: {100 * args.confidence:g}%; bootstrap replicates: {args.bootstrap_samples}.', '',
              '| Campaign | Games | Runs | Mean anchors/step | SE | CI |',
              '|---|---:|---:|---:|---:|---|']
    for row in summaries:
        se = f"{row['se']:.6f}" if row['se'] is not None else 'N/A'
        ci = f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]" if row['ci_low'] is not None else 'N/A'
        report.append(f"| {row['campaign']} | {row['games']} | {row['runs']} | "
                      f"{row['mean_anchors_per_step']:.6f} | {se} | {ci} |")
    report.extend(['', 'Run-level SE and CI use a circular moving-block bootstrap of consecutive updates.',
                   'Game-level SE and CI use independent run means; they are undefined for a single run.',
                   'Overall SE and CI resample game means with equal game weights, not individual updates.',
                   'Within-run intervals describe temporal resampling of the observed training trajectory; learning can be nonstationary.',
                   'They do not estimate variability across independently trained seeds.',
                   'Exact observed windows and partial-window flags are stored in per_run.csv.',
                   f'Skipped jobs: {len(skipped)}; invalid/unreadable runs: {len(failures)}.', ''])
    (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path, nargs='+', help='Default: all local TD-only campaign manifests')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/td_error_only_anchor_stats')
    parser.add_argument('--games', nargs='+')
    parser.add_argument('--threshold', type=float, help='Select an exact TD-error threshold')
    parser.add_argument('--start', type=int, help='First environment step, inclusive; default: retrieval warmup end')
    parser.add_argument('--stop', type=int, help='Last environment step, exclusive; default: final evaluation checkpoint')
    parser.add_argument('--include-running', action='store_true', help='Also analyze observed post-warmup prefixes of running jobs')
    parser.add_argument('--block-size', type=int, default=1000, help='Consecutive updates per bootstrap block (default: 1000)')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--refresh', action='store_true', help='Read event files even if their fingerprints match the cache')
    args = parser.parse_args(argv)
    if args.block_size < 1 or args.bootstrap_samples < 100 or not 0 < args.confidence < 1 or args.bootstrap_seed < 0:
        parser.error('Require block-size >= 1, bootstrap-samples >= 100, 0 < confidence < 1 and bootstrap-seed >= 0')
    if ((args.start is not None and args.start < 0) or (args.stop is not None and args.stop < 1)
            or (args.start is not None and args.stop is not None and args.start >= args.stop)):
        parser.error('Require 0 <= start < stop')
    root, output = args.root.resolve(), args.output.resolve()
    manifests = args.manifest or sorted((root / 'results/td_error_only_queue').glob('*/manifest.json'))
    if not manifests:
        parser.error('No TD-error-only campaign manifests found')
    runs, skipped = discover_runs(root, manifests, args.include_running, args.games, args.threshold)
    output.mkdir(parents=True, exist_ok=True)
    records, failures, provenance = [], [], []
    for index, run in enumerate(runs, 1):
        print(f"[{index}/{len(runs)}] Reading {run['run_name']}", flush=True)
        try:
            counts, source = load_counts(run, output, args.refresh)
            record = analyze_run(run, counts, args)
            records.append(record)
            provenance.append({'run_name': run['run_name'], **source})
            print(f"  mean={record['mean_anchors_per_step']:.6f}, SE={record['se']:.6f}, "
                  f"CI=[{record['ci_low']:.6f}, {record['ci_high']:.6f}], updates={record['updates']}", flush=True)
        except (ValueError, KeyError, OSError) as error:
            failures.append({'run_name': run['run_name'], 'reason': str(error)})
            print(f'  FAILED: {error}', flush=True)
    per_game, summaries = aggregate(records, args)
    write_outputs(output, records, per_game, summaries, skipped, failures, provenance, args)
    for row in summaries:
        print(f"Overall {row['campaign']}: mean={row['mean_anchors_per_step']:.6f}, "
              f"SE={row['se']}, CI=[{row['ci_low']}, {row['ci_high']}], games={row['games']}", flush=True)
    print(f'Results: {output}', flush=True)
    return 2 if failures or not records else 0


if __name__ == '__main__':
    raise SystemExit(main())
