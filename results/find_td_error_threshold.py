"""Find the measured TD-only threshold closest to a target anchor frequency.

Uses measure_additive_anchors.py to restore the same full checkpoint for every
candidate. See find_td_error_threshold.md. --dry-run requires no GPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import yaml

STORM = Path(__file__).resolve().parents[1]
PROBE = STORM.parent / 'measure_additive_anchors.py'
sys.path.insert(0, str(STORM.parent))
from measure_additive_anchors import resolve_checkpoint, resolve_settings
from td_error_only_anchor_stats import independent_statistics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def normalized_config(value):
    """YAML may decode 3E-4 as text although the training loader accepts it as a number."""
    if isinstance(value, dict):
        return {key: normalized_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_config(item) for item in value]
    if isinstance(value, str):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
    return value


def search_thresholds(evaluate, target=5.99, lower=0.0, upper=6.0, coarse_points=7,
                      rounds=4, tolerance=0.05, *, evaluate_many=None):
    """Coarse grid followed by local refinement; no monotonicity assumption."""
    if (not all(math.isfinite(x) for x in (target, lower, upper, tolerance))
            or target < 0 or lower < 0 or upper <= lower or tolerance < 0
            or coarse_points < 2 or rounds < 0):
        raise ValueError('Invalid target, search range, grid size, rounds or tolerance')
    measured = {}

    def measure(thresholds):
        pending = list(dict.fromkeys(float(f'{value:.12g}') for value in thresholds))
        pending = [value for value in pending if value not in measured]
        if not pending:
            return
        records = (evaluate_many(pending) if evaluate_many is not None
                   else {value: evaluate(value) for value in pending})
        if set(records) != set(pending):
            raise ValueError('Batch evaluator must return every requested threshold exactly once')
        for threshold in pending:
            record = dict(records[threshold])
            mean = record['mean_anchors_per_update']
            if not math.isfinite(mean) or mean < 0:
                raise ValueError('Probe returned an invalid mean')
            measured[threshold] = {**record, 'threshold': threshold,
                                   'absolute_error': abs(mean - target)}

    def best():
        return min(measured.values(), key=lambda row: (row['absolute_error'], row['threshold']))

    measure(np.linspace(lower, upper, coarse_points))
    spacing = (upper - lower) / (coarse_points - 1)
    for _ in range(rounds):
        center = best()['threshold']
        if best()['absolute_error'] <= tolerance:
            break
        spacing /= 2
        before = len(measured)
        measure(value for value in (center - spacing, center + spacing)
                if lower <= value <= upper)
        if len(measured) == before:
            break
    return best(), sorted(measured.values(), key=lambda row: row['threshold'])


def prepare_jobs(args):
    jobs, identities = [], set()
    for raw_path in args.checkpoint:
        checkpoint, metadata = resolve_checkpoint(raw_path)
        config, env, seed = resolve_settings(SimpleNamespace(
            config=args.config, env=args.env, seed=args.seed, score='td_error_only',
        ), checkpoint, metadata)
        source = yaml.safe_load(config.read_text())
        joint = source['JointTrainAgent']
        if (source['Task'] != 'JointTrainAgent' or joint['NumEnvs'] != 1
                or joint['UseDemonstration'] or joint['TrainDynamicsEverySteps'] != 1
                or joint['TrainAgentEverySteps'] != 1):
            raise ValueError(f'{checkpoint}: probe requires single-env STORM without demonstrations, update periods 1')
        if args.target > joint['BatchSize']:
            raise ValueError(f"Target exceeds maximum anchor count {joint['BatchSize']} per update")
        if (env, seed) in identities:
            raise ValueError(f'Duplicate game/training seed: {env}, {seed}; choose one checkpoint per run')
        identities.add((env, seed))
        original_config = checkpoint / 'config.yaml'
        if original_config.is_file():
            original = yaml.safe_load(original_config.read_text())
            if (normalized_config(original['Models']) != normalized_config(source['Models'])
                    or original['BasicSettings']['ImageSize'] != source['BasicSettings']['ImageSize']
                    or original['JointTrainAgent']['BufferMaxLength'] != joint['BufferMaxLength']):
                raise ValueError(f'{checkpoint}: override config changes model/image/replay-buffer dimensions')
        files = [checkpoint / name for name in ('training_state.pt', 'replay_buffer.npz')]
        jobs.append({
            'checkpoint': str(checkpoint), 'config': str(config), 'env': env, 'seed': seed,
            'batch_size': joint['BatchSize'],
            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
            'files': [{'path': str(path), 'size': path.stat().st_size,
                       'mtime_ns': path.stat().st_mtime_ns} for path in files],
        })
    return jobs


def aggregate_probes(records, bootstrap_samples, bootstrap_seed):
    games = defaultdict(list)
    for row in records:
        games[row['env']].append(row['mean_anchors_per_update'])
    means = [float(np.mean(values)) for _, values in sorted(games.items())]
    estimate = independent_statistics(means, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed)
    # One single-run probe has temporal uncertainty; it cannot yield game-level uncertainty.
    if len(records) == 1:
        se, ci = records[0]['mean_se'], records[0]['mean_ci95']
        unit = 'consecutive updates within this one resumed run (block bootstrap)'
    else:
        se = estimate['se']
        ci = [estimate['ci_low'], estimate['ci_high']] if se is not None else None
        unit = 'games, with run means averaged within each game (game bootstrap)'
        if len(games) == 1:
            estimate = independent_statistics([row['mean_anchors_per_update'] for row in records],
                                               bootstrap_samples=bootstrap_samples, seed=bootstrap_seed)
            se, ci = estimate['se'], [estimate['ci_low'], estimate['ci_high']]
            unit = 'independent training runs within this game (run bootstrap)'
    # A within-run SD is defined for one update trace, not for run/game means.
    within_run_sd = records[0]['within_run_sd_across_updates'] if len(records) == 1 else None
    return {'mean_anchors_per_update': float(np.mean(means)),
            'within_run_sd_across_updates': within_run_sd, 'se': se, 'ci95': ci,
            'uncertainty_unit': unit, 'games': len(games), 'runs': len(records)}


def validate_probe(directory, job, threshold, steps, burn_in):
    summary = json.loads((directory / 'summary.json').read_text())
    expected = {'checkpoint': job['checkpoint'], 'source_config': job['config'],
                'env': job['env'], 'seed': job['seed'], 'score_combination': 'td_error_only',
                'threshold': threshold, 'updates': steps, 'burn_in_updates': burn_in}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f'{directory}: cached probe does not match the requested measurement')
    with (directory / 'anchors.csv').open() as stream:
        trace = list(csv.DictReader(stream))
    if len(trace) != steps + burn_in:
        raise ValueError(f'{directory}: incomplete anchor trace')
    counts = []
    for offset, row in enumerate(trace):
        count = int(row['anchors'])
        if (int(row['env_step']) != summary['probe_env_step_start'] + offset
                or row['is_warmup'].lower() != 'false' or not 0 <= count <= job['batch_size']):
            raise ValueError(f'{directory}: invalid anchor trace')
        counts.append(count)
    measured = counts[burn_in:]
    if (sum(measured) != summary['total_anchor_insertions']
            or not math.isclose(float(np.mean(measured)), summary['mean_anchors_per_update'], abs_tol=1e-12)
            or not math.isclose(float(np.std(measured, ddof=1)),
                                summary['within_run_sd_across_updates'], abs_tol=1e-12)
            or not math.isfinite(summary['mean_se']) or summary['mean_se'] < 0):
        raise ValueError(f'{directory}: summary differs from recorded counts')
    return summary


def measure_candidate(threshold, jobs, args, output, *, validation=False):
    tag = f'threshold_{threshold:.12g}'
    parent = output / ('validation' if validation else 'probes') / tag
    steps = args.validation_steps if validation else args.steps
    burn_in = args.steps if validation else 0
    records = []
    print(f"{'Validation' if validation else 'Candidate'} threshold={threshold:g}", flush=True)
    for index, job in enumerate(jobs):
        if hashlib.sha256(Path(job['config']).read_bytes()).hexdigest() != job['config_sha256']:
            raise ValueError(f"{job['config']}: configuration changed during the search")
        for source in job['files']:
            info = Path(source['path']).stat()
            if info.st_size != source['size'] or info.st_mtime_ns != source['mtime_ns']:
                raise ValueError(f"{source['path']}: checkpoint changed during the search")
        run_dir = parent / f'run_{index:03d}'
        run_dir.mkdir(parents=True, exist_ok=True)
        completed = sorted(run_dir.glob('attempt_*/summary.json'))
        if completed:
            record = validate_probe(completed[-1].parent, job, threshold, steps, burn_in)
        else:
            # Keep interrupted attempts, then restart from the immutable source checkpoint.
            attempt = max([int(path.name.split('_')[-1]) for path in run_dir.glob('attempt_*')
                           if path.is_dir()], default=0) + 1
            destination = run_dir / f'attempt_{attempt:03d}'
            command = [sys.executable, str(PROBE), '--checkpoint', job['checkpoint'],
                       '--config', job['config'], '--env', job['env'], '--seed', str(job['seed']),
                       '--score', 'td_error_only', '--threshold', str(threshold),
                       '--steps', str(steps), '--burn-in', str(burn_in),
                       '--block-size', str(args.block_size), '--bootstrap-samples', str(args.bootstrap_samples),
                       '--bootstrap-seed', str(args.bootstrap_seed), '--output', str(destination)]
            environment = dict(os.environ)
            if args.gpu is not None:
                environment['CUDA_VISIBLE_DEVICES'] = args.gpu
            log_path = run_dir / f'attempt_{attempt:03d}.log'
            print(f"  threshold={threshold:g}, {job['env']}, seed {job['seed']}, "
                  f"GPU={args.gpu if args.gpu is not None else 'inherited'}; log: {log_path}", flush=True)
            with log_path.open('w') as stream:
                result = subprocess.run(command, cwd=STORM.parent, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Probe failed with exit code {result.returncode}; see {log_path}')
            record = validate_probe(destination, job, threshold, steps, burn_in)
        records.append(record)
        print(f"  mean={record['mean_anchors_per_update']:.5f}, "
              f"within-run SD={record['within_run_sd_across_updates']:.5f}, "
              f"SE={record['mean_se']:.5f}", flush=True)
    combined = aggregate_probes(records, args.bootstrap_samples, args.bootstrap_seed)
    write_json(parent / 'aggregate.json', {**combined, 'threshold': threshold, 'per_run': records})
    print(f"  Game-balanced mean={combined['mean_anchors_per_update']:.5f}", flush=True)
    return combined


def measure_candidates(thresholds, jobs, args, output, *, on_result=None):
    """Run each candidate on one GPU; schedule the next on whichever GPU finishes."""
    results = {}

    def completed(threshold, record):
        results[threshold] = record
        if on_result is not None:
            on_result(threshold, record)

    gpus = getattr(args, 'gpus', None)
    if not gpus:
        for threshold in thresholds:
            completed(threshold, measure_candidate(threshold, jobs, args, output))
        return results

    pending = iter(thresholds)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}

        def submit(gpu):
            threshold = next(pending, None)
            if threshold is not None:
                worker_args = SimpleNamespace(**{**vars(args), 'gpu': str(gpu)})
                future = executor.submit(measure_candidate, threshold, jobs, worker_args, output)
                futures[future] = (threshold, gpu)

        for gpu in gpus:
            submit(gpu)
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                available = []
                for future in done:
                    threshold, gpu = futures.pop(future)
                    completed(threshold, future.result())
                    available.append(gpu)
                for gpu in available:
                    submit(gpu)
        except BaseException:
            # Keep completed traces for resume and stop launching new candidates.
            for future in futures:
                future.cancel()
            raise
    return results


def save_curve(output, rows):
    with (output / 'search.csv').open('w', newline='') as stream:
        fields = ['threshold', 'mean_anchors_per_update', 'within_run_sd_across_updates',
                  'absolute_error', 'se', 'ci95',
                  'uncertainty_unit', 'games', 'runs']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, result, curve, args, jobs):
    def number(value):
        return f'{value:.5f}' if value is not None else '—'

    def ci(record):
        bounds = record['ci95']
        return f'[{number(bounds[0])}, {number(bounds[1])}]' if bounds else '—'

    best = result['calibration']
    lines = [
        '# TD-error-only threshold 탐색 결과', '',
        f"선택된 threshold: **{result['best_tested_threshold']:g}**. "
        f"목표: **{result['target']:g} anchors/update**, 허용 오차: {result['tolerance']:g}.", '',
        '선택 기준은 측정한 후보 중 목표와 평균 앵커 수의 절대 차이가 가장 작은 값입니다. '
        '동률이면 더 작은 threshold를 선택합니다.', '',
        '| 구간 | 측정 updates/run | 평균 anchors/update | Within-run SD across updates | 목표와 절대 차이 | SE | 평균의 95% CI | 허용 오차 충족 |',
        '|---|---:|---:|---:|---:|---:|---|---|',
        f"| 탐색 | {args.steps} | {number(best['mean_anchors_per_update'])} | "
        f"{number(best['within_run_sd_across_updates'])} | "
        f"{number(best['absolute_error'])} | {number(best['se'])} | {ci(best)} | "
        f"{result['calibration_within_tolerance']} |",
    ]
    validation = result['validation']
    if validation:
        lines.append(f"| 후속 확인 | {args.validation_steps} | "
                     f"{number(validation['mean_anchors_per_update'])} | "
                     f"{number(validation['within_run_sd_across_updates'])} | "
                     f"{number(validation['absolute_error'])} | {number(validation['se'])} | "
                     f"{ci(validation)} | {result['validation_within_tolerance']} |")
    lines += ['', '## 측정한 후보', '',
              '| Threshold | 평균 anchors/update | Within-run SD across updates | 목표와 절대 차이 | SE | 평균의 95% CI |',
              '|---:|---:|---:|---:|---:|---|']
    lines += [f"| {row['threshold']:g} | {number(row['mean_anchors_per_update'])} | "
              f"{number(row['within_run_sd_across_updates'])} | "
              f"{number(row['absolute_error'])} | {number(row['se'])} | {ci(row)} |" for row in curve]
    lines += ['', '## 측정 조건', '',
              *[f"- {job['env']}, seed {job['seed']}: `{job['checkpoint']}`" for job in jobs],
              f"- 탐색 범위: [{args.lower:g}, {args.upper:g}], 초기 후보 {args.coarse_points}개, "
              f"최대 세분화 {args.refinement_rounds}회.",
              f"- CI/SE의 재표집 단위: {best['uncertainty_unit']}.",
              '- Within-run SD across updates는 같은 run의 측정 구간 내 앵커 수의 표본 표준편차(ddof=1)입니다. '
              '평균의 SE 또는 게임·학습 seed 간 표준편차가 아닙니다.',
              '- 매 후보마다 같은 checkpoint를 복원하며, 앵커가 0개인 update도 평균에 포함합니다.']
    if len(jobs) > 1:
        lines.append('- 여러 run을 집계할 때 단일 Within-run SD는 표시하지 않습니다. '
                     '각 run의 SD는 probes/의 summary.json과 aggregate.json의 per_run에 기록됩니다.')
    if validation:
        lines.append(f'- 후속 확인은 checkpoint 복원 후 처음 {args.steps} updates를 제외하고 '
                     f'그 다음 {args.validation_steps} updates를 측정합니다. 독립 학습 run은 아닙니다.')
    lines += ['', '허용 오차 충족 여부는 평균과 목표의 절대 차이로 판정합니다. '
              '후속 확인이 허용 오차를 벗어나면 탐색 구간의 빈도 매칭이 유지되지 않은 것입니다. '
              '이 결과는 지정한 checkpoint의 짧은 재개 학습에 한정되며, 전체 학습의 평균을 보장하지 않습니다.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--config', type=Path, help='Default: preserve each checkpoint configuration')
    parser.add_argument('--env', help='For old checkpoints without environment metadata')
    parser.add_argument('--seed', type=int, help='For old checkpoints without training-seed metadata')
    parser.add_argument('--target', type=float, default=5.99)
    parser.add_argument('--lower', type=float, default=0.0)
    parser.add_argument('--upper', type=float, default=6.0)
    parser.add_argument('--coarse-points', type=int, default=7)
    parser.add_argument('--refinement-rounds', type=int, default=4)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--steps', type=int, default=100, help='Measured updates per calibration probe (default: 100)')
    parser.add_argument('--validation-steps', type=int, default=1000,
                        help='Measured updates after an excluded calibration-length prefix; 0 disables this check')
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    gpu_options = parser.add_mutually_exclusive_group()
    gpu_options.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES value for sequential child probes')
    gpu_options.add_argument('--gpus', type=int, nargs='+',
                             help='Parallel GPU IDs, one candidate per GPU, e.g. 0 1 2 3 4 5')
    parser.add_argument('--output', type=Path, help='Output directory; reusing it resumes matching completed probes')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and write the execution plan; no training')
    args = parser.parse_args(argv)
    if args.gpus and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)):
        parser.error('--gpus requires distinct, nonnegative GPU IDs')
    if (args.steps < 2 or args.validation_steps < 0 or args.validation_steps == 1
            or not 1 <= args.block_size <= args.steps // 2
            or (args.validation_steps and args.block_size > args.validation_steps // 2)
            or args.bootstrap_samples < 100 or args.bootstrap_seed < 0):
        parser.error('Invalid steps, validation length, block size or bootstrap settings')
    try:
        # Validate search arguments without running a probe.
        search_thresholds(lambda threshold: {'mean_anchors_per_update': 0}, args.target,
                          args.lower, args.upper, args.coarse_points, 0, args.tolerance)
        jobs = prepare_jobs(args)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = (args.output or STORM / 'results/td_error_only_threshold_search' / stamp).resolve()
        source_paths = [Path(__file__), PROBE, STORM / 'train.py', STORM / 'utils.py',
                        STORM.parent / 'retrieval.py', Path(independent_statistics.__code__.co_filename)]
        protocol = {'jobs': jobs, 'settings': {key: value for key, value in vars(args).items()
                    if key not in {'checkpoint', 'config', 'output', 'dry_run', 'env', 'seed'}},
                    'code_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_paths}}
        output.mkdir(parents=True, exist_ok=True)
        protocol_path = output / 'protocol.json'
        if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
            if (output / 'probes').exists() or (output / 'validation').exists():
                raise ValueError('Existing output uses a different protocol; choose a new --output directory')
            # A dry-run plan has no measured data to mix with the revised plan.
        write_json(protocol_path, protocol)
        maximum = args.coarse_points + 2 * args.refinement_rounds
        validation_plan = (f'{len(jobs)} × ({args.steps} excluded + {args.validation_steps} measured)'
                           if args.validation_steps else 'disabled')
        print(f'Plan: up to {maximum} thresholds × {len(jobs)} runs × {args.steps} updates; '
              f'validation: {validation_plan}.', flush=True)
        print('Initial thresholds: ' + ', '.join(f'{value:.12g}' for value in
              np.linspace(args.lower, args.upper, args.coarse_points)), flush=True)
        if args.gpus:
            print(f'Parallel workers: {len(args.gpus)}; GPUs: {args.gpus}; '
                  'one candidate per GPU. Validation uses the first GPU.', flush=True)
        print(f'Results: {output}', flush=True)
        if args.dry_run:
            return 0
        history = []

        def record_result(threshold, record):
            history.append({**record, 'threshold': threshold,
                            'absolute_error': abs(record['mean_anchors_per_update'] - args.target)})
            save_curve(output, sorted(history, key=lambda row: row['threshold']))

        def evaluate_many(thresholds):
            return measure_candidates(thresholds, jobs, args, output, on_result=record_result)

        best, curve = search_thresholds(None, args.target, args.lower, args.upper,
                                       args.coarse_points, args.refinement_rounds, args.tolerance,
                                       evaluate_many=evaluate_many)
        save_curve(output, curve)
        if args.gpus:
            args.gpu = str(args.gpus[0])
        validation = (measure_candidate(best['threshold'], jobs, args, output, validation=True)
                      if args.validation_steps else None)
        if validation:
            validation['absolute_error'] = abs(validation['mean_anchors_per_update'] - args.target)
        result = {
            'target': args.target, 'tolerance': args.tolerance,
            'best_tested_threshold': best['threshold'], 'calibration': best, 'validation': validation,
            'calibration_within_tolerance': best['absolute_error'] <= args.tolerance,
            'validation_within_tolerance': (abs(validation['mean_anchors_per_update'] - args.target)
                                            <= args.tolerance if validation else None),
            'best_at_search_boundary': best['threshold'] in (args.lower, args.upper),
            'observed_nonmonotonicity': any(b['mean_anchors_per_update'] > a['mean_anchors_per_update']
                                           for a, b in zip(curve, curve[1:])),
            'scope': 'Short resumed probes on the specified checkpoints; not a guarantee for full 50k–100k training.',
            'validation_scope': 'Subsequent updates under the selected threshold, not independent training runs.',
            'training_override': {'JointTrainAgent.Retrieval.td_error_z_score_threshold': best['threshold']},
        }
        write_json(output / 'result.json', result)
        write_report(output, result, curve, args, jobs)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        print(f"Selected threshold: {best['threshold']:g}; mean anchors/update: "
              f"{best['mean_anchors_per_update']:.5f}; absolute error: {best['absolute_error']:.5f}.", flush=True)
        print(f"Report: {output / 'report.md'}", flush=True)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
