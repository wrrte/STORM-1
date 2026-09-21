"""Incremental, offline experiment planning from the W&B classification CSV.

Run after classify_wandb_runs.py; --dry-run previews without dispatching jobs.
The CSV is authoritative for completion, the ledger reserves dispatched seeds,
and existing worker flock files protect queue appends. No checkpoints are deleted.
Only the standard library is required. See experiment_scheduler.md.
"""

import argparse
import ast
from collections import defaultdict
from contextlib import ExitStack
import copy
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import statistics
import tempfile


HERE = Path(__file__).resolve().parent
EMA_DATE = '2026-08-24T11:05:43Z'
PREFIX = 'JointTrainAgent.Retrieval.'
ACTIVE = {'pending', 'running', 'awaiting_result'}
VARIANTS = {
    'baseline': {'Retrieval Enable': 'False'},
    'retrieval': {'Retrieval Enable': 'True'},
    'target1': {'Retrieval Enable': 'True', 'Retrieval Target': '1'},
    'value': {'Retrieval Enable': 'True', 'Value Signal': 'value'},
    'add': {'Retrieval Enable': 'True', 'Score Combination': 'add'},
}


def literal_setting(path, name):
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{path}: {name} 설정을 찾을 수 없습니다.')


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def value(row, key, default=''):
    result = str(row.get(key, '')).strip()
    return default if result.lower() in {'', 'n/a', 'nan', 'none'} else result


def truth(item):
    return str(item).lower() in {'true', '1', 't'}


def timestamp(text):
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def modes(raw):
    raw = str(raw).strip()
    if raw.startswith('['):
        result = ast.literal_eval(raw)
        if not isinstance(result, list) or not result or any(v not in VARIANTS for v in result):
            raise ValueError(f'알 수 없는 실험 목록: {raw}')
        return result
    if raw.lower() == 'both':
        return ['retrieval', 'baseline']
    if raw.lower() in {'true', '1', 't'}:
        return ['retrieval']
    if raw.lower() in {'false', '0', 'f'}:
        return ['baseline']
    raise ValueError(f'알 수 없는 Retrieval.enable: {raw}')


def row_kind(row):
    """The two workbook rows only; baseline deliberately has no date cutoff."""
    enabled = value(row, 'Retrieval Enable').lower()
    if enabled in {'false', '0', 'f'}:
        return 'baseline'
    if enabled not in {'true', '1', 't'}:
        return None
    if timestamp(value(row, 'Created At')) < timestamp(EMA_DATE):
        return None
    anchor = value(row, 'Anchor Weight')
    if (number(row.get('Warmup Steps')) == 50000
            and value(row, 'Batch Size Reduction') == 'retrieved'
            and number(row.get('Z Score Threshold')) == 3.5
            and number(row.get('Retrieval Target')) == 16
            and (not anchor or number(anchor) == 0.0625)
            and value(row, 'Value Signal', 'value_diff') == 'value_diff'
            and value(row, 'Score Combination', 'multiply') == 'multiply'):
        return 'retrieval'
    return None


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        required = {'Run Name', 'Run ID', 'State', 'Seed', 'Eval Return', 'Retrieval Enable'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: 필요한 CSV 열이 없습니다: {sorted(required)}')
        rows = list(reader)
    for row in rows:
        row['game'] = row['Run Name'].split('_')[0]
        seed = number(row['Seed'])
        row['seed'] = int(seed) if seed is not None and seed.is_integer() else None
        row['kind'] = row_kind(row)
        row['score'] = number(row['Eval Return'])
        row['State'] = row['State'].lower().strip()
    return rows


def latest_scores(rows, excluded):
    groups = defaultdict(list)
    for row in rows:
        if (row['kind'] and row['seed'] is not None and row['State'] == 'finished'
                and row['score'] is not None and row['seed'] not in excluded.get(row['game'], set())):
            groups[row['game'], row['kind'], row['seed']].append(row)
    result = {}
    for key, group in groups.items():
        # Match the workbook's preference for hash_bits=10 within each cell.
        preferred = [r for r in group if number(r.get('Hash Bits')) == 10]
        result[key] = max(preferred or group, key=lambda r: timestamp(r.get('Created At')))
    return result


def local_checkpoint(path):
    """Keep paths portable between the machines that consume STORM queues."""
    parts = PurePosixPath(path).parts
    return str(PurePosixPath(*parts[parts.index('ckpt'):])) if 'ckpt' in parts else path


def warmup_for_row(row):
    path = value(row, 'Warmup Directory')
    if path:
        return local_checkpoint(path)
    base = value(row, 'Base Run Name')
    return f'ckpt/{base}_Shared' if base else ''


def warmup_root(path):
    path = PurePosixPath(local_checkpoint(path))
    return str(path.parent if path.name.startswith('shared_warmup_') else path)


def parse_command(line, gpu):
    tokens = shlex.split(line, comments=True)
    if not tokens:
        return None
    if not any(Path(token).name == 'train.py' for token in tokens):
        raise ValueError('train.py 명령이 아닙니다')
    if any(token in {';', '&&', '||', '|', '&'} for token in tokens):
        raise ValueError('복합 셸 명령의 시간을 추정할 수 없습니다')
    args = dict(zip(tokens, tokens[1:]))
    for token in tokens:
        if token.startswith('--') and '=' in token:
            key, val = token.split('=', 1)
            args[key] = val
    resume = args.get('--resume_warmup', '')
    base = args.get('-n', '')
    if not base and resume:
        parts = PurePosixPath(resume).parts
        base = next((p[:-7] for p in parts if p.endswith('_Shared')), '')
    seed = number(args.get('-seed', base.rsplit('-', 1)[-1]))
    env = args.get('-env_name', '')
    game = env.split('/')[-1].removesuffix('-v5') if env else base.split('-')[0]
    if not game or seed is None or not seed.is_integer():
        raise ValueError('게임/시드를 읽을 수 없습니다')
    names = modes(args.get(PREFIX + 'enable', 'Both'))
    config = {
        'Retrieval Enable': 'True', 'Warmup Steps': args.get(PREFIX + 'warmup_steps', '50000'),
        'Retrieval Target': args.get(PREFIX + 'target', '16'),
        'Batch Size Reduction': args.get(PREFIX + 'batch_size_reduction', 'retrieved'),
        'Z Score Threshold': args.get(PREFIX + 'z_score_threshold', '3.5'),
        'Anchor Weight': args.get(PREFIX + 'anchor_weight', '0.0625'),
        'Value Signal': args.get(PREFIX + 'value_signal', 'value_diff'),
        'Score Combination': args.get(PREFIX + 'score_combination', 'multiply'),
        'Created At': '2099-01-01T00:00:00Z',
    }
    kinds = {row_kind({**config, **VARIANTS[name]}) for name in names} - {None}
    return {
        'command': line.strip(), 'gpu': gpu, 'game': game, 'seed': int(seed),
        'names': names, 'kinds': sorted(kinds), 'resume': bool(resume),
        'warmup': local_checkpoint(resume) if resume else f'ckpt/{base}_Shared',
        'save_warmup': truth(args.get(PREFIX + 'save_warmup', 'False')),
    }


def validate_config(config):
    for key in ('anomaly_hns_gap', 'improvement_hns', 'lookahead_hours'):
        if number(config.get(key)) is None or config[key] < 0:
            raise ValueError(f'{key}: 0 이상의 숫자가 필요합니다.')
    for key in ('successful_pairs_per_game', 'max_new_jobs'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key}: 양의 정수가 필요합니다.')
    ranges = []
    historical = set()
    if set(config['gpus']) != {'pro6k', '3090', 'A6000', 'titan'}:
        raise ValueError('gpus에는 pro6k, 3090, A6000, titan이 필요합니다.')
    for gpu, spec in config['gpus'].items():
        if type(spec['count']) is not int or spec['count'] < 0:
            raise ValueError(f'{gpu}.count: 0 이상의 정수가 필요합니다.')
        for key in ('warmup_hours', 'retrieval_hours', 'baseline_hours'):
            if number(spec[key]) is None or spec[key] <= 0:
                raise ValueError(f'{gpu}.{key}: 양수가 필요합니다.')
        start, end = spec['new_seed_range']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < 2**32:
            raise ValueError(f'{gpu}: 잘못된 new_seed_range')
        if any(start <= b and a <= end for a, b in ranges):
            raise ValueError('GPU별 new_seed_range가 겹칩니다.')
        ranges.append((start, end))
        seeds = set(spec['historical_seeds'])
        if historical & seeds:
            raise ValueError('historical_seeds의 GPU 배정이 겹칩니다.')
        historical |= seeds
    if not sum(spec['count'] for spec in config['gpus'].values()):
        raise ValueError('사용 가능한 GPU가 없습니다.')


def gpu_for_seed(seed, config, jobs, queued):
    known = {item['gpu'] for item in [*jobs, *queued] if item['seed'] == seed}
    if len(known) == 1:
        return known.pop()
    if len(known) > 1:
        raise ValueError(f'시드 {seed}의 GPU 배정이 충돌합니다: {known}')
    for gpu, spec in config['gpus'].items():
        start, end = spec['new_seed_range']
        if seed in spec['historical_seeds'] or (seed is not None and start <= seed <= end):
            return gpu
    raise ValueError(f'시드 {seed}의 GPU를 모릅니다. 설정의 historical_seeds에 추가하세요.')


def duration(names, spec, warmup=True):
    return (spec['warmup_hours'] if warmup else 0) + sum(
        spec['baseline_hours'] if name == 'baseline' else spec['retrieval_hours'] for name in names)


def running_kinds(row):
    enabled = value(row, 'Retrieval Enable')
    if enabled.startswith('[') or enabled.lower() == 'both':
        configs = [{**row, **VARIANTS[name]} for name in modes(enabled)]
    else:
        configs = [row, *({**row, **part} for part in json.loads(
            value(row, 'Pending Retrieval Configs', '[]')))]
    return {row_kind(config) for config in configs} - {None}


def running_hours(row, spec, now):
    enabled = value(row, 'Retrieval Enable')
    shared = enabled.startswith('[') or enabled.lower() == 'both'
    if shared:
        names = modes(enabled)
        current = spec['warmup_hours']
        after = duration(names, spec, warmup=False)
    else:
        current = spec['baseline_hours'] if not truth(enabled) else spec['retrieval_hours']
        # Older exports lack phase metadata. save_warmup=False denotes a branch;
        # unidentifiable standalone runs conservatively include their warmup.
        if not value(row, 'Training Phase') and value(row, 'Save Warmup').lower() != 'false':
            current += spec['warmup_hours']
        pending = json.loads(value(row, 'Pending Retrieval Configs', '[]'))
        after = sum(spec['baseline_hours'] if not truth(part.get(
            'Retrieval Enable', enabled)) else spec['retrieval_hours'] for part in pending)
    elapsed = max(0, (now - timestamp(row.get('Created At'))).total_seconds() / 3600)
    # A run still reported running never becomes an idle slot purely by age.
    return max(0.25, current - elapsed) + after


def push_load(slots, gpu, hours):
    if not slots[gpu]:
        return math.inf
    index = min(range(len(slots[gpu])), key=slots[gpu].__getitem__)
    start = slots[gpu][index]
    slots[gpu][index] += hours
    return start


def make_command(game, seed, kind, warmup=''):
    if kind == 'baseline':
        return (f'python -u train.py --resume_warmup {shlex.quote(warmup)} '
                f'-n {shlex.quote(f"{game}-{seed}")} -seed {seed} '
                'JointTrainAgent.Retrieval.enable "[\'baseline\']"')
    return (
        f'python -u train.py -n "{game}-{seed}" -seed {seed} '
        f'-config_path "config_files/STORM.yaml" -env_name "ALE/{game}-v5" '
        f'-trajectory_path "D_TRAJ/{game}.pkl" '
        'JointTrainAgent.Retrieval.enable "[\'retrieval\']" '
        'JointTrainAgent.Retrieval.save_warmup True '
        'JointTrainAgent.Retrieval.target 16 JointTrainAgent.Retrieval.anchor_weight 0.0625 '
        'JointTrainAgent.Retrieval.warmup_steps 50000 '
        'JointTrainAgent.Retrieval.batch_size_reduction retrieved '
        'JointTrainAgent.Retrieval.z_score_threshold 3.5 '
        'JointTrainAgent.Retrieval.value_signal value_diff '
        'JointTrainAgent.Retrieval.score_combination multiply JointTrainAgent.Retrieval.hash_bits 10'
    )


def plan(rows, queue_texts, state, config, references, excluded, now, games=None, fail_jobs=()):
    """Pure planning: returns a new ledger and report, without touching disk."""
    state = copy.deepcopy(state)
    state.setdefault('version', 1)
    jobs = state.setdefault('jobs', [])
    if state['version'] != 1:
        raise ValueError('지원하지 않는 스케줄러 상태 파일입니다.')
    notices, queued, additions, cleanup = [], [], [], []
    for gpu, text in queue_texts.items():
        seen = set()
        for index, line in enumerate(text.splitlines(), 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            try:
                item = parse_command(line, gpu)
            except (ValueError, SyntaxError) as error:
                raise ValueError(f'job_queue_{gpu}.txt:{index}: {error}') from error
            key = (item['game'], item['seed'], tuple(item['names']))
            if key in seen:
                notices.append(f'{gpu} 기존 큐 중복: {key}; 기존 명령은 유지합니다.')
            seen.add(key)
            queued.append(item)
    scores = latest_scores(rows, excluded)
    summaries = {}
    for game, (random_score, human_score, paper_score) in references.items():
        scale = human_score - random_score
        means = {}
        for kind in ('baseline', 'retrieval'):
            values = [r['score'] for (g, k, _), r in scores.items() if g == game and k == kind]
            means[kind] = statistics.mean(values) if values else None
        gaps = {kind: (paper_score - mean) / scale if mean is not None else None
                for kind, mean in means.items()}
        anomalous = any(gap is not None and gap >= config['anomaly_hns_gap'] for gap in gaps.values())
        if (games is not None and game in games) or (games is None and anomalous):
            summaries[game] = {'means': means, 'paper_score': paper_score, 'hns_gaps': gaps,
                               'priority': max(gap or 0 for gap in gaps.values())}
    selected = set(summaries) | {job['game'] for job in jobs}
    if games is not None:
        unknown = set(games) - references.keys()
        if unknown:
            raise ValueError(f'알 수 없는 게임: {sorted(unknown)}')

    # Adopt pre-existing retrieval-only warmup experiments. Other configs consume
    # capacity but never become candidates or evaluation scores.
    ids = {job['id'] for job in jobs}
    candidates = []
    for item in queued:
        if item['game'] in selected and item['kinds'] == ['retrieval'] and item['save_warmup']:
            candidates.append({**item, 'row': None})
    for row in rows:
        if (row['game'] in selected and row['kind'] == 'retrieval' and row['seed'] is not None
                and any(truth(row.get(key)) for key in ('Save Warmup', 'Save Warmup Requested'))):
            gpu = gpu_for_seed(row['seed'], config, jobs, queued)
            candidates.append({'game': row['game'], 'seed': row['seed'], 'gpu': gpu,
                               'warmup': warmup_for_row(row), 'command': '', 'row': row})
    # EXCLUDED_SEEDS is read on every invocation. Freeze the comparison when a
    # trial is registered; an imported result cannot serve as its own reference.
    def threshold(game, candidate_seed=None):
        pool = [r for (g, kind, seed), r in scores.items()
                if g == game and kind == 'retrieval' and seed != candidate_seed]
        if not pool:
            return None
        worst = min(pool, key=lambda r: r['score'])
        random_score, human_score, _ = references[game]
        return {'reference_score': worst['score'], 'reference_seed': worst['seed'],
                'threshold_score': worst['score'] + config['improvement_hns'] * (human_score - random_score)}

    for item in candidates:
        job_id = f"{item['game']}:{item['seed']}:retrieval"
        if job_id in ids:
            continue
        comparison = threshold(item['game'], item['seed'])
        if comparison is None:
            notices.append(f"{item['game']}: 비교 가능한 기존 target 16 점수가 없습니다.")
            continue
        jobs.append({key: item[key] for key in ('game', 'seed', 'gpu', 'warmup', 'command')} | {
            'id': job_id, 'kind': 'retrieval', 'status': 'pending', 'adopted': True,
            'before_ids': [r['Run ID'] for r in rows if item['row'] is None
                           and (r['game'], r['seed'], r['kind']) == (item['game'], item['seed'], 'retrieval')],
            'registered_at': now.isoformat(), **comparison,
        })
        ids.add(job_id)

    running = [r for r in rows if r['State'] == 'running']
    occupied = {(item['game'], item['seed'], kind) for item in queued for kind in item['kinds']}
    occupied |= {(r['game'], r['seed'], kind) for r in running for kind in running_kinds(r)}
    for job in jobs:
        matching = [r for r in rows if r['game'] == job['game'] and r['seed'] == job['seed']
                    and r['kind'] == job['kind'] and r['Run ID'] not in job['before_ids']]
        latest = max(matching, key=lambda r: timestamp(r.get('Created At')), default=None)
        if (latest and job['kind'] == 'retrieval' and warmup_for_row(latest)
                and (job['status'] != 'complete' or latest['Run ID'] == job.get('run_id'))):
            job['warmup'] = warmup_for_row(latest)
        if job['id'] in fail_jobs:
            if (job['game'], job['seed'], job['kind']) in occupied:
                raise ValueError(f"{job['id']}: 아직 큐/CSV에 실행 중입니다. 먼저 상태를 확인하세요.")
            job['status'] = 'failed'
            continue
        if job['status'] in {'complete', 'failed'}:
            if job['status'] == 'complete' and job['kind'] == 'retrieval':
                job['accepted'] = (job['score'] >= job['threshold_score']
                                   and job['seed'] not in excluded.get(job['game'], set()))
            continue
        # A queued rerun with the same seed must not be completed using an old
        # finished CSV row, particularly during adoption of legacy queue jobs.
        if any((q['game'], q['seed']) == (job['game'], job['seed']) and job['kind'] in q['kinds']
               for q in queued):
            job['status'] = 'pending'
        elif any((r['game'], r['seed']) == (job['game'], job['seed'])
                 and job['kind'] in running_kinds(r) for r in running):
            job['status'] = 'running'
        elif latest and latest['State'] == 'finished' and latest['score'] is not None:
            job.update(status='complete', score=latest['score'], run_id=latest['Run ID'])
            if job['kind'] == 'retrieval':
                job['accepted'] = (latest['score'] >= job['threshold_score']
                                   and job['seed'] not in excluded.get(job['game'], set()))
        elif latest and latest['State'] in {'failed', 'crashed', 'killed'}:
            job['status'] = 'failed'
        else:
            job['status'] = 'awaiting_result'
            notices.append(f"{job['id']}: CSV/큐에서 완료를 확인할 수 없어 재등록하지 않습니다.")
    unknown_failures = set(fail_jobs) - {job['id'] for job in jobs}
    if unknown_failures:
        raise ValueError(f'없는 작업 ID: {sorted(unknown_failures)}')

    slots = {gpu: [0.0] * spec['count'] for gpu, spec in config['gpus'].items()}
    # Shared warmup/child transitions may briefly overlap in a W&B snapshot.
    loads = {}
    for row in running:
        gpu = gpu_for_seed(row['seed'], config, jobs, queued)
        key = gpu, row['game'], row['seed']
        loads[key] = max(loads.get(key, 0), running_hours(row, config['gpus'][gpu], now))
    for (gpu, _, _), hours in sorted(loads.items(), key=lambda item: -item[1]):
        push_load(slots, gpu, hours)
    for item in queued:
        push_load(slots, item['gpu'], duration(item['names'], config['gpus'][item['gpu']], not item['resume']))
    for job in jobs:
        if job['status'] == 'awaiting_result':
            push_load(slots, job['gpu'], duration([job['kind']], config['gpus'][job['gpu']],
                                                job['kind'] == 'retrieval'))
    initial_slots = copy.deepcopy(slots)
    used_seeds = {r['seed'] for r in rows} | {q['seed'] for q in queued} | {j['seed'] for j in jobs}
    used_seeds |= {seed for seeds in excluded.values() for seed in seeds}

    def add_job(game, gpu, kind, source=None):
        if source:
            seed, warmup = source['seed'], source['warmup']
            comparison = {}
        else:
            comparison = threshold(game)
            if comparison is None:
                return False
            start, end = config['gpus'][gpu]['new_seed_range']
            seed = next((s for s in range(start, end + 1) if s not in used_seeds), None)
            if seed is None:
                raise ValueError(f'{gpu}: 새 시드 범위를 모두 사용했습니다.')
            warmup = f'ckpt/{game}-{seed}_Shared'
        job = {'id': f'{game}:{seed}:{kind}', 'game': game, 'seed': seed, 'kind': kind,
               'gpu': gpu, 'warmup': warmup, 'status': 'pending', 'adopted': False,
               'registered_at': now.isoformat(),
               'before_ids': [r['Run ID'] for r in rows if (r['game'], r['seed'], r['kind']) == (game, seed, kind)],
               'command': make_command(game, seed, kind, warmup), **comparison}
        if source:
            job['source_job'] = source['id']
        hours = duration([kind], config['gpus'][gpu], kind == 'retrieval')
        job['estimated_start_hours'] = push_load(slots, gpu, hours)
        job['estimated_finish_hours'] = job['estimated_start_hours'] + hours
        previous = next((j for j in jobs if j['id'] == job['id']), None)
        if previous:
            job['attempt'] = previous.get('attempt', 1) + 1
            jobs[jobs.index(previous)] = job
        else:
            jobs.append(job)
        additions.append(job)
        used_seeds.add(seed)
        return True

    # Promote successful targets before allocating speculative seed searches.
    for job in list(jobs):
        if job['kind'] != 'retrieval' or job['status'] != 'complete' or not job.get('accepted'):
            continue
        if job['seed'] in excluded.get(job['game'], set()):
            continue
        baseline = next((j for j in jobs if j['game'] == job['game'] and j['seed'] == job['seed']
                         and j['kind'] == 'baseline'), None)
        if baseline and baseline['status'] != 'failed':
            continue
        # An already completed manual followup is valid only if it names this
        # same warmup, not merely an older baseline with the same seed.
        paired = [r for r in rows if r['kind'] == 'baseline'
                  and (r['game'], r['seed']) == (job['game'], job['seed'])
                  and r['State'] == 'finished' and r['score'] is not None
                  and warmup_for_row(r) and job['warmup']
                  and warmup_root(warmup_for_row(r)) == warmup_root(job['warmup'])]
        if paired:
            latest = max(paired, key=lambda r: timestamp(r.get('Created At')))
            completed = {'id': f"{job['game']}:{job['seed']}:baseline", 'kind': 'baseline',
                         'game': job['game'], 'seed': job['seed'], 'gpu': job['gpu'],
                         'warmup': job['warmup'], 'source_job': job['id'], 'status': 'complete',
                         'before_ids': [], 'command': '', 'adopted': True,
                         'score': latest['score'], 'run_id': latest['Run ID']}
            if baseline:
                baseline.update(completed)
            else:
                jobs.append(completed)
            continue
        if (job['game'], job['seed'], 'baseline') in occupied:
            notices.append(f"{job['id']}: 기존 baseline 실행/큐가 있어 후속 등록을 보류합니다.")
            continue
        if not job['warmup']:
            notices.append(f"{job['id']}: 통과했지만 warmup 경로가 없습니다. classify_wandb_runs.py를 다시 실행하세요.")
            continue
        if not slots[job['gpu']]:
            notices.append(f"{job['id']}: warmup이 있는 {job['gpu']}의 GPU count가 0입니다.")
            continue
        if len(additions) < config['max_new_jobs']:
            add_job(job['game'], job['gpu'], 'baseline', job)

    def active_count(game):
        tracked = {(j['game'], j['seed']) for j in jobs
                   if j['game'] == game and j['kind'] == 'retrieval' and j['status'] in ACTIVE}
        external = {(g, seed) for g, seed, kind in occupied if g == game and kind == 'retrieval'}
        return len(tracked | external)

    def enough_successes(game):
        return sum(j['game'] == game and j['kind'] == 'retrieval' and j['status'] == 'complete'
                   and j.get('accepted', False)
                   and j['seed'] not in excluded.get(game, set()) for j in jobs) >= config['successful_pairs_per_game']

    eligible = [game for game in summaries if not enough_successes(game) and threshold(game) is not None]
    eligible.sort(key=lambda game: (-summaries[game]['priority'], game))

    def best_gpu(allowed=None, fill=False):
        available = [gpu for gpu in (allowed or slots) if slots[gpu]]
        return min(available, key=lambda gpu: (
            min(slots[gpu]) if fill else 0,
            min(slots[gpu]) + duration(['retrieval'], config['gpus'][gpu]), gpu), default=None)

    # One initial/replacement trial per game; queued and running trials count.
    for game in eligible:
        if len(additions) >= config['max_new_jobs']:
            break
        if active_count(game) == 0:
            gpu = best_gpu()
            if gpu:
                add_job(game, gpu, 'retrieval')
    # Then fill only slots forecast to empty within the planning horizon.
    while eligible and len(additions) < config['max_new_jobs']:
        spare = [gpu for gpu in slots if slots[gpu] and min(slots[gpu]) < config['lookahead_hours']]
        if not spare:
            break
        game = min(eligible, key=lambda g: (active_count(g), -summaries[g]['priority'], g))
        if not add_job(game, best_gpu(spare, fill=True), 'retrieval'):
            break

    for job in jobs:
        if job['kind'] != 'retrieval' or not (job['status'] == 'failed' or (
                job['status'] == 'complete' and (not job.get('accepted')
                or job['seed'] in excluded.get(job['game'], set())))):
            continue
        # Do not recommend deletion while any variant might still use the source.
        busy = any(q['game'] == job['game'] and q['seed'] == job['seed'] for q in queued)
        busy |= any(r['game'] == job['game'] and r['seed'] == job['seed'] for r in running)
        busy |= any(j['game'] == job['game'] and j['seed'] == job['seed'] and j['status'] in ACTIVE for j in jobs)
        if not busy:
            cleanup.append({'job': job['id'], 'gpu': job['gpu'], 'warmup': job['warmup'],
                            'reason': 'failed' if job['status'] == 'failed' else 'below_threshold_or_excluded'})
    return state, {'new_jobs': additions, 'games': summaries, 'cleanup': cleanup,
                   'gpu_available_hours_before': initial_slots, 'gpu_available_hours_after': slots,
                   'notices': sorted(set(notices))}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def print_report(report, state, dry_run):
    print('미리보기 (큐/상태 저장 안 함)' if dry_run else '실험 큐 갱신 완료')
    for game, summary in report['games'].items():
        gaps = ', '.join(f'{kind} ΔHNS={gap:.3f}' if gap is not None else f'{kind} 없음'
                         for kind, gap in summary['hns_gaps'].items())
        print(f'대상 {game}: 논문 대비 부족분 {gaps}')
    for gpu, available in report['gpu_available_hours_before'].items():
        print(f'{gpu}: 기존 작업 후 GPU별 예상 여유 시점 {[round(v, 2) for v in available]} 시간')
    for job in state['jobs']:
        if job['kind'] == 'retrieval':
            result = f", 결과={job['score']:g}, {'통과' if job.get('accepted') else '미달/제외'}" if 'score' in job else ''
            print(f"{job['id']} [{job['status']}] 기준 seed {job['reference_seed']}: "
                  f"{job['reference_score']:g} → {job['threshold_score']:g} 이상{result}")
    for job in report['new_jobs']:
        print(f"추가 {job['gpu']} {job['id']} "
              f"(예상 시작 +{job['estimated_start_hours']:.2f}h, 완료 +{job['estimated_finish_hours']:.2f}h)")
        print(job['command'])
    print(f"새 명령 {len(report['new_jobs'])}개")
    for item in report['cleanup']:
        path = item['warmup'] or '경로 미확인: classify_wandb_runs.py를 다시 실행해 확인'
        print(f"warmup 삭제 필요 (자동 삭제 안 함): {item['job']} / {item['gpu']} / {path}")
    for notice in report['notices']:
        print(f'안내: {notice}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'experiment_scheduler.json')
    parser.add_argument('--csv', type=Path, default=HERE / 'wandb_runs_classification.csv')
    parser.add_argument('--queue-dir', type=Path, default=HERE.parent)
    parser.add_argument('--state', type=Path, default=HERE / 'experiment_scheduler_state.json')
    parser.add_argument('--report', type=Path, help='판정과 생성 명령을 JSON으로 저장')
    parser.add_argument('--games', nargs='+', help='자동 대상 선정 대신 지정한 게임만 탐색')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fail-job', action='append', default=[], metavar='GAME:SEED:KIND',
                        help='큐/실행 중에 없고 결과가 유실된 작업을 실패로 기록')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    references = literal_setting(HERE / 'convert_csv_to_excel.py', 'REFERENCE_SCORES')
    excluded = literal_setting(HERE / 'update_tex.py', 'EXCLUDED_SEEDS')
    args.queue_dir = args.queue_dir.resolve()
    args.state = args.state.resolve()
    with ExitStack() as stack:
        # Serialize planners independently of custom ledger paths, then use the
        # exact locks already used by 0_train.sh ... 7_train.sh.
        for path in [args.queue_dir / 'experiment_scheduler.lock', *(
                args.queue_dir / f'job_queue_{gpu}.lock' for gpu in sorted(config['gpus']))]:
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_rows(args.csv)
        paths = {gpu: args.queue_dir / f'job_queue_{gpu}.txt' for gpu in config['gpus']}
        texts = {gpu: path.read_text(encoding='utf-8') if path.exists() else '' for gpu, path in paths.items()}
        state = json.loads(args.state.read_text(encoding='utf-8')) if args.state.exists() else {}
        state, report = plan(rows, texts, state, config, references, excluded,
                             datetime.now(timezone.utc), args.games, args.fail_job)
        if not args.dry_run:
            # Reserve before dispatch. If a write fails, missing results remain
            # reserved for manual reconciliation; never duplicate expensive jobs.
            atomic_json(args.state, state)
            for gpu, path in paths.items():
                commands = [job['command'] for job in report['new_jobs'] if job['gpu'] == gpu]
                if commands:
                    with path.open('a', encoding='utf-8') as output:
                        if texts[gpu] and not texts[gpu].endswith('\n'):
                            output.write('\n')
                        output.write('\n'.join(commands) + '\n')
                        output.flush()
                        os.fsync(output.fileno())
        if args.report:
            atomic_json(args.report, report)
    print_report(report, state, args.dry_run)
    return report


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, SyntaxError) as error:
        raise SystemExit(f'스케줄링 오류: {error}') from error
