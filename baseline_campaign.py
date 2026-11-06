#!/usr/bin/env python3
"""Editable reservations for baseline-first STORM seed selection.

init only writes reservations. Training starts exclusively through worker or
run-job. No W&B connection is needed to advance the campaign.
"""

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
DEFAULT_STATE = ROOT / 'results/baseline_campaign/state.json'
PREFIX = 'JointTrainAgent.Retrieval.'
PARAMETER_SUFFIXES = {'.pt', '.pth', '.npz'}
TERMINAL = {'done', 'cancelled'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)


def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{name} not found in {path}')


def hns(score, reference):
    return (score - reference['random']) / (reference['human'] - reference['random'])


def satisfied(score_hns, reference, margin):
    threshold = hns(reference['reproduced'], reference) - margin
    return score_hns >= threshold or math.isclose(score_hns, threshold, rel_tol=0, abs_tol=1e-12)


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def snapshot_index(source, pairs):
    """A partially collected external run waits without occupying a GPU."""
    source = Path(source)
    index_path = source / 'warmups.json'
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text())
    if index.get('version') != 1 or index.get('kind') != 'storm_warmup_sweep':
        raise ValueError(f'Unsupported sweep: {source}')
    for target, _ in pairs:
        record = index['checkpoints'].get(str(target))
        if record is None:
            return None
        checkpoint = source / record['checkpoint']
        if not checkpoint.resolve().is_relative_to(source.resolve()):
            raise ValueError('Snapshot escapes its source directory')
        if not all((checkpoint / name).is_file() for name in (
                'training_state.pt', 'replay_buffer.npz', 'config.yaml', 'warmup_metadata.json')):
            return None
    return index


def latest_target(index, pairs):
    return max(pairs, key=lambda pair: index['checkpoints'][str(pair[0])]['actual_step'])


def continuation(state, attempt, kind, pairs):
    base = attempt['baseline_base'] if kind == 'baseline' else attempt['retrieval_base']
    return [state['python'], '-u', str(Path(state['root']) / 'train.py'),
            '--resume_warmup', attempt['source'], '-n', base,
            PREFIX + 'enable', "['baseline']" if kind == 'baseline' else 'True',
            PREFIX + 'warmup_offset_pairs', json.dumps(pairs)]


def fresh_command(state, attempt):
    return [state['python'], '-u', str(Path(state['root']) / 'train.py'),
            '-n', attempt['base'], '-seed', str(attempt['seed']),
            '-config_path', state['config'], '-env_name', f"ALE/{attempt['game']}-v5",
            '-trajectory_path', str(Path(state['root']) / 'D_TRAJ' / f"{attempt['game']}.pkl"),
            '--baseline_first', PREFIX + 'enable', 'False',
            PREFIX + 'warmup_offset_pairs', json.dumps(state['pairs']),
            'JointTrainAgent.SeedScreenSteps', '0']


def create_job(state, attempt, kind, command=None):
    job_id = ('b-' if kind == 'baseline' else 'r-') + attempt['id']
    job = {'id': job_id, 'attempt': attempt['id'], 'kind': kind,
           'round': attempt['round'], 'status': 'new' if command else 'waiting_snapshot',
           'command': command, 'created_at': utcnow()}
    state['jobs'][job_id] = job
    return job


def add_attempt(state, game, seed, *, source=None, origin='new', retrieval_done=False):
    attempt_id = f"a{len(state['attempts']) + 1:05d}"
    base = f"{game}-{seed}-bf-{state['campaign']}"
    attempt = {'id': attempt_id, 'game': game, 'seed': seed, 'origin': origin,
               'round': state['round'], 'base': base,
               'source': str(source or (Path(state['root']) / 'ckpt' / f'{base}_WarmupSweep')),
               'baseline_base': base if origin == 'new' else base + '-backfill',
               'retrieval_base': base + '-accepted', 'retrieval_done': retrieval_done,
               'score': None, 'hns': None, 'passed': None, 'retained': None}
    state['attempts'][attempt_id] = attempt
    state['used_seeds'].setdefault(game, []).append(seed)
    if origin == 'new':
        attempt['baseline_name'] = base + f"_w{max(p[0] for p in state['pairs'])}_baseline"
        create_job(state, attempt, 'baseline', fresh_command(state, attempt))
    else:
        create_job(state, attempt, 'baseline')
    return attempt


def game_passed(state, game):
    return any(attempt['game'] == game and attempt['passed'] is True
               for attempt in state['attempts'].values())


def failed_retention(state):
    """Equal scores keep older attempts; successful runs are outside this quota."""
    for game in state['game_order']:
        failed = [attempt for attempt in state['attempts'].values()
                  if attempt['game'] == game and attempt['passed'] is False]
        failed.sort(key=lambda attempt: (-attempt['hns'], attempt['id']))
        keep = {attempt['id'] for attempt in failed[:state['keep_failed']]}
        for attempt in failed:
            attempt['retained'] = attempt['id'] in keep


def plan(state):
    for attempt in list(state['attempts'].values()):
        if attempt.get('retrieval_external') and not attempt['retrieval_done']:
            external_result = Path(attempt['source']) / 'branch_results.json'
            if external_result.is_file():
                result = json.loads(external_result.read_text())
                if result.get('status') == 'completed':
                    attempt['retrieval_done'] = True
                elif result.get('status') == 'failed' and not externally_pinned(
                        state['root'], [Path(attempt['source'])]):
                    attempt['retrieval_external'] = False
        baseline_job = state['jobs']['b-' + attempt['id']]
        if baseline_job['status'] == 'waiting_snapshot':
            index = snapshot_index(attempt['source'], state['pairs'])
            if index:
                if index['seed'] != attempt['seed'] or index['env_name'] != f"ALE/{attempt['game']}-v5":
                    raise ValueError('Saved sweep identity differs from the reserved attempt')
                target = latest_target(index, state['pairs'])
                attempt['baseline_name'] = attempt['baseline_base'] + f'_w{target[0]}_baseline'
                baseline_job.update(command=continuation(state, attempt, 'baseline', [target]), status='new')
        if attempt['passed'] and not attempt['retrieval_done'] and not attempt.get('retrieval_external'):
            key = 'r-' + attempt['id']
            if key not in state['jobs']:
                create_job(state, attempt, 'retrieval')
            job = state['jobs'][key]
            if job['status'] == 'waiting_snapshot' and snapshot_index(attempt['source'], state['pairs']):
                job.update(command=continuation(state, attempt, 'retrieval', state['pairs']), status='new')
    failed_retention(state)
    # Baseline errors, pauses and in-flight jobs hold the round barrier.
    if any(job['kind'] == 'baseline' and job['status'] not in TERMINAL
           for job in state['jobs'].values()):
        return
    remaining = [game for game in state['game_order'] if not game_passed(state, game)]
    if not remaining:
        finished = all(job['status'] in TERMINAL for job in state['jobs'].values()) and all(
            not attempt['passed'] or attempt['retrieval_done'] for attempt in state['attempts'].values())
        state['phase'] = 'complete' if finished else 'retrieval'
        return
    counts = Counter(attempt['game'] for attempt in state['attempts'].values())
    if state['phase'] in {'backfill', 'balance'}:
        eligible = [game for game in remaining if counts[game] < state['balance_target']]
        if eligible:
            state['phase'] = 'balance'
        else:
            state['phase'] = 'retry'
            eligible = remaining
    else:
        eligible = remaining
    state['round'] += 1
    for game in eligible:
        used = set(state['used_seeds'].get(game, []))
        seed = state['start_seed']
        while seed in used:
            seed += 1
        if seed >= 2**32:
            raise ValueError(f'{game}: no valid seeds remain')
        add_attempt(state, game, seed)


def record_score(state, attempt, score):
    if not math.isfinite(score):
        raise ValueError('Baseline evaluation is not finite')
    attempt['score'] = score
    attempt['hns'] = hns(score, state['references'][attempt['game']])
    attempt['passed'] = satisfied(attempt['hns'], state['references'][attempt['game']], state['margin'])
    attempt['scored_at'] = utcnow()


def reservation(state_path, state, job_id):
    job = state['jobs'][job_id]
    attempt = state['attempts'][job['attempt']]
    return shlex.join([state['python'], '-u', str(Path(state['root']) / SCRIPT.name),
                       '--state', str(state_path), 'run-job', job_id]) + (
        f" # {attempt['game']} seed={attempt['seed']} {job['kind']} round={job['round']}")


def line_job(line, state_path):
    if not line.strip() or line.lstrip().startswith('#'):
        return None
    tokens = shlex.split(line)
    if 'run-job' not in tokens or '--state' not in tokens:
        raise ValueError('Queue entries must be campaign run-job reservations; use show JOB to inspect training arguments')
    if Path(tokens[tokens.index('--state') + 1]).resolve() != Path(state_path).resolve():
        raise ValueError('Queue entry belongs to another campaign')
    return tokens[tokens.index('run-job') + 1]


class Session:
    def __init__(self, state_path):
        self.path = Path(state_path).resolve()

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.lock = self.stack.enter_context(self.path.with_suffix('.lock').open('a'))
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            self.state = json.loads(self.path.read_text())
            self.queue = Path(self.state['queue'])
            self.queue_lock = self.stack.enter_context(self.queue.with_suffix('.lock').open('a'))
            fcntl.flock(self.queue_lock, fcntl.LOCK_EX)
            self.lines = self.queue.read_text().splitlines() if self.queue.exists() else []
            present = [line_job(line, self.path) for line in self.lines]
            ids = [item for item in present if item]
            if len(ids) != len(set(ids)) or any(key not in self.state['jobs'] for key in ids):
                raise ValueError('Queue contains duplicate or unknown job IDs')
            for job in self.state['jobs'].values():
                # Removing a reservation pauses it, instead of recreating it.
                if job['status'] == 'queued' and job['id'] not in present:
                    job['status'] = 'paused'
            self.lines = [line for line in self.lines
                          if not line_job(line, self.path) or self.state['jobs'][line_job(line, self.path)]['status'] not in TERMINAL | {'running'}]
            return self
        except BaseException:
            self.stack.close()
            raise

    def sync(self):
        recover_jobs(self.state)
        plan(self.state)
        for job in self.state['jobs'].values():
            if job['status'] == 'new':
                self.lines.append(reservation(self.path, self.state, job['id']))
                job['status'] = 'queued'

    def remove(self, job_id):
        self.lines = [line for line in self.lines if line_job(line, self.path) != job_id]

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                atomic_json(self.path, self.state)
                atomic_text(self.queue, '\n'.join(self.lines) + ('\n' if self.lines else ''))
                write_scores(self.path.parent / 'baseline_results.csv', self.state)
        finally:
            self.stack.close()


def recover_jobs(state):
    for job in state['jobs'].values():
        if job['status'] != 'running' or job.get('host') != socket.gethostname():
            continue
        try:
            os.kill(job['runner_pid'], 0)
            continue
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        if job.get('child_pid'):
            try:
                os.kill(job['child_pid'], 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        attempt = state['attempts'][job['attempt']]
        try:
            if job['kind'] == 'baseline':
                score = result_score(state, attempt['baseline_name'], attempt)
                record_score(state, attempt, score)
            else:
                for name in retrieval_names(state, attempt):
                    result_score(state, name, attempt)
                attempt['retrieval_done'] = True
            job.update(status='done', error=None, recovered_at=utcnow())
        except (OSError, ValueError, KeyError):
            job.update(status='interrupted', error='Owning worker exited; use retry to resume')


def protected_source(root, source):
    source = Path(source)
    checkpoint_root = (Path(root) / 'ckpt').resolve()
    if source.is_symlink() or not source.resolve().is_relative_to(checkpoint_root) or source.resolve() == checkpoint_root:
        raise ValueError(f'Unsafe checkpoint deletion path: {source}')
    index = source / 'warmups.json'
    if not index.is_file():
        raise ValueError(f'Deletion requires a sweep manifest: {source}')
    return source


def externally_pinned(root, paths):
    needles = [str(path.resolve()) for path in paths]
    needles += [str(path.resolve().relative_to(Path(root).resolve())) for path in paths]
    # Old queues may still have consumers of an imported checkpoint.
    texts = [path.read_text() for path in Path(root).glob('job_queue*.txt')]
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(entry.parent.name) == os.getpid():
                continue
            texts.append(entry.read_bytes().replace(b'\0', b' ').decode(errors='replace'))
        except (OSError, ValueError):
            continue
    return any(needle in text for needle in needles for text in texts)


def prune_failed(state):
    """Delete only checkpoint payloads; keep scores, logs and manifests."""
    failed_retention(state)
    for attempt in state['attempts'].values():
        if attempt['retained'] is not False or attempt.get('pruned_at'):
            continue
        related = [job for job in state['jobs'].values() if job['attempt'] == attempt['id']]
        if any(job['status'] not in TERMINAL for job in related):
            continue
        try:
            source = protected_source(state['root'], attempt['source'])
        except ValueError as exc:
            attempt['prune_pending'] = str(exc)
            continue
        baseline = Path(state['root']) / 'ckpt' / attempt['baseline_name']
        paths = [source, baseline]
        if externally_pinned(state['root'], paths):
            attempt['prune_pending'] = 'Referenced by another process or queue'
            continue
        removed = attempt.setdefault('deleted_parameters', [])
        for directory in paths:
            if directory.is_symlink() or not directory.resolve().is_relative_to((Path(state['root']) / 'ckpt').resolve()):
                raise ValueError('Unsafe checkpoint directory')
            for payload in directory.rglob('*'):
                if payload.suffix not in PARAMETER_SUFFIXES or not payload.is_file():
                    continue
                if payload.is_symlink() or any(parent.is_symlink() for parent in payload.parents if parent != directory.parent):
                    raise ValueError('Refusing to delete a symlinked checkpoint')
                try:
                    payload.unlink()
                    removed.append(str(payload))
                except OSError as exc:
                    attempt['prune_pending'] = str(exc)
                    break
            else:
                continue
            break
        else:
            attempt.update(pruned_at=utcnow(), prune_pending=None)


def result_score(state, run_name, attempt):
    root = Path(state['root'])
    result = json.loads((root / 'runs' / run_name / 'final_evaluation.json').read_text())
    completion = json.loads((root / 'ckpt' / run_name / 'training_complete.json').read_text())
    if (result['run_name'] != run_name or result['seed'] != attempt['seed']
            or result['env_name'] != f"ALE/{attempt['game']}-v5"
            or completion['next_step'] < result['sample_max_steps']):
        raise ValueError('Final evaluation identity or completion does not match its job')
    score = float(result['episode_avg_return'])
    if not math.isfinite(score):
        raise ValueError('Non-finite final evaluation')
    return score


def write_scores(path, state):
    fields = ['Game', 'Seed', 'Round', 'Origin', 'Baseline Return', 'Baseline HNS',
              'Reproduced HNS', 'Threshold HNS', 'Passed', 'Retained Failed Parameters',
              'Parameters Pruned', 'Baseline Run', 'Source']
    temporary = Path(path).with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for attempt in state['attempts'].values():
            reference_hns = hns(state['references'][attempt['game']]['reproduced'], state['references'][attempt['game']])
            writer.writerow(dict(zip(fields, [attempt['game'], attempt['seed'], attempt['round'], attempt['origin'],
                attempt['score'], attempt['hns'], reference_hns, reference_hns - state['margin'],
                attempt['passed'], attempt['retained'], bool(attempt.get('pruned_at')),
                attempt.get('baseline_name', ''), attempt['source']])))
    os.replace(temporary, path)


def retrieval_names(state, attempt):
    return [attempt['retrieval_base'] + f"_w{target}_offset_{'m' + str(-offset) if offset < 0 else offset}_retrieval"
            for target, offset in state['pairs']]


def claim(state_path, requested=None):
    with Session(state_path) as session:
        session.sync()
        candidates = [line_job(line, session.path) for line in session.lines]
        candidates = [key for key in candidates if key]
        job_id = requested or (candidates[0] if candidates else None)
        if job_id is None:
            return None
        job = session.state['jobs'][job_id]
        if job['status'] not in {'queued', 'paused'}:
            raise ValueError(f"{job_id}: cannot claim {job['status']} job")
        if job['kind'] == 'retrieval' and not session.state['attempts'][job['attempt']]['passed']:
            raise ValueError('Retrieval requires a passing baseline')
        session.remove(job_id)
        job.update(status='running', runner_pid=os.getpid(), host=socket.gethostname(), started_at=utcnow())
        return session.state, job.copy()


def run_job(state_path, requested=None, gpu=None):
    claimed = claim(state_path, requested)
    if claimed is None:
        return None
    state, job = claimed
    attempt = state['attempts'][job['attempt']]
    log_dir = Path(state_path).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{job['id']}.log"
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    process = None
    error = None
    interrupted = False
    handlers = {}
    def interrupt(*_):
        raise KeyboardInterrupt()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
        print(f"{job['id']} {attempt['game']} seed={attempt['seed']} {job['kind']}; log={log_path}", flush=True)
        with log_path.open('a') as output:
            output.write(utcnow() + ' ' + shlex.join(job['command']) + '\n')
            output.flush()
            process = subprocess.Popen(job['command'], cwd=state['root'], env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            with Session(state_path) as session:
                session.state['jobs'][job['id']].update(child_pid=process.pid, gpu=env.get('CUDA_VISIBLE_DEVICES'))
            if process.wait() != 0:
                raise RuntimeError(f"Training exited {process.returncode}; see {log_path}")
        if job['kind'] == 'baseline':
            score = result_score(state, attempt['baseline_name'], attempt)
        else:
            for name in retrieval_names(state, attempt):
                result_score(state, name, attempt)
    except KeyboardInterrupt:
        interrupted = True
        error = 'Interrupted by user; retry this reservation to continue'
    except Exception as exc:
        error = str(exc)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
    with Session(state_path) as session:
        current = session.state['jobs'][job['id']]
        current.update(status=('interrupted' if interrupted else 'error') if error else 'done',
                       error=error, finished_at=utcnow(), log=str(log_path))
        attempt = session.state['attempts'][job['attempt']]
        if not error:
            if job['kind'] == 'baseline':
                record_score(session.state, attempt, score)
            else:
                attempt['retrieval_done'] = True
            session.sync()
    if not error:
        # Persist the evaluation/gate before touching any checkpoint payloads.
        with Session(state_path) as session:
            prune_failed(session.state)
    if error:
        raise RuntimeError(error)
    return job['id']


def initialize(args):
    sys.path.insert(0, str(ROOT / 'results'))
    import compare_warmup as comparison
    workbook = Path(args.workbook).resolve()
    cells = comparison.load_scores(workbook)
    references = literal(ROOT / 'results/convert_csv_to_excel.py', 'REFERENCE_SCORES')
    reproduced = literal(ROOT / 'results/convert_csv_to_excel.py', 'REPRODUCED_SCORES')
    panel = comparison.matched_panel(cells.loc[cells.Usable])
    means = panel.groupby('Game').mean()
    game_order = args.games or sorted(reproduced, key=lambda game:
        -((reproduced[game] - references[game][0]) / (references[game][1] - references[game][0])
          - (means.loc[game].max() if game in means.index else 0)))
    if len(game_order) != len(set(game_order)) or set(game_order) - set(reproduced):
        raise ValueError('Unknown or duplicate games')
    state_path = Path(args.state).resolve()
    queue_path = Path(args.queue).resolve()
    if state_path.exists():
        raise FileExistsError(f'{state_path} already exists; use sync/status instead of init')
    if queue_path.exists() and queue_path.read_text().strip():
        raise FileExistsError(f'{queue_path} is not empty')
    campaign = args.name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    if not campaign or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in campaign):
        raise ValueError('Campaign name must be a simple filename component')
    state = {'version': 1, 'campaign': campaign, 'created_at': utcnow(), 'root': str(ROOT),
             'python': str(Path(args.python).resolve()), 'config': str(Path(args.config).resolve()),
             'queue': str(queue_path), 'workbook': str(workbook), 'margin': .1, 'keep_failed': 4,
             'pairs': [[30000, -2], [50000, -2], [70000, -2]], 'start_seed': args.start_seed,
             'game_order': game_order, 'references': {game: {
                 'random': references[game][0], 'human': references[game][1],
                 'reproduced': reproduced[game]} for game in game_order},
             'attempts': {}, 'jobs': {}, 'used_seeds': {}, 'phase': 'backfill', 'round': 0}
    # Reserve every seed already present in CSV or in any old training queue.
    for row in read_csv(ROOT / 'results/wandb_runs_classification.csv'):
        try:
            seed = int(float(row['Seed']))
            game = row['Run Name'].split('_')[0]
            state['used_seeds'].setdefault(game, []).append(seed)
        except (KeyError, ValueError):
            pass
    for path in ROOT.glob('job_queue*.txt'):
        for line in path.read_text().splitlines():
            try:
                tokens = shlex.split(line)
                seed = int(tokens[tokens.index('-seed') + 1])
                game = tokens[tokens.index('-env_name') + 1].split('/')[-1].split('-v')[0]
                state['used_seeds'].setdefault(game, []).append(seed)
            except (ValueError, IndexError):
                pass
    for directory in (ROOT / 'ckpt').iterdir():
        game = directory.name.split('-')[0]
        if game not in state['references']:
            continue
        match = re.match(r'^[A-Za-z]+-(\d+)(?:\D|$)', directory.name)
        match = match or re.search(r'(?:^|-)seed(\d+)(?:\D|$)', directory.name)
        if match:
            state['used_seeds'].setdefault(game, []).append(int(match[1]))
    # Include collectors that started since the last CSV/Excel export, but not
    # queued-only seeds whose collector has not been created yet.
    started = [bool(row.Usable or row['Has Running'] or (
        row['Raw Cell'] and (ROOT / 'ckpt' / f'{row.Game}-{row.Seed}_WarmupSweep' / 'warmups.json').is_file()))
        for _, row in cells.iterrows()]
    current = cells.loc[started]
    initial = {}
    missing = []
    for game in game_order:
        seeds = sorted(set(map(int, current.loc[current.Game.eq(game), 'Seed'])))
        initial[game] = seeds
    # Historical baselines also visit every game once before its next seed.
    panel_keys = set(panel.index)
    for position in range(max(map(len, initial.values()), default=0)):
        for game in game_order:
            if position >= len(initial[game]):
                continue
            seed = initial[game][position]
            source = ROOT / 'ckpt' / f'{game}-{seed}_WarmupSweep'
            if not (source / 'warmups.json').is_file():
                missing.append(str(source))
            done = (game, seed) in panel_keys
            attempt = add_attempt(state, game, seed, source=source, origin='existing', retrieval_done=done)
            attempt['retrieval_external'] = True
    state['initial_seeds'] = initial
    state['initial_completed_counts'] = panel.groupby('Game').size().reindex(game_order, fill_value=0).to_dict()
    state['balance_target'] = max(map(len, initial.values()), default=1)
    if missing:
        raise FileNotFoundError('Current runs lack reusable sweep indexes: ' + ', '.join(missing))
    plan(state)
    lines = []
    for job in state['jobs'].values():
        if job['status'] == 'new':
            lines.append(reservation(state_path, state, job['id']))
            job['status'] = 'queued'
    show_state(state)
    if args.dry_run:
        print('\nPreview only; no reservations or files written.')
        print('\n'.join(lines))
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialization is exclusive too; two init processes cannot overwrite a campaign.
    with state_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if state_path.exists():
            raise FileExistsError(state_path)
        with queue_path.with_suffix('.lock').open('a') as queue_lock:
            fcntl.flock(queue_lock, fcntl.LOCK_EX)
            if queue_path.exists() and queue_path.read_text().strip():
                raise FileExistsError(queue_path)
            atomic_json(state_path, state)
            atomic_text(queue_path, '\n'.join(lines) + ('\n' if lines else ''))
            write_scores(state_path.parent / 'baseline_results.csv', state)
    print(f'Queue: {queue_path}\nState: {state_path}\nTraining has not been started.')


def show_state(state):
    print(f"Campaign={state['campaign']} phase={state['phase']} round={state['round']}")
    print('Jobs:', dict(Counter(job['status'] for job in state['jobs'].values())))
    print('Game                    Initial  Attempts  Best baseline HNS  Passed  Kept failures')
    for game in state['game_order']:
        attempts = [attempt for attempt in state['attempts'].values() if attempt['game'] == game]
        scores = [attempt['hns'] for attempt in attempts if attempt['hns'] is not None]
        best = f'{max(scores):.4f}' if scores else '-'
        kept = sum(attempt['passed'] is False and attempt['retained'] for attempt in attempts)
        print(f"{game:23} {len(state['initial_seeds'][game]):7} {len(attempts):9} {best:18} {str(game_passed(state, game)):7} {kept}")


def retry_job(session, job_id):
    job = session.state['jobs'][job_id]
    if job['status'] not in {'error', 'interrupted'}:
        raise ValueError('retry requires an error/interrupted job; use resume for a paused reservation')
    attempt = session.state['attempts'][job['attempt']]
    if job['kind'] == 'baseline':
        try:
            score = result_score(session.state, attempt['baseline_name'], attempt)
        except (OSError, ValueError, KeyError):
            pass
        else:
            record_score(session.state, attempt, score)
            job.update(status='done', error=None, recovered_at=utcnow())
            return
        index = snapshot_index(attempt['source'], session.state['pairs'])
        if index:
            target = latest_target(index, session.state['pairs'])
            job['command'] = continuation(session.state, attempt, 'baseline', [target])
        elif attempt['origin'] == 'new':
            index_path = Path(attempt['source']) / 'warmups.json'
            if index_path.is_file():
                index = json.loads(index_path.read_text())
                if index['checkpoints']:
                    record = max(index['checkpoints'].values(), key=lambda item: item['actual_step'])
                    job['command'] = fresh_command(session.state, attempt) + [
                        '--resume_from', str(Path(attempt['source']) / record['checkpoint'])]
                else:
                    raise ValueError('No full snapshot was saved; cancel this attempt so a new seed can be reserved')
    else:
        missing = []
        for pair, name in zip(session.state['pairs'], retrieval_names(session.state, attempt)):
            try:
                result_score(session.state, name, attempt)
            except (OSError, ValueError, KeyError):
                missing.append(pair)
        if not missing:
            attempt['retrieval_done'] = True
            job.update(status='done', error=None, recovered_at=utcnow())
            return
        job['command'] = continuation(session.state, attempt, 'retrieval', missing)
    job.update(status='new', error=None)


def worker(args):
    # Preflight occurs only when the user explicitly starts this GPU worker.
    if ',' in args.gpu or not args.gpu.strip():
        raise ValueError('Start one worker per GPU; supply one GPU ID')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu}
    with Session(args.state) as session:
        interpreter = session.state['python']
    probe = subprocess.run([interpreter, '-c',
        'import torch; assert torch.cuda.is_available(), "Selected CUDA device is unavailable"; '
        'print(torch.cuda.get_device_name(0))'], env=env, check=True, capture_output=True, text=True)
    lock_dir = ROOT / 'diagnostics'
    lock_dir.mkdir(exist_ok=True)
    safe_gpu = args.gpu.replace('/', '_')
    with (lock_dir / f'baseline_gpu_{safe_gpu}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'A baseline worker already owns GPU {args.gpu}') from exc
        print(f'Worker GPU {args.gpu}: {probe.stdout.strip()}', flush=True)
        completed = 0
        while True:
            result = run_job(args.state, gpu=args.gpu)
            if result:
                completed += 1
                if args.once or (args.max_jobs and completed >= args.max_jobs):
                    return
            else:
                with Session(args.state) as session:
                    session.sync()
                    if session.state['phase'] == 'complete' or args.once:
                        return
                time.sleep(args.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    commands = parser.add_subparsers(dest='action', required=True)
    init = commands.add_parser('init', help='Create editable reservations, without starting training')
    init.add_argument('--queue', type=Path, default=ROOT / 'job_queue_baseline_first.txt')
    init.add_argument('--workbook', type=Path, default=ROOT / 'results/converted_results.xlsx')
    init.add_argument('--config', type=Path, default=ROOT / 'config_files/STORM.yaml')
    init.add_argument('--python', default=sys.executable)
    init.add_argument('--games', nargs='+')
    init.add_argument('--name')
    init.add_argument('--start-seed', type=int, default=10000)
    init.add_argument('--dry-run', action='store_true')
    commands.add_parser('status')
    commands.add_parser('sync', help='Discover ready snapshots and append new reservations')
    for action in ('show', 'pause', 'resume', 'retry', 'cancel', 'run-job'):
        sub = commands.add_parser(action)
        sub.add_argument('job_id')
    work = commands.add_parser('worker')
    work.add_argument('--gpu', required=True)
    work.add_argument('--once', action='store_true')
    work.add_argument('--max-jobs', type=int, default=0)
    work.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args(argv)
    if args.action == 'init':
        if not 0 <= args.start_seed < 2**32:
            parser.error('--start-seed must be a valid uint32 seed')
        initialize(args)
    elif args.action == 'worker':
        if args.poll_seconds <= 0 or args.poll_seconds > 60 or args.max_jobs < 0:
            parser.error('Use 0 < poll-seconds <= 60 and max-jobs >= 0')
        worker(args)
    elif args.action == 'run-job':
        run_job(args.state, args.job_id)
    else:
        with Session(args.state) as session:
            if args.action == 'status':
                show_state(session.state)
            elif args.action == 'sync':
                session.sync()
                prune_failed(session.state)
                show_state(session.state)
            else:
                job = session.state['jobs'][args.job_id]
                if args.action == 'show':
                    print(json.dumps(job, indent=2, ensure_ascii=False))
                    if job['command']:
                        print(shlex.join(job['command']))
                elif args.action == 'pause':
                    if job['status'] not in {'queued', 'new', 'waiting_snapshot', 'paused'}:
                        raise ValueError('Only pending reservations can be paused')
                    session.remove(args.job_id)
                    job['status'] = 'paused'
                elif args.action == 'resume':
                    if job['status'] != 'paused':
                        raise ValueError('resume requires a paused reservation')
                    job['status'] = 'new' if job['command'] else 'waiting_snapshot'
                    session.sync()
                elif args.action == 'retry':
                    retry_job(session, args.job_id)
                    session.sync()
                elif args.action == 'cancel':
                    if job['status'] in {'running', 'done'}:
                        raise ValueError('Only pending/interrupted reservations can be cancelled')
                    job['status'] = 'cancelled'
                    session.remove(args.job_id)
                    session.sync()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
