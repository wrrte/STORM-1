"""Frozen queue cohorts shared by the hash/target collector and Excel exporter."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from statistics import fmean, stdev
import tempfile

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MANIFEST = HERE / 'hash_target_campaign.json'
DEFAULT_CSV = HERE / 'hash_target_runs.csv'
DEFAULT_EXCEL = HERE / 'hash_target_results.xlsx'
SCHEMA_VERSION = 1
PREFIX = 'JointTrainAgent.Retrieval.'
FIELDS = ['Schema Version', 'Campaign ID', 'Job ID', 'Stage', 'Game', 'GPU Type', 'Seed',
          'Setting', 'Hash Bits', 'Retrieval Target', 'Warmup ID', 'Warmup Directory',
          'Resume Warmup', 'Actual Warmup Step', 'Local Run Name', 'Config SHA256',
          'State', 'Eval Return', 'Eval Checkpoint Step', 'Eval Precision', 'Eval Episodes',
          'Eval Seed', 'Episode SD', 'Run ID', 'Source', 'Created At', 'Updated At',
          'Attempt Count', 'Notes']
STATIC_FIELDS = FIELDS[:16]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def experiment_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',value):
        raise ValueError('Use 1–80 letters/digits, dots, underscores or hyphens for --experiment, starting with a letter/digit')
    return value


def report_paths(experiment=None,manifest=None,csv=None,excel=None,results_dir=HERE):
    """An experiment name selects all three output paths; no code/file edits needed."""
    directory=Path(results_dir)
    if experiment is not None:
        directory=directory/'experiments'/experiment_name(experiment)
        defaults=('manifest.json','runs.csv','results.xlsx')
    else:
        defaults=('hash_target_campaign.json','hash_target_runs.csv','hash_target_results.xlsx')
    return tuple(Path(path) if path is not None else directory/name
                 for path,name in zip((manifest,csv,excel),defaults))


def list_experiments(jobs,results_dir=HERE):
    groups={}
    for job in jobs.values():
        name=job.get('experiment_id')
        if name:
            groups.setdefault(name,set()).add(job['id'])
    for path in sorted((Path(results_dir)/'experiments').glob('*/manifest.json')):
        manifest=load_manifest(path)
        groups.setdefault(path.parent.name,set()).update(job['id'] for job in manifest['jobs'])
    return [(name,len(identities)) for name,identities in sorted(groups.items())]


def number(value):
    if isinstance(value, bool) or value in (None, ''):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def get(config, key, default=None):
    if key in config:
        return config[key]
    node = config
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def effective_config(job):
    conf = yaml.safe_load(job['config_text'])
    for key, value in job['overrides'].items():
        node = conf
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = deepcopy(value)
    if job['game'] == 'Freeway' and 'JointTrainAgent.UseDemonstration' not in job['overrides']:
        conf['JointTrainAgent']['UseDemonstration'] = True
    return conf


def stage_specs(job, pairs):
    for bits, target in pairs:
        yield dict(key=f'retrieval_h{bits}_t{target}', setting=f'h{bits} / t{target}',
                   hash_bits=bits, target=target,
                   local_name=f'{job["name"]}_h{bits}_t{target}_retrieval')
    yield dict(key='baseline', setting='Baseline', hash_bits=None, target=None,
               local_name=f'{job["name"]}_baseline')


def stage_config(job, stage):
    conf = deepcopy(job['config'])
    ret = conf['JointTrainAgent']['Retrieval']
    ret.update(enable=stage['key'] != 'baseline', save_warmup=False,
               warmup_offset_pairs=[], hash_target_pairs=[])
    if stage['key'] != 'baseline':
        ret.update(hash_bits=stage['hash_bits'], target=stage['target'])
    return conf


def config_mismatch(actual, expected, prefix=''):
    """Saved YAML may have additional default keys, but every frozen key must match."""
    for key, value in expected.items():
        name = prefix + key
        if isinstance(value, dict):
            reason = config_mismatch(actual, value, name + '.')
            if reason:
                return reason
        elif get(actual, name) != value:
            return f'configuration mismatch: {name}'
    return None


def read_jobs(database):
    """Do not instantiate TrainingQueue: reporting must never recover/claim/mutate jobs."""
    database = Path(database).resolve()
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=10) as db:
        rows = db.execute('SELECT id,status,payload,created,worker FROM jobs ORDER BY seq').fetchall()
    return {identity: dict(json.loads(payload), status=status, created=created, worker=worker)
            for identity, status, payload, created, worker in rows}


def command_arg(command, key):
    values = [command[i + 1] for i, token in enumerate(command[:-1]) if token == key]
    return values[-1] if values else None


def same_path(first, second):
    return bool(first and second) and Path(first).resolve() == Path(second).resolve()


def plan_proves_source(job, stage, live):
    command = (live or {}).get('plan', {}).get(stage['key'])
    source = command_arg(command, '--resume_from') if command else job.get('plan_sources', {}).get(stage['key'])
    if source is None:
        return False
    if not (same_path(source, job['resume_warmup']) or same_path(source, job['warmup_source'])):
        raise ValueError('Queue plan changed the frozen shared warmup')
    if command and (command_arg(command, '-seed') != str(job['seed'])
                    or command_arg(command, '-env_name') != f'ALE/{job["game"]}-v5'):
        raise ValueError('Queue plan changed the frozen game/seed')
    return True


def check_source(source, job, stage, root, plan_proof):
    if same_path(source, job['resume_warmup']) or same_path(source, job['warmup_source']):
        return
    path = Path(source).resolve() if source else None
    # Interrupted branches resume their own full checkpoint, retaining the original plan.
    if (plan_proof and path and path.parent == (root / 'ckpt' / stage['local_name']).resolve()
            and path.name.startswith('queue_resume_')):
        return
    raise ValueError('Run resumed a different or unverifiable shared warmup')


def evaluation_fields(score, returns, precision, step, episodes, seed, manifest):
    required = manifest['evaluation']
    value = number(score)
    if value is None or number(step) != required['checkpoint_step'] or precision != required['precision']:
        raise ValueError('Missing final score or incompatible evaluation checkpoint/precision')
    if number(episodes) != required['episodes'] or number(seed) != required['seed']:
        raise ValueError('Incompatible evaluation episodes/seed')
    if len(returns) != required['episodes'] or any(number(v) is None for v in returns):
        raise ValueError('Final evaluation does not contain all episode returns')
    returns = [float(v) for v in returns]
    if not math.isclose(value, fmean(returns), rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError('Final score disagrees with its episode returns')
    return {'State': 'completed', 'Eval Return': value, 'Eval Checkpoint Step': int(step),
            'Eval Precision': precision, 'Eval Episodes': len(returns), 'Eval Seed': int(seed),
            'Episode SD': stdev(returns) if len(returns) > 1 else None}


def local_result(root, job, stage, live, manifest, *, expected_config=None):
    directory = root / 'runs' / stage['local_name']
    result_path = directory / 'final_evaluation.json'
    if not result_path.exists():
        return None
    conf = yaml.safe_load((directory / 'config.yaml').read_text())
    expected = stage_config(job, stage) if expected_config is None else expected_config
    reason = config_mismatch(conf, expected)
    if reason:
        raise ValueError(reason)
    result = json.loads(result_path.read_text())
    if (result.get('run_name') != stage['local_name'] or result.get('seed') != job['seed']
            or result.get('env_name') != f'ALE/{job["game"]}-v5'
            or result.get('sample_max_steps') != manifest['evaluation']['sample_max_steps']):
        raise ValueError('Local final evaluation identity/training budget mismatch')
    proof = plan_proves_source(job, stage, live)
    shared = directory / 'shared_warmup.json'
    if shared.exists():
        check_source(json.loads(shared.read_text()).get('checkpoint_dir'), job, stage, root, proof)
    elif not proof:
        raise ValueError('Cannot verify local shared warmup provenance')
    marker = root / 'ckpt' / stage['local_name'] / 'queue_stage_complete.json'
    if not marker.exists():
        return {'State': 'evaluating', 'Notes': 'Final evaluation exists; waiting for branch completion marker'}
    if json.loads(marker.read_text()).get('stage') != f'{job["id"]}:{stage["key"]}':
        raise ValueError('Completion marker belongs to another queue branch')
    fields = evaluation_fields(result.get('episode_avg_return'), result.get('episode_returns', []),
                               result.get('precision'), result.get('checkpoint_step'),
                               len(result.get('episode_returns', [])), get(conf, 'JointTrainAgent.EvalSeed'), manifest)
    return dict(fields, Source='local', **{'Updated At': datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc).isoformat()})


def resolved_warmup(job, root):
    """Use the queue's own compatibility rules without discovery, writes, or GPU work."""
    source, resume = job.get('warmup_source'), job.get('resume_warmup')
    if not source or not resume:
        import sys
        sys.path.insert(0, str(ROOT))
        from warmup_catalog import choose_warmup
        inventory = json.loads((root / 'training_queue/warmup_inventory.json').read_text())
        record, reasons = choose_warmup(job, inventory)
        if record is None:
            raise ValueError(f'{job["game"]}/{job["seed"]}: no reusable warmup; {reasons}')
        source, resume = record['source'], record['resume_warmup']
    path = Path(resume)
    if not path.is_absolute():
        path = root / path
    metadata = json.loads((path / 'warmup_metadata.json').read_text())
    if metadata['seed'] != job['seed'] or metadata['env_name'] != f'ALE/{job["game"]}-v5':
        raise ValueError(f'Warmup identity mismatch: {path}')
    for name in ('training_state.pt', 'replay_buffer.npz'):
        if not (path / name).is_file():
            raise ValueError(f'Missing warmup payload: {path / name}')
    return str((root / source).resolve()), str(path.resolve()), metadata['step']


def build_manifest(jobs, root, gpu_type='3090', job_ids=None, resolver=resolved_warmup,experiment=None):
    root = Path(root).resolve()
    selected = [job for identity, job in jobs.items()
                if (not gpu_type or job['gpu_type'] == gpu_type) and (not job_ids or identity in job_ids)
                and (experiment is None or job.get('experiment_id')==experiment_name(experiment))
                and get(effective_config(job), PREFIX + 'hash_target_pairs')]
    if job_ids and set(job_ids) - {job['id'] for job in selected}:
        raise ValueError('Some requested Job IDs are missing, in another GPU pool, or not hash/target sweeps')
    if not selected:
        raise ValueError('No hash/target jobs in the selected queue. Enqueue first, or use an existing --manifest.')
    frozen, pairs, evaluation, identities = [], None, None, set()
    for live in selected:
        conf = effective_config(live)
        requested = get(conf, PREFIX + 'hash_target_pairs')
        if any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or v < 1 for v in p)
               or p[0] > 62 for p in requested):
            raise ValueError('Invalid hash/target pairs in queue')
        grid = sorted(tuple(p) for p in requested)
        if len(set(grid)) != len(grid):
            raise ValueError('Duplicate hash/target pairs')
        if pairs is None:
            pairs = grid
        if pairs != grid:
            raise ValueError('Queue contains different grids; select one cohort with repeated --job-id')
        if get(conf, PREFIX + 'enable') not in ('Both', ['retrieval', 'baseline'], ['baseline', 'retrieval']):
            raise ValueError('This report requires retrieval pairs and one shared baseline (enable Both)')
        mode = get(conf, 'JointTrainAgent.EvalPrecision')
        # Never infer auto precision from the collector's hardware.
        primary = 'bf16' if mode in ('bf16', 'both') else 'fp32' if mode == 'fp32' else None
        settings = dict(checkpoint_step=get(conf, 'JointTrainAgent.FinalEvalStep'),
                        episodes=get(conf, 'JointTrainAgent.EvalEpisodes'),
                        seed=get(conf, 'JointTrainAgent.EvalSeed'), precision=primary,
                        sample_max_steps=get(conf, 'JointTrainAgent.SampleMaxSteps'))
        if primary is None or not settings['checkpoint_step'] or not settings['episodes']:
            raise ValueError('Set an explicit EvalPrecision, FinalEvalStep, and positive EvalEpisodes')
        if get(conf, 'JointTrainAgent.EvalMode') not in ('active', 'final_only'):
            raise ValueError('Final evaluation is disabled')
        if evaluation is None:
            evaluation = settings
        if evaluation != settings:
            raise ValueError('Jobs use different evaluation conditions; choose one cohort')
        identity = (live['game'], live['gpu_type'], live['seed'])
        if identity in identities:
            raise ValueError(f'Multiple jobs for {identity}; select exact --job-id values instead of mixing repeats')
        identities.add(identity)
        source, resume, step = resolver(live, root)
        job = {key: live[key] for key in ('id', 'name', 'game', 'gpu_type', 'seed', 'created')}
        job.update(config=conf, warmup_source=source, resume_warmup=resume, warmup_step=step,
                   warmup_id=digest([*identity, source])[:20])
        job['stages'] = list(stage_specs(job, pairs))
        job['plan_sources'] = {}
        for stage, command in live.get('plan', {}).items():
            saved = command_arg(command, '--resume_from')
            if not (same_path(saved, resume) or same_path(saved, source)):
                raise ValueError(f'Queue plan uses a different warmup: {live["id"]}/{stage}')
            job['plan_sources'][stage] = saved
        frozen.append(job)
    body = dict(schema_version=SCHEMA_VERSION, root=str(root), pairs=[list(p) for p in pairs],
                evaluation=evaluation, jobs=frozen)
    body['campaign_id'] = 'hash-grid-' + digest(body)[:16]
    body['created_at'] = now()
    if experiment is not None:
        body['experiment_name']=experiment
    return body


@contextmanager
def report_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION or not manifest.get('jobs'):
        raise ValueError('Unsupported/empty hash-target campaign manifest')
    body = {key: manifest[key] for key in ('schema_version', 'root', 'pairs', 'evaluation', 'jobs')}
    if manifest.get('campaign_id') != 'hash-grid-' + digest(body)[:16]:
        raise ValueError('Campaign manifest was modified; create a new manifest for a new experiment')
    return manifest


def expected_rows(manifest):
    for job in manifest['jobs']:
        for stage in job['stages']:
            row = dict(zip(STATIC_FIELDS, [SCHEMA_VERSION, manifest['campaign_id'], job['id'], stage['key'],
                       job['game'], job['gpu_type'], job['seed'], stage['setting'], stage['hash_bits'],
                       stage['target'], job['warmup_id'], job['warmup_source'], job['resume_warmup'],
                       job['warmup_step'], stage['local_name'], digest(stage_config(job, stage))]))
            yield job, stage, row


def validate_row(row, expected, manifest):
    for key in STATIC_FIELDS:
        if str(row.get(key) if row.get(key) is not None else '') != str(expected.get(key) if expected.get(key) is not None else ''):
            raise ValueError(f'CSV row outside the frozen experiment: {key}')
    if row.get('State') == 'completed':
        evaluation = manifest['evaluation']
        if number(row.get('Eval Return')) is None:
            raise ValueError('Completed row has no finite score')
        for column, key in [('Eval Checkpoint Step', 'checkpoint_step'), ('Eval Episodes', 'episodes'), ('Eval Seed', 'seed')]:
            if number(row.get(column)) != evaluation[key]:
                raise ValueError(f'Completed row has incompatible {column}')
        if row.get('Eval Precision') != evaluation['precision']:
            raise ValueError('Completed row has incompatible precision')


def read_csv_rows(path, manifest, *, allow_missing=False):
    import csv
    path = Path(path)
    if not path.exists() and allow_missing:
        return {}
    expected = {(row['Job ID'], row['Stage']): row for _, _, row in expected_rows(manifest)}
    result = {}
    with path.open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if not set(STATIC_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError('This is a historical/unsupported CSV. Run classify_wandb_runs.py to collect the new campaign.')
        for row in reader:
            key = (row.get('Job ID'), row.get('Stage'))
            if key not in expected or row.get('Campaign ID') != manifest['campaign_id']:
                raise ValueError('CSV contains results from another campaign')
            if key in result:
                raise ValueError(f'Duplicate CSV observation: {key}')
            validate_row(row, expected[key], manifest)
            result[key] = row
    return result
