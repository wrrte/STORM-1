"""Collect only a frozen hash/target queue cohort, including unfinished branches.

    python results/classify_wandb_runs.py --excel
    python results/classify_wandb_runs.py --offline --excel
    python results/classify_wandb_runs.py --experiment grid-v2 --excel
    python results/classify_wandb_runs.py --list-experiments

Named experiments use results/experiments/NAME/{manifest.json,runs.csv,results.xlsx}.
With --excel, both matched comparisons use equal total weight per game for IQM.
The default experiment is configured by DEFAULT_EXPERIMENT in this file.
No queue is mutated by reporting.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import io
import json
import math
from pathlib import Path
import sqlite3
from statistics import fmean, stdev
import sys
import tempfile

import yaml

from hash_target_common import (
    ROOT, FIELDS, atomic_text,
    build_manifest, command_arg, config_mismatch, expected_rows, get, load_manifest,
    now, number, read_csv_rows, read_jobs, report_lock, same_path, stage_config,
    report_paths, list_experiments,
)


# Update this when switching the default report to a different experiment.
DEFAULT_EXPERIMENT = 'hash-grid-anchor-rehash-20261125'


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


def local_result(root, job, stage, live, manifest):
    directory = root / 'runs' / stage['local_name']
    result_path = directory / 'final_evaluation.json'
    if not result_path.exists():
        return None
    conf = yaml.safe_load((directory / 'config.yaml').read_text())
    reason = config_mismatch(conf, stage_config(job, stage))
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


def get_run_args(run):
    with tempfile.TemporaryDirectory(prefix='storm-grid-metadata-') as directory:
        downloaded = run.file('wandb-metadata.json').download(root=directory, replace=True)
        path = Path(downloaded.name)
        downloaded.close()
        args = json.loads(path.read_text()).get('args')
    if not isinstance(args, list):
        raise ValueError('W&B run has no usable training arguments')
    return args


def classify_run(run, job, stage, manifest, root, live=None):
    """An exact UUID-containing local run name is required; names/seeds alone never match."""
    if get(run.config, 'RunMetadata.local_run_name') != stage['local_name']:
        return None
    if (get(run.config, 'RunMetadata.seed') != job['seed']
            or get(run.config, 'RunMetadata.game') != job['game']):
        raise ValueError('W&B game/seed mismatch')
    created = datetime.fromisoformat(str(run.created_at).replace('Z', '+00:00'))
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if created.timestamp() + 1 < job['created']:
        raise ValueError('W&B run predates its frozen queue job')
    reason = config_mismatch(run.config, stage_config(job, stage))
    if reason:
        raise ValueError(reason)
    common = {'Run ID': run.id, 'Created At': created.isoformat(), 'Source': 'wandb',
              'Updated At': getattr(run, 'updated_at', None) or now()}
    if run.state != 'finished':
        return dict(common, State={'running': 'running', 'crashed': 'failed', 'failed': 'failed',
                                   'killed': 'interrupted'}.get(run.state, 'pending'))
    precision = manifest['evaluation']['precision']
    summary = run.summary
    score = summary.get(f'eval/{precision}/episode_avg_return', summary.get('eval/episode_avg_return'))
    if number(score) is None or summary.get('eval/checkpoint_step') is None:
        # The queue's SIGTERM handler calls wandb.finish() before exiting 143.
        # W&B may therefore say finished even though no final evaluation occurred.
        return dict(common, State='interrupted', Notes='Run ended before final evaluation; awaiting queue continuation')
    args = get_run_args(run)
    proof = plan_proves_source(job, stage, live)
    check_source(command_arg(args, '--resume_from'), job, stage, root, proof)
    returns = [summary.get(f'eval/{precision}/episode_return_{i}', summary.get(f'eval/episode_return_{i}'))
               for i in range(manifest['evaluation']['episodes'])]
    fields = evaluation_fields(score, returns, precision, summary.get('eval/checkpoint_step'),
                               get(run.config, 'JointTrainAgent.EvalEpisodes'), summary.get('eval/seed'), manifest)
    return dict(common, **fields)


def fetch_results(manifest, live_jobs, root, workers=8, project='choemj-kaist/STORM', api=None):
    if api is None:
        import wandb
        api = wandb.Api(timeout=20)
    lookup = {stage['local_name']: (job, stage) for job, stage, _ in expected_rows(manifest)}
    names = sorted(lookup)
    runs, seen = [], set()
    # Server-side scope filter avoids downloading the historical project.
    for start in range(0, len(names), 80):
        for run in api.runs(project, filters={'config.RunMetadata.local_run_name': {'$in': names[start:start+80]}},
                            per_page=100, lazy=False):
            local_name = get(run.config, 'RunMetadata.local_run_name')
            if local_name in lookup and run.id not in seen:
                runs.append(run)
                seen.add(run.id)
    def inspect(run):
        name = get(run.config, 'RunMetadata.local_run_name')
        job, stage = lookup[name]
        try:
            row = classify_run(run, job, stage, manifest, root, live_jobs.get(job['id']))
            return (job['id'], stage['key']), row, None
        except Exception as error:
            return (job['id'], stage['key']), None, f'{run.id}: {error}'
    grouped, issues = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for key, row, error in executor.map(inspect, runs):
            if row:
                grouped.setdefault(key, []).append(row)
            if error:
                issues.setdefault(key, []).append(error)
    print(f'W&B: {len(runs)} run(s) matching this campaign only.')
    return grouped, issues


def base_state(job, stage, live):
    if not live:
        return 'pending'
    remaining = [key for key in (live.get('plan') or {s['key']: None for s in job['stages']})
                 if key not in live.get('completed_branches', [])]
    if remaining and stage['key'] == remaining[0]:
        return {'running': 'running', 'failed': 'failed'}.get(live['status'], 'pending')
    return 'pending'


def collect_rows(manifest, live_jobs, root, cached=None, remote=None, remote_issues=None):
    output = []
    for job, stage, expected in expected_rows(manifest):
        key = job['id'], stage['key']
        live = live_jobs.get(job['id'])
        row = dict(expected, State=base_state(job, stage, live), **{'Attempt Count': 0, 'Notes': ''})
        notes = list((remote_issues or {}).get(key, []))
        candidates = list((remote or {}).get(key, []))
        previous = (cached or {}).get(key)
        if previous and previous.get('State') == 'completed':
            candidates.append({**previous, 'Source': 'cached:' + previous.get('Source', '').removeprefix('cached:')})
        try:
            plan_proves_source(job, stage, live)
        except ValueError as error:
            row.update(State='invalid', Notes=str(error))
            output.append(row)
            continue
        try:
            local = local_result(root, job, stage, live, manifest)
            if local:
                candidates.append(local)
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as error:
            notes.append(f'local: {error}')
        complete = [candidate for candidate in candidates if candidate.get('State') == 'completed']
        if complete:
            # One observation per branch; never average retry attempts.
            complete.sort(key=lambda c: (c.get('Source') == 'local', c.get('Source') == 'wandb',
                                         str(c.get('Created At', '')), str(c.get('Run ID', ''))), reverse=True)
            chosen = complete[0]
            row.update({k: v for k, v in chosen.items() if k not in expected})
            if len(complete) > 1:
                notes.append(f'{len(complete)} completed records; selected {chosen.get("Source")}, no retry averaging')
        elif candidates:
            chosen = max(candidates, key=lambda c: (c.get('State') == 'evaluating', str(c.get('Created At', ''))))
            # The queue owns scheduling state. A stale running/finished W&B
            # attempt must not turn a pending retry into an active/completed run.
            row.update({k: v for k, v in chosen.items() if k not in expected and k != 'State'})
            if live is None or (row['State'] == 'running' and chosen.get('State') == 'evaluating'):
                row['State'] = chosen['State']
        elif notes:
            row['State'] = 'invalid'
        row['Attempt Count'] = len([c for c in candidates if c.get('Run ID')])
        row['Notes'] = '; '.join(filter(None, [row.get('Notes'), *notes]))
        # Incomplete results never carry a numeric final score into Excel.
        if row['State'] != 'completed':
            for column in ('Eval Return', 'Eval Checkpoint Step', 'Eval Precision', 'Eval Episodes', 'Eval Seed', 'Episode SD'):
                row[column] = None
        output.append(row)
    return output


def main(workers=8, output_path=None, offline=False, *, manifest_path=None,
         root=ROOT, queue_db=None, gpu_type=None, job_ids=None, project='choemj-kaist/STORM',
         experiment=None, excel_path=None):
    root=Path(root).resolve()
    manifest_path,output,_=report_paths(experiment,manifest=manifest_path,csv=output_path,results_dir=root/'results')
    database = Path(queue_db) if queue_db else root / 'training_queue/queue.sqlite3'
    manifest_path = Path(manifest_path)
    with report_lock(manifest_path):
        live = read_jobs(database) if database.exists() else {}
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
            if experiment and manifest.get('experiment_name') not in (None,experiment):
                raise ValueError('The selected manifest belongs to another --experiment')
            if experiment:
                extra={identity for identity,job in live.items() if job.get('experiment_id')==experiment} - {job['id'] for job in manifest['jobs']}
                if extra:
                    raise ValueError('New jobs reused a frozen experiment name. Use a different --experiment name.')
        else:
            pool=gpu_type if gpu_type is not None else None if experiment else '3090'
            manifest = build_manifest(live, root, gpu_type=pool, job_ids=job_ids,experiment=experiment)
            atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            print(f'Frozen campaign: {len(manifest["jobs"])} jobs → {manifest_path}')
        try:
            cached = read_csv_rows(output, manifest, allow_missing=True)
        except ValueError as error:
            print(f'Ignoring previous CSV: {error}', file=sys.stderr)
            cached = {}
        if offline:
            remote, issues = {}, {}
        else:
            try:
                remote, issues = fetch_results(manifest, live, root, workers, project)
            except Exception as error:
                raise RuntimeError('W&B query failed. Use --offline --excel to refresh local results.') from error
        rows = collect_rows(manifest, live, root, cached, remote, issues)
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(output, buffer.getvalue())
        completed = sum(row['State'] == 'completed' for row in rows)
        print(f'{completed}/{len(rows)} completed branches → {output}')
        if excel_path is not None:
            from convert_csv_to_excel import main as convert_excel
            convert_excel(csv_path=output, output_path=excel_path, manifest_path=manifest_path, experiment=experiment)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--experiment', help=f'Experiment name used by enqueue_train.py (default: {DEFAULT_EXPERIMENT!r})')
    parser.add_argument('--list-experiments', action='store_true', help='List named queues and saved experiment reports without contacting W&B')
    parser.add_argument('--output', type=Path, help='Optional override of the experiment CSV path')
    parser.add_argument('--manifest', type=Path, help='Optional override of the experiment manifest path')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--queue-db', type=Path)
    parser.add_argument('--gpu-type', help='Optional GPU pool filter for first capture (legacy default: 3090)')
    parser.add_argument('--job-id', action='append', help='Optional exact Job IDs for first capture')
    parser.add_argument('--project', default='choemj-kaist/STORM')
    parser.add_argument('--offline', action='store_true', help='Read local final evaluations and this campaign CSV only')
    parser.add_argument('--excel', action='store_true', help='Refresh the settings-first and samples-first Excel comparisons')
    parser.add_argument('--excel-output', type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    try:
        if args.list_experiments:
            database=args.queue_db or args.root/'training_queue/queue.sqlite3'
            live=read_jobs(database) if database.exists() else {}
            for name,count in list_experiments(live,args.root/'results'):
                print(f'{name}: {count} job(s)')
            raise SystemExit(0)
        experiment = args.experiment
        if experiment is None and not (args.manifest or args.job_id or args.gpu_type):
            experiment = DEFAULT_EXPERIMENT
        manifest,csv_path,excel_path=report_paths(experiment,manifest=args.manifest,csv=args.output,
                                                excel=args.excel_output,results_dir=args.root/'results')
        main(args.workers, csv_path, args.offline, manifest_path=manifest, root=args.root,
             queue_db=args.queue_db, gpu_type=args.gpu_type, job_ids=args.job_id, project=args.project,
             experiment=experiment, excel_path=excel_path if args.excel else None)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
        raise SystemExit(str(error))
