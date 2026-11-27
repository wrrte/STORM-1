"""Preview or delete local shared warmups with missing required statistics or broken payloads."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import socket
import sqlite3
import zipfile

from queue_train import ROOT, gpu_type, locked
from warmup_catalog import (warmup_sources, inspect_warmup, source_stamp, compatibility,
                            imported_path)
from warmup_registry import load_manifest, scan, identifier, save, timestamp


# These are payload-validation failures, not configuration/profile differences.
BROKEN_PAYLOAD = (
    'Missing/empty training_state.pt', 'Missing/empty replay_buffer.npz',
    'Incomplete training state:', 'Incomplete RNG state', 'Incomplete replay buffer',
    'Checkpoint is not a finished shared warmup', 'Invalid/inconsistent saved warmup step',
    'Invalid replay buffer', 'Invalid replay obs header', 'Invalid replay action header',
    'Invalid replay reward header', 'Invalid replay termination header', 'Truncated replay ',
    'Metadata claims statistics missing from the checkpoint',
    'Invalid td_error statistics', 'Invalid value_diff statistics', 'Invalid value statistics',
    'Cannot load file containing pickled data when allow_pickle=False',
)


def safe_source(root, source):
    """Delete only real snapshot directories immediately below ckpt/RUN/."""
    root = Path(root).resolve()
    source = Path(source)
    path = root / source
    if (source.is_absolute() or len(source.parts) != 3 or source.parts[0] != 'ckpt'
            or '..' in source.parts or not re.fullmatch(r'shared_warmup_\d+', source.name)):
        raise ValueError(f'Not a local shared warmup directory: {source}')
    if path.resolve() != path or not path.is_dir():
        raise ValueError(f'Missing or symlinked shared warmup directory: {source}')
    return path


def inspect_candidate(root, source, model, profile):
    path = safe_source(root, source)
    before = source_stamp(path)
    record = None
    try:
        record = inspect_warmup(path, root=root, queue_dir=root / 'training_queue')
        record['gpu_type'] = model
        reasons = [compatibility(record, dict(game=record['game'], seed=record['seed'],
                   gpu_type=model, config_text=profile['config_text'], overrides=overrides))
                   for overrides in profile['requests']]
        # Other warmup targets remain useful for their own sweeps.
        missing = any(reason and reason.startswith('missing statistics') for reason in reasons)
        only_statistics_or_target = all(reason and (reason.startswith('missing statistics') or
                                                    reason.startswith('warmup target ')) for reason in reasons)
        category = 'missing_statistics' if missing and only_statistics_or_target else None
    except (ValueError, KeyError, TypeError, RuntimeError, OSError, EOFError,
            AssertionError, zipfile.BadZipFile, pickle.UnpicklingError) as error:
        reasons = [f'{type(error).__name__}: {error}']
        published = (path / 'warmup_metadata.json').is_file() or (path / 'branches.json').is_file()
        broken = isinstance(error, ValueError) and str(error).startswith(BROKEN_PAYLOAD)
        broken = broken or isinstance(error, (EOFError, zipfile.BadZipFile, pickle.UnpicklingError))
        broken = broken or (isinstance(error, RuntimeError) and str(error).startswith((
            'PytorchStreamReader failed', 'mmap can only be used with files saved with',
            'Invalid magic number', 'Invalid protocol version', 'unexpected EOF')))
        category = 'invalid_checkpoint' if published and broken else None
    if before != source_stamp(path):
        return None, None, 'changed during inspection'
    if not category:
        return None, record, '; '.join(reason for reason in reasons if reason) or 'compatible'
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
    return dict(source=str(source), category=category, reasons=reasons, bytes=size,
                stamp=before, game=record['game'] if record else None,
                seed=record['seed'] if record else None), record, None


@contextmanager
def queue_guard(root):
    """Prevent a queue claim/enqueue from changing protection while deleting."""
    path = root / 'training_queue/queue.sqlite3'
    if not path.exists():
        yield []
        return
    db = sqlite3.connect(path, timeout=60)
    try:
        db.execute('BEGIN IMMEDIATE')
        jobs = [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM jobs WHERE status IN ('pending','running','failed')")]
        yield jobs
        db.commit()
    finally:
        db.close()


def protected(root, source, record, jobs):
    path = root / source
    references = {path.resolve(), imported_path(path, root / 'training_queue').resolve()}
    for job in jobs:
        paths = [job.get('warmup_source'), job.get('resume_warmup')]
        for command in job.get('plan', {}).values():
            paths.extend(command[i + 1] for i, flag in enumerate(command[:-1])
                         if flag in ('--resume_from', '--resume_warmup'))
        if any((root / value).resolve() in references for value in paths if value):
            return f"pinned by queued/retryable job {job['id']}"
        if record and (job.get('game'), job.get('seed')) == (record['game'], record['seed']):
            if compatibility(record, job) is None:
                return f"compatible with queued/retryable job {job['id']}"
    return None


def remove_references(root, deleted, *, seed_file=None):
    """Keep seed selections and local discovery cache consistent with disk."""
    deleted = set(deleted)
    absolute = {str(root / value) for value in deleted}
    seed_files = (root / seed_file,) if seed_file is not None else (root / 'config_files/game_seeds').glob('type_*.json')
    for path in seed_files:
        data = json.loads(path.read_text())
        changed = False
        for entry in data['game_seeds']:
            old = entry.get('warmups', [])
            entry['warmups'] = [name for name in old if name not in deleted and
                               str((root / name).resolve()) not in absolute]
            changed |= old != entry['warmups']
        if changed:
            save(path, data)
    path = root / 'training_queue/warmup_inventory.json'
    if path.exists():
        data = json.loads(path.read_text())
        for key in ('available', 'excluded'):
            data[key] = [item for item in data.get(key, []) if item['source'] not in deleted]
        save(path, data)


def cleanup(round_id, model, machine=None, *, apply=False, root=ROOT, cache=None):
    root, model = Path(root).resolve(), gpu_type(model)
    cache = Path(cache) if cache else root / 'training_queue/shared_warmups'
    machine = identifier(machine or socket.gethostname())
    manifest = load_manifest(cache, round_id)
    if model not in manifest['gpu_types']:
        raise ValueError(f'{model} does not participate in round {round_id}')
    report = dict(round=round_id, gpu_type=model, machine=machine, apply=apply,
                  profile_sha256=manifest['profile_sha256'], created_at=timestamp(),
                  candidates=[], protected=[], kept=[], deleted=[], failed=[])
    audit = root / 'training_queue/cleanup' / (report['created_at'].replace(':', '-') + '.json')
    with locked(cache / 'sync.lock'), locked(root / 'training_queue/catalog.lock'), queue_guard(root) as jobs:
        for source in warmup_sources(root, include_unpublished=True):
            name = str(source.relative_to(root))
            try:
                item, record, reason = inspect_candidate(root, name, model, manifest['profile'])
            except (ValueError, OSError) as error:
                report['kept'].append(dict(source=name, reason=str(error)))
                continue
            if not item:
                report['kept'].append(dict(source=name, reason=reason))
                continue
            guard = protected(root, name, record, jobs)
            if guard:
                report['protected'].append(dict(source=name, reason=guard))
            else:
                report['candidates'].append(item)
        save(audit, report)  # Persist exact targets/reasons before any deletion.
        if apply:
            for item in report['candidates']:
                name = item['source']
                try:
                    path = safe_source(root, name)
                    if source_stamp(path) != item['stamp']:
                        raise ValueError('Checkpoint changed since inspection')
                    retired = path.with_name(path.name + '.retired')
                    if retired.exists() or retired.is_symlink():
                        raise ValueError('A retired checkpoint already exists')
                    os.replace(path, retired)
                    shutil.rmtree(retired)
                    report['deleted'].append(name)
                except (ValueError, OSError) as error:
                    report['failed'].append(dict(source=name, reason=str(error)))
                save(audit, report)
            remove_references(root, report['deleted'])
            # A scan replaces this machine's report; Git publication is a separate sync.
            scan(cache, round_id, model, machine, root=root)
    report['audit'] = str(audit)
    save(audit, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--gpu-type', required=True, help='GPU model of this machine')
    parser.add_argument('--machine', help='Same machine ID used by warmup_registry.py')
    parser.add_argument('--apply', action='store_true', help='Delete candidates; default only previews')
    args = parser.parse_args(argv)
    try:
        report = cleanup(args.round, args.gpu_type, args.machine, apply=args.apply)
        for item in report['candidates']:
            state = 'DELETED' if item['source'] in report['deleted'] else 'CANDIDATE'
            print(f"{state} {item['source']} ({item['bytes'] / 2**20:.1f} MiB): {'; '.join(item['reasons'])}")
        for key in ('protected', 'failed'):
            for item in report[key]:
                print(f"{key.upper()} {item['source']}: {item['reason']}")
        size = sum(item['bytes'] for item in report['candidates'])
        print(f"{len(report['candidates'])} candidate(s), {size / 2**30:.2f} GiB; "
              f"{len(report['deleted'])} deleted, {len(report['protected'])} protected.")
        print('Audit:', report['audit'])
        if not args.apply:
            print('Preview only. Add --apply to delete after a fresh inspection.')
        else:
            print(f"Publish updated inventory: python warmup_registry.py sync --round {args.round} "
                  f"--gpu-type {gpu_type(args.gpu_type)}" + (f" --machine {args.machine}" if args.machine else ''))
        return int(bool(report['failed']))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
