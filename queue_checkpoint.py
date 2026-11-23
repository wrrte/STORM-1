"""Small queue-only hooks; normal interactive training keeps its existing behavior."""
import json
import os
from pathlib import Path
import signal
import uuid
import shutil

STOP_REQUESTED = False


def install_handlers():
    if os.environ.get('STORM_QUEUE_JOB'):
        def stop(*_):
            global STOP_REQUESTED
            STOP_REQUESTED = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)


def publish(path, value):
    from queue_train import write_atomic
    write_atomic(Path(path), json.dumps(value, indent=2) + '\n')


def warmup_saved(checkpoint):
    # Record model provenance for manually launched warmups as well as queue jobs.
    from queue_train import gpu_type
    import socket
    import torch
    name = torch.cuda.get_device_name()
    kind = gpu_type(name)
    expected = os.environ.get('STORM_GPU_TYPE')
    if expected and kind != gpu_type(expected):
        raise ValueError(f'Expected {expected}, but warmup was created on {name}')
    publish(Path(checkpoint) / 'queue_origin.json', dict(
        gpu_type=kind, gpu_name=name, host=socket.gethostname(), job_id=os.environ.get('STORM_QUEUE_JOB')))


def save_resume(run_name, save):
    root = Path('ckpt') / run_name
    root.mkdir(parents=True, exist_ok=True)
    locator = root / 'queue_resume.json'
    previous = json.loads(locator.read_text())['checkpoint'] if locator.exists() else None
    checkpoint = root / ('queue_resume_' + uuid.uuid4().hex)
    save(str(checkpoint))
    # Only a fully written model + replay checkpoint becomes eligible for resume.
    publish(locator, dict(checkpoint=str(checkpoint.resolve())))
    if previous:
        previous = Path(previous)
        if previous.parent == root.resolve() and previous.name.startswith('queue_resume_'):
            shutil.rmtree(previous, ignore_errors=True)


def stage_complete(run_name):
    if os.environ.get('STORM_QUEUE_JOB'):
        root = Path('ckpt') / run_name
        root.mkdir(parents=True, exist_ok=True)
        publish(root / 'queue_stage_complete.json', dict(stage=os.environ['STORM_QUEUE_STAGE']))
        # Completed stages no longer need their large temporary resume checkpoints.
        locator = root / 'queue_resume.json'
        if locator.exists():
            locator.unlink()
        for checkpoint in root.glob('queue_resume_*'):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)
