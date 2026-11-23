"""Execute one claimed experiment, retaining branch progress and warmup provenance."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from queue_train import ROOT, TrainingQueue, json_line, write_atomic


def stage_run_name(command):
    name = command[command.index('-n') + 1]
    if '--branch_experiment' in command:
        return name + '_' + command[command.index('--branch_experiment') + 1]
    return name + '_Shared'


def resume_command(command, *, root=ROOT):
    command = list(command)
    locator = root / 'ckpt' / stage_run_name(command) / 'queue_resume.json'
    if locator.exists():
        record = json.loads(locator.read_text())
        checkpoint = Path(record['checkpoint'])
        if not checkpoint.is_relative_to(locator.parent.resolve()):
            raise ValueError('Invalid queue resume locator')
        if all((checkpoint / name).is_file() for name in ('training_state.pt', 'replay_buffer.npz')):
            if '--resume_from' in command:
                command[command.index('--resume_from') + 1] = str(checkpoint)
            else:
                command += ['--resume_from', str(checkpoint)]
            if '--rebuild_retrieval_on_resume' in command:
                command.remove('--rebuild_retrieval_on_resume')
    return command


class JobRunner:
    def __init__(self, queue, job, root=ROOT):
        self.queue, self.job, self.root = queue, job, Path(root)
        self.stopping = False
        self.child = None
        self.fds = tuple(int(fd) for fd in os.environ.get('STORM_QUEUE_FDS', '').split(',') if fd)

    def stop(self, *_):
        self.stopping = True

    def execute(self, command, stage, *, collect=False):
        if self.stopping:
            return 143
        name = stage_run_name(command)
        marker = self.root / 'ckpt' / name / 'queue_stage_complete.json'
        if marker.exists() and json.loads(marker.read_text()).get('stage') == f"{self.job['id']}:{stage}":
            return 0
        env = {**os.environ, 'STORM_QUEUE_JOB': self.job['id'],
               'STORM_QUEUE_STAGE': f"{self.job['id']}:{stage}",
               'STORM_GPU_TYPE': self.job['gpu_type'],
               'STORM_QUEUE_COLLECT_ONLY': '1' if collect else '0'}
        command = resume_command(command, root=self.root)
        print(f"[{self.job['gpu_type']}] {stage}: {' '.join(command[:5])}", flush=True)
        self.child = subprocess.Popen(command, cwd=self.root, env=env, start_new_session=True, pass_fds=self.fds)
        deadline = None
        try:
            while self.child.poll() is None:
                if self.stopping and deadline is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    deadline = time.monotonic() + max(0, float(os.environ.get("STORM_QUEUE_STOP_GRACE", "120")))
                if deadline is not None and time.monotonic() >= deadline:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    break
                time.sleep(0.2)
            return self.child.wait()
        finally:
            # Reap environment subprocesses even when the trainer failed.
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.child = None

    def run(self):
        from warmup_catalog import discover, choose_warmup, request_config, branch_commands, prune_replaced
        job, queue = self.job, self.queue
        if not job.get('plan'):
            report = discover(job['gpu_type'], root=self.root, queue_dir=queue.directory, seed_dir=queue.seed_directory)
            record, incompatible = choose_warmup(job, report)
            if record is None:
                conf = request_config(job)
                conf.JointTrainAgent.Retrieval.enable = ['baseline']
                conf.JointTrainAgent.Retrieval.save_warmup = True
                conf.JointTrainAgent.Retrieval.warmup_offset_pairs = []
                conf.JointTrainAgent.Retrieval.hash_target_pairs = []
                directory = queue.directory / 'requests' / job['id']
                directory.mkdir(parents=True, exist_ok=True)
                config = directory / 'collector.yaml'
                write_atomic(config, conf.dump())
                name = job['name'] + '_collector'
                command = [job['python'], '-u', str(self.root / 'train.py'), '-n', name,
                           '-seed', str(job['seed']), '-config_path', str(config),
                           '-env_name', f"ALE/{job['game']}-v5",
                           '-trajectory_path', str(self.root / 'D_TRAJ' / f"{job['game']}.pkl")]
                code = self.execute(command, 'collect', collect=True)
                if code or self.stopping:
                    return code or 143
                report = discover(job['gpu_type'], root=self.root, queue_dir=queue.directory, seed_dir=queue.seed_directory)
                record, _ = choose_warmup(job, report)
                if record is None:
                    raise ValueError('Collector completed without publishing a compatible shared warmup')
            if self.stopping:
                return 143
            job['warmup_source'] = record['source']
            job['resume_warmup'] = record['resume_warmup']
            job['plan'] = branch_commands(job, record, root=self.root)
            queue.update_job(job)  # Pin the source and exact plan before training starts.
            removed = prune_replaced(job, record, incompatible, queue, root=self.root)
            for source in removed:
                print(f"Removed replaced incompatible warmup: {source}", flush=True)
        for stage, command in job['plan'].items():
            if stage in job.get('completed_branches', []):
                continue
            code = self.execute(command, stage)
            if code or self.stopping:
                return code or 143
            job.setdefault('completed_branches', []).append(stage)
            queue.update_job(job)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args(argv)
    queue = TrainingQueue(args.queue_dir)
    with queue.connect() as db:
        row = db.execute("SELECT payload,status FROM jobs WHERE id=?", (args.job_id,)).fetchone()
    if row is None or row['status'] != 'running':
        parser.error('Job must already be claimed by a worker')
    runner = JobRunner(queue, json.loads(row['payload']))
    handlers = {sig: signal.signal(sig, runner.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    code = 1
    try:
        code = runner.run()
    except Exception as error:
        print(f"Job failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        with queue.connect() as db:
            queue.event(db, args.job_id, 'error', message=str(error))
    finally:
        status = 'pending' if runner.stopping or code < 0 or code in (130, 143) else ('completed' if code == 0 else 'failed')
        queue.finish(args.job_id, status, code)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
