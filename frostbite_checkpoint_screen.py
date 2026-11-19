"""Screen saved Frostbite policies, then evaluate critic alignment on each GPU.

The coordinator uses only the standard library. Each subprocess sees one GPU
and loads PyTorch only after CUDA_VISIBLE_DEVICES has been set.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import traceback


ROOT = Path(__file__).resolve().parent
EVENT_PREFIX = 'SCREEN_EVENT '


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def discover_checkpoints(root=ROOT, runs=None, exclude_runs=()):
    """Discover nonzero paired weights, final weights, and full shared states."""
    root = Path(root).resolve()
    candidates = []
    for directory in sorted((root / 'ckpt').glob('Frostbite*')):
        if not directory.is_dir() or directory.name in exclude_runs:
            continue
        if runs and directory.name not in runs:
            continue
        config = root / 'runs' / directory.name / 'config.yaml'
        if not config.is_file():
            raise ValueError(f'Missing saved config for {directory.name}: {config}')
        sources = []
        for wm in sorted(directory.glob('world_model_*.pth')):
            token = wm.stem.removeprefix('world_model_')
            if token != 'final' and not (token.isdigit() and int(token) > 0):
                continue
            agent = directory / f'agent_{token}.pth'
            if agent.is_file():
                sources.append((token, dict(kind='weights', world_model=file_stamp(wm),
                                           agent=file_stamp(agent))))
        for state in sorted(directory.glob('shared_warmup_*/training_state.pt')):
            sources.append((state.parent.name, dict(kind='shared', training_state=file_stamp(state))))
        for token, source in sources:
            spec = dict(run_name=directory.name, checkpoint=token, source=source,
                        config=str(config.resolve()),
                        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            spec['fingerprint'] = digest(spec)
            spec['id'] = f'{directory.name}__{token}__{spec["fingerprint"][:10]}'
            candidates.append(spec)
    return candidates


def return_parameters(options):
    return dict(episodes=options['episodes'], seed=options['return_seed'],
                precision=options['precision'], environment='ALE/Frostbite-v5',
                eval_code_sha256=options['eval_code_sha256'])


def emit(phase, **fields):
    print(EVENT_PREFIX + json.dumps(dict(phase=phase, **fields)), flush=True)


class GPUBackend:
    def __init__(self, job):
        import ale_py
        import gymnasium
        import torch
        from train import build_agent, build_world_model
        from utils import configure_performance, load_config

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in the worker environment')
        self.job = job
        self.conf = load_config(job['candidate']['config'])
        configure_performance(self.conf)
        gymnasium.register_envs(ale_py)
        env = gymnasium.make('ALE/Frostbite-v5', full_action_space=False, frameskip=1)
        try:
            action_dim = env.action_space.n
        finally:
            env.close()
        self.world_model = build_world_model(self.conf, action_dim)
        self.agent = build_agent(self.conf, action_dim)
        source = job['candidate']['source']
        if source['kind'] == 'shared':
            # These are the user's local resume checkpoints, including RNG and
            # retrieval objects. Only model state dicts are restored for inference.
            saved = torch.load(source['training_state']['path'], map_location='cpu', weights_only=False)
            wm_state, agent_state = saved['world_model_state_dict'], saved['agent_state_dict']
            self.checkpoint_step = int(saved['total_steps'])
        else:
            wm_state = torch.load(source['world_model']['path'], map_location='cpu', weights_only=True)
            agent_state = torch.load(source['agent']['path'], map_location='cpu', weights_only=True)
            token = job['candidate']['checkpoint']
            self.checkpoint_step = int(token) if token.isdigit() else None
        self.world_model.load_state_dict(wm_state)
        self.agent.load_state_dict(agent_state)
        for model in (self.world_model, self.agent):
            model.eval()
            model.requires_grad_(False)

    def evaluate_return(self):
        from eval import eval_episodes
        options = self.job['options']
        return eval_episodes(num_episode=options['episodes'], num_envs=options['episodes'],
            env_name='ALE/Frostbite-v5', max_steps=self.conf.JointTrainAgent.SampleMaxSteps,
            image_size=self.conf.BasicSettings.ImageSize, world_model=self.world_model,
            agent=self.agent, precision=options['precision'], seed=options['return_seed'])

    def evaluate_alignment(self, output):
        import cv2
        import numpy as np
        import eval_frostbite_value as value_eval
        from frostbite_value_stats import analyze_values, make_blocks

        options = self.job['options']
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        raw_path = output / 'model_values.npz'
        with np.load(options['data'], allow_pickle=False) as stored:
            data = {key: stored[key] for key in stored.files}
        required = ('obs', 'action', 'ram', 'episode_id', 'episode_step', 'episode_end')
        if any(key not in data for key in required):
            raise ValueError('Shared data is missing observations, actions, RAM or episode metadata')
        n = len(data['obs'])
        if data['obs'].shape != (n, 64, 64, 3) or any(len(data[key]) != n for key in required):
            raise ValueError('Invalid shared evaluation data shapes')
        data['action'] = data['action'].reshape(-1)
        if not raw_path.exists():
            templates = []
            for stage in range(17):
                path = Path(options['template_dir']) / f'stage_{stage:02d}.png'
                template = cv2.imread(str(path))
                if template is None or template.shape != (7, 16, 3):
                    raise ValueError(f'Expected a 7x16 igloo template: {path}')
                templates.append(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
            stages = np.array([value_eval.get_igloo_stage(r[77]) for r in data['ram']])
            blocks = make_blocks(stages, data['episode_id'])
            if not blocks:
                raise ValueError('Shared data has no valid context blocks')
            value_eval.seed_all(options['value_seed'])
            values = value_eval.evaluate_values(self.world_model, self.agent, data,
                                                templates, blocks, options['value_seed'])
            frame_ids = np.concatenate([np.arange(start + 8, end) for start, end in blocks])
            block_ids = np.concatenate([np.full(end - start - 8, i) for i, (start, end) in enumerate(blocks)])
            temporary = output / 'model_values.tmp.npz'
            np.savez_compressed(temporary, values=values,
                run_name=np.array(self.job['candidate']['run_name']), frame_id=frame_ids,
                block_id=block_ids, episode_id=data['episode_id'][frame_ids],
                episode_step=data['episode_step'][frame_ids])
            temporary.replace(raw_path)
            np.savez_compressed(output / 'used_templates.npz', templates=np.stack(templates))
        write_json(output / 'protocol.json', dict(evaluation_mode='single',
            checkpoint=self.job['candidate'], checkpoint_step=self.checkpoint_step,
            data_path=options['data'], collection_run=str(data.get('collection_run', 'unknown')),
            seed=options['value_seed'], bootstrap=options['bootstrap'], bootstrap_unit='episode',
            context_warmup=8, max_block_length=64,
            latent_sampling='random_sample; reset seed per block, shared across stages/models',
            uncertainty_scope='Evaluation contexts of one fixed checkpoint, not training-seed variation'))
        analyze_values(output, options['bootstrap'], options['value_seed'], 'episode')


def run_worker(job, backend_factory=GPUBackend):
    """A GPU stays exclusively assigned through return screening and alignment."""
    candidate, options = job['candidate'], job['options']
    folder = Path(job['directory'])
    folder.mkdir(parents=True, exist_ok=True)
    source = candidate['source']
    stamps = [source['training_state']] if source['kind'] == 'shared' else [source['world_model'], source['agent']]
    if any(file_stamp(stamp['path']) != stamp for stamp in stamps):
        raise ValueError('Checkpoint files changed after discovery; restart to rebuild the job list')
    if hashlib.sha256(Path(candidate['config']).read_bytes()).hexdigest() != candidate['config_sha256']:
        raise ValueError('Saved run config changed after discovery; restart to rebuild the job list')
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            backend = backend_factory(job)
        return backend

    parameters = return_parameters(options)
    cached = read_json(folder / 'return_evaluation.json')
    reusable = (cached and cached.get('fingerprint') == candidate['fingerprint']
                and cached.get('parameters') == parameters)
    if reusable:
        score = cached['episode_avg_return']
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError('Cached return is nonfinite')
        emit('return_cached', score=score)
    else:
        emit('return_start')
        _, returns = get_backend().evaluate_return()
        if len(returns) != options['episodes'] or not all(math.isfinite(float(v)) for v in returns):
            raise ValueError('Return evaluation did not produce the requested number of finite episode returns')
        score = math.fsum(float(v) for v in returns) / len(returns)
        cached = dict(fingerprint=candidate['fingerprint'], parameters=parameters,
                      episode_avg_return=score, episode_returns=[float(v) for v in returns])
        write_json(folder / 'return_evaluation.json', cached)
        emit('return_done', score=score)
    result = dict(candidate=candidate, gpu=job['gpu'], episode_avg_return=score,
                  qualified=score > options['min_return'], alignment=None, status='below_threshold')
    if result['qualified']:
        alignment_signature = dict(data=options['data_stamp'], templates=options['templates_sha256'],
            seed=options['value_seed'], bootstrap=options['bootstrap'],
            code=options['value_code_sha256'])
        alignment = folder / f'alignment_{digest(alignment_signature)[:10]}'
        result['alignment'] = str(alignment)
        if not (alignment / 'statistics.json').is_file() or not (alignment / 'report_ko.md').is_file():
            emit('alignment_start', score=score, output=str(alignment))
            get_backend().evaluate_alignment(alignment)
        else:
            emit('alignment_cached', output=str(alignment))
        if not (alignment / 'statistics.json').is_file() or not (alignment / 'report_ko.md').is_file():
            raise RuntimeError('Alignment did not produce statistics.json and report_ko.md')
        result['status'] = 'complete'
        emit('alignment_done', output=str(alignment))
    write_json(folder / 'result.json', result)
    return result


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another screen is already using {output}') from None
        yield


def terminate_worker(process):
    """Stop the policy worker and its vector-environment subprocesses together."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def failed_result(job, error):
    result = dict(candidate=job['candidate'], gpu=job['gpu'], status='failed', error=str(error))
    cached = read_json(Path(job['directory']) / 'return_evaluation.json')
    if (cached and cached.get('fingerprint') == job['candidate']['fingerprint']
            and cached.get('parameters') == return_parameters(job['options'])):
        score = cached.get('episode_avg_return')
        if isinstance(score, (float, int)) and math.isfinite(score):
            result.update(episode_avg_return=score, qualified=score > job['options']['min_return'])
    return result


def coordinate(jobs, gpus, output):
    pending = Queue()
    for job in jobs:
        pending.put(job)
    active, results = {}, []
    mutex, stopping = threading.Lock(), threading.Event()

    def run_gpu(gpu):
        while not stopping.is_set():
            try:
                job = pending.get_nowait()
            except Empty:
                return
            job = dict(job, gpu=str(gpu))
            folder = Path(job['directory'])
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / 'job.json', job)
            name = f'{job["candidate"]["run_name"]}@{job["candidate"]["checkpoint"]}'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['PYTHONUNBUFFERED'] = '1'
            env['OMP_NUM_THREADS'] = '1'
            env['MKL_NUM_THREADS'] = '1'
            env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT.parent), str(ROOT / 'probing'),
                                               env.get('PYTHONPATH', '')])
            print(f'[GPU {gpu}] {name}', flush=True)
            process = None
            try:
                with (folder / 'worker.log').open('a') as log:
                    with mutex:
                        if stopping.is_set():
                            return
                        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            '--worker', str(folder / 'job.json')], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
                        active[gpu] = process
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith(EVENT_PREFIX):
                            print(f'[GPU {gpu}] {name}: {line[len(EVENT_PREFIX):].strip()}', flush=True)
                    process.stdout.close()
                    code = process.wait()
                result = read_json(folder / 'result.json') if code == 0 else None
                if result is None:
                    raise RuntimeError(f'Worker exited with code {code}; see {folder / "worker.log"}')
            except Exception as error:
                if process is not None and process.poll() is None:
                    terminate_worker(process)
                result = failed_result(job, error)
                write_json(folder / 'result.json', result)
                print(f'[GPU {gpu}] FAILED {name}: {error}', flush=True)
            finally:
                with mutex:
                    active.pop(gpu, None)
            with mutex:
                results.append(result)
                ranked = sorted(results, key=lambda row: row.get('episode_avg_return', float('inf')))
                write_json(output / 'summary.json', dict(completed=len(results), total=len(jobs),
                    results=ranked, qualified=[row for row in ranked if row.get('qualified')]))

    executor = ThreadPoolExecutor(max_workers=len(gpus))
    interrupted = False
    try:
        futures = [executor.submit(run_gpu, gpu) for gpu in gpus]
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        interrupted = True
        stopping.set()
        with mutex:
            processes = list(active.values())
        for process in processes:
            terminate_worker(process)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return 130 if interrupted else int(any(row['status'] == 'failed' for row in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', default=['3', '4', '5'])
    parser.add_argument('--run', action='append', help='Only scan this run; repeat to select several.')
    parser.add_argument('--exclude-run', action='append', default=[])
    parser.add_argument('--min-return', type=float, default=1000, help='Alignment requires mean return strictly greater than this.')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--return-seed', type=int, default=0)
    parser.add_argument('--value-seed', type=int, default=2027)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--data', default='results/old/seed_3710/shared_contexts.npz')
    parser.add_argument('--template-dir', default='frostbite_igloo_templates')
    parser.add_argument('--output', default='results/frostbite_checkpoint_screen')
    parser.add_argument('--dry-run', action='store_true', help='List checkpoints without importing PyTorch or starting workers.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        job = read_json(args.worker)
        try:
            run_worker(job)
        except Exception as error:
            write_json(Path(job['directory']) / 'result.json', failed_result(job, error))
            traceback.print_exc()
            return 1
        return 0
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        parser.error('GPU identifiers must be unique')
    if args.episodes < 1 or args.bootstrap < 100 or not math.isfinite(args.min_return):
        parser.error('episodes must be positive, bootstrap >=100, and min-return finite')
    if not 0 <= args.return_seed <= 2**32 - args.episodes or not 0 <= args.value_seed <= 2**32 - 1:
        parser.error('Seeds must fit unsigned 32-bit episode RNG ranges')
    resolve = lambda path: (ROOT / path).resolve()
    data, templates, output = map(resolve, (args.data, args.template_dir, args.output))
    if not data.is_file():
        parser.error(f'Shared context data does not exist: {data}')
    template_paths = [templates / f'stage_{i:02d}.png' for i in range(17)]
    if any(not path.is_file() for path in template_paths):
        parser.error(f'Need all 17 stage templates in {templates}')
    options = dict(episodes=args.episodes, return_seed=args.return_seed, value_seed=args.value_seed,
        precision=args.precision, min_return=args.min_return, bootstrap=args.bootstrap,
        data=str(data), data_stamp=file_stamp(data), template_dir=str(templates),
        templates_sha256=hashlib.sha256(b''.join(path.read_bytes() for path in template_paths)).hexdigest(),
        eval_code_sha256=hashlib.sha256((ROOT / 'eval.py').read_bytes()).hexdigest(),
        value_code_sha256=hashlib.sha256(b''.join((ROOT / 'probing' / name).read_bytes()
            for name in ('eval_frostbite_value.py', 'frostbite_value_stats.py'))).hexdigest())
    candidates = discover_checkpoints(ROOT, args.run, args.exclude_run)
    if not candidates:
        parser.error('No matching checkpoint pairs or shared checkpoints found')
    if args.run and set(args.run) - {candidate['run_name'] for candidate in candidates}:
        parser.error('One or more requested runs have no eligible saved checkpoints')
    jobs = [dict(candidate=candidate, options=options,
                 directory=str(output / candidate['id'])) for candidate in candidates]
    print(f'{len(jobs)} checkpoints; GPUs {", ".join(args.gpus)}; '
          f'{args.episodes} episodes; align when mean return > {args.min_return:g}', flush=True)
    if args.dry_run:
        for candidate in candidates:
            print(f'{candidate["run_name"]}@{candidate["checkpoint"]} ({candidate["source"]["kind"]})')
        return 0
    with output_lock(output):
        write_json(output / 'manifest.json', dict(gpus=args.gpus, jobs=jobs))
        return coordinate(jobs, args.gpus, output)


if __name__ == '__main__':
    # Shared resume checkpoints contain objects from modules next to STORM-1.
    sys.path[:0] = [str(ROOT), str(ROOT.parent), str(ROOT / 'probing')]
    raise SystemExit(main())
