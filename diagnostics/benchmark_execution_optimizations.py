"""Bounded synthetic timings on an explicitly selected idle GPU; no W&B/env run.

Baseline already enables vectorized replay and disables distribution validation.
The learning cycle includes both optimizer updates and warmup retrieval statistics,
not environment collection, real log writers, hash insertion, evaluation or saves.
"""
import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, help='Explicit physical GPU UUID')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root.parent)]
    warnings.filterwarnings('ignore', category=FutureWarning)
    import numpy as np
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function
    from train import build_world_model, build_agent
    from replay_buffer import ReplayBuffer
    from retrieval import RetrievalContextManager
    from utils import load_config, seed_np_torch, log_scalar_metrics

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seed_np_torch(3710)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.distributions.Distribution.set_default_validate_args(False)
    conf = load_config(str(root / 'config_files/STORM.yaml'))
    world, agent = build_world_model(conf, 18), build_agent(conf, 18)
    replay = ReplayBuffer((64, 64, 3), 1, max_length=4096, warmup_length=1024,
                          store_on_gpu=True, vectorized_sampling=True)
    replay.obs_buffer.random_(256)
    replay.action_buffer.random_(18)
    replay.reward_buffer.zero_()
    replay.termination_buffer.zero_()
    replay.length, replay.last_pointer = 4096, 4095
    ret_conf = dict(conf.JointTrainAgent.Retrieval)
    ret_conf.update(enable=['target1', 'value', 'add'], context_length=8)
    manager = RetrievalContextManager(1, ret_conf, 1024)
    context_obs, context_action, *_ = replay.sample(1024, 0, 8)
    stat_signals = {name: torch.randn(16, 55, device='cuda') for name in manager.statistics_signals}
    stat_mask = torch.ones(16, 55, device='cuda', dtype=torch.bool)
    stat_envs = torch.zeros(16, device='cuda', dtype=torch.int64)
    metrics_input = {str(i): torch.randn((), device='cuda') for i in range(14)}

    class Sink:
        def __init__(self):
            self.metrics = {}
        def log(self, tag, value):
            self.metrics[tag] = value
    logger = Sink()

    def cpu_snapshot(state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, dict):
            return {key: cpu_snapshot(value) for key, value in state.items()}
        if isinstance(state, list):
            return [cpu_snapshot(value) for value in state]
        return copy.deepcopy(state)

    models = cpu_snapshot([world.state_dict(), agent.state_dict()])
    optimizers = copy.deepcopy([world.optimizer.state_dict(), agent.optimizer.state_dict()])
    scalers = copy.deepcopy([world.scaler.state_dict(), agent.scaler.state_dict()])
    retrieval_state = cpu_snapshot(manager.state_dict())
    initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())

    variants = {
        'baseline': (False, 'legacy', False),
        'batched': (True, 'batched', False),
        'optimized': (True, 'vectorized', True),
    }

    def reset(name):
        logging, stats, projected = variants[name]
        for i, model in enumerate((world, agent)):
            model.load_state_dict(models[i])
            model.optimizer.load_state_dict(copy.deepcopy(optimizers[i]))
            model.scaler.load_state_dict(copy.deepcopy(scalers[i]))
            model.optimizer.zero_grad(set_to_none=True)
            model.batch_scalar_logging = logging
        agent.lowerbound_ema.scalar = agent.upperbound_ema.scalar = 0.0
        world.storm_transformer.projected_kv_cache = projected
        manager.statistics_mode = stats
        manager.load_state_dict(retrieval_state)
        random.setstate(initial_rng[0])
        np.random.set_state(initial_rng[1])
        torch.set_rng_state(initial_rng[2])
        torch.cuda.set_rng_state(initial_rng[3])

    @torch.no_grad()
    def imagine():
        world.eval()
        agent.eval()
        return world.imagine_data(agent, context_obs, context_action, 1024, 16, False, None)

    def cycle(trace=False):
        scope = record_function if trace else lambda _: nullcontext()
        with scope('replay/world'):
            obs, action, reward, done, indexes, envs = replay.sample(16, 0, 64)
        with scope('world/update'):
            latent, features = world.update(obs, action, reward, done, logger=logger)
        with scope('retrieval/warmup_stats'), torch.no_grad():
            values = agent.value(torch.cat([latent, features], dim=-1)).squeeze(-1)
            manager.add_batch_transitions(values, reward, done, agent.gamma, indexes,
                                          envs, replay.max_length, skip_len=8, is_warmup=True)
        with scope('replay/imagination'):
            sample_obs, sample_action, *_ = replay.sample(1024, 0, 8)
            weights = torch.ones(1024, device='cuda')
        with scope('imagination'), torch.no_grad():
            world.eval()
            agent.eval()
            latent, action, reward, done = world.imagine_data(
                agent, sample_obs, sample_action, 1024, 16, False, None)
        with scope('agent/update'):
            agent.update(latent, action, None, None, reward, done, logger=logger, weights=weights)

    records = []
    def measure(name, fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        row = dict(name=name, median_ms=statistics.median(samples), min_ms=min(samples),
                   max_ms=max(samples), samples_ms=samples)
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)

    for round_index in range(2):
        order = list(variants) if round_index == 0 else list(reversed(variants))
        for variant in order:
            reset(variant)
            measure(f'r{round_index}/{variant}/imagination', imagine)
            measure(f'r{round_index}/{variant}/statistics',
                    lambda: manager._update_ema_stats(stat_signals, stat_mask, stat_envs))
            measure(f'r{round_index}/{variant}/scalar_readout',
                    lambda: log_scalar_metrics(logger, metrics_input, variants[variant][0]))
            reset(variant)
            measure(f'r{round_index}/{variant}/learning_cycle', cycle)
            assert all(np.isfinite(v) for v in logger.metrics.values())

    trace_counts = {}
    if args.trace:
        for variant in variants:
            reset(variant)
            cycle()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                cycle(trace=True)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(output / f'{variant}.json'))
            (output / f'{variant}_operators.txt').write_text(
                prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40))
            trace_counts[variant] = {e.key: e.count for e in prof.key_averages()
                                    if e.key in ('aten::item', 'aten::nonzero', 'cudaStreamSynchronize')}
    # Fixed input tokens isolate numerical differences from stochastic rollout divergence.
    with torch.no_grad():
        transformer = world.storm_transformer.eval()
        fixed_tokens = torch.nn.functional.one_hot(
            torch.randint(32, (32, 24, 32), device='cuda'), 32).flatten(-2).float()
        fixed_actions = torch.randint(18, (32, 24), device='cuda')
        forced_outputs = []
        rng_before = torch.cuda.get_rng_state()
        for projected in (False, True):
            transformer.projected_kv_cache = projected
            transformer.reset_kv_cache_list(32, world.amp_dtype)
            with torch.autocast('cuda', dtype=world.amp_dtype):
                forced_outputs.append(torch.cat([
                    transformer.forward_with_kv_cache(fixed_tokens[:, t:t+1], fixed_actions[:, t:t+1])
                    for t in range(24)], dim=1).float())
        delta = forced_outputs[1] - forced_outputs[0]
        numerical_comparison = dict(
            max_abs=delta.abs().max().item(), mean_abs=delta.abs().mean().item(),
            relative_l2=(delta.norm() / forced_outputs[0].norm()).item(),
            rng_unchanged=torch.equal(rng_before, torch.cuda.get_rng_state()))
        torch.testing.assert_close(forced_outputs[1], forced_outputs[0], atol=.05, rtol=.03)
    result = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), gpu_uuid=args.gpu,
                  amp_dtype=str(world.amp_dtype), cpu_threads=torch.get_num_threads(),
                  baseline='VectorizedReplaySampling=True, DisableDistributionValidation=True',
                  limitations=['Synthetic GPU replay; newly initialized models; component boundaries synchronized',
                               'Learning cycle excludes env/collection, hash insertion, real log writers and saves',
                               'Changed reduction/projection shapes do not promise bitwise equality'],
                  records=records, trace_counts=trace_counts, numerical_comparison=numerical_comparison)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    print(f'Artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
