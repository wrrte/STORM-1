"""Bounded, synthetic GPU diagnostics; does not launch training or contact W&B.

Run with the training Python environment and an explicitly selected idle GPU:
  python diagnostics/benchmark_shared_paths.py --gpu GPU-... --output /tmp/storm-diag

These component timings are NOT an end-to-end Hero training benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Explicit physical GPU UUID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    warnings.filterwarnings("ignore", category=FutureWarning)

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    from replay_buffer import ReplayBuffer
    from train import build_agent, build_world_model
    from utils import load_config

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3710)
    np.random.seed(3710)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    conf = load_config(str(Path(__file__).resolve().parents[1] / "config_files/STORM.yaml"))
    # This older benchmark isolates replay indexing and distribution validation.
    conf.defrost()
    conf.Performance.ProjectedKVCache = False
    conf.Performance.BatchScalarLogging = False
    conf.freeze()
    # Smaller capacity than the full run, with the same observation/batch shapes.
    replay = ReplayBuffer((64, 64, 3), 1, max_length=4096,
                          warmup_length=1024, store_on_gpu=True)
    replay.obs_buffer.random_(0, 256)
    replay.action_buffer.random_(0, 18)  # ALE/Hero-v5 minimal action space.
    replay.reward_buffer.zero_()
    replay.termination_buffer.zero_()
    replay.length = 4096
    replay.last_pointer = 4095

    def vector_sample(batch_size, batch_length):
        # Exercise the production opt-in path, restoring the legacy default.
        previous = replay.vectorized_sampling
        replay.vectorized_sampling = True
        try:
            return replay.sample(batch_size, 0, batch_length)
        finally:
            replay.vectorized_sampling = previous

    equality = {}
    for batch, length in [(16, 64), (1024, 8)]:
        state = np.random.get_state()
        original = replay.sample(batch, 0, length)
        after_original = np.random.get_state()
        np.random.set_state(state)
        alternative = vector_sample(batch, length)
        after_alternative = np.random.get_state()
        equality[f"{batch}x{length}"] = all(
            torch.equal(a, b) if isinstance(a, torch.Tensor) else np.array_equal(a, b)
            for a, b in zip(original, alternative)) and all(
                np.array_equal(a, b) for a, b in zip(after_original, after_alternative))
        assert equality[f"{batch}x{length}"]
        del original, alternative

    world = build_world_model(conf, 18).eval()
    agent = build_agent(conf, 18).eval()
    context_obs, context_action, *_ = replay.sample(1024, 0, 8)

    @torch.no_grad()
    def imagine():
        world.eval()
        agent.eval()
        return world.imagine_data(agent, context_obs, context_action, 1024, 16,
                                  log_video=False, logger=None)

    records = []

    def measure(name, fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            elapsed.append(1000 * (time.perf_counter() - start))
        record = {"name": name, "median_ms": statistics.median(elapsed),
                  "min_ms": min(elapsed), "max_ms": max(elapsed), "samples_ms": elapsed}
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "samples_ms"}), flush=True)

    for round_index in range(2):
        for batch, length in [(16, 64), (1024, 8)]:
            measure(f"r{round_index}/sample_original_{batch}x{length}",
                    lambda: replay.sample(batch, 0, length))
            measure(f"r{round_index}/sample_vector_{batch}x{length}",
                    lambda: vector_sample(batch, length))
        for validate in ([True, False] if round_index == 0 else [False, True]):
            torch.distributions.Distribution.set_default_validate_args(validate)
            measure(f"r{round_index}/imagine_validate_{validate}", imagine)

    # Profiler is kept separate from the timing comparison: its overhead is large.
    for validate in (True, False):
        torch.distributions.Distribution.set_default_validate_args(validate)
        imagine()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("replay_sample_1024x8"):
                replay.sample(1024, 0, 8)
            with record_function("imagination_1024x16"):
                imagine()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / f"validate_{validate}.json"))
        (output / f"validate_{validate}_operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
        events = [{"key": e.key, "count": e.count,
                   "cpu_total_us": e.cpu_time_total, "self_cpu_us": e.self_cpu_time_total}
                  for e in prof.key_averages()]
        (output / f"validate_{validate}_operators.json").write_text(json.dumps(events, indent=2))

    torch.distributions.Distribution.set_default_validate_args(True)
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "gpu_uuid": args.gpu, "amp_dtype": str(world.amp_dtype),
              "torch_num_threads": torch.get_num_threads(), "equal_outputs_and_numpy_rng": equality,
              "limitations": ["Synthetic replay data; untrained models; no env, optimizer or logging timing",
                              "Timings synchronize at component boundaries",
                              "This benchmark measures one GPU-resident environment without demonstrations"],
              "records": records}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"Artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
