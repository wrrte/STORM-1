# Hero-3710 performance investigation — 2026-09-29

## Scope and current run

The user observes roughly 60–75% GPU utilization during shared warmup, and
recalls similarly low utilization in retrieval-on and retrieval-off runs.
No production training code or configuration was changed during this investigation.
The running training process was not restarted or attached to a profiler.

The local active run was PID 2819807, on physical GPU 7 (RTX A6000).
Its W&B local directory was `wandb/run-20260929_143953-vu0liwvf`.
The actual command also overrides `JointTrainAgent.Retrieval.enable` with
`['target1', 'value', 'add']`; its effective configuration is
`runs/Hero-3710_Shared/config.yaml`. This differs from the unoverridden YAML's
`Both` branch selection, but both execute the common shared-warmup path.
One live log sample reported approximately 5.82 iterations/s. Separate GPU
snapshots showed 68–69% utilization; these are not a controlled throughput baseline.

Relevant configuration: one environment, GPU-resident replay, world-model batch
16 × 64, imagination context 1024 × 8, imagination horizon 16, BF16 AMP.
Demonstrations are disabled, so `D_TRAJ/Hero.pkl` is not loaded by this run.
Retrieval searches and periodic global rebuilding do not execute in ordinary
shared-warmup iterations, but hashing and retrieval statistics still do.

## Controlled component measurements

The standalone `benchmark_shared_paths.py` uses the existing training Python
environment and model implementations. It ran on idle physical GPU 6, using
synthetic GPU-resident replay with 4096 frames and newly initialized models.
It never initializes W&B or starts an environment/training run.

Each timing below is the median of 12 calls after four warmup calls. Two rounds
were run, reversing the validation comparison order in the second round.
CUDA is synchronized at component boundaries. These are component wall times,
not end-to-end training step times; do not add the savings to predict throughput.

| Component | Existing code, ms (rounds 1 / 2) | Diagnostic alternative, ms (rounds 1 / 2) |
|---|---:|---:|
| Replay sample 16 × 64 | 0.533 / 0.544 | 0.382 / 0.378 |
| Replay sample 1024 × 8 | 12.283 / 13.229 | 2.332 / 2.321 |
| Imagination 1024 × 16 | 97.175 / 97.026 | 70.635 / 70.882 |

The replay alternative constructs a batch of indices and gathers tensors in
one operation per field. It keeps the original NumPy random draws. All six
returned values and the post-call NumPy RNG state matched exactly for both
tested batch shapes. This alternative only covers the current one-environment,
GPU-resident, no-demonstration case; it is not a production replacement.

The imagination alternative only changes
`torch.distributions.Distribution.set_default_validate_args(False)` in the
diagnostic process. It disables distribution validation, not model computation.
This diagnostic setting must not be confused with running all Python under `-O`.
Output/gradient equivalence and numerical-error detection in real training
have not yet been validated.

Separate CPU/CUDA traces captured one original replay sample followed by one
imagination call. With validation enabled, the trace contains 91 calls each to
`aten::_is_all_true`, `aten::item`, and `cudaStreamSynchronize`. None of those
events appear in the corresponding validation-disabled trace. An explicit
end-of-capture device synchronization is present in both cases.
The traces include profiler overhead and are not the source of the timing table.

Code locations:

- `replay_buffer.py:65–71`: four Python lists of slices for every sample;
  the imagination batch creates 4096 slice/select pairs.
- `sub_models/world_models.py:369`: `OneHotCategorical` without an explicit
  `validate_args` setting.
- `agents.py:127–135`: `Categorical` without an explicit `validate_args` setting.
- `sub_models/world_models.py:408–430`: eight context steps and sixteen
  imagined steps repeatedly construct these distributions.
- Installed `torch/distributions/distribution.py`: validation defaults to
  `__debug__` and converts tensor validity checks to a Python condition.

Artifacts from the completed component run:

- `/tmp/storm-hero-diagnostic-20260929/summary.json`
- `/tmp/storm-hero-diagnostic-20260929/validate_True.json`
- `/tmp/storm-hero-diagnostic-20260929/validate_False.json`
- Operator tables and machine-readable operator summaries in the same directory.

The diagnostic process exited successfully and released GPU 6.

## Swap investigation

The screenshot's 99% SWP is system swap occupancy, not current swap traffic.
Read-only measurements during the running Hero job found:

| Measurement | Observed value |
|---|---:|
| Swap total / used | 2,097,148 / 2,085,020 KiB (99.42%) |
| RAM `MemAvailable` | 291,393,300 KiB (277.89 GiB) |
| Hero `VmRSS` | 2,240,428 KiB (2.14 GiB) |
| Hero `VmSwap` | 0 KiB |
| `vmstat 1 8`: seven interval samples, excluding first since-boot row | `si=0`, `so=0`, `wa=0` throughout |
| Memory PSI `some` / `full`, avg10 / avg60 / avg300 | all 0.00 |
| I/O PSI `some` / `full`, avg10 / avg60 / avg300 | all 0.00 |

These observations do not support active swapping as the cause of the sustained
utilization issue at the time measured. They cannot exclude a past or future
transient stall. Swap settings were not changed.

## Next discriminating experiment

The strongest current evidence points to repeated distribution-validation
synchronization and Python replay sampling overhead. Their contribution to the
actual complete Hero step has not yet been measured.

Measure a bounded window of ordinary shared-warmup training, excluding startup,
video/checkpoint writes and the branch boundary. Label collection/env step,
replay sampling, world-model update, retrieval statistics, imagination,
actor-critic update, and logging. Use CUDA timeline traces for gaps and an
unprofiled elapsed-time window for throughput; per-phase Python timers alone
misattribute asynchronously queued GPU work.

Compare original code, replay indexing alone, distribution-validation change
alone, and both, using identical starting state and settings. Verify output/RNG
behavior and loss finiteness before promoting an optimization. Keep batch size,
training frequency and rollout length fixed. The target metric is lower wall
time per training step, with GPU-util only a supporting observation.

References for metric interpretation:

- NVIDIA GPU utilization: https://docs.nvidia.com/deploy/nvidia-smi/
- Linux memory pressure: https://www.kernel.org/doc/html/latest/accounting/psi.html
- vmstat counters and first-row semantics: https://www.man7.org/linux/man-pages/man8/vmstat.8.html
