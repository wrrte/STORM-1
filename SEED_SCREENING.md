# Frostbite seed screening

## Paired random-neighbor / Full FLASH campaign (2026-11-07)

The active campaign uses `frostbite_queue_campaign.py` and the frozen
`config_files/STORM.frostbite_screen_20261107.yaml`. It searches new seeds from
11000 upward, reserving at most three candidates across all queue workers.
There is no baseline branch.

Each candidate trains the common prefix once. If any episode's cumulative
training score strictly exceeds 500 by environment step 20,000, including the
current unfinished episode, it continues without resetting model, replay, or
RNG. Otherwise it exits at exactly 20,000 and the worker tries the next seed.
This is a training-trajectory criterion, not a separate evaluation average.
Exactly 500 does not pass; a success at exactly step 20,000 does pass.

At the first episode boundary at or after 50,000 steps, each accepted seed
saves one shared checkpoint and runs `randomneighor` and `retrieval` sequentially
from that identical checkpoint. Both use offset -2, target 16, fixed anchor
weight 0.0625 (`anchor_weight_r: null`), the existing 102k training budget, and
20 evaluation episodes in BF16 and FP32 from the 100k checkpoint. The children
disable screening because the common prefix already passed. An accepted seed
is never replaced based on its final score.

Eight priority entries in `job_queue_3090.txt` refer to the same campaign,
one per physical GPU worker. Only three candidates can train at once; other
workers wait. Keep one `N_train.sh` worker per physical GPU. Existing jobs may
finish, then join this barrier. After **three seeds have completed both
branches**, the remaining priority entries are removed and the original queue
continues in its original order. One worker can also complete the entire
search sequentially. Unexpected training errors pause the campaign and keep
the ordinary queue waiting; only the explicit screening rejection permits
moving to another seed. Ctrl+C/SIGTERM instead marks that attempt as
`interrupted` and queues the **same seed** for retry, with a new run name and
log such as `seed_11001_retry2.log`. Earlier logs and checkpoints remain in
place. Stopping one worker does not pause the other workers. Each restarted
worker replenishes the priority entries so restarts cannot exhaust the barrier.

The campaign directory under `diagnostics/frostbite_neighbor_screen_*` contains:

- `plan.json`: frozen settings, config hash, and queue command.
- `state.json`: all attempts, their status, screening decisions, errors, and paired final scores.
- `results.csv`: one row per attempted seed, including paired scores on completion.
- `seed_<seed>.log`: full training and branch-supervisor output.
- `seed_<seed>_retry<N>.log`: separate output for an interrupted seed's retry.
- `queue_before.txt` / `queue_after.txt`: queue snapshots taken under its normal lock.
- `worker_gpu_<gpu>.log`: output from workers started for this campaign.

Inspect the campaign without touching GPUs:

```bash
python frostbite_queue_campaign.py --status diagnostics/<campaign-directory>
```

An interrupted worker from the initial launcher version could leave an empty
global error (`Seed 11001: `). After confirming that its child training process
has stopped, this legacy state can be repaired without changing other running
candidates or deleting any files:

```bash
python frostbite_queue_campaign.py --recover-interrupted 11001 \
  --directory diagnostics/<campaign-directory>
```

This command saves the previous state and queues that same seed for retry.
It refuses to clear a genuine training failure with a diagnostic message.

To stage a new campaign after the active one has finished:

```bash
python frostbite_queue_campaign.py \
  --prepare config_files/STORM.frostbite_screen_20261107.yaml --start-seed 12000
bash 0_train.sh
```

The selected seeds are conditioned on early training performance. These are
additional screened comparisons; they do not replace the frozen paper or
paired-bootstrap seed selections. Paired final BF16 means are in `results.csv`;
each branch retains its usual local evaluation, W&B metrics, and checkpoints.

## Legacy baseline-only launcher

Run from an environment with the STORM dependencies and access to the selected GPUs:

```bash
cd /home/choemj/STORM-1
conda activate storm
python -u screen_frostbite_seeds.py --gpus 0 1 2 --start-seed 10000
```

Each GPU trains one seed at a time with `Retrieval.enable=False`,
`save_warmup=False`, and `warmup_offset_pairs=[]`. GPU 0 tries seeds
10000, 10003, 10006, ...; GPU 1 tries 10001, 10004, ...; GPU 2 tries
10002, 10005, .... Existing job queues are not used.

The seed passes as soon as any episode's cumulative score strictly exceeds
500 within the first 20,000 environment steps (80,000 Atari frames with the
existing frame skip). Scores at exactly step 20,000 count, including the
terminal reward and any unfinished episode's current score. A score of exactly
500 does not pass. Scores from separate episodes are never added together.
At step 20,000 a rejected run exits without final evaluation and the same GPU
immediately starts its next seed. Genuine training errors stop the search.

An accepted run continues the same training process with its existing model,
replay buffer and RNG state. The existing training budget, checkpoint schedule,
and evaluation precision come from `config_files/STORM.yaml`. By default the
100k checkpoint is evaluated on 100 episodes per configured precision, then
each GPU stops after one fully trained and evaluated model (three models total).

Useful options:

```bash
# Preview commands without starting training or writing files.
python screen_frostbite_seeds.py --gpus 0 1 2 --dry-run

# Keep searching after each accepted model finishes. Ctrl+C stops all workers.
python -u screen_frostbite_seeds.py --gpus 0 1 2 --successes-per-gpu 0

# Bound the search and change the score gate or evaluation episode count.
python -u screen_frostbite_seeds.py --gpus 0 1 2 --max-attempts-per-gpu 20 \
    --screen-steps 20000 --min-return 500 --eval-episodes 100
```

`--python /path/to/python` selects the training interpreter explicitly. GPU
preflight fails before starting workers if the selected CUDA devices are
unavailable. A fresh campaign name prevents different invocations from
overwriting checkpoints. Reusing a start seed intentionally repeats those
training seeds in a new campaign; choose another start seed for a new seed range.

The launcher prints its log directory under `diagnostics/frostbite_seed_screen_*`:

- `campaign.json`: settings and seed assignment.
- `gpu_<id>_seed_<seed>.log`: each training process's complete output.
- `gpu_<id>.jsonl`: rejected, completed, and failed outcomes with run paths.
- `runs/Frostbite-seed-screen-<campaign>-seed<seed>_X/seed_screen.json`:
  threshold decision and maximum cumulative score up to that decision.
- `ckpt/Frostbite-seed-screen-<campaign>-seed<seed>_X/`: model checkpoints.

Threshold decisions are also logged to W&B as `seed_screen/*`. Rejected runs
have no `eval/episode_avg_return`. Seed screening is disabled in ordinary
training (`JointTrainAgent.SeedScreenSteps=0`).
