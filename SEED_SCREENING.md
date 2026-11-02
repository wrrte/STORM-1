# Frostbite baseline seed search

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
