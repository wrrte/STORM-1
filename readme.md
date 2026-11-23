# Implementation of STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning



export CUDA_VISIBLE_DEVICES=5
source .bashrc
cd STORM-1
conda activate storm



News: This project is no longer maintained, and we recommend using the new [OC-STORM](https://github.com/weipu-zhang/OC-STORM) repo, which also contains full code for running STORM on Atari games.

This repo contains an implementation of STORM. 

Following the **Training and Evaluating Instructions** to reproduce the main results presented in our paper. One may also find **Additional Useful Information** useful when debugging and observing intermediate results. To reproduce the speed metrics mentioned in the paper, please see **Reproducing Speed Metrics**.

## Training and Evaluating Instructions

1. Install the necessary dependencies. Note that we conducted our experiments using `python 3.10`.
    ```shell
    pip install -r requirements.txt
    ```
    Installing `AutoROM.accept-rom-license` may take several minutes.

2. Train the agent.
    ```shell
    chmod +x train.sh
    ./train.sh
    ```

    The `train.sh` file controls the environment and the running name of a training process. 
    ```shell
    env_name=MsPacman
    python -u train.py \
        -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
        -seed 1 \
        -config_path "config_files/STORM.yaml" \
        -env_name "ALE/${env_name}-v5" \
        -trajectory_path "trajectory/${env_name}.pkl"
    ```

    - The `env_name` on the first line can be any Atari game, which can be found [here](https://gymnasium.farama.org/environments/atari/).
    
    - `-n` option is the name for the tensorboard logger and checkpoint folder. You can change it to your preference, but we recommend keeping the environment's name first. The tensorboard logging folder is `runs`, and the checkpoint folder is `ckpt`.

    - The `-seed` parameter controls the running seed during the training. We evaluated our method using 5 seeds and report the mean return in Table 1.

    - The `-config_path` points to a YAML file that controls the model's hyperparameters. The configuration in `config_files/STORM.yaml` is the same as in our paper.

    - With `EvalMode: final_only`, training finishes at `SampleMaxSteps` before evaluation. `FinalEvalStep: 100000` loads `ckpt/<run_name>/world_model_100000.pth` and `agent_100000.pth` for that evaluation; both checkpoints must exist. Final training weights are still saved separately as `world_model_final.pth` and `agent_final.pth`. Set `JointTrainAgent.FinalEvalStep None` on the command line (or `FinalEvalStep: null` in YAML) to evaluate the end-of-training weights. This setting also selects the final checkpoint when `EvalMode: active`; periodic evaluations continue to use the current weights. The selected step is logged as `eval/checkpoint_step`.

    - The `-trajectory_path` is only useful when the option `UseDemonstration` in the YAML file is set to `True` (by default it's `False`). This corresponds to the ablation studies in Section 5.3. We provide the pre-collected trajectories in the `D_TRAJ.7z` file, and you need to decompress it for use.
    

3. Evaluate the agent. The evaluation results will be presented in a CSV file located in the `eval_result` folder.
    ```shell
    chmod +x eval.sh
    ./eval.sh
    ```

    The `eval.sh` file controls the environment and the running name when testing an agent.

    ```shell
    env_name=MsPacman
    python -u eval.py \
        -env_name "ALE/${env_name}-v5" \
        -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
        -config_path "config_files/STORM.yaml" 
    ```

    The `-run_name` option is the same as the `-n` option in `train.sh`. It should be kept the same as in the training script.

    **BF16 / FP32 evaluation comparison:** `JointTrainAgent.EvalPrecision` accepts
    `auto`, `bf16`, `fp32`, or `both`. The supplied STORM config uses `both`.
    Intermediate and final evaluations run both precisions on the same weights.
    `EvalSeed: 0` fixes the policy RNG and assigns distinct environment seeds
    `0, 1, ..., EvalEpisodes-1` in both arms. Each environment contributes only its
    first complete episode, including the terminal reward.

    Existing checkpoints can be compared without starting training:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python -u eval.py \
        -config_path config_files/STORM.yaml \
        -env_name ALE/Krull-v5 -run_name Krull-2170_baseline \
        -step 100000 -precision both -seed 10000 -episodes 100
    ```

    Standalone evaluation writes timestamped mean and per-episode CSV files under
    `eval_result/`, with precision and environment-seed columns. Explicit BF16 and
    FP32 evaluation disable TF32; FP32 disables autocast in both the world model
    and the agent. Model precision, train/eval flags, backend settings, and Python,
    NumPy, CPU/CUDA RNG states are restored afterwards. Training precision and
    optimizer/scaler settings are preserved. Equal initial RNG states do not imply
    identical actions or trajectories once numerical results differ.

    Training logs contain `eval/bf16/episode_avg_return`,
    `eval/fp32/episode_avg_return`, per-episode returns under each prefix, and
    `eval/fp32_minus_bf16`. The existing `eval/episode_avg_return` and per-episode
    tags use the BF16 arm when `both` is selected. Override training settings with
    `JointTrainAgent.EvalPrecision both JointTrainAgent.EvalEpisodes 100
    JointTrainAgent.EvalSeed 10000` on the training command. `auto` uses the
    model's existing AMP configuration; older YAML files default to `auto`.

## Additional Useful Information

### GPU model queues and reusable warmups

A **seed belongs to one GPU model**, independently of the game. All A6000
workers on this machine share one priority queue; a seed is never tied to a
physical card number. Workers identify the actual model and UUID using
`nvidia-smi`. Different models cannot consume each other's jobs.

`config_files/game_seeds/type_A6000.json` lists the selected games, seeds,
and saved warmups. Set `enabled: false` to exclude a pair from future default
enqueues. Explicit `--games`/`--game-seeds` can request it again. Discovery runs
on enqueue and immediately before execution, so newly published shared
warmups become available automatically. Both explicit pairs and saved
selections prefer compatible existing snapshots.

```bash
cd /home/choemj/STORM-1
conda activate storm

# Preview using the existing A6000 selections. No jobs or training are started.
python enqueue_train.py --gpu-type A6000 \
  --warmup-offset-pairs '[[50000,-2]]' \
  --hash-target-pairs '[[9,4],[10,8],[12,16]]' \
  --set JointTrainAgent.Retrieval.enable Both \
  --priority 10 --dry-run
```

Remove `--dry-run` to enqueue. Higher priority numbers run first; ties use
insertion order. New high-priority jobs do not preempt running training.
Identical pending/running experiments are deduplicated. Configuration is
snapshotted when enqueued, so later edits to `STORM.yaml` do not alter an
already queued experiment. The pair options default to `[]`, replacing the
YAML pair lists. With no warmup pairs, the selected config's scalar
`warmup_steps`/`anchor_offset` applies. Each warmup/offset pair is a separate
job; its selected experiment/hash variants retain individual completion state.

For STORM, Freeway defaults to `JointTrainAgent.UseDemonstration: true`, while
other games retain the configured value (false in `STORM.yaml`). The same
game-specific setting applies to warmup compatibility, new collection, and
retrieval/baseline continuations. An explicit
`--set JointTrainAgent.UseDemonstration False` requests the no-demonstration
ablation instead.

```bash
# In separate tmux sessions, start only the cards you want to use.
bash 0_train.sh
bash 4_train.sh
bash 7_train.sh

# Any physical index is supported; there is no 0–7 restriction.
bash queue_train.sh 12 --once

python queue_train.py status --gpu-type A6000
python queue_train.py status --all
python queue_train.py priority JOB_ID 100
python queue_train.py retry FAILED_JOB_ID  # preserves completed branches
```

Stopping workers 0–3 leaves their unstarted jobs available to workers 4–7.
Ctrl-C/SIGTERM terminates the active job's process tree and returns its job to
the same model pool. The next free worker takes the highest-priority runnable
job. Completed branches are skipped. Collection for the same game/seed is
serialized until a warmup is published; continuations can then run on multiple
cards of that model concurrently.

Queue training requests a full model/optimizer/replay/RNG checkpoint at the
next episode boundary when stopped. If it finishes saving within the shutdown
grace period (120 seconds by default; set `STORM_QUEUE_STOP_GRACE` to change it), another worker resumes from that checkpoint.
Otherwise it uses the last complete checkpoint or the shared warmup, restarting
only the unfinished branch. This is not an exact mid-episode ALE-state restore.
Checkpoint publication is atomic; partially written checkpoints are ignored.
Final evaluation must also finish before a branch is marked complete. Temporary
resume checkpoints are removed after branch success.

A dead worker's running job is recovered when its lease lock is released.
A surviving trainer retains that lock, preventing duplicate execution after a
worker-only SIGKILL. Errors are recorded as failed jobs rather than retried in
an endless loop. `--once` handles at most one job; `--max-jobs N` limits a worker.

New seeds can be allocated without choosing numbers manually:

```bash
python enqueue_train.py --gpu-type A6000 --games Alien Pong --new-seeds 2 --dry-run
# Or use existing/exact pairs, still preferring reusable warmups:
python enqueue_train.py --gpu-type A6000 --game-seeds '[["Alien",6000],["Pong",6010]]' --dry-run
python queue_train.py seeds
```

New A6000 seeds use `1000000–1999999`; RTX 3090 uses `2000000–2999999`,
TITAN RTX `3000000–3999999`, PRO 6000 Blackwell `4000000–4999999`, RTX 4090
`5000000–5999999`, RTX 5090 `6000000–6999999`, and RTX 6000 Ada
`7000000–7999999`. Other models receive persisted nonoverlapping ranges.
Legacy seeds retain their original values and explicit model mappings, so
existing checkpoints remain reusable. A seed already registered to another
model, or belonging to another model's reserved range, is rejected for every
game. `--new-seeds --dry-run` previews numbers without reserving them.

Compatibility checks include model dimensions, replay/training-prefix settings,
statistics EMA settings, warmup targets, dynamic warmup settings and the
statistics needed by the requested retrieval preset. Post-warmup hash/target,
offset and retrieval experiment settings may vary. If no compatible source
exists, the job collects a new shared warmup with all three statistics signals
and then runs its continuations. The original config file is never edited.

After a compatible replacement is verified, incompatible old snapshots for the
same pair/covered warmup targets are removed to reclaim disk space. A source
pinned by a running/retryable failed job or useful to another pending experiment is retained;
different warmup targets are retained as well. Deletion is restricted to shared
warmup directories under this repository's `ckpt/`. A dry run never deletes
checkpoints. Catalog inspection can also be requested explicitly:

```bash
python warmup_catalog.py --gpu-type A6000 --refresh
```

Runtime data is stored under `training_queue/` (excluded from Git):

| File | Purpose |
| --- | --- |
| `queue.sqlite3` | Authoritative pending/running/completed/failed jobs, priorities, seed ownership and execution events |
| `warmup_inventory.json` | Inspected warmups, compatibility metadata and excluded-source reasons |
| `leases/` | Process-held locks preventing duplicate claims/recovery |
| `requests/` | Frozen collector configs |
| `imported_warmups/` | Metadata/config adapters and symlinks for reusable legacy snapshots |
| `legacy_backup/` | Original per-card selections, queues and history from migration |

### Shared warmup inventory across A6000, RTX 3090 and TITAN RTX

`warmup_registry.py` collects **game → GPU model → seed → snapshots** and
exchanges the inspection results through the `warmup-inventory` Git branch on
`origin`. This branch contains only JSON metadata. Checkpoint/replay payloads
stay on their original machine. Synchronization uses a temporary repository,
so the source checkout's branch, staged changes and uncommitted edits are kept.
The script must first be available in each machine's source checkout.

A collection **round** freezes the configuration and requested warmup targets,
so every machine uses the same compatibility criteria even if its local YAML
differs. Create the round **once, on the first machine**:

```bash
conda activate storm
python warmup_registry.py start --round warmups-20261123 \
  --gpu-types A6000 RTX3090 titan
```

The default profile is `config_files/STORM.yaml` with the scalar warmup target
and no pair sweep, matching the default enqueue command. `start` also accepts
`--config`, `--warmup-offset-pairs`, `--hash-target-pairs`, and repeated
`--set KEY VALUE`, just like `enqueue_train.py`. Use a new round name to change
the profile or to collect fresh results from all three models again. `start`
alone records no scan, and old rounds never satisfy a new round's coverage.

Run the following command **on each corresponding machine**, starting with the
machine that created the round. Git authentication and a Git author identity
must already be configured for the repository's `origin` remote.

```bash
# A6000 machine: publish the round and its local inspection.
python warmup_registry.py sync --round warmups-20261123 --gpu-type A6000

# RTX 3090 machine: fetch the same round, inspect local files, and publish.
python warmup_registry.py sync --round warmups-20261123 --gpu-type RTX3090

# TITAN RTX machine: the same operation completes the three-model collection.
python warmup_registry.py sync --round warmups-20261123 --gpu-type titan

# Back on any machine: fetch the final overview without rescanning/publishing.
python warmup_registry.py pull --round warmups-20261123
python warmup_registry.py report --round warmups-20261123 --json
```

`sync` fetches existing reports before scanning, commits only this machine's
report and the round manifest, and pushes without force. Concurrent writers
are retried against the updated branch. Failed publication preserves the
local scan; retry `sync` after fixing connectivity/authentication. `--remote`
and `--branch` select a different Git remote or dedicated inventory branch.
No source-code commits are included in this push. Ordinary `git pull` of
`main` does not refresh the inventory; use the script's `pull` command.

The display lists pending GPU models, each machine's scan time/publication
status, usable seeds by game/model, incompatible settings, excluded files,
and seed ownership conflicts across **all games**. A seed observed under two
models, or inside another model's reserved range, is flagged and excluded from
the usable count. Conflicting historical seeds are never silently renamed.
Collection completion means at least one scan for each of the three models;
Git completion additionally requires every locally displayed scan to have
been published. These indicators do not hide incompatibilities or conflicts.

`--machine ID` overrides the hostname; use a distinct, stable ID for each
machine, including multiple machines of the same model. GPU indices are not
used. The inventory is accurate **as of each scan**, not a live remote disk
check. Rescanning replaces that machine's report, removing deleted snapshots
from the current inventory. New rounds keep prior round history separate.

Inspection reuses the queue's CPU checkpoint/configuration checks. Checkpoints
remain on the machine that generated them, so `--gpu-type` assigns every local
snapshot to that machine's GPU model. Historical `queue_origin.json` and W&B
GPU logs are not required and do not override this assignment. Reports record
the machine/model as `provenance.kind: machine_location`. After updating an
older scanner, rerun `sync` on any machine whose reports excluded snapshots
for missing GPU history; existing published reports are replaced on that scan.
A structurally valid snapshot can
still be incompatible with the frozen profile. With multiple requested warmup
targets, “usable” means compatible with **at least one** request; the JSON
includes the compatibility result and reason for every request/snapshot.

The scanner does not enqueue training, change game/seed selections, or remove
checkpoints. Local-only inspection and reporting also work offline once the
round manifest is cached:

```bash
python warmup_registry.py scan --round warmups-20261123 --gpu-type A6000
python warmup_registry.py report --round warmups-20261123
```

Local files are in `training_queue/shared_warmups/rounds/ROUND/manifest.json`
and `machines/GPU_TYPE/MACHINE.json` beneath that round directory. They are
excluded from the source branch and published on the dedicated inventory
branch. The initial A6000 inspection found 53 snapshots / 49 valid pairs;
all 49 pairs match the default profile, including Freeway's demonstration
snapshots. The extra four snapshots are duplicate game/seed pairs, not
unidentified pairs. Each round's report shows which other machines have
published an inspection.

### Cleaning warmups with missing statistics or broken payloads

`cleanup_warmups.py` re-inspects local files using a cached round's frozen
profile. It selects snapshots missing statistics required by that profile,
and published snapshots with recognized checkpoint/replay corruption. Other
warmup targets (such as 30k/70k when requesting 50k), other configuration
variants, and seed ownership conflicts alone are retained. Unknown inspection
errors and directories without publication metadata are retained as well.

Snapshot counts and game/seed counts differ: the RTX 3090 inventory had 264
snapshots for 181 pairs. Its 83 extra snapshots are not all invalid. For this
round, 171 snapshots matched the requested 50k profile, 76 covered 30k/70k
targets, and 17 lacked required statistics. TITAN had 6 statistics-deficient
snapshots across 6 pairs, including one pair with another compatible snapshot.
The cleanup selects the 17 / 6 deficient snapshots, subject to a fresh local
inspection and protection for queued/retryable work.

After updating the source code on each machine, use its own GPU type. The
default command previews targets, reasons and disk size:

```bash
# On the RTX 3090 machine:
python cleanup_warmups.py --round warmups-20261123 --gpu-type RTX3090

# On the TITAN machine:
python cleanup_warmups.py --round warmups-20261123 --gpu-type titan
```

Add `--apply` to delete, then publish that machine's updated inventory:

```bash
# On the RTX 3090 machine:
python cleanup_warmups.py --round warmups-20261123 --gpu-type RTX3090 --apply
python warmup_registry.py sync --round warmups-20261123 --gpu-type RTX3090

# On the TITAN machine:
python cleanup_warmups.py --round warmups-20261123 --gpu-type titan --apply
python warmup_registry.py sync --round warmups-20261123 --gpu-type titan
```

Use the same `--machine ID` as inventory collection if you overrode the hostname.
Only real `ckpt/RUN/shared_warmup_STEP` directories are eligible. Pending,
running, and retryable failed jobs protect their pinned sources and sources
compatible with their requests. Files changed since inspection are skipped.
The audit under `training_queue/cleanup/` records candidates, reasons, deletion
results and protected sources. Applying cleanup updates seed-file references,
the local discovery cache and the local machine inventory. It starts no
training; later queued requests collect replacements as needed. After both
machines publish, run `python warmup_registry.py pull --round warmups-20261123`
on A6000 to view the combined result.

### Migrating older queues

For an older checkout, stop its old queue workers and migrate once:

```bash
python queue_train.py migrate --gpu-type A6000
```

Migration preserves seed values and original files in a backup, imports pending
and interrupted jobs, and writes `type_A6000.json`. It refuses migration while
an old worker holds its lock. This workspace's 53 existing snapshots have A6000
model evidence in their saved W&B metadata; the 49 selected game/seed pairs are
now in the A6000 pool. Existing checkpoints were not deleted during migration.

**Legacy shell queues:** pass an explicit queue filename to keep the previous
one-shell-command-per-line behavior. These queues have no permanent game/seed
ownership or interruption recovery; use the default model queue for those features.

```bash
bash 0_train.sh job_queue_random_neighbor.txt
bash 0_train.sh job_queue_random_neighbor.txt --once
bash 0_train.sh job_queue_td_error_only.txt
```

Legacy queue pops use their corresponding `.lock` files. Blank lines and
comments are skipped, and each job has its own shell environment and working
directory. Relative paths resolve from STORM-1. To run the baseline-first
campaign, use `bash baseline_train.sh 0`; its `job_queue_baseline_first.txt`
contains campaign reservations.

### First TD-error-only result for every game

Use the dedicated `config_files/STORM.td_error_only.yaml`. It disables the
warmup/offset sweep, uses a single 50k retrieval warmup, and evaluates the 100k
checkpoint. Edit `td_error_z_score_threshold` there before preparing the queue:

```shell
../miniconda3/envs/storm/bin/python td_error_only_queue.py prepare
bash 0_train.sh job_queue_td_error_only.txt
```

Additional GPUs must explicitly select the same legacy queue, for example
`bash 1_train.sh job_queue_td_error_only.txt`.
Run the preparation command with the STORM Python
environment; its executable is saved in every queue command.

Preparation adds exactly one selected seed per game, with no second-seed jobs.
It chooses the smallest existing Full FLASH seed number for each game from
`results/main_performance_std/paired_scores.csv`, without inspecting its score.
Use `prepare --seed 7` to explicitly select one common seed instead. Same seed
numbers alone do not establish identical warmup trajectories.

The preparation output gives a manifest path under
`results/td_error_only_queue/<campaign>/manifest.json`. Inspect progress with:

```shell
../miniconda3/envs/storm/bin/python td_error_only_queue.py status \
  --manifest results/td_error_only_queue/<campaign>/manifest.json
```

Failed jobs, including zero-exit jobs without a valid final evaluation, are
appended after the remaining games. The default limit is three attempts per
game, set by `prepare --max-attempts`. Each retry starts a fresh run with a new
attempt name and log. A valid local `final_evaluation.json`, rather than an
exit code alone, establishes completion. Per-game locks prevent simultaneous
duplicate jobs when workers share the same filesystem. Repeating `prepare`
adds missing jobs while skipping completed, currently locked, and queued jobs.
Failures reaching the attempt limit remain visible in `status` and their logs.

Preparation freezes the effective YAML in the campaign directory. Edit the
source YAML and run `prepare` again for a new threshold/configuration campaign;
use a separate `--queue` if the previous campaign still has pending work.
The existing GPU-specific queues are separate from this dedicated queue.

### Paired warmup and anchor-offset experiments

Set `JointTrainAgent.Retrieval.warmup_offset_pairs` to the exact pairs to run:

```yaml
JointTrainAgent:
  Retrieval:
    enable: True
    warmup_offset_pairs:
      - [20000, -4]
      - [50000, -2]
      - [50000, 0]
      - [80000, -2]
      - [dynamic_warmup, -2]
    dynamic_warmup_multiplier: 2.6
    dynamic_warmup_delay_steps: 0
```

Each entry is `[warmup_steps, anchor_offset]`; these are explicit pairs, not a
Cartesian product. A nonempty list overrides the single `warmup_steps` for
collection. The collector trains once without retrieval and saves full
model/optimizer/replay/statistics/RNG state at
the first episode boundary at or after each target. Targets reaching the same
boundary share one snapshot. All branches then run sequentially from their own
snapshot to the original `SampleMaxSteps`, using their specified offset and the
other common settings, including `anchor_weight_r`.

`enable: Both` runs retrieval for each pair and one baseline per distinct
requested warmup. Existing named experiment lists also work: each preset is
combined with each pair, with baseline deduplicated per warmup. An empty pair
list preserves the original single-run/Both behavior. Only fixed nonnegative
integer warmups and `dynamic_warmup` (also written `-1`) are supported; offsets
must be integers. `NumEnvs` must be 1, and the actual boundary
must precede both the training end and the selected final evaluation step.

Dynamic detection runs on the same retrieval-free trajectory, using completed
episode returns including terminal rewards. After `min_warmup_steps`, the
detector compares the last five-episode average against the recent smoothed
mean plus `dynamic_warmup_multiplier` times its standard deviation (at least
25 completed episodes are required). It checks only when a new episode ends,
or once on becoming eligible/resuming, and latches the first detection step.
For detection at step `d`, the dynamic target is:

```text
min(max_warmup_steps, max(dynamic_warmup_target_steps, d + dynamic_warmup_delay_steps))
```

Without a detection, the target is `max_warmup_steps`. The collector saves at
the first real episode end (`terminated` or `truncated`, not a life loss) at or
after the target. Thus the actual snapshot may exceed the cap. The cap must
precede the selected final evaluation step. Detection does not enable retrieval
in the collector: it continues until every fixed and dynamic target has a
snapshot, then stops immediately. Multiple dynamic offsets share one snapshot;
fixed and dynamic targets reaching the same boundary also share a snapshot.
With an empty pair list, `warmup_steps: -1` uses the same delay and boundary rule.

Append this to the usual training command to select pairs from the shell:

```shell
JointTrainAgent.Retrieval.enable True JointTrainAgent.Retrieval.warmup_offset_pairs "[[20000, -4], [50000, -2], [50000, 0], [80000, -2]]"
```

The collector uses `runs/<name>_WarmupSweep/` and
`ckpt/<name>_WarmupSweep/`. Its `warmups.json` maps requested targets to actual
snapshot steps. Branch names include the requested warmup, offset and preset,
for example `<name>_w50000_offset_m2_retrieval` or
`<name>_wdynamic_offset_m2_retrieval`. Snapshot creation preserves the
collector's RNG and hash/PCA state. Hash rebuilding happens after restoring a
branch, equally for retrieval and baseline. As with Both, branches start a new
episode; emulator state is not saved.

Saved targets can be reused later with new offsets:

```shell
python -u train.py --resume_warmup ckpt/Pong-17_WarmupSweep \
  JointTrainAgent.Retrieval.enable True \
  JointTrainAgent.Retrieval.warmup_offset_pairs "[[20000, -1], [50000, -3]]"
```

Only targets present in `warmups.json` can be selected; a maximum-step snapshot
cannot reconstruct an earlier unsaved model. Followups keep the original
checkpoints/manifests and use separate job names unless `-n` is provided.
Saved dynamic targets can be reused with new offsets, provided their detection
settings are unchanged. Changing the multiplier, delay, minimum, target or cap
requires a new collection. The index and snapshot metadata record the detection,
target and actual snapshot steps. Pending detection/delay state is also retained
when resuming an interrupted collector.
CSV records include `Anchor Offset` and `Warmup Offset Pairs`; Excel keeps
nondefault warmups/offsets on separate rows. The actual shared boundary is
recorded separately from the requested warmup. Keep a fresh `-n` for a new
collection; `--resume_from` with the same collector name/config can continue a
saved collection checkpoint if all earlier target snapshots are available.

FLASH supports the `RandomNeighor` ablation: it runs hash retrieval and lazy
re-verification, then replaces each verified neighbor with a valid random replay
context while preserving the anchor, final group size, weights and batch rules.
Set `JointTrainAgent.Retrieval.enable RandomNeighor` on an existing training
command, or use `enable: [retrieval, RandomNeighor]` in YAML to compare both from
one shared warmup. From a saved reusable warmup:

```shell
python -u train.py --resume_warmup ckpt/<run>_Shared JointTrainAgent.Retrieval.enable RandomNeighor
```

The run suffix is `_randomneighor`. See [the training guide](../retrieval_training.md)
for sampling, weighting and checkpoint details.

### Shared-warmup hash-bit and target sweep

Set `hash_target_pairs` to the explicit `[hash_bits, target]` pairs to train.
For a 3-by-3 grid, list all nine pairs. `target` includes the anchor.
Use an empty `warmup_offset_pairs` to collect exactly one warmup target:

```yaml
JointTrainAgent:
  Retrieval:
    enable: Both
    warmup_steps: 50000
    warmup_offset_pairs: []
    hash_target_pairs:
      - [9, 4]
      - [9, 8]
      - [9, 16]
      - [10, 4]
      - [10, 8]
      - [10, 16]
      - [12, 4]
      - [12, 8]
      - [12, 16]
```

Launch the usual `train.py` command with this config. Warmup is trained once,
up to the first episode boundary at or after 50k. Each pair then starts from
the same model, optimizer, replay, EMA statistics and RNG snapshot. `Both` adds
one baseline continuation per requested warmup, rather than one per hash/target
pair. `enable: True` runs only the retrieval pairs. All continuations execute
sequentially. A nonempty `warmup_offset_pairs` combines every hash/target pair
with every requested warmup/offset pair.

Reuse an existing WarmupSweep without repeating collection:

```shell
python -u train.py --resume_warmup ckpt/Pong-17_WarmupSweep \
  -n Pong-17-grid \
  JointTrainAgent.Retrieval.enable Both \
  JointTrainAgent.Retrieval.warmup_offset_pairs '[[50000, -2]]' \
  JointTrainAgent.Retrieval.hash_target_pairs '[[9, 4], [10, 8], [12, 16]]'
```

For a reusable ordinary `_Shared` checkpoint, supply
`JointTrainAgent.Retrieval.enable "['retrieval', 'baseline']"`, omit the
`warmup_offset_pairs` override, and use the same `hash_target_pairs` override.
Snapshot metadata supplies the original environment and seed; each snapshot
must contain the EMA signals required by the selected experiments.

Before any hash lookup in a grid branch, the old projection, buckets and
queued anchors are discarded while EMA statistics are restored. The new
`hash_bits` controls both the bit vector and the PCA projection (or the initial
random projection when `use_pca: False`); the complete replay index is rebuilt.
Rebuild randomness is isolated from the restored training RNG, so changing
the PCA width does not itself shift the branch's training RNG at activation.
Subsequent learning trajectories can differ as retrieval changes.

Names include `_h9_t4`, `_h10_t8`, etc.; W&B records each branch's effective
config and Excel keeps grid runs on separate hash/target rows. The original
snapshots are retained. Sweep followups use their own manifests/results under
`followups/`. Reusing a WarmupSweep without a new `-n` assigns a fresh job name;
use a fresh `-n` for each ordinary shared-checkpoint followup.

`hash_bits` must be an integer from 1 through 62 and `target` a positive integer.
Duplicate pairs are rejected. Each explicit pair overrides the scalar settings
for that branch. Keep `max_anchors`, `multiplier`, `max_contexts` and weighting
settings fixed when comparing pairs; groups remain subject to `max_contexts`.
The `target1` named preset conflicts with a pair's target; use a pair with target
1 instead. An empty `hash_target_pairs` preserves the existing sweep behavior.

### TD-error-only anchor ablation

Set `score_combination: td_error_only` to select anchors using only
`ReLU(z_TD)`, where `z_TD` normalizes the absolute one-step TD error with the
preceding per-environment EMA mean and variance. The value-difference term is
omitted. Sequence-maximum selection, anchor offsets, hash-neighbor retrieval,
group weights and imagination-batch allocation follow the existing settings.

```yaml
JointTrainAgent:
  Retrieval:
    enable: True
    trigger_mode: z_score
    score_combination: td_error_only
    td_error_z_score_threshold: 3.5
    warmup_steps: 50000
    warmup_offset_pairs: []
```

Edit `td_error_z_score_threshold` in `config_files/STORM.yaml` to tune this
mode's threshold. `z_score_threshold` and `additive_z_score_threshold` continue
to control the multiplicative and additive modes, respectively. The value 3.5
is a starting setting; it does not establish a matched trigger frequency.
Use `Retrieval/triggered_anchors_step` to compare counts with Full FLASH.

For a paired comparison from one 50k warmup, leave the common
`score_combination: multiply` and set:

```yaml
JointTrainAgent:
  Retrieval:
    enable: [retrieval, td_error_only]
    score_combination: multiply
    td_error_z_score_threshold: 3.5
    warmup_steps: 50000
    warmup_offset_pairs: []
```

The `td_error_only` preset overrides only `enable` and `score_combination`.
Its run suffix is `_td_error_only`. A saved reusable warmup can also launch it:

```shell
python -u train.py --resume_warmup ckpt/<run>_Shared \
  JointTrainAgent.Retrieval.enable td_error_only \
  JointTrainAgent.Retrieval.td_error_z_score_threshold 3.5
```

TD-error-only needs only TD-error EMA statistics when resuming. Shared warmup
collection still saves the union needed by its selected experiments;
`save_warmup: True` retains both value statistics for future comparisons.
W&B records the scoring mode and threshold in the effective config. CSV exports
`TD Error Z Score Threshold`, and Excel separates TD-error-only results by
threshold so threshold sweeps are not averaged together.

Retrieval groups support fixed or relative anchor weights:

```yaml
JointTrainAgent:
  Retrieval:
    anchor_weight: 0.0625
    anchor_weight_r: null
```

With `anchor_weight_r: null` (the default, including older configs), the anchor
gets `anchor_weight` and each of its `K` final neighbors gets
`(1 - anchor_weight) / K`. Set `anchor_weight_r` to a finite positive number `r`
to use `r / (K + r)` for the anchor and `1 / (K + r)` for each neighbor instead.
The relative setting takes precedence over `anchor_weight`; `r=1` gives equal
weights and `r=2` makes the anchor twice as heavy as each neighbor. Counts are
taken after re-verification, validity filtering and `max_contexts` truncation.
Each group's weights still sum to one; a lone anchor always has weight one.
The same rules apply to `RandomNeighor`.

Append `JointTrainAgent.Retrieval.anchor_weight_r 2` to a training command,
including a `--resume_warmup` command, to select relative weights. Append
`JointTrainAgent.Retrieval.anchor_weight_r None` to return to fixed weights.
Restoring a checkpoint keeps the receiving run's weighting settings.
W&B stores `anchor_weight_r`, CSV exports `Anchor Weight R`, and Excel lists relative runs on
separate rows such as `target: 16 (anchor r: 2)`.

STORM retrieval uses a per-call CPU snapshot of termination flags and batched
replay reads to reduce CUDA synchronization while preserving sampling and model
input order. See [retrieval execution optimizations](diagnostics/retrieval_execution_optimizations.md)
for equivalence checks, measured timings and reproduction commands.

You can use Tensorboard to visualize the training curve and the imagination videos:
```shell
 chmod +x TensorBoard.sh
 ./TensorBoard.sh
 ```


## Reproducing Speed Metrics
To reproduce the speed metrics mentioned in the paper, please consider the following:
- Hardware requirements: NVIDIA GeForce RTX 3090 with a high frequence CPU, we use `11th Gen Intel(R) Core(TM) i9-11900K` in our experiments. Low frequence CPUs may lead to a GPU idle and slow down the traning. To make full use of a powerful GPU, one can traing several agents at the same time on one device.
- Software requiements: `PyTorch>=2.0.0` is required.

## Troubleshooting
### Mixed precision on other devices
- By default (`STORM_AMP_DTYPE=auto`, or unset), **NVIDIA TITAN RTX** uses FP16 and other GPUs use BF16. Detection respects `CUDA_VISIBLE_DEVICES`; there is no general fallback for older GPUs such as V100. The policy is defined in `sub_models/precision.py`.
- To use the same FP16 path on an A6000, RTX 3090, or another GPU, prefix that job with `STORM_AMP_DTYPE=fp16`. For example, put the following single line in `job_queue_A6000.txt` (run workers from the `STORM` directory):
  ```shell
  STORM_AMP_DTYPE=fp16 python -u train.py -n Pong-fp16-seed1 -seed 1 -config_path config_files/STORM.yaml -env_name ALE/Pong-v5 -trajectory_path Pong.pkl
  ```
  Existing `0_train.sh` and other workers pass this through without changes. The prefix applies to that command and its child processes, including shared-warmup branches; it does not change later queue jobs. For a separately queued resume/followup or standalone evaluation, repeat the prefix. Use `STORM_AMP_DTYPE=bf16` to force BF16 or `STORM_AMP_DTYPE=auto` for GPU detection.
- The selected dtype applies to world-model/agent autocast, imagination buffers, and KV caches. FP16 attention uses `-6e4` for masking to prevent overflow; BF16 keeps `-1e9`. This selects mixed precision, not a conversion of all model/checkpoint tensors. Optimizer settings are unchanged.
- The startup message `AMP dtype: torch.float16` confirms FP16 selection. Changes apply to newly started training processes; an already-running process keeps its loaded code.
- On devices like the NVIDIA A100, using bfloat16 may slow down the training. In this case, you can toggle the `self.use_amp = True` option in both `agents.py` and `sub_models/world_models.py`.

### Windows and WSL
We've recently observed if one **clones the repo** from `Powershell` and then calls `train.sh` under `WSL shell`, then it may throw an error related to arg parse. This may be due to invisible newlines in the files somehow generated when cloning with git. The solution is to download the zip or clone directly inside `WSL`.

### ALE error
pip install "gymnasium[atari,accept-rom-license]==0.29.1"

## Code references
We've referenced several other projects during the development of this code:
- [Attention is all you need pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) For Transformer structure, attention operation, and other building blocks.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) For trainable positional encoding.
- [DreamerV3](https://github.com/danijar/dreamerv3) For Symlog loss, layer & kernel configuration in VAE.

## Bibtex

```
@inproceedings{
    zhang2023storm,
    title={{STORM}: Efficient Stochastic Transformer based World Models for Reinforcement Learning},
    author={Weipu Zhang and Gang Wang and Jian Sun and Yetian Yuan and Gao Huang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=WxnrX42rnS}
}
```
