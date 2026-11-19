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

### GPU job queues

Activate the STORM Python environment, then run `bash 0_train.sh` (or
`1_train.sh` through `7_train.sh`) to consume ordinary shell commands, one per
line. All workers default to the shared `job_queue_td_error_only.txt`,
regardless of GPU model. Relative paths in jobs resolve from the STORM-1 directory. Queue pops
use the corresponding `.lock` file, so multiple GPUs can share a queue.
Blank lines and comments are skipped. Failed jobs are reported and the worker
continues; each job has its own shell environment and working directory.

Pass a queue path to select a separate queue, for example:

```bash
bash 0_train.sh job_queue_random_neighbor.txt
bash 0_train.sh job_queue_random_neighbor.txt --once
```

`--once` executes at most one queued job and exits, including when the queue is
empty. `--max-jobs N` exits after N jobs. These options return the last job's
exit status. To run the baseline-first campaign, use `bash baseline_train.sh 0`
instead; its `job_queue_baseline_first.txt` contains campaign reservations.

### First TD-error-only result for every game

Use the dedicated `config_files/STORM.td_error_only.yaml`. It disables the
warmup/offset sweep, uses a single 50k retrieval warmup, and evaluates the 100k
checkpoint. Edit `td_error_z_score_threshold` there before preparing the queue:

```shell
../miniconda3/envs/storm/bin/python td_error_only_queue.py prepare
bash 0_train.sh
```

Additional GPUs use the same default queue, for example `bash 1_train.sh`.
Passing a queue path explicitly selects that queue instead.
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
