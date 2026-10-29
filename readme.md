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
```

Each entry is `[warmup_steps, anchor_offset]`; these are explicit pairs, not a
Cartesian product. A nonempty list overrides the single `warmup_steps` for
collection. The collector trains once without retrieval through the largest
requested warmup and saves full model/optimizer/replay/statistics/RNG state at
the first episode boundary at or after each target. Targets reaching the same
boundary share one snapshot. All branches then run sequentially from their own
snapshot to the original `SampleMaxSteps`, using their specified offset and the
other common settings, including `anchor_weight_r`.

`enable: Both` runs retrieval for each pair and one baseline per distinct
requested warmup. Existing named experiment lists also work: each preset is
combined with each pair, with baseline deduplicated per warmup. An empty pair
list preserves the original single-run/Both behavior. Only fixed nonnegative
integer warmups and integer offsets are supported; `-1` dynamic warmup and `.5`
offsets cannot be used in a pair. `NumEnvs` must be 1, and the actual boundary
must precede both the training end and the selected final evaluation step.

Append this to the usual training command to select pairs from the shell:

```shell
JointTrainAgent.Retrieval.enable True JointTrainAgent.Retrieval.warmup_offset_pairs "[[20000, -4], [50000, -2], [50000, 0], [80000, -2]]"
```

The collector uses `runs/<name>_WarmupSweep/` and
`ckpt/<name>_WarmupSweep/`. Its `warmups.json` maps requested targets to actual
snapshot steps. Branch names include the requested warmup, offset and preset,
for example `<name>_w50000_offset_m2_retrieval`. Snapshot creation preserves the
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
