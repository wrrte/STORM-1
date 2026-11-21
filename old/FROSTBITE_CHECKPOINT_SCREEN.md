# Frostbite return screening and critic alignment

Activate the `storm` environment and run this command from STORM-1:

```bash
python frostbite_checkpoint_screen.py --gpus 3 4 5 \
  --exclude-run Frostbite-life_done-wm_2L512D8H-100k-seed6000_X \
  --exclude-run Frostbite-life_done-wm_2L512D8H-100k-seed6000_O
```

This excludes the two already tested controls assigned to GPUs 0 and 2. It currently finds
16 other saved checkpoints. Each GPU runs one checkpoint at a time. The worker first
evaluates 20 complete Frostbite episodes with BF16 and evaluation seed 0. If the mean
return is **strictly greater than 1,000**, it immediately evaluates critic alignment on
that same GPU. The other GPUs continue processing their own checkpoints.

The script discovers nonzero `world_model_<step>.pth` / `agent_<step>.pth` pairs,
`*_final.pth` pairs, and `shared_warmup_*/training_state.pt`. It uses the saved config
under `runs/<run>/config.yaml`. It restores only model weights from shared states;
it does not resume training or change the saved checkpoints. Zero-step models are excluded.
For intermediate and shared checkpoints, the new return belongs to that specific saved
state; it is not labeled as the return of a completed 100k run.

The initial invocation evaluates each selected checkpoint under this common return
protocol. Historical TensorBoard and manual CSV scores are not used as the cache.
Afterward, rerunning the same command reuses this pipeline's saved returns when the
checkpoint files, config, and return evaluation parameters match. Completed alignment
results are also reused. Changed context data, templates, seed, or bootstrap settings
produce a new alignment subdirectory. If raw values were saved but analysis did not
finish, they are reused when retrying the analysis.

The alignment phase reuses `results/old/seed_3710/shared_contexts.npz`, the existing 17
igloo templates, and latent seed 2027. It calculates both aggregation orders, each with
Spearman ρ, Kendall τ_b, and episode-bootstrap 95% CIs from 10,000 replicates:

1. Average the correlations computed separately for each observation-action context.
2. Average the predicted values at each stage, then correlate that 17-point profile with stage.

Results are saved under `results/frostbite_checkpoint_screen/`:

- `manifest.json`: candidate files, configs, parameters, and requested GPUs.
- `summary.json`: completed jobs ordered by measured mean return; `qualified` lists those above the threshold.
- `<checkpoint-id>/return_evaluation.json`: exact evaluation parameters and all episode returns.
- `<checkpoint-id>/worker.log`: return and alignment output, including errors.
- `<checkpoint-id>/alignment_<id>/report_ko.md`: both alignment estimates and CIs.
- `<checkpoint-id>/alignment_<id>/statistics.json`: unrounded estimates and uncertainty details.

The console prints which GPU is evaluating each checkpoint, its return, and when
alignment begins. A failed checkpoint is recorded and the GPU proceeds to the next one.
Ctrl+C stops workers and their environment subprocesses; the same command can resume
from completed results. A second coordinator using the same output directory is rejected.

Useful options:

```bash
# List candidates without requiring CUDA or importing PyTorch.
python frostbite_checkpoint_screen.py --dry-run

# Evaluate only a particular run, including its intermediate/shared/final states.
python frostbite_checkpoint_screen.py --gpus 3 4 5 --run Frostbite-6000_Shared

# Choose another threshold, evaluation precision, or output directory.
python frostbite_checkpoint_screen.py --gpus 3 4 5 --min-return 1500 \
  --precision fp32 --output results/frostbite_checkpoint_screen_fp32
```

`--episodes`, `--return-seed`, `--value-seed`, `--bootstrap`, `--data`, and
`--template-dir` are also configurable. There is no five-model limit: all selected
checkpoints are screened, and all that exceed the threshold receive alignment evaluation.

The coordinator and its scheduling/cache tests require only Python's standard library:

```bash
python -m unittest -v test_frostbite_checkpoint_screen
```

Workers need the existing STORM, Atari/ALE, OpenCV, SciPy, and Matplotlib dependencies.
