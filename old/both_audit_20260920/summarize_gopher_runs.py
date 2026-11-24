"""Summarize downloaded W&B evidence without further network requests."""

import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GROUPS = [
    (6000, "s12t06bx", "ori7lpmm", "w30fmuzh"),
    (6010, "p9dfsvag", "tepmvj6g", "3d8lc1uk"),
    (9999, "9uwipoit", "uz4318p4", "hrca6yyg"),
]
REWARD = "sample/ALE/Gopher-v5_reward"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex="col", constrained_layout=True)
    rows = []
    colors = ("#666666", "#c45c2a", "#247aa5")
    for column, (seed, parent_id, on_id, off_id) in enumerate(GROUPS):
        runs = [json.loads((ROOT / run_id / "run_data.json").read_text()) for run_id in (parent_id, on_id, off_id)]
        parent_config = copy.deepcopy(runs[0]["config"])
        parent_config["JointTrainAgent"]["Retrieval"].pop("enable")
        first_losses = []
        for stage, color, run in zip(("Warmup", "Retrieval True", "Retrieval False"), colors, runs):
            cfg = copy.deepcopy(run["config"])
            cfg["JointTrainAgent"]["Retrieval"].pop("enable")
            assert cfg == parent_config, (seed, stage, "unexpected config difference")
            first = run["first_updates"][0]
            # Warmup starts updating at environment step 1024; any earlier
            # episode-only logs account for its W&B step offset. Children log
            # every collection/update iteration from environment step 50000.
            offset = 1024 - first["_step"] if stage == "Warmup" else 50000
            if stage != "Warmup":
                first_losses.append(first["WorldModel/total_loss"])
            episodes = run["episode_rewards"]
            steps = np.array([row["_step"] + offset for row in episodes])
            values = np.array([row[REWARD] for row in episodes], dtype=float)
            smoothed = np.array([values[max(0, i-9):i+1].mean() for i in range(len(values))])
            axes[0, column].scatter(steps / 1000, values, s=8, color=color, alpha=.2)
            axes[0, column].plot(steps / 1000, smoothed, color=color, label=stage, lw=2)
            history = [row for row in run["history"] if "ActorCritic/entropy_loss" in row]
            hsteps = np.array([row["_step"] + offset for row in history])
            entropy = np.array([row["ActorCritic/entropy_loss"] for row in history])
            axes[1, column].plot(hsteps / 1000, entropy, color=color, lw=1, alpha=.7)
            evals = [value for key, value in run["summary"].items() if key.startswith("eval/episode_return_")]
            rows.append({
                "seed": seed, "stage": stage, "run_id": run["id"], "gpu": run["metadata"]["gpu"],
                "episode_count": len(values), "first_10_train_mean": values[:10].mean(),
                "last_10_train_mean": values[-10:].mean(),
                "eval_mean": run["summary"].get("eval/episode_avg_return"),
                "eval_episode_count": len(evals),
                "first_world_model_loss": first["WorldModel/total_loss"],
            })
        assert first_losses[0] == first_losses[1], (seed, first_losses)
        axes[0, column].set_title(f"Gopher, seed {seed}")
        for axis in axes[:, column]:
            axis.axvline(50, color="black", ls="--", lw=1, alpha=.55)
            axis.grid(alpha=.15)
            axis.set_xlim(0, 102)
        axes[1, column].set_xlabel("Environment steps (thousands)")
        axes[1, column].set_ylim(0, 2.15)
    axes[0, 0].set_ylabel("Training return (10-episode moving mean)")
    axes[1, 0].set_ylabel("Actor entropy (sampled W&B history)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Shared warmup and resumed branches; dashed line = branch at 50k\nTraining returns are not fixed-seed evaluation scores", fontsize=12)
    fig.savefig(ROOT / "gopher_training_curves.png", dpi=180)
    fig.savefig(ROOT / "gopher_training_curves.pdf")
    with (ROOT / "gopher_summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
