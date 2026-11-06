"""Replay dynamic detection on six saved retrieval-free seed-3333 traces.

Run with the STORM Python environment. Reads existing TFRecord event files;
does not import training entry points, access W&B remotely, or launch training.
"""
import ast
import bisect
import csv
import json
import os
from pathlib import Path
import struct

import numpy as np
import pandas as pd
from tensorboard.compat.proto.event_pb2 import Event

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
GAMES = ("Alien", "Assault", "Boxing", "Breakout", "Pong", "Seaquest")
COEFFICIENTS = (1.5, 2.0, 2.6, 3.2)


def read_scalars(directory, tags):
    needles = tuple(tag.encode() for tag in tags)
    values = {tag: [] for tag in tags}
    for path in sorted(directory.glob("events.out.tfevents.*")):
        # TFRecord framing: length, length CRC, payload, payload CRC. Unrelated
        # records need no protobuf conversion. Files are local completed logs;
        # CRCs are skipped, but record lengths and protobuf fields are checked.
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            while stream.tell() < size:
                header = stream.read(12)
                if len(header) != 12:
                    raise ValueError(f"Incomplete TFRecord header: {path}")
                length = struct.unpack("<Q", header[:8])[0]
                if length > size - stream.tell() - 4:
                    raise ValueError(f"Incomplete TFRecord payload: {path}")
                raw = stream.read(length)
                stream.read(4)
                if not any(needle in raw for needle in needles):
                    continue
                event = Event.FromString(raw)
                for value in event.summary.value:
                    if value.tag in values and value.HasField("simple_value"):
                        values[value.tag].append((event.wall_time, event.step, value.simple_value))
    return {tag: sorted(rows) for tag, rows in values.items()}


def load_detector():
    tree = ast.parse((ROOT / "train.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_dynamic_warmup")
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<STORM detector>", "exec"), namespace)
    return namespace["check_dynamic_warmup"]


def analyze_game(game, detector):
    directory = ROOT / "runs" / f"{game}-3333_WarmupSweep"
    config = __import__("yaml").safe_load((directory / "config.yaml").read_text())
    assert config["JointTrainAgent"]["NumEnvs"] == 1
    tag = f"sample/ALE/{game}-v5_reward"
    dense_tag = "Retrieval/triggered_anchors_step"
    anchor_tag = "Retrieval/shared_warmup_ended_at_step"
    source = read_scalars(directory, (tag, dense_tag, anchor_tag))
    dense, anchors, rewards = source[dense_tag], source[anchor_tag], source[tag]
    times = [r[0] for r in dense]
    offsets = []
    for wall_time, _, step in anchors:
        i = bisect.bisect_left(times, wall_time)
        if i < len(dense):
            offsets.append(int(step) + 1 - dense[i][1])
    if not offsets or len(set(offsets)) != 1:
        raise ValueError(f"Cannot establish a consistent environment-step mapping: {game}, {offsets}")
    offset = offsets[0]
    assert all(a[1] < b[1] for a, b in zip(dense, dense[1:])), "Resumed/duplicate dense ordinals require separate handling"
    rows = []
    for wall_time, _, reward in rewards:
        i = bisect.bisect_left(times, wall_time)
        if i == len(dense):
            raise ValueError(f"No subsequent training event for episode return: {game}")
        # A return is logged before that iteration's world-model update. Before
        # the first update, exact episode steps are unknown; these early points
        # are marked and cannot affect detection after the 5k minimum.
        rows.append(dict(step=dense[i][1] + offset, reward=reward,
                         early_step_approximation=wall_time < times[0]))
    frame = pd.DataFrame(rows)
    frame["episode"] = np.arange(1, len(frame) + 1)
    smooth = frame["reward"].rolling(5, min_periods=1).mean()
    mean = smooth.rolling(20, min_periods=5).mean()
    std = smooth.rolling(20, min_periods=5).std()
    z = (smooth - mean) / std.replace(0, np.nan)
    frame["smooth5"] = smooth
    frame["recent20_mean"] = mean
    frame["recent20_std"] = std
    frame["z"] = z
    min_step = config["JointTrainAgent"]["Retrieval"].get("min_warmup_steps", 5000)
    max_step = config["JointTrainAgent"]["Retrieval"].get("max_warmup_steps", 90000)
    observed_end = int(anchors[-1][2])
    steps = frame["step"].tolist()
    history = frame["reward"].tolist()
    candidates = sorted({min_step, *(step for step in steps if step >= min_step)})
    detections = {}
    for coefficient in COEFFICIENTS:
        hit = None
        for step in candidates:
            if step > observed_end or step >= max_step:
                break
            count = bisect.bisect_right(steps, step)
            if count < 25:
                continue
            result = detector(history[max(0, count - 200):count], multiplier=coefficient)
            assert result == bool(z.iloc[count - 1] > coefficient)
            if result:
                hit = dict(step=int(step), episode=int(count), reward=history[count - 1],
                           smooth5=float(smooth.iloc[count - 1]), z=float(z.iloc[count - 1]))
                break
        detections[str(coefficient)] = hit
    bins = []
    for lo, hi in ((0, 20000), (20000, 40000), (40000, 50000), (50000, 60000), (60000, 80000)):
        sample = frame[(frame.step >= lo) & (frame.step < hi)]
        bins.append(dict(start=lo, end=min(hi, observed_end), episodes=len(sample),
                         mean=float(sample.reward.mean()) if len(sample) else None,
                         median=float(sample.reward.median()) if len(sample) else None,
                         max=float(sample.reward.max()) if len(sample) else None))
    frame.to_csv(OUT / f"{game}_episodes.csv", index=False)
    summary = dict(game=game, seed=3333, episodes=len(frame), observed_end=observed_end,
                   episode25_step=int(steps[24]) if len(steps) >= 25 else None,
                   first_eligible_step=max(min_step, int(steps[24])) if len(steps) >= 25 else None,
                   environment_step_offset=offset, independently_checked_offsets=offsets,
                   detections=detections, reward_bins=bins)
    return frame, summary


def plot_results(frames, summaries):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/storm_dynamic_warmup_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for ax, frame, summary in zip(axes.flat, frames, summaries):
        x = frame.step / 1000
        ax.plot(x, frame.reward, color="0.65", lw=.7, alpha=.7, label="Episode return")
        ax.plot(x, frame.smooth5, color="#1c6da5", lw=1.7, label="Last 5 episodes mean")
        hit = summary["detections"]["2.6"]
        if hit:
            ax.axvline(hit["step"] / 1000, color="#c83c3c", ls="--", lw=1.3, label="First detection: c=2.6")
        if summary["episode25_step"] is not None:
            ax.axvline(summary["episode25_step"] / 1000, color="0.35", ls=":", lw=1, label="25th completed episode")
        ax.set(title=f"{summary['game']} (seed 3333, {summary['episodes']} episodes)",
               xlabel="Environment steps (thousands)", ylabel="Training episode return")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Retrieval-free shared warmup reward curves (seed 3333)", fontsize=13)
    fig.savefig(OUT / "reward_curves.png", dpi=160)
    plt.close(fig)

    frame = frames[0]
    summary = summaries[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(frame.step / 1000, frame.reward, color="0.65", lw=.8, label="Episode return")
    axes[0].plot(frame.step / 1000, frame.smooth5, color="#1c6da5", lw=2, label="Last 5 episodes mean")
    axes[0].plot(frame.step / 1000, frame.recent20_mean, color="#da8a19", lw=1.3, label="Recent 20 smoothed values mean")
    axes[0].set(title="Alien: local reward improvement and subsequent fluctuations", ylabel="Return")
    axes[0].legend(loc="upper left")
    colors = ("#7b9970", "#da8a19", "#c83c3c", "#8557aa")
    eligible = frame.episode >= 25
    axes[1].plot(frame.loc[eligible, "step"] / 1000, frame.loc[eligible, "z"], color="#1c6da5", lw=1.2)
    for c, color in zip(COEFFICIENTS, colors):
        axes[1].axhline(c, color=color, ls=":", lw=1, label=f"c={c}")
        hit = summary["detections"][str(c)]
        if hit:
            axes[1].plot(hit["step"] / 1000, hit["z"], "o", color=color)
    axes[1].set(xlabel="Environment steps (thousands)", ylabel="Local standardized reward signal")
    axes[1].legend(loc="upper left", ncol=4)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "alien_detail.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    detector = load_detector()
    frames, summaries = [], []
    for game in GAMES:
        frame, summary = analyze_game(game, detector)
        frames.append(frame)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="") as stream:
        columns = ["game", "seed", "episodes", "observed_end", "episode25_step"]
        columns += [f"detected_step_c{c}" for c in COEFFICIENTS]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary[key] for key in columns[:5]}
            row.update({f"detected_step_c{c}": (summary["detections"][str(c)] or {}).get("step")
                        for c in COEFFICIENTS})
            writer.writerow(row)
    plot_results(frames, summaries)
    print("Artifacts:", OUT, flush=True)
