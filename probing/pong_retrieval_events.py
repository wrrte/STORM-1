"""Render saved anchor/neighbor trajectories, without model inference.

Choose neighbor numbers from an existing analysis, keeping its exact retrieval::

    python probing/pong_retrieval_events.py \
        --analysis results/pong_retrieval_analysis \
        --neighbors 2 5 11 --output results/pong_selected_events

Anchor is always included. Each trajectory is a row with time running left to
right, matching all_point_events.png.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def event_id(rank):
    return "anchor" if rank == 0 else f"neighbor_{rank:02d}"


def collect_point_events(replay, records, selected, past_steps=2, future_steps=6):
    """Copy actual replay frames, stopping at recorded termination boundaries.

    An action and reward at offset k belong to the outgoing transition from
    obs[k]. Invalid frames remain masked; zero-filled storage is never plotted.
    """
    if past_steps < 0 or future_steps < 1:
        raise ValueError("past_steps must be nonnegative and future_steps positive")
    offsets = np.arange(-past_steps, future_steps + 1, dtype=np.int64)
    shape = (len(records), len(offsets))
    data = dict(
        offsets=offsets,
        obs=np.zeros((*shape, *replay.obs_buffer.shape[2:]), dtype=np.uint8),
        obs_valid=np.zeros(shape, dtype=bool),
        pointers=np.full(shape, -1, dtype=np.int64),
        actions=np.full(shape, -1, dtype=np.int64),
        rewards=np.full(shape, np.nan, dtype=np.float32),
        termination=np.zeros(shape, dtype=bool),
        missing_reason=np.full(shape, "", dtype="U96"),
    )
    for rank, record in enumerate(records):
        pointer, env = record["pointer"], record["env"]
        origin = (pointer - replay.oldest) % replay.capacity
        for column, offset in enumerate(offsets):
            position = origin + int(offset)
            if not 0 <= position < replay.length:
                data["missing_reason"][rank, column] = "outside saved replay"
                continue
            # Include the terminal transition's starting observation, but never
            # mistake a reset observation for its unsaved terminal successor.
            between = (np.arange(min(origin, position), max(origin, position)) + replay.oldest) % replay.capacity
            if np.any(replay.termination_buffer[between, env] > .5):
                data["missing_reason"][rank, column] = "recorded termination boundary; final observation unavailable"
                continue
            current = (replay.oldest + position) % replay.capacity
            data["obs"][rank, column] = replay.obs_buffer[current, env]
            data["obs_valid"][rank, column] = True
            data["pointers"][rank, column] = current
            data["actions"][rank, column] = int(replay.action_buffer[current, env])
            data["rewards"][rank, column] = replay.reward_buffer[current, env]
            data["termination"][rank, column] = replay.termination_buffer[current, env] > .5
    metadata = dict(
        version=1, past_steps=past_steps, future_steps=future_steps,
        event_ids=[event_id(rank) for rank in range(len(records))],
        records=records, selected=selected,
        alignment="t=0 is each retrieved observation; not aligned to each trajectory's scoring event",
        action_reward_alignment="a[t], r[t] describe the outgoing transition obs[t] -> obs[t+1]",
        boundary_policy="only saved frames connected to t=0 without crossing recorded termination",
    )
    return data, metadata


def save_event_archive(data, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "point_events.npz", **data)
    (output / "point_events.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def frame_annotations(data, metadata, rank, column):
    offset = int(data["offsets"][column])
    notes = []
    if offset == 0:
        notes.append("ANCHOR" if rank == 0 else "RETRIEVED NEIGHBOR")
    if data["rewards"][rank, column] > 0:
        notes.append("SCORING ACTION")
    if (column > 0 and data["obs_valid"][rank, column - 1]
            and data["rewards"][rank, column - 1] > 0):
        notes.append("AFTER POINT")
    if (rank == 0 and data["pointers"][rank, column] == metadata["selected"]["surprise_pointer"]):
        notes.append("SURPRISE TRANSITION")
    if data["termination"][rank, column]:
        notes.append("TERMINAL ACTION")
    return " / ".join(notes)


def render_event_grid(data, metadata, output_stem, ranks, *, layout="rows", title=None):
    """rows: horizontal timeline per trajectory; columns: trajectories side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = len(data["offsets"])
    columns = len(ranks) if layout == "columns" else steps
    rows = steps if layout == "columns" else len(ranks)
    # Add two title-font heights between rows, with matching extra canvas height.
    # tight_layout's h_pad is expressed in units of the default font size.
    extra_row_gap_points = 20.0
    figure_height = 2.35 * rows + .8 + max(0, rows - 1) * extra_row_gap_points / 72.
    row_padding = 1.08 + extra_row_gap_points / float(plt.rcParams["font.size"])
    fig, axes = plt.subplots(rows, columns, figsize=(2.25 * columns + .7, figure_height), squeeze=False)
    labels = metadata.get("action_labels")
    for group, rank in enumerate(ranks):
        record = metadata["records"][rank]
        name = "Anchor" if rank == 0 else f"Neighbor {rank:02d}"
        for column, offset in enumerate(data["offsets"]):
            axis = axes[column, group] if layout == "columns" else axes[group, column]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            frame_title = f"t{int(offset):+d}"
            if layout == "columns" and column == 0:
                frame_title = f"{name} | env={record['env']}\n" + frame_title
            elif layout == "rows" and column == 0:
                axis.set_ylabel(f"{name}\nenv={record['env']}", fontsize=10)
            if not data["obs_valid"][rank, column]:
                reason = str(data["missing_reason"][rank, column])
                caption = "Outside saved replay" if reason.startswith("outside") else "Termination boundary"
                axis.text(.5, .5, f"Unavailable\n{caption}", ha="center", va="center",
                          fontsize=9, transform=axis.transAxes)
                axis.set_title(frame_title, fontsize=10)
                continue
            pointer = int(data["pointers"][rank, column])
            axis.imshow(data["obs"][rank, column], interpolation="nearest")
            axis.set_title(f"{frame_title} | p={pointer}", fontsize=10)
            action = int(data["actions"][rank, column])
            action_label = f" ({labels[action]})" if labels and 0 <= action < len(labels) else ""
            caption = (f"a={action}{action_label}, r={data['rewards'][rank, column]:+g}\n"
                       + frame_annotations(data, metadata, rank, column))
            axis.set_xlabel(caption, fontsize=8)
            if offset == 0:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#276FBF")
                    spine.set_linewidth(2)
    fig.suptitle(title or "Actual replay trajectories | t=0: retrieved observation\n"
                 "Actions and rewards belong to the outgoing transition from each image", fontsize=11)
    fig.tight_layout(h_pad=row_padding, rect=(0, 0, 1, 1 - .65 / figure_height))
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix("." + extension), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_all_point_events(data, metadata, output):
    """Individual strips and one overview, with stable retrieval-order numbering."""
    save_event_archive(data, metadata, output)
    for rank, name in enumerate(metadata["event_ids"]):
        render_event_grid(data, metadata, output / "events" / name / "point_event", [rank], layout="rows")
    render_event_grid(data, metadata, output / "all_point_events",
                      list(range(len(metadata["records"]))), layout="rows")


def load_saved_events(analysis_dir, past_steps, future_steps, checkpoint_override=None):
    """Use the cached frames; older analyses can use their recorded replay indices."""
    archive_path = analysis_dir / "point_events.npz"
    metadata_path = analysis_dir / "point_events.json"
    if archive_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        with np.load(archive_path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        if data["offsets"][0] <= -past_steps and data["offsets"][-1] >= future_steps:
            keep = (data["offsets"] >= -past_steps) & (data["offsets"] <= future_steps)
            data = {name: values[keep] if name == "offsets" else values[:, keep]
                    for name, values in data.items()}
            metadata.update(past_steps=past_steps, future_steps=future_steps)
            return data, metadata
    # Compatibility with results generated before future trajectories were saved.
    # Reuse their exact selected anchor and neighbor order; no search/model calls.
    report = json.loads((analysis_dir / "analysis.json").read_text())
    if not report.get("transitions") or not report.get("selected"):
        raise ValueError("The analysis has no selected retrieval to visualize")
    checkpoint = Path(checkpoint_override or report["checkpoint"])
    if (checkpoint / "warmup.json").is_file():
        checkpoint = checkpoint / json.loads((checkpoint / "warmup.json").read_text())["checkpoint"]
    replay = SimpleNamespace()
    with np.load(checkpoint / "replay_buffer.npz", allow_pickle=False) as archive:
        for name in ("obs", "action", "reward", "termination"):
            setattr(replay, name + "_buffer", archive[name])
        replay.capacity = int(archive["max_length"]) // int(archive["num_envs"])
        replay.length = min(int(archive["length"]), replay.capacity)
        replay.oldest = ((int(archive["last_pointer"]) + 1) % replay.capacity
                         if replay.length == replay.capacity else 0)
    data, metadata = collect_point_events(replay, report["transitions"], report["selected"], past_steps, future_steps)
    metadata.update(checkpoint=str(checkpoint.resolve()), action_labels=report.get("action_labels"))
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Existing analysis output directory")
    parser.add_argument("--output", type=Path, required=True, help="New empty directory")
    parser.add_argument("--neighbors", type=int, nargs="+", help="1-based neighbor IDs; anchor is always included")
    parser.add_argument("--past-steps", type=int, default=2)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument("--layout", choices=("rows", "columns"), default="rows",
                        help="rows (default): horizontal timelines, matching all_point_events.png; "
                             "columns: vertical timelines side by side")
    parser.add_argument("--checkpoint", type=Path, help="Replay location if the original checkpoint was moved")
    args = parser.parse_args()
    if args.past_steps < 0 or args.future_steps < 1:
        parser.error("past-steps must be nonnegative and future-steps positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory")
    data, metadata = load_saved_events(args.analysis.resolve(), args.past_steps, args.future_steps, args.checkpoint)
    if args.neighbors is None:
        # Conveniently expand an older analysis into every per-neighbor strip.
        save_all_point_events(data, metadata, args.output)
        print(f"Saved every anchor/neighbor trajectory to {args.output}")
        return
    if len(set(args.neighbors)) != len(args.neighbors):
        parser.error("neighbors must not contain duplicate IDs")
    maximum = len(metadata["records"]) - 1
    if any(rank < 1 or rank > maximum for rank in args.neighbors):
        parser.error(f"Neighbor IDs must be between 1 and {maximum}")
    ranks = [0, *args.neighbors]
    save_event_archive(data, metadata, args.output)
    render_event_grid(data, metadata, args.output / "comparison", ranks, layout=args.layout)
    selection = dict(source_analysis=str(args.analysis.resolve()),
                     event_ids=[metadata["event_ids"][rank] for rank in ranks],
                     records=[metadata["records"][rank] for rank in ranks],
                     past_steps=args.past_steps, future_steps=args.future_steps, layout=args.layout)
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Saved anchor + neighbors {args.neighbors} to {args.output / 'comparison.png'}")


if __name__ == "__main__":
    main()
