"""Collect synchronized ALE screenshots/RAM or inspect an existing obs+ram NPZ.

Run from STORM-1: python probing/verify_ram_states.py --game all
See probing/RAM_VERIFICATION.md. No checkpoints, CUDA, or OCAtari required.
"""

import argparse
import csv
import html
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .ram_states import ALE_NAMES, GAMES, SOURCE_URLS, canonical_game, extract_state, numeric_features, read_ale_ram
except ImportError:
    from ram_states import ALE_NAMES, GAMES, SOURCE_URLS, canonical_game, extract_state, numeric_features, read_ale_ram


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"player": "#ff5050", "enemy": "#36dcff", "diver": "#ff66ff",
          "enemy_missile": "#ffb640", "player_missile": "#ffff60", "ball": "#ffff40",
          "floe": "#40e0ff", "igloo": "#f060ff", "oxygen_bar": "#ffff50"}


def collect_rollout(game, steps, seed, frame_skip):
    """Pair the RAM and final RGB screen BEFORE taking the next action.

    obs is STORM-style max(last two frames), resized to 64x64; screen is the
    unpooled final frame. RAM corresponds exactly to screen, not both pooled frames.
    Random actions are for mapping verification, not retrieval benchmarking.
    """
    import cv2
    import gymnasium as gym
    import ale_py

    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
    env = gym.make(f"ALE/{ALE_NAMES[game]}-v5", frameskip=1,
                   repeat_action_probability=0.0, full_action_space=True)
    rng = np.random.default_rng(seed)
    arrays = dict(obs=np.empty((steps, 64, 64, 3), np.uint8),
                  screen=np.empty((steps, 210, 160, 3), np.uint8),
                  ram=np.empty((steps, 128), np.uint8), action=np.empty(steps, np.int32),
                  reward=np.empty(steps, np.float32), termination=np.empty(steps, bool),
                  episode_id=np.empty(steps, np.int32), episode_step=np.empty(steps, np.int32),
                  ale_frame_number=np.empty(steps, np.int64))
    episode_id, episode_step = 0, 0
    try:
        obs, _ = env.reset(seed=seed)
        pooled = obs
        fire = env.unwrapped.get_action_meanings().index("FIRE")
        for i in range(steps):
            arrays["obs"][i] = cv2.resize(pooled, (64, 64), interpolation=cv2.INTER_AREA)
            arrays["screen"][i] = env.unwrapped.ale.getScreenRGB()
            arrays["ram"][i] = read_ale_ram(env.unwrapped.ale)
            arrays["episode_id"][i], arrays["episode_step"][i] = episode_id, episode_step
            arrays["ale_frame_number"][i] = env.unwrapped.ale.getFrameNumber()
            # Start the game/restart a serve, then sample from the full action set.
            action = fire if episode_step == 0 else int(rng.integers(env.action_space.n))
            arrays["action"][i] = action
            frames, reward, ended = [], 0.0, False
            for _ in range(frame_skip):
                obs, rew, terminated, truncated, _ = env.step(action)
                frames.append(obs)
                reward += rew
                ended = bool(terminated or truncated)
                if ended:
                    break
            arrays["reward"][i], arrays["termination"][i] = reward, ended
            pooled = np.maximum(frames[-2], frames[-1]) if len(frames) > 1 else frames[-1]
            episode_step += 1
            if ended:
                episode_id, episode_step = episode_id + 1, 0
                obs, _ = env.reset()
                pooled = obs
    finally:
        env.close()
    arrays["game"] = np.asarray(game)
    arrays["capture_metadata"] = np.asarray(json.dumps(dict(
        policy="random", seed=seed, frame_skip=frame_skip,
        numpy_version=np.__version__, ale_version=ale_py.__version__,
        gymnasium_version=gym.__version__, ram_reader="read_ale_ram: explicit output buffer when supported",
        alignment="screen[t] and ram[t] before action[t]; obs[t] is max-pooled",
        repeat_action_probability=0.0, full_action_space=True)))
    return arrays


def load_data(path, game):
    with np.load(path, allow_pickle=False) as archive:
        if not {"obs", "ram"}.issubset(archive.files):
            raise ValueError(f"{path}: requires obs and ram arrays")
        keys = ("obs", "ram", "screen", "episode_id", "episode_step", "game", "capture_metadata")
        data = {k: archive[k] for k in keys if k in archive.files}
    if "game" in data and canonical_game(str(data["game"].item())) != game:
        raise ValueError("--game does not match the archive's game metadata")
    ram = data["ram"]
    if ram.ndim != 2 or ram.shape[1] != 128 or not len(ram):
        raise ValueError("Expected nonempty ram with shape (N, 128)")
    for key in ("obs", "screen"):
        if key not in data:
            continue
        images = data[key]
        if images.ndim != 4 or images.shape[-1] != 3 or len(images) != len(ram):
            raise ValueError(f"{key} must have shape (N, H, W, 3), aligned with ram")
        if images.dtype != np.uint8:
            raise ValueError(f"{key} must be RGB uint8 [0,255]; convert explicitly before using")
    for key in ("episode_id", "episode_step"):
        if key in data and data[key].shape != (len(ram),):
            raise ValueError(f"{key} must have shape (N,)")
    return data


def _event_signature(state):
    s, game = state["scalars"], state["game"]
    keys = {"seaquest": ("collected_divers", "visible_divers"),
            "frostbite": ("igloo_stage", "sinking_raw"),
            "pong": ("player_score", "enemy_score", "ball_present"),
            "boxing": ("player_score_raw", "enemy_score_raw")}[game]
    return tuple(s[k] for k in keys)


def select_frames(states, count, seed, episode_ids=None):
    """Mix time coverage, before/after semantic changes, and seeded random frames."""
    n, chosen = len(states), {}
    count = min(count, n)
    for index in np.linspace(0, n - 1, max(1, count // 3), dtype=int):
        chosen[int(index)] = "uniform"
    events = [i for i in range(1, n)
              if _event_signature(states[i]) != _event_signature(states[i - 1])
              and (episode_ids is None or episode_ids[i] == episode_ids[i - 1])]
    if events:
        budget = min(len(events), max(1, count // 4))
        for index in np.linspace(0, len(events) - 1, budget, dtype=int):
            for frame, reason in ((events[index] - 1, "before change"), (events[index], "after change")):
                if len(chosen) < count:
                    chosen[frame] = reason
    # Fill from a permutation so duplicates cannot cause an endless loop.
    for index in np.random.default_rng(seed).permutation(n):
        if len(chosen) >= count:
            break
        chosen.setdefault(int(index), "random")
    return dict(sorted(chosen.items()))


def draw_overlay(pixels, state, scale=3):
    image = Image.fromarray(pixels).resize((480, 630), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    font = _font(14)
    for number, obj in enumerate(state["objects"], 1):
        x, y, w, h = obj["xywh"]
        color = COLORS.get(obj["kind"], "#70ff80")
        # Native ALE coordinates -> uncropped display, independent of obs resolution.
        parts = [(x, y, w, h)]
        if obj["attributes"].get("wraps_right"):
            parts.append((x - 160, y, w, h))
        for px, py, pw, ph in parts:
            left, top = max(0, px * scale), max(0, py * scale)
            right, bottom = min(479, (px + pw) * scale - 1), min(629, (py + ph) * scale - 1)
            draw.rectangle((left, top, right, bottom), outline=color, width=2)
            # Numeric labels keep a dense Seaquest/Frostbite frame readable.
            draw.text((left, max(0, top - 15)), str(number), font=font, fill=color,
                      stroke_width=1, stroke_fill="black")
    return image


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_report(game, data, output, samples, seed, explicit_frames=None, image_key="auto", source_path=None):
    output.mkdir(parents=True, exist_ok=True)
    states = [extract_state(game, ram) for ram in data["ram"]]
    selected = (select_frames(states, samples, seed, data.get("episode_id")) if explicit_frames is None
                else dict.fromkeys(explicit_frames, "explicit"))
    if any(i < 0 or i >= len(states) for i in selected):
        raise ValueError(f"Frame index must lie in [0, {len(states) - 1}]")
    image_key = ("screen" if "screen" in data else "obs") if image_key == "auto" else image_key
    if image_key not in data:
        raise ValueError(f"Archive has no {image_key!r} array")
    warning_frames = [i for i, s in enumerate(states) if s["warnings"]]
    with (output / "states.jsonl").open("w") as stream:
        for i, state in enumerate(states):
            stream.write(json.dumps(dict(frame_index=i, **state), ensure_ascii=False) + "\n")
    features = [numeric_features(s) for s in states]
    columns = sorted(set().union(*(f.keys() for f in features)))
    values = np.full((len(states), len(columns)), np.nan, np.float32)
    with (output / "features.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", *columns])
        for i, f in enumerate(features):
            row = [f.get(k, 0 if k.endswith(".present") or ".kind." in k else None) for k in columns]
            writer.writerow([i, *row])
            values[i] = [np.nan if value is None else value for value in row]
    np.savez_compressed(output / "features.npz", values=values, names=np.asarray(columns),
                        valid=np.isfinite(values), frame_index=np.arange(len(states)))
    metadata = dict(game=game, frames=len(states), image_key=image_key,
                    image_shape=list(data[image_key].shape[1:]),
                    selected_frames=selected, warning_frames=warning_frames,
                    coordinate_system="uncropped ALE 160x210, top-left xywh, pixels",
                    source=SOURCE_URLS[game],
                    input_archive=str(source_path.resolve()) if source_path else None,
                    alignment=(str(data["capture_metadata"].item()) if "capture_metadata" in data else
                               "Existing archive: obs/RAM temporal alignment is unverified"))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    cards, sections = [], []
    for i, reason in selected.items():
        state, stem = states[i], f"frame_{i:06d}"
        pixels = data[image_key][i]
        original = Image.fromarray(pixels).resize((480, 630), Image.Resampling.NEAREST)
        overlay = draw_overlay(pixels, state)
        original.save(frames_dir / f"{stem}_raw.png")
        overlay.save(frames_dir / f"{stem}_overlay.png")
        card = Image.new("RGB", (960, 730), "#141821")
        card.paste(original, (0, 34))
        card.paste(overlay, (480, 34))
        draw = ImageDraw.Draw(card)
        draw.text((8, 7), f"{game} #{i} ({reason}) | original / RAM", font=_font(18), fill="white")
        summary = ", ".join(f"{k}={v}" for k, v in state["scalars"].items())
        for line, text_line in enumerate(textwrap.wrap(summary, width=98)[:3]):
            draw.text((8, 670 + 18 * line), text_line, font=_font(15), fill="white")
        card.save(frames_dir / f"{stem}.png")
        cards.append(card.resize((480, 365), Image.Resampling.LANCZOS))
        payload = dict(frame_index=i, selection_reason=reason, ram=data["ram"][i].tolist(), **state)
        (frames_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        rows = []
        for number, obj in enumerate(state["objects"], 1):
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                        (number, obj["id"], obj["kind"], obj["xywh"], obj["ram"], obj["attributes"])) + "</tr>")
        obs_note = ""
        if image_key == "screen":
            Image.fromarray(data["obs"][i]).resize((256, 256), Image.Resampling.NEAREST).save(
                frames_dir / f"{stem}_obs.png")
            obs_note = f'<details><summary>STORM obs (max-pooled)</summary><img src="frames/{stem}_obs.png"></details>'
        sections.append(f'<article id="f{i}"><h2>Frame {i} · {reason}</h2>'
                        f'<a href="frames/{stem}.png"><img class="pair" src="frames/{stem}.png"></a>'
                        f'<pre>{html.escape(json.dumps(state["scalars"], indent=2))}</pre>{obs_note}'
                        f'<p><a href="frames/{stem}.json">Full RAM + extracted state (JSON)</a></p>'
                        '<table><tr><th>#</th><th>ID</th><th>Kind</th><th>xywh</th><th>RAM bytes</th><th>Attributes</th></tr>'
                        + "".join(rows) + "</table>"
                        + f'<pre>{html.escape(str(state["warnings"]))}</pre></article>')
    cols = min(3, len(cards))
    sheet = Image.new("RGB", (cols * 480, ((len(cards) + cols - 1) // cols) * 365), "#141821")
    for j, card in enumerate(cards):
        sheet.paste(card, ((j % cols) * 480, (j // cols) * 365))
    sheet.save(output / "contact_sheet.jpg", quality=92)
    note = ("Exact final screen / RAM pairs from this collector. RAM boxes can describe flickering sprites."
            if "capture_metadata" in data and image_key == "screen" else
            "Existing/resized/pooled observations: check temporal alignment and cropping. "
            "The overlay assumes a full, uncropped 160x210 ALE image.")
    report = ('<!doctype html><meta charset="utf-8"><title>RAM verification</title>'
              '<style>body{font:16px system-ui;background:#151922;color:#eee;margin:24px}'
              'a{color:#8cdbff}.pair{width:min(960px,100%)}table{border-collapse:collapse;font:13px monospace}'
              'td,th{border:1px solid #666;padding:6px}article{margin:48px 0}pre{white-space:pre-wrap}</style>'
              f'<h1>{ALE_NAMES[game]}: RAM verification</h1><p>{note}</p>'
              '<p>Left: original pixels. Right: RAM boxes. Numbers match table rows. '
              'Coordinates are top-left xywh in native ALE pixels. '
              'Numeric range checks do not prove semantic correctness.</p>'
              f'<p>{len(states)} frames; {len(selected)} displayed; {len(warning_frames)} frames with mapping warnings. '
              '<a href="metadata.json">Metadata</a> · <a href="features.csv">All numeric features</a> · '
              '<a href="states.jsonl">All object states</a></p>'
              '<nav>' + " · ".join(f'<a href="#f{i}">{i}</a>' for i in selected) + '</nav>'
              '<p>Boxing scores and Frostbite temperature include BCD-decoded values and raw bytes. '
              'Arm and animation fields are raw codes. '
              'Boxing boxes cover the body, not extended fists. '
              'A RAM-only snapshot does not provide ball velocity.</p>' + "".join(sections))
    (output / "report.html").write_text(report)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("all", *GAMES), default="all", type=str.lower)
    parser.add_argument("--data", type=Path, help="Existing RGB uint8 obs+ram NPZ; requires a single game")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ram_verification")
    parser.add_argument("--steps", type=int, default=2000, help="Action decisions to collect per game")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--frames", nargs="+", type=int, help="Explicit archive row indices to display")
    parser.add_argument("--image-key", choices=("auto", "obs", "screen"), default="auto")
    args = parser.parse_args()
    if min(args.steps, args.samples, args.frame_skip) < 1:
        parser.error("--steps, --samples and --frame-skip must be positive")
    if args.data and args.game == "all":
        parser.error("--data requires --game seaquest/frostbite/pong/boxing")
    for game in GAMES if args.game == "all" else (args.game,):
        output = args.output / game
        output.mkdir(parents=True, exist_ok=True)
        if args.data:
            data = load_data(args.data, game)
        else:
            if (output / "rollout.npz").exists():
                parser.error(f"{output}/rollout.npz already exists; reuse via --data or choose another --output")
            print(f"[{game}] collecting {args.steps} random-policy decisions", flush=True)
            data = collect_rollout(game, args.steps, args.seed, args.frame_skip)
            np.savez_compressed(output / "rollout.npz", **data)
        metadata = write_report(game, data, output, args.samples, args.seed, args.frames, args.image_key,
                                source_path=args.data or output / "rollout.npz")
        print(f"[{game}] {metadata['frames']} states, {len(metadata['selected_frames'])} previews, "
              f"{len(metadata['warning_frames'])} mapping warnings: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
