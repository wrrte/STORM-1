"""Render actual igloo templates and probe noise preprocessing without GPU inference."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_probing_mlp import inject_background_noise


def contact_sheet(rows, row_labels, column_labels, path):
    scale, margin, label_width, title_height = 4, 12, 130, 38
    tile = 64 * scale
    canvas = Image.new(
        "RGB",
        (label_width + len(column_labels) * (tile + margin),
         title_height + len(rows) * (tile + margin)),
        "#eeeeee",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for col, label in enumerate(column_labels):
        draw.text((label_width + col * (tile + margin), 10), label, fill="black", font=font)
    for row, (images, label) in enumerate(zip(rows, row_labels)):
        y = title_height + row * (tile + margin)
        draw.text((10, y + tile // 2), label, fill="black", font=font)
        for col, pixels in enumerate(images):
            if pixels is not None:
                enlarged = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST)
                canvas.paste(enlarged, (label_width + col * (tile + margin), y))
    canvas.save(path)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "probing_data/Frostbite_20260822_201417.npz")
    parser.add_argument("--template-dir", type=Path, default=root / "frostbite_igloo_templates")
    parser.add_argument("--frame-index", type=int, default=305,
                        help="Base observation index; 305 is an in-game frame in the default dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=root / "results/frostbite_noise_preview")
    args = parser.parse_args()

    with np.load(args.data, allow_pickle=False) as data:
        if not 0 <= args.frame_index < len(data["obs"]):
            parser.error("--frame-index is outside the dataset")
        base = data["obs"][args.frame_index:args.frame_index + 1].copy()
        ram_stage = int(data["ram"][args.frame_index, 77])
        source_stage = 0 if ram_stage == 255 else ram_stage + 1
    assert base.shape == (1, 64, 64, 3) and base.dtype == np.uint8
    templates = []
    for stage in range(17):
        # Same loading and RGB conversion as eval_frostbite_value.py.
        path = args.template_dir / f"stage_{stage:02d}.png"
        img = cv2.imread(str(path))
        if img is None or img.shape != (7, 16, 3):
            raise ValueError(f"Expected a 7x16 RGB template: {path}")
        templates.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base[0]).save(args.output / "source_frame.png")
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    background = np.ones((64, 64), dtype=bool)
    background[10:17, 42:58] = False
    variants = []
    for strength in strengths:
        stage_images = []
        folder = args.output / f"alpha_{strength}"
        folder.mkdir(exist_ok=True)
        for stage in range(17):
            mod_obs = base.copy()
            # Exact template assignment used by evaluate_values().
            mod_obs[:, 10:17, 42:58, :] = templates[stage]
            # Invoke the actual training preprocessing; reset seed per variant
            # to compare stages with identical random background pixels.
            noisy = inject_background_noise(mod_obs, "frostbite_igloo", strength, args.seed)[0]
            assert np.array_equal(noisy[10:17, 42:58], templates[stage])
            if stage_images:
                assert np.array_equal(noisy[background], stage_images[0][background])
            if strength == 0.0:
                assert np.array_equal(noisy[background], base[0][background])
            Image.fromarray(noisy).save(folder / f"stage_{stage:02d}.png")
            stage_images.append(noisy)
        variants.append(stage_images)
        padded = stage_images + [None]
        contact_sheet(
            [padded[start:start + 6] for start in (0, 6, 12)],
            ["Stages 0-5", "Stages 6-11", "Stages 12-16"],
            [f"Offset +{i}" for i in range(6)],
            args.output / f"all_stages_alpha_{strength}.png",
        )

    selected = [0, 4, 8, 12, 16]
    contact_sheet(
        [[images[stage] for stage in selected] for images in variants],
        [f"Alpha {strength:g}" for strength in strengths],
        [f"Stage {stage}" for stage in selected],
        args.output / "comparison.png",
    )
    metadata = {
        "data": str(args.data.resolve()), "frame_index": args.frame_index,
        "source_ram_stage": source_stage, "seed": args.seed,
        "strengths": strengths, "stages": list(range(17)),
        "template_dir": str(args.template_dir.resolve()),
        "target_region": "obs[:, 10:17, 42:58, :]",
        "stage_method": "Template replacement from eval_frostbite_value.py; not performed by probe training itself",
        "noise_method": "Direct call to train_probing_mlp.inject_background_noise",
        "noise_seed_policy": "Same seed for each single-frame variant; identical background across stages",
        "display": "Raw PNGs are 64x64 RGB; contact sheets use 4x nearest-neighbor enlargement",
        "checks": "Target matches template exactly; background identical across stages; clean background matches source",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved 85 raw images and 6 contact sheets to {args.output}")
    print(f"Source frame: {args.frame_index}; RAM stage: {source_stage}; pixel checks passed.")


if __name__ == "__main__":
    main()
