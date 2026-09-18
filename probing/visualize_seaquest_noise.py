"""Preview Seaquest observations using the probe's actual noise preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from train_probing_mlp import inject_background_noise
from visualize_frostbite_noise import contact_sheet


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=root / "probing_data/Seaquest_20260822_133245.npz")
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=root / "results/seaquest_noise_preview")
    args = parser.parse_args()
    if any(not 0.0 <= strength <= 1.0 for strength in args.strengths):
        parser.error("--strengths must be between 0 and 1")

    with np.load(args.data, allow_pickle=False) as data:
        labels = data["ram"][:, 62].astype(np.int64)
        obs = data["obs"]
        if obs.shape != (len(labels), 64, 64, 3) or obs.dtype != np.uint8:
            parser.error("Expected uint8 obs with shape (N, 64, 64, 3) and matching RAM")
        # Use real observations rather than synthesizing diver counts.
        selected = []
        for count in range(7):
            indices = np.flatnonzero(labels == count)
            if len(indices):
                frame_index = int(indices[len(indices) // 2])
                selected.append((count, frame_index, obs[frame_index:frame_index + 1].copy()))
    if not selected:
        parser.error("No observations with diver counts 0-6 found")

    args.output.mkdir(parents=True, exist_ok=True)
    rows, row_labels, sources = [], [], []
    for count, frame_index, clean in selected:
        folder = args.output / f"divers_{count}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(clean[0]).save(folder / "source.png")
        images = []
        for strength in args.strengths:
            # Directly call the same function used before probe training.
            # Reset the seed so noise pixels are shared across strengths.
            noisy = inject_background_noise(
                clean, "diver_count", strength=strength, seed=args.seed,
            )[0]
            assert np.array_equal(noisy[53:57, 18:46], clean[0, 53:57, 18:46])
            if strength == 0.0:
                assert np.array_equal(noisy, clean[0])
            Image.fromarray(noisy).save(folder / f"alpha_{strength}.png")
            images.append(noisy)
        rows.append(images)
        row_labels.append(f"Divers {count}")
        sources.append({"diver_count_ram": count, "frame_index": frame_index})

    contact_sheet(rows, row_labels, [f"Alpha {s:g}" for s in args.strengths],
                  args.output / "comparison.png")
    metadata = {
        "data": str(args.data.resolve()), "sources": sources,
        "strengths": args.strengths, "seed": args.seed,
        "preserved_region": "obs[:, 53:57, 18:46, :]",
        "noise_function": "train_probing_mlp.inject_background_noise",
        "selection": "One real observation per RAM count; backgrounds may differ between rows",
        "display": "Raw images are 64x64; comparison uses 4x nearest-neighbor enlargement",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved comparison: {args.output / 'comparison.png'}")
    print("Target preservation checks passed for all images.")


if __name__ == "__main__":
    main()
