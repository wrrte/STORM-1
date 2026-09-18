"""Create the paper's 2-by-7 Frostbite/Seaquest noise comparison without inference."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train_probing_mlp import inject_background_noise


ROOT = Path(__file__).resolve().parent
STRENGTHS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


def generate_examples(output=ROOT.parent / "experiment", frostbite_frame=1131,
                      seaquest_frame=1044, seed=42):
    sources = [
        ("Frostbite", "frostbite_igloo", "Frostbite_20260822_201417.npz",
         frostbite_frame, slice(10, 17), slice(42, 58), 77),
        ("Seaquest", "diver_count", "Seaquest_20260822_133245.npz",
         seaquest_frame, slice(53, 57), slice(18, 46), 62),
    ]
    rows, metadata = [], []
    for game, task, filename, frame_index, y, x, ram_byte in sources:
        path = ROOT / "probing_data" / filename
        with np.load(path, allow_pickle=False) as data:
            if not 0 <= frame_index < len(data["obs"]):
                raise ValueError(f"Frame index {frame_index} outside {filename}")
            clean = data["obs"][frame_index:frame_index + 1].copy()
            label = int(data["ram"][frame_index, ram_byte])
        if task == "frostbite_igloo":
            label = 0 if label == 255 else label + 1
        assert clean.shape == (1, 64, 64, 3) and clean.dtype == np.uint8
        images = []
        for alpha in STRENGTHS:
            # Same function as training and the earlier visualize scripts.
            # Each column starts from the original observation and seed.
            pixels = inject_background_noise(clean, task, strength=alpha, seed=seed)[0]
            assert np.array_equal(pixels[y, x], clean[0, y, x])
            if alpha == 0:
                assert np.array_equal(pixels, clean[0])
            images.append(pixels)
        # Check shared noise fields and blending across all seven strengths.
        for alpha, pixels in zip(STRENGTHS, images):
            expected = np.rint(
                (1-alpha)*images[0].astype(np.float32)
                + alpha*images[-1].astype(np.float32)
            ).clip(0, 255).astype(np.uint8)
            assert np.array_equal(pixels, expected)
        rows.append(images)
        metadata.append({"game": game, "data": str(path), "frame_index": frame_index,
                         "ram_label": label,
                         "preserved_y": [y.start, y.stop], "preserved_x": [x.start, x.stop]})

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(2, len(STRENGTHS), figsize=(7.4, 2.45))
        fig.subplots_adjust(left=0.047, right=0.998, bottom=0.02, top=0.86,
                            wspace=0.025, hspace=0.04)
        for row, (game, images) in enumerate(zip(("Frostbite", "Seaquest"), rows)):
            for col, (alpha, pixels) in enumerate(zip(STRENGTHS, images)):
                ax = axes[row, col]
                ax.imshow(pixels, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                if row == 0:
                    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=10, pad=5)
                if col == 0:
                    ax.set_ylabel(game, fontsize=10, labelpad=5)
        fig.savefig(output / "background_noise_examples.png", dpi=400, facecolor="white")
        fig.savefig(output / "background_noise_examples.pdf", facecolor="white")
        plt.close(fig)
    np.savez_compressed(output / "background_noise_examples_pixels.npz",
                        images=np.array(rows), strengths=np.array(STRENGTHS))
    (output / "background_noise_examples_metadata.json").write_text(json.dumps({
        "sources": metadata, "strengths": STRENGTHS, "seed": seed,
        "noise_function": "train_probing_mlp.inject_background_noise",
        "layout": "2 rows (Frostbite, Seaquest), 7 columns (noise strengths)",
        "checks": "Original at alpha=0; target preserved; same noise field across strengths",
    }, indent=2) + "\n")
    print(f"Saved {output / 'background_noise_examples.png'}")
    print(f"Saved {output / 'background_noise_examples.pdf'}")
    print("All 14 images passed pixel checks; no training or inference performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "experiment")
    parser.add_argument("--frostbite-frame", type=int, default=1131,
                        help="Observation index; default has igloo stage 16")
    parser.add_argument("--seaquest-frame", type=int, default=1044,
                        help="Observation index; default has 5 collected divers")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_examples(args.output, args.frostbite_frame, args.seaquest_frame, args.seed)


if __name__ == "__main__":
    main()
