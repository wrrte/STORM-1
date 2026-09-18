"""Read-only analysis of the 2026-09-10 workbook; writes only to this directory.

Run with /home/ai2lab/miniconda3/envs/storm/bin/python <this file>.
Bootstrap unit is a training run, never an individual evaluation episode.
Fixed-game, percentile intervals are exploratory and not selection-adjusted.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "storm-target-analysis-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import compare_target_results as source

REPS = 50000
RNG_SEED = 20260910
BASE = source.BASELINE
PER = source.ANCHOR_ONLY
C12 = "target: 12 (anchor: 0.12)"
C4 = "target: 4 (anchor 미설정)"
C16 = "target: 16 (anchor 미설정)"
LABELS = {BASE: "STORM", PER: "PER-only", C4: "n=4, uniform",
          C12: "n=12, a=0.12", C16: "n=16, uniform",
          "target: 8 (anchor: 0.3)": "n=8, a=0.3",
          "target: 12 (anchor: 0.2)": "n=12, a=0.2",
          "target: 16 (anchor: 0.4)": "n=16, a=0.4"}


def weighted_iqm(values, weights):
    """Exact central 50% of a weighted empirical distribution, last axis.

    A game has total mass 1/G, a run within it has mass 1/(G*n_g).
    This unequal-n extension is not a call to rliable.aggregate_iqm.
    """
    order = np.argsort(values, axis=-1)
    x = np.take_along_axis(values, order, axis=-1)
    w = np.broadcast_to(weights, values.shape)
    w = np.take_along_axis(w, order, axis=-1)
    end = np.cumsum(w, axis=-1)
    central = np.maximum(0, np.minimum(end, .75) - np.maximum(end-w, .25))
    return (x * central).sum(axis=-1) / .5


def metrics(arrays):
    # arrays: one (..., runs, configs) array per fixed game.
    gm = np.stack([x.mean(axis=-2) for x in arrays], axis=-2)
    g = len(arrays)
    trim = g // 4
    vals = np.concatenate(arrays, axis=-2).swapaxes(-1, -2)
    weights = np.concatenate([np.full(x.shape[-2], 1/(g*x.shape[-2])) for x in arrays])
    return {
        "Mean": gm.mean(axis=-2),
        "Median": np.median(gm, axis=-2),
        "IQM_game_workbook": np.sort(gm, axis=-2)[..., trim:g-trim, :].mean(axis=-2),
        "Gap_game_workbook": np.maximum(0, 1-gm).mean(axis=-2),
        "IQM_run_equal_game": weighted_iqm(vals, weights),
        "Gap_run_equal_game": (np.maximum(0, 1-vals)*weights).sum(axis=-1),
    }


def analyze_panel(panel, configs, norm, panel_name, pairs, independent=False):
    panel = panel[configs].dropna().sort_index()
    groups = list(panel.groupby(level="Game"))
    arrays = []
    for game, rows in groups:
        r, h = norm.loc[game, ["Random", "Human"]]
        arrays.append((rows.to_numpy()-r)/(h-r))
    point = metrics(arrays)
    rng = np.random.default_rng(RNG_SEED)
    boot_arrays = []
    for x in arrays:
        n, k = x.shape
        if independent:
            ix = rng.integers(n, size=(REPS, n, k))
            boot_arrays.append(x[ix, np.arange(k)])
        else:
            ix = rng.integers(n, size=(REPS, n))
            boot_arrays.append(x[ix])
    boot = metrics(boot_arrays)
    result = []
    raw_means = panel.groupby(level="Game").mean()
    for candidate, reference in pairs:
        ci, ri = configs.index(candidate), configs.index(reference)
        game_delta = raw_means[candidate]-raw_means[reference]
        wins, losses = int((game_delta>0).sum()), int((game_delta<0).sum())
        for name in point:
            delta = float(point[name][ci]-point[name][ri])
            lo, hi = np.quantile(boot[name][:, ci]-boot[name][:, ri], [.025, .975])
            result.append(dict(panel=panel_name, config=candidate, reference=reference,
                sampling="independent" if independent else "paired", games=len(groups),
                seed_cells=len(panel), min_n=min(len(x) for x in arrays),
                max_n=max(len(x) for x in arrays), wins=wins, losses=losses,
                game_sign_p=binomtest(wins, wins+losses).pvalue if wins+losses else 1,
                metric=name, reference_value=float(point[name][ri]),
                config_value=float(point[name][ci]), delta=delta,
                improvement_pct=delta/abs(point[name][ri])*100*(-1 if name.startswith("Gap") else 1),
                ci_low=lo, ci_high=hi, included_games=", ".join(g for g, _ in groups)))
    return result, point


def main():
    workbook = ROOT / "target_comparison.xlsx"
    sheets = pd.read_excel(workbook, sheet_name=None)
    runs = sheets["Selected Runs"]
    norm = sheets["Normalization"].set_index("Game")
    latest = source.load_latest(ROOT / "wandb_runs_classification.csv")
    keys = ["Game", "Config", "Seed", "Run ID", "Eval Return"]
    pd.testing.assert_frame_equal(runs[keys].sort_values(keys[:3]).reset_index(drop=True),
        latest[keys].sort_values(keys[:3]).reset_index(drop=True), check_dtype=False)
    assert source.load_normalization(ROOT.parent / "iclr2027_conference.tex") == {
        g: tuple(row) for g, row in norm.iterrows()}
    panel = runs.pivot(index=["Game", "Seed"], columns="Config", values="Eval Return")
    candidates = [c for c in panel.columns if c not in [BASE, PER]]
    records = []
    for candidate in candidates:
        configs = [BASE, PER, candidate]
        rows, _ = analyze_panel(panel, configs, norm, "three_way", [(candidate, BASE), (candidate, PER)])
        records.extend(rows)
        rows, _ = analyze_panel(panel, [BASE, candidate], norm, "baseline_all_available", [(candidate, BASE)])
        records.extend(rows)
    # Fair ranking: every config evaluated on exactly the same game/seed cells.
    configs = [BASE, PER] + candidates
    rows, point = analyze_panel(panel, configs, norm, "all_configs_identical_cells",
        [(c, r) for c in candidates for r in [BASE, PER]] + [(C12, C4)])
    records.extend(rows)
    strict = pd.DataFrame(point, index=configs)
    strict.index.name = "Config"
    strict.to_csv(OUT / "identical_panel_metrics.csv")
    panel.dropna().to_csv(OUT / "identical_panel_scores.csv")
    # Direct finalist comparison with both controls, and sensitivity to pairing.
    rows, _ = analyze_panel(panel, [BASE, PER, C4, C12], norm, "finalists_identical_cells", [(C12, C4)])
    records.extend(rows)
    rows, _ = analyze_panel(panel, [BASE, PER, C12], norm, "three_way",
        [(C12, BASE), (C12, PER)], independent=True)
    records.extend(rows)
    # n=16 mixes two hash widths in the workbook. Do not treat the mixture as
    # a single fully specified configuration when recommending a new run.
    for bits in [9, 10]:
        selected = runs[(runs.Config==C16) & (runs["Hash Bits"]==bits)].set_index(["Game", "Seed"]).index
        rows, _ = analyze_panel(panel.loc[selected], [BASE, C16], norm,
            "n16_hash"+str(bits), [(C16, BASE)])
        records.extend(rows)
    rows, _ = analyze_panel(panel, [BASE, C4, C16], norm,
        "n16_n4_matched_baseline", [(C16, C4)])
    records.extend(rows)
    paired16 = panel[[BASE, C16]].dropna()
    repeated_games = paired16.groupby(level="Game").size().loc[lambda x:x>=2].index
    rows, _ = analyze_panel(paired16.loc[repeated_games], [BASE, C16], norm,
        "n16_exclude_single_seed_games", [(C16, BASE)])
    records.extend(rows)
    # Original workbook point estimates must reproduce exactly for primary metrics.
    result = pd.DataFrame(records)
    metric_map = {"IQM_game_workbook": "IQM ↑", "Gap_game_workbook": "Optimality Gap ↓",
                  "Mean": "Mean ↑", "Median": "Median ↑"}
    for row in result[(result.panel=="three_way") & (result.sampling=="paired")].itertuples():
        if row.metric not in metric_map:
            continue
        original = sheets["Dual Summary"]
        m = original[(original.Config==row.config) & (original.Reference==row.reference)
                     & (original.Metric==metric_map[row.metric])].iloc[0]
        assert np.isclose(row.config_value, m["Config Value"], atol=1e-12)
        assert np.isclose(row.reference_value, m["Reference Value"], atol=1e-12)
    result.to_csv(OUT / "bootstrap_comparisons.csv", index=False)
    counts = runs.groupby(["Game", "Config"]).size().unstack(fill_value=0).reindex(norm.index, fill_value=0)
    counts.to_csv(OUT / "coverage.csv")
    runs.to_csv(OUT / "selected_runs.csv", index=False)
    sheets["Dual Comparisons"].to_csv(OUT / "game_comparisons.csv", index=False)
    # Leave-one-game-out sensitivity keeps all three methods paired.
    lop = panel[[BASE, PER, C12]].dropna()
    loo = []
    for omitted in lop.index.get_level_values("Game").unique():
        smaller = lop.drop(index=omitted, level="Game")
        arrays = [(x.to_numpy()-norm.loc[g,"Random"])/(norm.loc[g,"Human"]-norm.loc[g,"Random"])
                  for g, x in smaller.groupby(level="Game")]
        m = metrics(arrays)
        for metric, v in m.items():
            for i, ref in enumerate([BASE, PER]):
                loo.append(dict(omitted=omitted, reference=ref, metric=metric, delta=v[2]-v[i]))
    pd.DataFrame(loo).to_csv(OUT / "leave_one_game_out.csv", index=False)
    # Compact exportable chart, using the workbook-compatible IQM estimand.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    order = [C4, C12, C16, "target: 8 (anchor: 0.3)", "target: 12 (anchor: 0.2)", "target: 16 (anchor: 0.4)"]
    for ax, ref in zip(axes, [BASE, PER]):
        rr = result[(result.panel=="three_way") & (result.sampling=="paired") &
                    (result.metric=="IQM_game_workbook") & (result.reference==ref)].set_index("config").loc[order]
        yy = np.arange(len(order))
        ax.hlines(yy, rr.ci_low, rr.ci_high, color="#3775aa", linewidth=2)
        ax.scatter(rr.delta, yy, color="#17385b", zorder=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title("vs " + LABELS[ref]); ax.set_xlabel("Difference in workbook IQM (HNS units)")
        ax.set_yticks(yy, [LABELS[c] for c in order]); ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle("Matched STORM / PER-only / candidate comparisons\n50,000 paired training-run bootstrap draws; fixed games; exploratory 95% intervals", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT/"iqm_intervals.png", dpi=180); fig.savefig(OUT/"iqm_intervals.pdf")
    metadata = {"bootstrap_replicates": REPS, "random_seed": RNG_SEED,
        "unit": "training-run evaluation mean", "sampling": "fixed games, seeds within each game",
        "interval": "percentile 95%; no correction for candidate selection",
        "verified": "420 selected runs and all workbook three-way point estimates reproduced",
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [workbook, ROOT/"wandb_runs_classification.csv", ROOT.parent/"iclr2027_conference.tex"]}}
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    show = result[(result.metric.isin(["IQM_game_workbook", "IQM_run_equal_game"])) &
                  (result.config.isin([C12, C4, C16]))]
    print(show.drop(columns=["included_games", "game_sign_p"]).to_string(index=False))
    print("ALL-CONFIG IDENTICAL PANEL\n", strict.to_string())
    print("Verified workbook and CSV. Outputs:", OUT)


if __name__ == "__main__":
    main()
