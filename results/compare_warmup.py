"""Compare ONLY the three anchor-r=1 warmup configurations in Results.

Run from any directory:
    python /path/to/STORM-1/results/compare_warmup.py

Dependencies: numpy, pandas, openpyxl, matplotlib (unless --no-plots).
See README_warmup_comparison.md for score selection and metric definitions.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook


HERE = Path(__file__).resolve().parent
# This is deliberately an exact allowlist, including when adding new metrics.
# Never select rows using startswith/contains or the hidden Ablation Results sheet.
CONFIGS = {
    30000: 'target: 16 (anchor r: 1) [warmup: 30000]',
    50000: 'target: 16 (anchor r: 1)',
    70000: 'target: 16 (anchor r: 1) [warmup: 70000]',
}
WARMUPS = tuple(CONFIGS)
METRICS = ('Mean', 'Median', 'IQM', 'Optimality Gap')
PLOT_FILES = ('metrics_vs_warmup.png', 'metrics_vs_warmup.pdf', 'hns_by_game.png')
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
SCORE_ENTRY = re.compile(rf'^\s*({NUMBER})\s*(?:\(([^()]*)\))?\s*$')


def parse_score_cell(value):
    """Select the first finite completed result; metadata commas are not runs.

    The converter orders runs newest first (after its hash-bit preference).
    A numeric entry after RUNNING is an older completed result, not a live score.
    Repeated runs in one seed cell must never become independent seed samples.
    """
    entries = re.split(r',\s*(?![^()]*\))|\n', '' if value is None else str(value))
    completed = []
    for entry in entries:
        match = SCORE_ENTRY.fullmatch(entry)
        if match and np.isfinite(float(match[1])):
            completed.append((float(match[1]), match[2] or '', entry.strip()))
    if not completed:
        return np.nan, np.nan, '', 0
    score, metadata, entry = completed[0]
    actual = re.search(rf'\bw:\s*({NUMBER})', metadata)
    return score, float(actual[1]) if actual else np.nan, entry, len(completed)


def load_scores(path):
    """Read allowed seed cells and their normalization references only."""
    config_to_warmup = {config: warmup for warmup, config in CONFIGS.items()}
    records = []
    references = {}
    found = set()
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if 'Results' not in workbook.sheetnames:
            raise ValueError('The workbook must contain the offset-separated Results sheet.')
        rows = workbook['Results'].iter_rows()
        header = [cell.value for cell in next(rows)]
        if header[:2] != ['Game', 'Config']:
            raise ValueError('Expected Game and Config in columns A and B.')
        seed_columns = [(i, int(str(value).strip())) for i, value in enumerate(header)
                        if str(value).strip().isdigit()]
        if not seed_columns or len({seed for _, seed in seed_columns}) != len(seed_columns):
            raise ValueError('Expected distinct numeric seed column headers.')
        game = None
        for row in rows:
            # Game and Random/Human references occupy a merged cell. This is the
            # only information read from rows outside the three configurations.
            if row[0].value:
                text = str(row[0].value)
                game = text.splitlines()[0].strip()
                random_match = re.search(rf'Random:\s*({NUMBER})', text)
                human_match = re.search(rf'Human:\s*({NUMBER})', text)
                if random_match and human_match:
                    references[game] = (float(random_match[1]), float(human_match[1]))
            config = row[1].value
            if config not in config_to_warmup:
                continue
            if game is None:
                raise ValueError('An allowed configuration has no game name.')
            found.add(config)
            for column, seed in seed_columns:
                cell = row[column]
                raw = '' if cell.value is None else str(cell.value)
                score, actual, selected, count = parse_score_cell(raw)
                excluded = bool(cell.font and cell.font.strike)
                records.append({
                    'Game': game, 'Seed': seed, 'Config': config,
                    'Warmup Steps': config_to_warmup[config],
                    'Eval Return': score, 'Actual Warmup Steps': actual,
                    'Excluded': excluded,
                    'Usable': bool(np.isfinite(score) and not excluded),
                    'Has Running': 'RUNNING' in raw.upper(),
                    'Has Queued': 'QUEUED' in raw.upper(),
                    'Completed Entries': count, 'Selected Entry': selected,
                    'Cell': cell.coordinate if raw else '', 'Raw Cell': raw,
                })
    finally:
        workbook.close()
    missing = set(CONFIGS.values()) - found
    if missing:
        raise ValueError(f'Missing required configuration rows: {sorted(missing)}')
    cells = pd.DataFrame(records)
    if cells.duplicated(['Game', 'Seed', 'Warmup Steps']).any():
        raise ValueError('Duplicate game/config rows: expected one cell per training seed.')
    for game in cells.Game.unique():
        if game not in references:
            raise ValueError(f'{game}: Random/Human references are missing from the game cell.')
        random_score, human_score = references[game]
        if not np.isfinite([random_score, human_score]).all() or human_score <= random_score:
            raise ValueError(f'{game}: invalid Random/Human references.')
    cells['Random'] = cells.Game.map(lambda game: references[game][0])
    cells['Human'] = cells.Game.map(lambda game: references[game][1])
    cells['HNS'] = (cells['Eval Return'] - cells.Random) / (cells.Human - cells.Random)
    return cells


def matched_panel(scores, min_seeds=1):
    """Every method receives exactly the same (game, seed) observations."""
    panel = scores.pivot(index=['Game', 'Seed'], columns='Warmup Steps', values='HNS')
    panel = panel.reindex(columns=WARMUPS).dropna().sort_index()
    counts = panel.groupby(level='Game').size()
    return panel.loc[panel.index.get_level_values('Game').isin(counts[counts >= min_seeds].index)]


def calculate_metrics(arrays):
    """Match update_tex.py conventions without rounding game means.

    Each array has shape (..., seeds, configurations), one array per game.
    Mean/Median/Gap use game means; IQM trims pooled seed HNS by floor(0.25*N).
    In particular Gap is not 1-Mean, nor the mean of per-seed gaps.
    """
    game_means = np.stack([values.mean(axis=-2) for values in arrays], axis=-2)
    pooled = np.sort(np.concatenate(arrays, axis=-2), axis=-2)
    n = pooled.shape[-2]
    trim = n // 4
    return {
        'Mean': game_means.mean(axis=-2),
        'Median': np.median(game_means, axis=-2),
        'IQM': pooled[..., trim:n - trim, :].mean(axis=-2),
        'Optimality Gap': np.maximum(0.0, 1.0 - game_means).mean(axis=-2),
    }


def bootstrap_metrics(arrays, reps, random_seed):
    """Paired seed bootstrap within each fixed game; do not resample games."""
    # Resampling singleton games gives no uncertainty information.
    if reps == 0 or all(len(values) == 1 for values in arrays):
        return {}
    rng = np.random.default_rng(random_seed)
    chunks = {metric: [] for metric in METRICS}
    for start in range(0, reps, 256):
        batch = min(256, reps - start)
        sampled = [values[rng.integers(len(values), size=(batch, len(values)))]
                   for values in arrays]
        for metric, values in calculate_metrics(sampled).items():
            chunks[metric].append(values)
    return {metric: np.concatenate(values) for metric, values in chunks.items()}


def summarize(panel, reps=0, confidence=0.95, random_seed=0):
    arrays = [rows.to_numpy() for _, rows in panel.groupby(level='Game', sort=True)]
    point = calculate_metrics(arrays) if arrays else {
        metric: np.full(len(panel.columns), np.nan) for metric in METRICS}
    boot = bootstrap_metrics(arrays, reps, random_seed) if arrays else {}
    counts = [len(values) for values in arrays]
    quantiles = [(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0]
    rows = []
    for column, warmup in enumerate(panel.columns):
        row = {'Warmup Steps': warmup, 'Config': CONFIGS[warmup],
               'Games': len(arrays), 'Game-Seed Cells': len(panel),
               'Min Seeds/Game': min(counts, default=0),
               'Max Seeds/Game': max(counts, default=0),
               'Single-Seed Games': counts.count(1)}
        for metric in METRICS:
            row[metric] = point[metric][column]
            lo, hi = np.quantile(boot[metric][:, column], quantiles) if boot else (np.nan, np.nan)
            row[f'{metric} CI Low'] = lo
            row[f'{metric} CI High'] = hi
        rows.append(row)
    deltas = []
    if 50000 in panel.columns:
        reference = list(panel.columns).index(50000)
        for column, warmup in enumerate(panel.columns):
            if warmup == 50000:
                continue
            for metric in METRICS:
                delta = point[metric][column] - point[metric][reference]
                lo, hi = np.quantile(boot[metric][:, column] - boot[metric][:, reference],
                                     quantiles) if boot else (np.nan, np.nan)
                deltas.append({'Warmup Steps': warmup, 'Reference Warmup': 50000,
                               'Metric': metric, 'Delta': delta, 'CI Low': lo, 'CI High': hi,
                               'Better': 'lower' if metric == 'Optimality Gap' else 'higher',
                               'Games': len(arrays), 'Game-Seed Cells': len(panel)})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def build_tables(cells, min_seeds=1, reps=2000, confidence=0.95, random_seed=0):
    scores = cells.loc[cells.Usable].copy()
    panel = matched_panel(scores, min_seeds)
    summary, deltas = summarize(panel, reps, confidence, random_seed)
    available = []
    for warmup in WARMUPS:
        single = scores.loc[scores['Warmup Steps'].eq(warmup)].set_index(['Game', 'Seed'])[['HNS']]
        single.columns = pd.Index([warmup], name='Warmup Steps')
        diagnostic, _ = summarize(single)
        available.append(diagnostic)
    panel_keys = set(panel.index)
    cells = cells.copy()
    cells['In Matched Panel'] = [(game, seed) in panel_keys and usable
                                 for game, seed, usable in
                                 zip(cells.Game, cells.Seed, cells.Usable)]
    coverage = cells.groupby(['Game', 'Warmup Steps'], sort=True).agg(
        Usable=('Usable', 'sum'), Matched=('In Matched Panel', 'sum'),
        Running=('Has Running', 'sum'), Queued=('Has Queued', 'sum'),
        Excluded=('Excluded', 'sum'),
    ).reset_index()
    # Also retain games with no matched observations in the per-game output.
    per_game = coverage[['Game', 'Warmup Steps', 'Usable', 'Matched']].copy()
    means = panel.groupby(level='Game').mean().rename_axis(columns='Warmup Steps')
    long_means = means.reset_index().melt(id_vars='Game', var_name='Warmup Steps', value_name='HNS')
    per_game = per_game.merge(long_means, on=['Game', 'Warmup Steps'], how='left')
    per_game['Delta vs 50000'] = per_game.HNS - per_game.Game.map(means[50000])
    selected = cells.loc[cells.Usable].sort_values(['Game', 'Seed', 'Warmup Steps'])
    return {
        'Matched Summary': summary, 'Delta vs 50000': deltas,
        'Available Diagnostic': pd.concat(available, ignore_index=True),
        'Per Game': per_game, 'Coverage': coverage, 'Selected Scores': selected,
        'Matched HNS': panel.reset_index(),
        'Cell Audit': cells.loc[cells['Raw Cell'].ne('')].sort_values(['Game', 'Warmup Steps', 'Seed']),
    }


def plot_results(tables, output_dir, confidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = tables['Matched Summary']
    if summary.Games.max() == 0:
        return
    x = summary['Warmup Steps'].to_numpy() / 1000
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        y = summary[metric].to_numpy()
        ax.plot(x, y, 'o-', color='#2563eb', linewidth=2)
        lo = summary[f'{metric} CI Low'].to_numpy()
        hi = summary[f'{metric} CI High'].to_numpy()
        if np.isfinite(lo).all():
            ax.fill_between(x, lo, hi, color='#2563eb', alpha=0.18,
                            label=f'{confidence:.0%} paired seed bootstrap CI')
            ax.legend(fontsize=8)
        for step, value in zip(x, y):
            ax.annotate(f'{value:.4f}', (step, value), xytext=(0, 7),
                        textcoords='offset points', ha='center', fontsize=9)
        direction = 'lower is better' if metric == 'Optimality Gap' else 'higher is better'
        ax.set(title=f'{metric} ({direction})', xlabel='Requested warmup steps (thousands)',
               ylabel='HNS', xticks=x)
        ax.margins(x=0.15, y=0.25)
        ax.grid(alpha=0.25)
    games, cells = summary.iloc[0][['Games', 'Game-Seed Cells']]
    fig.suptitle(f'Anchor r=1, target=16 | {games} common games, {cells} common game-seed cells\n'
                 'Same observations at 30k / 50k / 70k', fontsize=12)
    fig.savefig(output_dir / 'metrics_vs_warmup.png', dpi=180)
    fig.savefig(output_dir / 'metrics_vs_warmup.pdf')
    plt.close(fig)

    game_hns = tables['Matched HNS'].groupby('Game')[list(WARMUPS)].mean()
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(game_hns) * 0.4)), constrained_layout=True)
    positions = np.arange(len(game_hns))
    for i, (warmup, color) in enumerate(zip(WARMUPS, ('#60a5fa', '#2563eb', '#f59e0b'))):
        ax.barh(positions + (i - 1) * 0.24, game_hns[warmup], height=0.22,
                label=f'{warmup // 1000}k', color=color)
    ax.set(yticks=positions, yticklabels=game_hns.index,
           xlabel='Mean HNS over common seeds', title='Per-game warmup comparison')
    ax.invert_yaxis()
    ax.axvline(1, color='#64748b', linestyle='--', linewidth=1, label='Human = 1')
    ax.legend()
    ax.grid(axis='x', alpha=0.2)
    fig.savefig(output_dir / 'hns_by_game.png', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=HERE / 'converted_results.xlsx')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'warmup_comparison')
    parser.add_argument('--min-seeds', type=int, default=1, help='Minimum common seeds per game.')
    parser.add_argument('--bootstrap-reps', type=int, default=2000, help='0 disables confidence intervals.')
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap random seed.')
    parser.add_argument('--no-plots', action='store_true', help='Export tables without importing matplotlib.')
    args = parser.parse_args(argv)
    if args.min_seeds < 1 or args.bootstrap_reps < 0 or not 0 < args.confidence < 1 or args.seed < 0:
        parser.error('Require min-seeds >= 1, bootstrap-reps >= 0, 0 < confidence < 1, seed >= 0.')
    output_dir = args.output_dir.resolve()
    if args.input.resolve().parent == output_dir and args.input.name == 'warmup_comparison.xlsx':
        parser.error('The output workbook must not overwrite the input workbook.')
    cells = load_scores(args.input)
    tables = build_tables(cells, args.min_seeds, args.bootstrap_reps, args.confidence, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        ('Input', str(args.input.resolve())),
        ('Allowed configurations', '\n'.join(CONFIGS.values())),
        ('Sheet', 'Results only; all other config rows and sheets are ignored.'),
        ('HNS', '(return - Random) / (Human - Random); Random=0, Human=1; no clipping.'),
        ('Seed score', 'First finite completed entry in workbook order; ignore RUNNING/QUEUED entries.'),
        ('Exclusions', 'Struck-through cells are excluded, matching workbook EXCLUDED_SEEDS marks.'),
        ('Matched panel', 'Intersection of (game, seed) across exactly the three allowed configurations.'),
        ('Minimum common seeds/game', args.min_seeds),
        ('Mean / Median', 'Mean / median of per-game seed-mean HNS; no intermediate rounding.'),
        ('IQM', 'Pool matched seed HNS, sort, discard floor(N/4) at each end, then average.'),
        ('IQM weighting', 'Each seed has equal weight; games with more seeds contribute more to IQM.'),
        ('Optimality Gap', 'Mean over games of max(0, 1 - seed-mean HNS); lower is better.'),
        ('Available Diagnostic', 'Uses each warmup\'s available games/seeds; not a fair ranking.'),
        ('Confidence intervals', f'{args.confidence:.0%} percentile; {args.bootstrap_reps} paired seed bootstrap replicates, RNG seed {args.seed}.'),
        ('CI scope', 'Fixed games, resample seeds within game jointly across warmups; no game resampling.'),
        ('Singleton games', 'No CI if every game has one common seed; singleton games stay fixed otherwise.'),
        ('Warmup axis', 'Requested warmup (30k/50k/70k); actual step retained in Selected Scores.'),
        ('Delta vs 50000', 'Candidate minus 50k on the same matched panel; negative gap delta improves.'),
    ]
    tables['Methodology'] = pd.DataFrame(methods, columns=['Item', 'Definition'])
    with pd.ExcelWriter(output_dir / 'warmup_comparison.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
    for name, filename in [('Matched Summary', 'summary.csv'), ('Delta vs 50000', 'delta_vs_50000.csv'),
                           ('Available Diagnostic', 'available_diagnostic.csv'), ('Per Game', 'per_game.csv'),
                           ('Coverage', 'coverage.csv'), ('Selected Scores', 'selected_scores.csv')]:
        tables[name].to_csv(output_dir / filename, index=False)
    # Do not leave charts from an earlier dataset beside newly written tables,
    # including when --no-plots is used or the common panel becomes empty.
    for filename in PLOT_FILES:
        (output_dir / filename).unlink(missing_ok=True)
    if not args.no_plots:
        plot_results(tables, output_dir, args.confidence)
    summary = tables['Matched Summary']
    print('Only target: 16 (anchor r: 1), warmup 30000 / 50000 / 70000. HNS: Human=1.')
    print(summary[['Warmup Steps', 'Games', 'Game-Seed Cells', *METRICS]].to_string(index=False, float_format='{:.6f}'.format))
    if summary.Games.max() == 0:
        print('No common completed game/seed cells: matched metrics are empty. See Coverage and Available Diagnostic.')
    elif summary['Max Seeds/Game'].max() == 1:
        print('Each common game has only one seed: seed-bootstrap confidence intervals are unavailable.')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
