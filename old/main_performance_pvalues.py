"""Two-sided paired label-swap tests for the frozen paper comparisons.

The four original metric definitions are retained. Games are fixed, and method
labels are swapped independently within each matching game/training-seed pair.
This generates a null distribution, not a new bootstrap CI distribution.
"""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from compare_warmup import calculate_metrics
from main_performance_std import ROOT, HERE, SOURCES, read_frozen_scores, sha256


METHOD_REFERENCE = 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html'
METRICS = ('Mean', 'Median', 'IQM', 'Optimality Gap')


def holm_adjust(pvalues):
    """Holm adjustment, preserving input order and allowing dependent tests."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError('Expected a finite one-dimensional array of p-values in [0, 1]')
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(pvalues[order]*np.arange(len(pvalues), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def absolute_tail(observed, null):
    # Include ties; avoid floating-point roundoff excluding equal permutations.
    tolerance = 100*np.finfo(float).eps*max(1., abs(float(observed)))
    return int(np.count_nonzero(np.abs(null) >= abs(observed)-tolerance))


def swapped_metric_differences(panel, swaps):
    """Evaluate all four statistics for explicit pairwise swap masks."""
    values = panel.to_numpy(dtype=float)
    swaps = np.asarray(swaps, dtype=bool)
    if swaps.ndim != 2 or swaps.shape[1] != len(values):
        raise ValueError('Each mask must specify one swap decision per paired observation')
    permuted = np.where(swaps[..., None], values[None, :, ::-1], values[None, :, :])
    groups = panel.groupby(level='Game', sort=True).indices
    metrics = calculate_metrics([permuted[:, indices, :] for indices in groups.values()])
    return {name: value[:, 1]-value[:, 0] for name, value in metrics.items()}


def aggregate_test(panel, permutations=200000, seed=2027, batch=1024):
    if permutations < 1 or batch < 1:
        raise ValueError('Permutation count and batch size must be positive')
    if not np.isfinite(panel.to_numpy()).all() or panel.shape[1] != 2:
        raise ValueError('Expected a complete paired two-method panel')
    point = calculate_metrics([values.to_numpy() for _, values in panel.groupby(level='Game', sort=True)])
    rng = np.random.default_rng(seed)
    null = {metric: np.empty(permutations) for metric in METRICS}
    for start in range(0, permutations, batch):
        count = min(batch, permutations-start)
        # Fixed-width draws keep RNG consumption independent of batch size.
        swaps = rng.integers(0, 2, size=(count, len(panel)), dtype=np.int64).astype(bool)
        for metric, values in swapped_metric_differences(panel, swaps).items():
            null[metric][start:start+count] = values
    rows = []
    for metric in METRICS:
        observed = float(point[metric][1]-point[metric][0])
        extreme = absolute_tail(observed, null[metric])
        p = (extreme+1)/(permutations+1)
        rows.append({'Metric': metric, 'Baseline': point[metric][0], 'Full FLASH': point[metric][1],
                     'FLASH - Baseline': observed, 'p Two-sided': p,
                     'Monte Carlo SE Approx': np.sqrt(p*(1-p)/(permutations+1)),
                     'Extreme Permutations': extreme, 'Permutations': permutations,
                     'Seed Pairs': len(panel), 'Games': len(panel.groupby(level='Game')),
                     'RNG Seed': seed, 'Test': 'paired label-swap; Monte Carlo; absolute two-sided tail'})
    return pd.DataFrame(rows), null


def exact_game_tests(backbone, raw, hns):
    rows = []
    for game, group in raw.groupby(level='Game', sort=True):
        difference = group.iloc[:, 1].to_numpy()-group.iloc[:, 0].to_numpy()
        n = len(group)
        if not 2 <= n <= 20:
            raise ValueError(f'{game}: exact enumeration requires 2..20 pairs')
        signs = np.asarray(list(product((-1., 1.), repeat=n)))
        null = (signs*difference).mean(axis=1)
        observed = difference.mean()
        extreme = absolute_tail(observed, null)
        normalized = hns.loc[game].iloc[:, 1]-hns.loc[game].iloc[:, 0]
        hns_null = (signs*normalized.to_numpy()).mean(axis=1)
        if extreme != absolute_tail(normalized.mean(), hns_null):
            raise ValueError(f'{game}: raw-score and HNS tests disagree')
        rows.append({'Backbone': backbone, 'Game': game, 'N': n,
                     'Baseline Mean': group.iloc[:, 0].mean(), 'Full FLASH Mean': group.iloc[:, 1].mean(),
                     'FLASH - Baseline Mean': observed, 'FLASH - Baseline HNS': normalized.mean(),
                     'p Exact Two-sided': extreme/len(null), 'Extreme Assignments': extreme,
                     'Exact Assignments': len(null),
                     'Seeds': ', '.join(map(str, group.index.get_level_values('Seed'))),
                     'Test': 'exact paired sign-flip test of the game mean difference'})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--permutations', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=ROOT/'main_performance_pvalues.xlsx')
    parser.add_argument('--output-dir', type=Path, default=HERE/'main_performance_pvalues')
    args = parser.parse_args(argv)
    if args.permutations < 1 or args.seed < 0 or args.batch < 1:
        parser.error('Require permutations >= 1, seed >= 0, batch >= 1')
    metric_tables, game_tables, provenance = [], [], []
    null_results = {}
    for backbone, (results_dir, audit) in SOURCES.items():
        directory, metadata, scores, raw, hns = read_frozen_scores(backbone, results_dir, audit)
        tests, null = aggregate_test(hns, args.permutations, args.seed, args.batch)
        tests.insert(0, 'Backbone', backbone)
        prior = pd.read_csv(directory/'summary.csv').set_index('Metric')
        for _, row in tests.iterrows():
            old = prior.loc[row['Metric']]
            np.testing.assert_allclose([old[backbone], old['Full FLASH']],
                                       [row['Baseline'], row['Full FLASH']], atol=1e-12, rtol=0)
        # Merge the original difference CIs to keep estimates and both kinds of inference visible.
        tests = tests.merge(prior[['Delta CI Low', 'Delta CI High']].reset_index(), on='Metric', validate='one_to_one')
        metric_tables.append(tests)
        game_tables.append(exact_game_tests(backbone, raw, hns))
        null_results.update({f'{backbone}_{name}': values for name, values in null.items()})
        for path in (audit, directory/'selected_scores.csv', directory/'metadata.json', directory/'summary.csv'):
            provenance.append({'Backbone': backbone, 'Source': str(path.resolve()), 'SHA256': sha256(path)})
        print(f'{backbone}: {args.permutations:,} paired permutations completed.', flush=True)
    metrics = pd.concat(metric_tables, ignore_index=True)
    games = pd.concat(game_tables, ignore_index=True)
    metrics['p Holm All 8 Metrics'] = holm_adjust(metrics['p Two-sided'])
    metrics['Raw p Below .05'] = metrics['p Two-sided'] < .05
    metrics['Holm p Below .05'] = metrics['p Holm All 8 Metrics'] < .05
    games['p Holm All 52 Games'] = holm_adjust(games['p Exact Two-sided'])
    notes = pd.DataFrame([
        ('Inputs', 'Original frozen paper observations used for CI: STORM 103 pairs, DRAMA 102 pairs, 26 fixed games each.'),
        ('Null assumption', 'Joint baseline/FLASH outcomes are exchangeable under independent method-label swaps within each selected game/training-seed pair.'),
        ('Two-sided test', 'Compare abs(FLASH-baseline statistic) to its paired-swap null distribution; include ties within 100*eps*max(1,abs(observed)).'),
        ('Monte Carlo correction', 'Aggregate p=(extreme_count+1)/(permutations+1). Monte Carlo SE is approximate numerical simulation error, not scientific effect uncertainty.'),
        ('Metric definitions', 'Mean/Median/Optimality Gap use equal-weight game mean HNS. IQM pools seed HNS and trims floor(N/4) observations from each tail, as in the paper.'),
        ('Game tests', 'Enumerate all 2^N swaps; test each game mean difference. Raw-score and HNS tests are checked to agree.'),
        ('Multiplicity', 'Holm correction for all 8 aggregate tests; separate Holm family for all 52 game tests. Raw p-values are also retained.'),
        ('Conditioning', 'Conditional on these fixed 26 games and selected paper seeds. No adjustment for historical seed selection; no screened Frostbite or hypothetical scores.'),
        ('CI versus p', 'The previously saved percentile bootstrap CIs and these paired-swap tests use different inference constructions and need not agree exactly.'),
        ('Reference', METHOD_REFERENCE),
    ], columns=['Item', 'Definition'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir/'metric_pvalues.csv', index=False)
    games.to_csv(args.output_dir/'game_pvalues.csv', index=False)
    np.savez_compressed(args.output_dir/'permutation_null_samples.npz', **null_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        for name, frame in [('Metric P-values', metrics), ('Game P-values', games),
                            ('Definitions', notes), ('Provenance', pd.DataFrame(provenance))]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'C2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='17365D')
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            sheet.row_dimensions[1].height = 34
            for index, column in enumerate(frame.columns, 1):
                sheet.column_dimensions[get_column_letter(index)].width = (
                    90 if column in ('Definition', 'Source', 'SHA256') else min(50, max(14, len(column)+2)))
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = '0.000000'
    manifest = {'method': 'two-sided paired label-swap permutation test', 'permutations': args.permutations,
                'rng_seed': args.seed, 'batch': args.batch, 'metric_tests': len(metrics), 'game_tests': len(games),
                'p_correction': '(extreme+1)/(B+1) for Monte Carlo; extreme/2^N for exact tests',
                'multiple_testing': {'aggregate_family': 8, 'game_family': 52, 'adjustment': 'Holm'},
                'sources': provenance, 'reference': METHOD_REFERENCE, 'output': str(args.output.resolve())}
    (args.output_dir/'metadata.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(metrics[['Backbone', 'Metric', 'FLASH - Baseline', 'p Two-sided', 'p Holm All 8 Metrics',
                   'Monte Carlo SE Approx']].to_string(index=False))
    print(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
