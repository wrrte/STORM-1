"""Report seed sample SDs and saved bootstrap SEs for both paper backbones.

Use only the exact observations and replicates saved by the earlier CI runs.
No new training, seed selection, score imputation, or bootstrap draws occur.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from compare_warmup import calculate_metrics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCES = {
    'STORM': (HERE, HERE/'main_performance_seeds.xlsx'),
    'DRAMA': (ROOT/'Drama/results', ROOT/'Drama/results/main_performance_sources.json'),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frozen_scores(backbone, results_dir, audit):
    directory = results_dir/'main_performance_bootstrap'
    metadata = json.loads((directory/'metadata.json').read_text())
    if sha256(audit) != metadata['inputs'][str(audit.resolve())]:
        raise ValueError(f'{backbone}: frozen audit differs from the original CI input')
    scores = pd.read_csv(directory/'selected_scores.csv')
    if backbone == 'STORM':
        original = pd.read_excel(audit, sheet_name='시드별 점수').rename(columns={
            '게임': 'Game', '방법': 'Method', '시드': 'Seed', '선택 점수': 'Score'})
        original['Method'] = original.Method.replace({'STORM+ours': 'Full FLASH'})
    else:
        original = pd.json_normalize(json.loads(audit.read_text())['records']).rename(columns={
            'game': 'Game', 'method': 'Method', 'seed': 'Seed', 'score': 'Score'})
    keys = ['Game', 'Seed', 'Method']
    if scores.duplicated(keys).any() or original.duplicated(keys).any():
        raise ValueError(f'{backbone}: duplicate training-seed observations')
    selected, audited = [frame.set_index(keys).sort_index().Score for frame in (scores, original)]
    pd.testing.assert_index_equal(selected.index, audited.index)
    np.testing.assert_allclose(selected, audited, rtol=1e-12, atol=1e-9)
    normalized = (scores.Score-scores.Random)/(scores.Human-scores.Random)
    np.testing.assert_allclose(scores.HNS, normalized, rtol=1e-12, atol=1e-12)
    if not np.isfinite(scores[['Score', 'HNS', 'Random', 'Human']].to_numpy()).all():
        raise ValueError(f'{backbone}: non-finite source score')
    if not (scores.Human > scores.Random).all():
        raise ValueError(f'{backbone}: invalid HNS references')
    if set(scores.Method) != {backbone, 'Full FLASH'}:
        raise ValueError(f'{backbone}: unexpected methods')
    raw = scores.pivot(index=['Game', 'Seed'], columns='Method', values='Score')
    hns = scores.pivot(index=['Game', 'Seed'], columns='Method', values='HNS')
    raw, hns = [p.reindex(columns=[backbone, 'Full FLASH']).sort_index() for p in (raw, hns)]
    if raw.isna().any().any() or hns.isna().any().any():
        raise ValueError(f'{backbone}: incomplete baseline/FLASH seed pairs')
    if len(raw) != metadata['seed_pairs'] or len(raw.groupby(level='Game')) != 26:
        raise ValueError(f'{backbone}: source panel differs from the CI panel')
    return directory, metadata, scores, raw, hns


def game_statistics(backbone, panel):
    rows = []
    for game, values in panel.groupby(level='Game', sort=True):
        base = values[backbone].to_numpy()
        flash = values['Full FLASH'].to_numpy()
        delta = flash-base
        if len(values) < 2:
            raise ValueError(f'{backbone}/{game}: sample SD requires at least two seeds')
        rows.append({'Backbone': backbone, 'Game': game, 'N': len(values),
                     'Baseline Mean': base.mean(), 'Baseline SD': base.std(ddof=1),
                     'Full FLASH Mean': flash.mean(), 'Full FLASH SD': flash.std(ddof=1),
                     'FLASH - Baseline Mean': delta.mean(),
                     'FLASH - Baseline SD': delta.std(ddof=1),
                     'Seeds': ', '.join(map(str, values.index.get_level_values('Seed')))})
        covariance = np.cov(base, flash, ddof=1)[0, 1]
        np.testing.assert_allclose(delta.var(ddof=1),
                                   base.var(ddof=1)+flash.var(ddof=1)-2*covariance,
                                   rtol=1e-9, atol=1e-8)
    return pd.DataFrame(rows)


def metric_standard_errors(backbone, directory, metadata, hns):
    summary = pd.read_csv(directory/'summary.csv').set_index('Metric')
    point = calculate_metrics([group.to_numpy() for _, group in hns.groupby(level='Game', sort=True)])
    q = [(1-metadata['confidence'])/2, (1+metadata['confidence'])/2]
    rows = []
    with np.load(directory/'bootstrap_samples.npz') as samples:
        for metric, values in point.items():
            recorded = summary.loc[metric]
            np.testing.assert_allclose(values, [recorded[backbone], recorded['Full FLASH']], atol=1e-12, rtol=0)
            draws = samples[metric]
            if draws.shape != (metadata['replicates'], 2) or not np.isfinite(draws).all():
                raise ValueError(f'{backbone}/{metric}: invalid saved bootstrap replicates')
            delta = draws[:, 1]-draws[:, 0]
            # Check that these are the replicates used for the published CI report.
            for series, low, high in (
                    (draws[:, 0], f'{backbone} CI Low', f'{backbone} CI High'),
                    (draws[:, 1], 'Full FLASH CI Low', 'Full FLASH CI High'),
                    (delta, 'Delta CI Low', 'Delta CI High')):
                np.testing.assert_allclose(np.quantile(series, q), [recorded[low], recorded[high]], atol=1e-12, rtol=0)
            rows.append({'Backbone': backbone, 'Metric': metric,
                         'Baseline Estimate': values[0], 'Baseline Bootstrap SE': draws[:, 0].std(ddof=1),
                         'Full FLASH Estimate': values[1], 'Full FLASH Bootstrap SE': draws[:, 1].std(ddof=1),
                         'FLASH - Baseline Estimate': values[1]-values[0],
                         'FLASH - Baseline Bootstrap SE': delta.std(ddof=1),
                         'Replicates': len(draws)})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'main_performance_std.xlsx')
    parser.add_argument('--output-dir', type=Path, default=HERE/'main_performance_std')
    args = parser.parse_args(argv)
    raw_tables, hns_tables, metric_tables, paired_tables, provenance = [], [], [], [], []
    for backbone, (results_dir, audit) in SOURCES.items():
        directory, metadata, scores, raw, hns = read_frozen_scores(backbone, results_dir, audit)
        raw_table, hns_table = game_statistics(backbone, raw), game_statistics(backbone, hns)
        references = scores.groupby('Game').first()
        for field in ('Baseline SD', 'Full FLASH SD', 'FLASH - Baseline SD'):
            denominator = raw_table.Game.map(references.Human-references.Random).to_numpy()
            np.testing.assert_allclose(hns_table[field], raw_table[field]/denominator, atol=1e-12, rtol=1e-12)
        raw_tables.append(raw_table)
        hns_tables.append(hns_table)
        metric_tables.append(metric_standard_errors(backbone, directory, metadata, hns))
        pairs = raw.rename(columns={backbone: 'Baseline Score', 'Full FLASH': 'Full FLASH Score'})
        pairs['FLASH - Baseline Score'] = pairs['Full FLASH Score']-pairs['Baseline Score']
        pairs['Baseline HNS'] = hns[backbone]
        pairs['Full FLASH HNS'] = hns['Full FLASH']
        pairs['FLASH - Baseline HNS'] = hns['Full FLASH']-hns[backbone]
        pairs.insert(0, 'Backbone', backbone)
        paired_tables.append(pairs.reset_index())
        for path in (audit, directory/'selected_scores.csv', directory/'metadata.json',
                     directory/'summary.csv', directory/'bootstrap_samples.npz'):
            provenance.append({'Backbone': backbone, 'Source': str(path.resolve()), 'SHA256': sha256(path)})
    raw = pd.concat(raw_tables, ignore_index=True)
    hns = pd.concat(hns_tables, ignore_index=True)
    metrics = pd.concat(metric_tables, ignore_index=True)
    pairs = pd.concat(paired_tables, ignore_index=True)
    notes = pd.DataFrame([
        ('Game SD', 'Sample SD across selected training seeds: sqrt(sum((x-mean)^2)/(n-1)); ddof=1.'),
        ('Observation', 'One final evaluation episode mean per training seed; SD is not over evaluation episodes.'),
        ('Pairing', 'Difference SD is SD(FLASH_seed - baseline_seed), not SD(FLASH) - SD(baseline).'),
        ('Game HNS SD', 'The same statistics after (score-random)/(human-random), without clipping.'),
        ('Metric Bootstrap SE', 'Sample SD of the original 50,000 saved paired-bootstrap metric estimates; an estimator SE, not game-level seed SD.'),
        ('Inputs', 'Frozen paper main-performance observations only: STORM 103 pairs; DRAMA 102 pairs. No hypothetical or newly screened seeds.'),
        ('Seed counts', 'STORM: Gopher/KungFuMaster/UpNDown 3, Krull/PrivateEye 5, others 4; DRAMA: Freeway 2, others 4.'),
    ], columns=['Item', 'Definition'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [('game_sd', raw), ('game_hns_sd', hns), ('metric_bootstrap_se', metrics), ('paired_scores', pairs)]:
        frame.to_csv(args.output_dir/f'{name}.csv', index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        for name, frame in [('Game SD', raw), ('Game HNS SD', hns), ('Metric Bootstrap SE', metrics),
                            ('Paired Seed Scores', pairs), ('Definitions', notes), ('Provenance', pd.DataFrame(provenance))]:
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = 'D2' if name.startswith('Game') else 'C2'
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = PatternFill('solid', fgColor='17365D')
                cell.font = Font(color='FFFFFF', bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            sheet.row_dimensions[1].height = 32
            for index, column in enumerate(frame.columns, 1):
                width = 80 if column in ('Definition', 'Source', 'SHA256') else (36 if column == 'Seeds' else max(14, len(column)+2))
                sheet.column_dimensions[get_column_letter(index)].width = width
                for cell in list(sheet.columns)[index-1][1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = '0.000000' if name != 'Game SD' else '0.00'
    manifest = {'ddof': 1, 'game_rows': len(raw), 'seed_pairs': len(pairs),
                'metric_rows': len(metrics), 'output': str(args.output.resolve()), 'sources': provenance}
    (args.output_dir/'metadata.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(args.output.resolve())
    print(raw[raw.Game == 'Frostbite'].to_string(index=False))
    print(metrics[metrics.Metric == 'IQM'].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
