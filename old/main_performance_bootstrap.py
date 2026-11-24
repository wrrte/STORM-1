"""Paired seed-bootstrap CIs for the paper's baseline vs Full FLASH.

Run from any directory. Inputs are the frozen, audited main-table seed scores,
not the newest score in each live Excel cell. All 26 Atari 100k games are
required. Games remain fixed; seed pairs are resampled within each game.
Use --backbone drama for DRAMA; the default remains STORM.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from compare_warmup import calculate_metrics, bootstrap_metrics, parse_score_cell
from update_tex import main_table_rows, metric_name, number_text, update_table


HERE = Path(__file__).resolve().parent
DRAMA_HERE = HERE.parent.parent/'Drama'/'results'
METHODS = ('STORM', 'Full FLASH')
METRICS = ('Mean', 'Median', 'IQM', 'Optimality Gap')
COMBINED_OUTPUT = HERE.parent.parent/'main_performance_ci.xlsx'


def write_combined_report(output=COMBINED_OUTPUT, result_dirs=None):
    """Put both saved comparisons and all 24 CIs in one overview worksheet."""
    result_dirs = result_dirs or {'STORM': HERE/'main_performance_bootstrap',
                                 'DRAMA': DRAMA_HERE/'main_performance_bootstrap'}
    records, provenance, settings = [], [], []
    for baseline in ('STORM', 'DRAMA'):
        directory = Path(result_dirs[baseline])
        csv_path, meta_path = directory/'summary.csv', directory/'metadata.json'
        summary = pd.read_csv(csv_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if len(summary) != 4 or set(summary.Metric) != set(METRICS):
            raise ValueError(f'{baseline}: expected exactly the four metric results')
        summary = summary.set_index('Metric').loc[list(METRICS)].reset_index()
        for column, key in [('Games', 'games'), ('Seed Pairs', 'seed_pairs'),
                            ('Confidence', 'confidence'), ('Replicates', 'replicates'),
                            ('Bootstrap Seed', 'seed')]:
            if not summary[column].eq(metadata[key]).all():
                raise ValueError(f'{baseline}: summary and metadata disagree on {column}')
        settings.append(f"{baseline}: {metadata['games']} games / {metadata['seed_pairs']} pairs, "
                        f"{metadata['confidence']:.0%} CI, {metadata['replicates']:,} replicates, seed {metadata['seed']}")
        for _, row in summary.iterrows():
            records.append({'Backbone': baseline, 'Metric': row.Metric,
                            'Baseline': row[baseline], 'Baseline CI Low': row[f'{baseline} CI Low'],
                            'Baseline CI High': row[f'{baseline} CI High'], 'Full FLASH': row['Full FLASH'],
                            'Full FLASH CI Low': row['Full FLASH CI Low'],
                            'Full FLASH CI High': row['Full FLASH CI High'],
                            'FLASH - Baseline': row[f'Delta FLASH-{baseline}'],
                            **{key: row[key] for key in ('Delta CI Low', 'Delta CI High',
                                                        'CI Excludes Zero', 'Better', 'Games', 'Seed Pairs',
                                                        'Confidence', 'Replicates', 'Bootstrap Seed')}})
        for path in (csv_path, meta_path):
            provenance.extend([{'Backbone': baseline, 'Item': 'Result file', 'Value': str(path.resolve())},
                               {'Backbone': baseline, 'Item': 'Result SHA256',
                                'Value': hashlib.sha256(path.read_bytes()).hexdigest()}])
        for key, value in metadata.items():
            provenance.append({'Backbone': baseline, 'Item': key,
                               'Value': json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)})

    numeric = pd.DataFrame(records)
    numeric_columns = ['Baseline', 'Baseline CI Low', 'Baseline CI High', 'Full FLASH',
                       'Full FLASH CI Low', 'Full FLASH CI High', 'FLASH - Baseline',
                       'Delta CI Low', 'Delta CI High']
    if not np.isfinite(numeric[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError('Combined CI results must be finite')
    confidence = numeric.Confidence.unique()
    ci_label = f'{confidence[0]:.0%} CI' if len(confidence) == 1 else 'CI'
    overview = []
    for row in records:
        overview.append([row['Backbone'], row['Metric'], row['Baseline'],
                         f"[{row['Baseline CI Low']:.6f}, {row['Baseline CI High']:.6f}]",
                         row['Full FLASH'], f"[{row['Full FLASH CI Low']:.6f}, {row['Full FLASH CI High']:.6f}]",
                         row['FLASH - Baseline'], f"[{row['Delta CI Low']:.6f}, {row['Delta CI High']:.6f}]",
                         '예' if row['CI Excludes Zero'] else '아니오'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(overview, columns=['Backbone', 'Metric', 'Point estimate', ci_label,
                                       'Point estimate', ci_label, 'Difference', ci_label,
                                       '차이 CI가 0 제외']).to_excel(writer, sheet_name='Overview', startrow=4, index=False)
        numeric.to_excel(writer, sheet_name='Numeric Results', index=False)
        pd.DataFrame(provenance).to_excel(writer, sheet_name='Provenance', index=False)
        sheet = writer.book['Overview']
        sheet.sheet_view.showGridLines = False
        sheet.merge_cells('A1:I1')
        sheet['A1'] = f'Main performance: STORM & DRAMA — {ci_label}'
        sheet['A1'].font = Font(name='Calibri', size=18, bold=True, color='17365D')
        sheet.row_dimensions[1].height = 32
        sheet.merge_cells('A2:I2')
        sheet['A2'] = '같은 게임·시드의 baseline / Full FLASH를 함께 재표집한 percentile CI. 차이 = Full FLASH − baseline.'
        sheet.row_dimensions[2].height = 25
        for address, label in [('A4:B4', 'Comparison'), ('C4:D4', 'Baseline'),
                               ('E4:F4', 'Full FLASH'), ('G4:H4', 'Full FLASH − baseline')]:
            sheet.merge_cells(address)
            sheet[address.split(':')[0]] = label
        for row in sheet.iter_rows(min_row=4, max_row=5):
            for cell in row:
                cell.fill = PatternFill('solid', fgColor='17365D')
                cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
                cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        sheet.row_dimensions[4].height = 25
        sheet.row_dimensions[5].height = 32
        for row in sheet.iter_rows(min_row=6, max_row=13):
            for cell in row:
                cell.fill = PatternFill('solid', fgColor='EDF3FA' if cell.row < 10 else 'F1F6EC')
                cell.font = Font(name='Calibri', size=11)
                cell.alignment = Alignment(vertical='center', horizontal='left' if cell.column < 3 else 'center')
                if cell.row == 10:
                    cell.border = Border(top=Side(style='medium', color='8090A0'))
            sheet.row_dimensions[row[0].row].height = 28
            for column in (3, 5):
                sheet.cell(row[0].row, column).number_format = '0.000000'
            sheet.cell(row[0].row, 7).number_format = '+0.000000;-0.000000;0.000000'
            if row[8].value == '예':
                for cell in row[6:9]:
                    cell.fill = PatternFill('solid', fgColor='D9EAD3')
                    cell.font = Font(name='Calibri', size=11, bold=True, color='215E21')
        for column, width in enumerate((12, 20, 16, 27, 16, 27, 16, 27, 16), start=1):
            sheet.column_dimensions[get_column_letter(column)].width = width
        notes = [*settings,
                 'Optimality Gap은 차이가 음수일 때 개선. 그 외 지표는 양수일 때 개선. 지표별 CI이며 다중비교 보정은 하지 않음.',
                 '논문 표시값 재현을 먼저 검사했으며, 위 추정치와 CI는 반올림 전 시드 점수로 계산함.',
                 '숫자 전체 정밀도와 CI 하한·상한은 Numeric Results, 계산 조건과 출처는 Provenance 시트 참조.',
                 'DRAMA: Freeway 2쌍, 나머지 게임 4쌍. 204개 중 185개 점수는 W&B 대조, 19개는 기존 Excel·CSV 기록 대조.']
        for index, note in enumerate(notes, start=15):
            sheet.merge_cells(start_row=index, start_column=1, end_row=index, end_column=9)
            cell = sheet.cell(index, 1, note)
            cell.font = Font(name='Calibri', size=11, color='454545')
            cell.alignment = Alignment(vertical='center', wrap_text=True)
            sheet.row_dimensions[index].height = 25
        sheet.freeze_panes = 'C6'
        sheet.print_options.horizontalCentered = True
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 1
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.print_area = 'A1:I20'
        for name in ('Numeric Results', 'Provenance'):
            detail = writer.book[name]
            detail.freeze_panes = 'C2'
            detail.auto_filter.ref = detail.dimensions
            for cell in detail[1]:
                cell.font = Font(bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='17365D')
            for cells in detail.columns:
                detail.column_dimensions[cells[0].column_letter].width = min(80, max(15, len(str(cells[0].value)) + 3))
        writer.book['Provenance'].column_dimensions['C'].width = 120
        for row in writer.book['Provenance'].iter_rows(min_row=2):
            row[2].alignment = Alignment(wrap_text=True, vertical='top')
    return output


def comparison_methods(baseline):
    if baseline not in ('STORM', 'DRAMA'):
        raise ValueError(f'Unsupported baseline: {baseline}')
    return baseline, 'Full FLASH'


def paper_columns(baseline):
    comparison_methods(baseline)
    return (3, 4) if baseline == 'STORM' else (6, 7)


def drama_renderer():
    # Both result directories have update_tex.py. Load DRAMA under a distinct
    # module name so STORM's module and column constants remain untouched.
    spec = importlib.util.spec_from_file_location('drama_main_table', DRAMA_HERE/'update_tex.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.update_table


def read_paper(path, baseline='STORM'):
    lines = Path(path).read_text(encoding='utf-8').splitlines(keepends=True)
    base_column, flash_column = paper_columns(baseline)
    games, aggregates = {}, {}
    for _, parts, _ in main_table_rows(lines):
        metric = metric_name(parts[0])
        if metric:
            aggregates[metric] = tuple(number_text(parts[i]) for i in (base_column, flash_column))
            continue
        game = parts[0].strip()
        if game in games:
            raise ValueError(f'Duplicate paper game: {game}')
        random_score, human_score = map(float, (parts[1], parts[2]))
        if not np.isfinite([random_score, human_score]).all() or human_score <= random_score:
            raise ValueError(f'{game}: invalid normalization references')
        games[game] = {'Random': random_score, 'Human': human_score,
                       f'Paper {baseline}': float(number_text(parts[base_column])),
                       'Paper Full FLASH': float(number_text(parts[flash_column]))}
    if len(games) != 26:
        raise ValueError(f'Full Atari 100k requires 26 paper games; found {len(games)}')
    return lines, games, aggregates


def load_panel(audit_path, games, baseline='STORM'):
    """Read one audited evaluation mean per method and (game, training seed)."""
    methods = comparison_methods(baseline)
    if baseline == 'DRAMA':
        manifest = json.loads(Path(audit_path).read_text(encoding='utf-8'))
        frame = pd.json_normalize(manifest['records']).rename(
            columns={'game': 'Game', 'method': 'Method', 'seed': 'Seed', 'score': 'Score'})
    else:
        frame = pd.read_excel(audit_path, sheet_name='시드별 점수').rename(
            columns={'게임': 'Game', '방법': 'Method', '시드': 'Seed', '선택 점수': 'Score'})
    required = {'Game', 'Method', 'Seed', 'Score'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Audit columns missing: {sorted(required - set(frame.columns))}')
    if baseline == 'STORM':
        frame['Method'] = frame.Method.replace({'STORM+ours': 'Full FLASH'})
    if set(frame.Method) != set(methods):
        raise ValueError(f'Expected only the audited {baseline} and Full FLASH methods')
    frame['Seed'] = pd.to_numeric(frame.Seed, errors='raise')
    if not np.isfinite(frame.Seed).all() or (frame.Seed < 0).any() or (frame.Seed % 1 != 0).any():
        raise ValueError('Training seeds must be finite nonnegative integers')
    frame['Seed'] = frame.Seed.astype(int)
    frame['Score'] = pd.to_numeric(frame.Score, errors='raise')
    if not np.isfinite(frame.Score).all():
        raise ValueError('Every selected score must be finite')
    if frame.duplicated(['Game', 'Seed', 'Method']).any():
        raise ValueError('Duplicate game/seed/method score; reruns are not independent seeds')
    if set(frame.Game) != set(games):
        raise ValueError(f'Expected exactly all 26 games: missing={set(games)-set(frame.Game)}, '
                         f'extra={set(frame.Game)-set(games)}')
    frame['Random'] = frame.Game.map(lambda g: games[g]['Random'])
    frame['Human'] = frame.Game.map(lambda g: games[g]['Human'])
    frame['HNS'] = (frame.Score - frame.Random) / (frame.Human - frame.Random)
    panel = frame.pivot(index=['Game', 'Seed'], columns='Method', values='HNS')
    panel = panel.reindex(columns=methods).sort_index()
    if panel.isna().any().any():
        raise ValueError('Missing baseline/FLASH partner; incomplete pairs must not be dropped silently')
    if (panel.groupby(level='Game').size() < 2).any():
        raise ValueError('At least two matched training seeds are required for every game')
    return panel, frame


def verify_source_workbook(frame, path):
    """Find the frozen score among current cell entries, including older reruns."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook['Ablation Results']
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        seeds = {int(str(value).strip()): i for i, value in enumerate(header)
                 if str(value).strip().isdigit()}
        cells, game = {}, None
        for row in rows:
            if row[0]:
                game = str(row[0]).splitlines()[0].strip()
            if row[1] in ('Retrieval 미사용', 'target: 16 (anchor 미설정)'):
                for seed, column in seeds.items():
                    cells[(game, row[1], seed)] = row[column]
        for row in frame.itertuples():
            config = 'Retrieval 미사용' if row.Method == 'STORM' else 'target: 16 (anchor 미설정)'
            raw = cells.get((row.Game, config, row.Seed))
            # parse_score_cell removes status and metadata; inspect every
            # completed entry to retain the paper's historical Krull choices.
            entries = re.split(r',\s*(?![^()]*\))|\n', '' if raw is None else str(raw))
            if not any(parse_score_cell(entry)[0] == row.Score for entry in entries):
                raise ValueError(f'Frozen paper score absent from current workbook: '
                                 f'{row.Game}, {row.Method}, seed={row.Seed}, score={row.Score}')
    finally:
        workbook.close()


def verify_drama_sources(audit_path, workbook_path, csv_path):
    """Check the recorded Excel/CSV scores and the six W&B-only additions.

    No network refresh or score reselection is performed. Missing live W&B
    evaluations remain explicitly marked as Excel/CSV-only in the frozen data.
    """
    records = json.loads(Path(audit_path).read_text(encoding='utf-8'))['records']
    csv_rows = pd.read_csv(csv_path).set_index('Run ID')
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        run_rows = pd.read_excel(workbook_path, sheet_name='Runs').set_index('Run ID')
        for row in records:
            label = f"{row['game']}, {row['mode']}, seed={row['seed']}"
            if row['mode'] not in ('X', 'O') or row['method'] != ('DRAMA' if row['mode'] == 'X' else 'Full FLASH'):
                raise ValueError(f'DRAMA method/mode mismatch: {label}')
            config = row['retrieval']
            if (config['warmup_steps'], config['anchor_offset'], config['anchor_weight'],
                config['target'], config['enable']) != (50000, -2, .0625, 16, row['mode'] == 'O'):
                raise ValueError(f'DRAMA configuration differs: {label}')
            logged = row['wandb_eval_score']
            if logged is not None and logged != row['score']:
                raise ValueError(f'Frozen W&B score differs: {label}')
            if not row['workbook_cell']:
                if logged is None:
                    raise ValueError(f'No recorded score source: {label}')
                continue
            sheet, address = row['workbook_cell'].split('!')
            if float(workbook[sheet][address].value) != row['score']:
                raise ValueError(f'Frozen score differs from current workbook: {label}')
            for source in (csv_rows, run_rows):
                if row['run_id'] not in source.index or source.index.tolist().count(row['run_id']) != 1:
                    raise ValueError(f'Missing or duplicate recorded run ID: {label}')
                recorded = source.loc[row['run_id']]
                if (recorded['Game'], int(recorded['Seed']), recorded['Retrieval'],
                    float(recorded['Eval Return'])) != (row['game'], row['seed'], row['mode'], row['score']):
                    raise ValueError(f'Frozen score or run metadata differs from Excel/CSV: {label}')
    finally:
        workbook.close()


def verify_paper(frame, lines, baseline='STORM'):
    """Require every game score and all five paper aggregate rows to reproduce."""
    methods = comparison_methods(baseline)
    results = {}
    for game, rows in frame.groupby('Game', sort=True):
        raw = rows.pivot(index='Seed', columns='Method', values='Score').reindex(columns=methods)
        results[game] = tuple(raw[method].tolist() for method in methods)
    renderer = update_table if baseline == 'STORM' else drama_renderer()
    rendered = renderer(lines, results)
    checks = []
    for (_, old, _), (_, new, _) in zip(main_table_rows(lines), main_table_rows(rendered)):
        label = metric_name(old[0]) or old[0].strip()
        for method, column in zip(methods, paper_columns(baseline)):
            expected, actual = number_text(old[column]), number_text(new[column])
            checks.append({'Row': label, 'Method': method, 'Paper': expected,
                           'Reproduced': actual, 'Matches': expected == actual})
    checks = pd.DataFrame(checks)
    if not checks.Matches.all():
        raise ValueError('Input scores do not reproduce the paper:\n'
                         + checks.loc[~checks.Matches].to_string(index=False))
    return checks


def paired_bootstrap(panel, reps=50000, confidence=0.95, seed=2027, baseline='STORM'):
    """Use one shared seed draw for both methods in every game and replicate.

    Games stay fixed. Keep each game's observed number of seed pairs, and
    recompute nonlinear metrics on every bootstrap sample.
    Intervals for differences come from paired replicate differences, not
    subtraction of the two marginal CI endpoints.
    """
    if reps < 2 or not 0 < confidence < 1:
        raise ValueError('Need reps >= 2 and 0 < confidence < 1')
    if list(panel.columns) != list(comparison_methods(baseline)) or not np.isfinite(panel.to_numpy()).all():
        raise ValueError(f'Expected complete paired {baseline} / Full FLASH observations')
    arrays = [rows.to_numpy(dtype=float) for _, rows in panel.groupby(level='Game', sort=True)]
    if not arrays or any(len(values) < 2 for values in arrays):
        raise ValueError('Need at least two complete seed pairs per game')
    point = calculate_metrics(arrays)
    boot = bootstrap_metrics(arrays, reps, seed)
    quantiles = [(1-confidence)/2, (1+confidence)/2]
    rows = []
    for metric in METRICS:
        base_value, flash = point[metric]
        delta_samples = boot[metric][:, 1] - boot[metric][:, 0]
        baseline_ci = np.quantile(boot[metric][:, 0], quantiles)
        flash_ci = np.quantile(boot[metric][:, 1], quantiles)
        delta_ci = np.quantile(delta_samples, quantiles)
        rows.append({'Metric': metric, baseline: base_value,
                     f'{baseline} CI Low': baseline_ci[0], f'{baseline} CI High': baseline_ci[1],
                     'Full FLASH': flash, 'Full FLASH CI Low': flash_ci[0],
                     'Full FLASH CI High': flash_ci[1], f'Delta FLASH-{baseline}': flash-base_value,
                     'Delta CI Low': delta_ci[0], 'Delta CI High': delta_ci[1],
                     'CI Excludes Zero': bool(delta_ci[0] > 0 or delta_ci[1] < 0),
                     'Better': 'lower' if metric == 'Optimality Gap' else 'higher',
                     'Games': len(arrays), 'Seed Pairs': len(panel),
                     'Confidence': confidence, 'Replicates': reps, 'Bootstrap Seed': seed})
    return pd.DataFrame(rows), boot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', type=str.lower, choices=('storm', 'drama'), default='storm')
    parser.add_argument('--audit', type=Path,
                        help='STORM audited seed XLSX or DRAMA frozen source JSON')
    parser.add_argument('--tex', type=Path, default=HERE.parent.parent/'iclr2027_conference.tex')
    parser.add_argument('--source-workbook', type=Path)
    parser.add_argument('--source-csv', type=Path, help='DRAMA source CSV (checked offline)')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--combine-only', action='store_true',
                        help='Combine existing STORM and DRAMA results without resampling')
    parser.add_argument('--combined-output', type=Path, default=COMBINED_OUTPUT)
    parser.add_argument('--reps', type=int, default=50000)
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args(argv)
    if args.reps < 2 or not 0 < args.confidence < 1 or args.seed < 0:
        parser.error('Require --reps >= 2, 0 < --confidence < 1 and --seed >= 0')
    if args.combine_only:
        print(f'Combined results: {write_combined_report(args.combined_output).resolve()}')
        return
    baseline = args.backbone.upper()
    result_dir = HERE if baseline == 'STORM' else DRAMA_HERE
    args.audit = args.audit or result_dir/('main_performance_seeds.xlsx' if baseline == 'STORM'
                                           else 'main_performance_sources.json')
    args.source_workbook = args.source_workbook or result_dir/('converted_results.xlsx' if baseline == 'STORM'
                                                             else 'drama_results.xlsx')
    args.output_dir = args.output_dir or result_dir/'main_performance_bootstrap'
    inputs = [args.audit, args.tex, args.source_workbook]
    lines, games, _ = read_paper(args.tex, baseline)
    panel, scores = load_panel(args.audit, games, baseline)
    if baseline == 'STORM':
        verify_source_workbook(scores, args.source_workbook)
        source_notes = ['This uses the frozen audited inputs, including Kangaroo seed exclusions and historical Krull rerun scores.']
    else:
        args.source_csv = args.source_csv or result_dir/'drama_wandb_runs.csv'
        inputs.append(args.source_csv)
        verify_drama_sources(args.audit, args.source_workbook, args.source_csv)
        logged = int(scores.wandb_eval_score.notna().sum())
        source_notes = [
            'Freeway uses seeds 2030 and 2710 (two pairs); all other games have four pairs.',
            'Krull uses seeds 1710, 2000, 3710 and 6000; the six Freeway/Krull scores missing from the workbook come from frozen W&B summaries.',
            f'{logged} selected scores matched W&B evaluate/score; {len(scores)-logged} have only Excel/CSV score verification.',
            'Freeway seed 2030, rather than the identical zero-score seed 2040, is inferred from the paper commit and run timestamps.',
        ]
    reproduction = verify_paper(scores, lines, baseline)
    summary, boot = paired_bootstrap(panel, args.reps, args.confidence, args.seed, baseline)
    delta_column = f'Delta FLASH-{baseline}'
    per_game = panel.groupby(level='Game').mean()
    per_game[delta_column] = per_game['Full FLASH'] - per_game[baseline]
    per_game['Seed Pairs'] = panel.groupby(level='Game').size()
    counts = per_game['Seed Pairs'].value_counts().to_dict()
    metadata = {
        'comparison': f'{baseline} baseline vs Full FLASH; paper main performance',
        'games': len(games), 'seed_pairs': len(panel), 'games_by_seed_count': counts,
        'confidence': args.confidence, 'replicates': args.reps, 'seed': args.seed,
        'resampling': 'Paired training-seed draws within each of 26 fixed games; percentile CI',
        'delta': f'Full FLASH minus {baseline}; lower Optimality Gap is better',
        'normalization': '(score - random) / (human - random); no clipping',
        'metrics': {
            'Mean': 'Equal-weight mean across game mean HNS',
            'Median': 'Median across game mean HNS',
            'IQM': 'Pool seed HNS, sort and discard floor(N/4) at each tail, as in the paper; games with more seeds have more weight',
            'Optimality Gap': 'Equal-weight mean across games of max(0, 1 - game mean HNS)',
        },
        'notes': ['Evaluation episode means are the observations; episodes are not bootstrapped as independent training runs.',
                  'Intervals are pointwise and conditional on the selected paper seeds and fixed benchmark games.',
                  'Paper-rounded displayed scores are checked for reproduction; calculations use unrounded seed scores.',
                  *source_notes],
        'method_reference': 'https://github.com/google-research/rliable/blob/master/rliable/library.py',
        'inputs': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir/'summary.csv', index=False)
    markdown = [f'# {baseline} baseline vs Full FLASH: paired bootstrap', '',
                f'26 games, {len(panel)} matched training-seed pairs; '
                f'{args.confidence:.0%} percentile CI, {args.reps:,} replicates, seed {args.seed}.', '',
                f'| Metric | {baseline} | Full FLASH | FLASH−{baseline} | Difference CI |',
                '|---|---:|---:|---:|---|']
    for _, row in summary.iterrows():
        markdown.append(f"| {row['Metric']} | {row[baseline]:.6f} | {row['Full FLASH']:.6f} | "
                        f"{row[delta_column]:+.6f} | "
                        f"[{row['Delta CI Low']:.6f}, {row['Delta CI High']:.6f}] |")
    markdown.extend(['', 'Difference CIs are paired; games stay fixed. '
                     'Negative Optimality Gap differences favor FLASH.',
                     'These intervals are conditional on the selected paper seeds and benchmark games.',
                     *source_notes])
    (args.output_dir/'summary.md').write_text('\n'.join(markdown)+'\n')
    scores.to_csv(args.output_dir/'selected_scores.csv', index=False)
    reproduction.to_csv(args.output_dir/'paper_reproduction.csv', index=False)
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(args.output_dir/'bootstrap_samples.npz', **boot)
    with pd.ExcelWriter(args.output_dir/'paired_bootstrap_ci.xlsx', engine='openpyxl') as writer:
        for name, table in [('Summary', summary), ('Paired HNS', panel.reset_index()),
                            ('Per Game', per_game.reset_index()), ('Selected Scores', scores),
                            ('Paper Reproduction', reproduction)]:
            table.to_excel(writer, sheet_name=name, index=False)
            writer.sheets[name].freeze_panes = 'A2'
    display = ['Metric', baseline, 'Full FLASH', delta_column, 'Delta CI Low', 'Delta CI High']
    print(f'Paper reproduced: 26 games, {len(panel)} paired training seeds; '
          f'{args.confidence:.0%} CI, {args.reps:,} paired bootstrap replicates.')
    print(summary[display].to_string(index=False, float_format=lambda v: f'{v:.6f}'))
    print(f'Results: {args.output_dir.resolve()}')
    combined_dirs = {'STORM': HERE/'main_performance_bootstrap',
                     'DRAMA': DRAMA_HERE/'main_performance_bootstrap'}
    combined_dirs[baseline] = args.output_dir
    if all((directory/'summary.csv').is_file() and (directory/'metadata.json').is_file()
           for directory in combined_dirs.values()):
        print(f'Combined results: {write_combined_report(args.combined_output, combined_dirs).resolve()}')


if __name__ == '__main__':
    main()
