"""Update appendix ablations and a compact main-text summary from Excel seed scores.

Usage: python STORM-1/results/update_tex_ablation.py [--excel PATH] [--tex PATH]
       [--main-games Frostbite Gopher KungFuMaster] [--random-study STUDY]
Edit MAIN_GAMES below to set the default game rows (in display order). The
--main-games option overrides that list for one run; --main-games alone shows
only metrics. Re-running regenerates the entire summary, including its layout.
Uses the same result loading, duplicate-result parsing, and aggregation as update_tex.py.
Neighbor retrieval, value signal, and score combination are paired independently
with default FLASH on common training seeds. Random neighbors reads the current
study's Paper Random Neighbor worksheet. Only completed study runs are paired
with their same-game/seed Full FLASH scores. Audited historical results may be
explicitly reused by the selected study; intermediate evaluations are excluded.
Aggregate metrics include only games with scores for both methods, while game
rows still show every available score.
The main summary shares Full FLASH for value/add only when all paired seed IDs
and baseline scores agree. Neighbor retrieval and random-neighbor comparisons
always keep their own baselines.
Main-summary metrics match the appendix's available-game metrics exactly,
regardless of the selected game rows (including when fewer than 26 are available).
Bold marks the best displayed score within each comparison group (including ties),
with lower scores preferred only for Optimality Gap.
Only the marked ablation tables are written; the main summary's markers are
inserted in subsec:ablation_main on the first run. Once installed, the markers
locate the table independently of surrounding prose, headings, and section labels.
Keep each marker pair around just its table and float barrier. Malformed or
ambiguous blocks raise an error before saving. Main performance is untouched.
Each table ends with a float barrier to keep it within its ablation subsection.
The target document must load the placeins package.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import pandas as pd

from update_tex import (
    BASE_COLUMN,
    OURS_COLUMN,
    extract_float,
    load_results,
    main_table_rows,
    metric_name,
    update_table,
)


FLASH_CONFIG = 'target: 16 (anchor 미설정)'
RANDOM_STUDY = 'neighbor_reuse_20261107'
# 본문에 표시할 게임을 원하는 순서로 지정하세요. CLI --main-games로도 변경 가능합니다.
# Current selections use two HNS changes per direction and one percentage change;
# an overlapping percentage choice is replaced by the third HNS change.
MAIN_GAMES = [
    'Assault',
    'BankHeist',
    'Boxing',
    'ChopperCommand',
    'Gopher',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MsPacman',
    'Pong',
    'PrivateEye',
]
ATARI_GAME_COUNT = 26
MAIN_BEGIN_MARKER = '% BEGIN AUTO MAIN ABLATION'
MAIN_END_MARKER = '% END AUTO MAIN ABLATION'


@dataclass(frozen=True)
class Ablation:
    variant_config: str
    variant_name: str
    marker_name: str
    label: str
    caption: str

    @property
    def configs(self):
        return (FLASH_CONFIG, self.variant_config)

    @property
    def method_names(self):
        return ('Full FLASH', self.variant_name)

    @property
    def begin_marker(self):
        return f'% BEGIN AUTO {self.marker_name} ABLATION'

    @property
    def end_marker(self):
        return f'% END AUTO {self.marker_name} ABLATION'


ABLATIONS = {
    'neighbor': Ablation(
        'target: 1 (anchor 미설정)', 'No neighbor', 'NEIGHBOR RETRIEVAL',
        'tab:neighbor_retrieval_ablation',
        r'Neighbor retrieval ablation on STORM. Full FLASH uses retrieval '
        r'target $n=16$; No neighbor uses $n=1$ (anchor only), both with the default '
        r'anchor setting. ',
    ),
    'random': Ablation(
        FLASH_CONFIG + ' [RandomNeighor]', 'Random neighbors', 'RANDOM NEIGHBOR',
        'tab:random_neighbor_ablation',
        r'Neighbor source ablation on STORM. Full FLASH uses hash-matched '
        r'neighbors; Random neighbors replaces the non-anchor contexts with '
        r'uniformly sampled valid replay contexts. Both use retrieval target '
        r'$n=16$ and the same anchor-selection and weighting rules. '
        r'Scores are averaged over matched training seeds. ',
    ),
    'value': Ablation(
        FLASH_CONFIG + ' [value]', 'Absolute value', 'VALUE SIGNAL',
        'tab:value_signal_ablation',
        r'Value signal ablation on STORM. Full FLASH uses the signed temporal '
        r'value difference $V(s_t)-V(s_{t-1})$; Absolute value uses the state '
        r'value $V(s_t)$. Both use retrieval target $n=16$, the default anchor '
        r'setting, and multiplicative score combination. ',
    ),
    'add': Ablation(
        FLASH_CONFIG + ' [add]', 'Additive', 'SCORE COMBINATION',
        'tab:score_combination_ablation',
        r'Score combination ablation on STORM. Full FLASH multiplies the '
        r'ReLU-transformed normalized TD-error and softplus-transformed '
        r'normalized temporal value difference; Additive sums these terms. '
        r'Both use retrieval target $n=16$ and the default anchor setting. ',
    ),
}

# Preserve the original neighbor-table defaults for callers updating one table.
NEIGHBOR = ABLATIONS['neighbor']
CONFIGS = NEIGHBOR.configs
METHOD_NAMES = NEIGHBOR.method_names
BEGIN_MARKER = NEIGHBOR.begin_marker
END_MARKER = NEIGHBOR.end_marker


def bold_best_scores(label, cells):
    """Highlight displayed maxima (minima for Optimality Gap), including ties."""
    values = [extract_float(cell) for cell in cells]
    available = [value for value in values if value is not None]
    if not available:
        return cells
    best = (min if metric_name(label) == 'Optimality Gap' else max)(available)
    return [rf'\textbf{{{cell}}}' if value == best else cell
            for cell, value in zip(cells, values)]


def load_random_neighbor_results(excel_path, study=RANDOM_STUDY, include_seeds=False):
    """Pair the selected study's completed scores by game and training seed."""
    try:
        frame = pd.read_excel(excel_path, sheet_name='Paper Random Neighbor')
    except ValueError as error:
        raise ValueError('Missing Paper Random Neighbor worksheet. Refresh the workbook '
                         'with classify_wandb_runs.py --offline --excel first.') from error
    flash_column = 'Full FLASH' if 'Full FLASH' in frame else 'Paper Full FLASH'
    required = {'Study', 'Game', 'Seed', 'Status', flash_column, 'Random Neighbor',
                'Warmup Steps', 'Anchor Offset', 'Local Run Name', 'Eval Checkpoint Step'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Paper Random Neighbor columns missing: {sorted(required-set(frame.columns))}')
    frame = frame.loc[frame.Study.eq(study)].copy()
    if frame.empty:
        raise ValueError(f'Random-neighbor study not found in workbook: {study}')
    frame['Seed'] = pd.to_numeric(frame.Seed, errors='raise')
    if any(not math.isfinite(seed) or seed < 0 or seed % 1 for seed in frame.Seed):
        raise ValueError('Random-neighbor study seeds must be nonnegative integers')
    frame['Seed'] = frame.Seed.astype(int)
    if frame.duplicated(['Game', 'Seed']).any() or frame['Local Run Name'].duplicated().any():
        raise ValueError('Duplicate game/seed or run in the random-neighbor study')
    if (not pd.to_numeric(frame['Warmup Steps'], errors='coerce').eq(50000).all()
            or not pd.to_numeric(frame['Anchor Offset'], errors='coerce').eq(-2).all()):
        raise ValueError('Expected random-neighbor warmup_steps=50000 and anchor_offset=-2')
    frame = frame.loc[frame.Status.astype(str).str.upper().eq('FINISHED')].copy()
    for column in (flash_column, 'Random Neighbor'):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not all(math.isfinite(score) for score in frame[column]):
            raise ValueError(f'Completed random-neighbor result has no finite {column} score')
    checkpoints = pd.to_numeric(frame['Eval Checkpoint Step'], errors='raise').dropna()
    if not checkpoints.eq(100000).all():
        raise ValueError('Random-neighbor results must evaluate the 100k checkpoint')
    results, paired_seeds = {}, {}
    for game, rows in frame.sort_values(['Game', 'Seed']).groupby('Game', sort=True):
        results[game] = (rows[flash_column].tolist(), rows['Random Neighbor'].tolist())
        paired_seeds[game] = rows.Seed.tolist()
    print(f'Random neighbors ({study}): {len(frame)} paired scores across {len(results)} games.')
    return (results, paired_seeds) if include_seeds else results


def ablation_table_rows(lines, results):
    """Show available game scores; aggregate only games with both methods present."""
    method_rows = []
    for method in (0, 1):
        available = {game: (scores[method], scores[method])
                     for game, scores in results.items() if scores[method]}
        method_rows.append(main_table_rows(update_table(lines, available)))
    common_results = {game: scores for game, scores in results.items() if all(scores)}
    metric_rows = main_table_rows(update_table(lines, common_results))
    for (_, parts, _), (_, variant_parts, _), (_, metric_parts, _) in zip(
            *method_rows, metric_rows, strict=True):
        if metric_name(parts[0]):
            parts[BASE_COLUMN] = metric_parts[BASE_COLUMN]
            parts[OURS_COLUMN] = metric_parts[OURS_COLUMN]
        else:
            parts[OURS_COLUMN] = variant_parts[OURS_COLUMN]
    return method_rows[0]


def render_ablation_table(lines, results, ablation=NEIGHBOR):
    # Reuse the main-table calculation in memory for identical normalization,
    # rounding, and pooled per-seed IQM. Project only the two score columns.
    calculated_rows = ablation_table_rows(lines, results)
    games = [parts[0].strip() for _, parts, _ in calculated_rows
             if metric_name(parts[0]) is None]
    missing_references = set(results) - set(games)
    if missing_references:
        raise ValueError(f"Games missing from the main table: {sorted(missing_references)}")

    table = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\small',
        r'\caption{' + ablation.caption +
        r'Mean, Median, and Optimality Gap use human-normalized game means; IQM pools '
        r'the unrounded human-normalized scores. '
        r'Bold indicates the best score in each pair (lower for Optimality Gap), '
        r'including ties.}',
        rf'\label{{{ablation.label}}}',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        f'Game & {ablation.method_names[0]} & {ablation.method_names[1]} ' + r'\\',
        r'\midrule',
    ]
    metrics_started = False
    for _, parts, _ in calculated_rows:
        label = parts[0].strip()
        is_metric = metric_name(label) is not None
        if ablation == ABLATIONS['random'] and not is_metric and label not in results:
            continue
        if is_metric and not metrics_started:
            table.append(r'\midrule')
            metrics_started = True
        full, variant = bold_best_scores(label, [parts[BASE_COLUMN].strip(),
                                                parts[OURS_COLUMN].strip()])
        table.append(f'{label} & {full} & {variant} ' + r'\\')
    # Flush the table before the next subsection or Extended Related Work.
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\FloatBarrier'])
    return '\n'.join(table) + '\n'


def update_ablation_table(document, results, ablation=NEIGHBOR):
    """Replace exactly one marked table while preserving all surrounding text."""
    lines = document.splitlines(keepends=True)
    bounds = marked_table_bounds(lines, ablation.begin_marker, ablation.end_marker,
                                 ablation.label)
    if bounds is None:
        raise ValueError(f'Missing {ablation.marker_name} ablation table markers.')
    table = render_ablation_table(lines, results, ablation)
    return replace_marked_table(lines, bounds, table)


def tex_code(line):
    """Ignore LaTeX comments when inspecting labels and section boundaries."""
    return re.split(r'(?<!\\)%', line, maxsplit=1)[0]


def marked_table_bounds(lines, begin_marker, end_marker, label):
    """Reject ambiguous or oversized blocks before replacing any document text."""
    begins = [i for i, line in enumerate(lines) if line.strip() == begin_marker]
    ends = [i for i, line in enumerate(lines) if line.strip() == end_marker]
    if not begins and not ends:
        return None
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise ValueError(f'Expected exactly one ordered pair of {begin_marker} markers.')
    begin, end = begins[0], ends[0]
    body = '\n'.join(tex_code(line) for line in lines[begin + 1:end])
    label_text = rf'\label{{{label}}}'
    if (not re.fullmatch(r'\s*\\begin\{table\}.*\\end\{table\}\s*'
                         r'(?:\\FloatBarrier\s*)?', body, flags=re.DOTALL)
            or body.count(r'\begin{table}') != 1
            or body.count(r'\end{table}') != 1
            or body.count(label_text) != 1
            or sum(tex_code(line).count(label_text) for line in lines) != 1
            or any(line.strip().startswith(('% BEGIN AUTO ', '% END AUTO '))
                   for line in lines[begin + 1:end])):
        raise ValueError(f'Markers for {label} must enclose only its single table '
                         'and optional FloatBarrier, with a unique table label.')
    return begin, end


def replace_marked_table(lines, bounds, table):
    begin, end = bounds
    newline = '\r\n' if lines[begin].endswith('\r\n') else '\n'
    return (''.join(lines[:begin + 1]) + table.replace('\n', newline)
            + ''.join(lines[end:]))


def shared_trigger_baseline(results, paired_seeds):
    """Compare seed identities and unrounded scores across the entire benchmark."""
    def signature(key):
        return {
            game: dict(zip(paired_seeds[key][game], scores[0], strict=True))
            for game, scores in results[key].items()
        }

    return signature('value') == signature('add')


def render_main_ablation_table(lines, results, paired_seeds, games):
    """Use the same available-game cells as the appendix, then select display rows."""
    source_rows = main_table_rows(lines)
    references = {parts[0].strip(): parts for _, parts, _ in source_rows
                  if metric_name(parts[0]) is None}
    if len(references) != ATARI_GAME_COUNT:
        raise ValueError(f'Main ablation summary requires {ATARI_GAME_COUNT} game references; '
                         f'found {len(references)} in tab:main_performance.')
    for game, parts in references.items():
        random, human = (extract_float(parts[column]) for column in (1, 2))
        if (random is None or human is None or not math.isfinite(random)
                or not math.isfinite(human) or random == human):
            raise ValueError(f'Invalid Random/Human references for {game}.')
    unknown = set(games) - references.keys()
    if unknown:
        raise ValueError(f'Unknown --main-games entries: {sorted(unknown)}. '
                         f'Choose from: {", ".join(references)}')
    if len(games) != len(set(games)):
        raise ValueError('--main-games must not contain duplicate games.')

    cells = {}
    for key in ABLATIONS:
        unknown_results = results[key].keys() - references.keys()
        if unknown_results:
            raise ValueError(f'Games missing from the main table: {sorted(unknown_results)}')
        for game, scores in results[key].items():
            if (len(scores) != 2
                    or (not any(scores) if key == 'random' else
                        not scores[0] or len(scores[0]) != len(scores[1]))
                    or any(not math.isfinite(score) for method in scores for score in method)):
                raise ValueError(f'Invalid scores for {key}: {game}.')
        # Never pass the selected display games to the aggregation routine.
        cells[key] = {parts[0].strip(): parts
                      for _, parts, _ in ablation_table_rows(lines, results[key])}

    shared = shared_trigger_baseline(results, paired_seeds)
    groups = [('Neighbor retrieval', [('neighbor', BASE_COLUMN, 'Full FLASH'),
                                      ('neighbor', OURS_COLUMN, 'No neighbor')]),
              ('Neighbor source', [('random', BASE_COLUMN, 'Full FLASH'),
                                   ('random', OURS_COLUMN, 'Random neighbors')])]
    if shared:
        groups.append(('Value signal / Combination', [
            ('value', BASE_COLUMN, 'Full FLASH'),
            ('value', OURS_COLUMN, 'Absolute value'),
            ('add', OURS_COLUMN, 'Additive'),
        ]))
    else:
        groups.extend([
            ('Value signal', [('value', BASE_COLUMN, 'Full FLASH'),
                              ('value', OURS_COLUMN, 'Absolute value')]),
            ('Combination', [('add', BASE_COLUMN, 'Full FLASH'),
                             ('add', OURS_COLUMN, 'Additive')]),
        ])
    columns = [column for _, group in groups for column in group]
    caption = (
        r'Ablations on STORM (Tables~\ref{tab:neighbor_retrieval_ablation}, '
        r'\ref{tab:random_neighbor_ablation}, \ref{tab:value_signal_ablation}, '
        r'and \ref{tab:score_combination_ablation}). '
        r'Aggregate metrics match the corresponding appendix tables. '
    )
    caption += (r'Bold indicates the best score within each comparison group '
                r'(lower for Optimality Gap), including ties.')
    table = [
        r'\begin{table}[!htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\caption{' + caption + '}', r'\label{tab:ablation_main}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{@{}l' + 'r' * len(columns) + r'@{}}', r'\toprule',
        ' & ' + ' & '.join(rf'\multicolumn{{{len(group)}}}{{c}}{{{title}}}'
                          for title, group in groups) + r' \\',
    ]
    offset = 2
    rules = []
    for _, group in groups:
        rules.append(rf'\cmidrule(lr){{{offset}-{offset + len(group) - 1}}}')
        offset += len(group)
    headers = [r'\shortstack{' + name.replace(' ', r'\\') + '}'
               for _, _, name in columns]
    table.extend([' '.join(rules),
                  'Game / Metric & ' + ' & '.join(headers) + r' \\',
                  r'\midrule'])
    metrics = [parts[0].strip() for _, parts, _ in source_rows if metric_name(parts[0])]
    for label in [*games, *metrics]:
        if games and label == metrics[0]:
            table.append(r'\midrule')
        values = [value for _, group in groups
                  for value in bold_best_scores(label, [cells[key][label][column].strip()
                                                        for key, column, _ in group])]
        table.append(' & '.join([label, *values]) + r' \\')
    table.extend([r'\bottomrule', r'\end{tabular}', '}', r'\end{table}', r'\FloatBarrier'])
    return '\n'.join(table) + '\n'


def update_main_ablation_table(document, results, paired_seeds, games=MAIN_GAMES):
    """Install or replace the generated block without changing the user's prose."""
    lines = document.splitlines(keepends=True)
    bounds = marked_table_bounds(lines, MAIN_BEGIN_MARKER, MAIN_END_MARKER,
                                 'tab:ablation_main')
    if bounds is not None:
        table = render_main_ablation_table(lines, results, paired_seeds, games)
        return replace_marked_table(lines, bounds, table)

    # Without markers, never append a second copy of an existing summary.
    if any(r'\label{tab:ablation_main}' in tex_code(line) for line in lines):
        raise ValueError('Existing tab:ablation_main is missing its AUTO MAIN ABLATION '
                         'markers; restore them around the table before updating.')
    labels = [i for i, line in enumerate(lines)
              for _ in re.finditer(re.escape(r'\label{subsec:ablation_main}'), tex_code(line))]
    if len(labels) != 1:
        raise ValueError('Expected exactly one \\label{subsec:ablation_main}.')
    start = labels[0] + 1
    stop = next((i for i in range(start, len(lines))
                 if re.match(r'\s*\\(?:subsection|section|appendix)\b|\s*\\end\{document\}',
                             tex_code(lines[i]))), len(lines))
    table = render_main_ablation_table(lines, results, paired_seeds, games)
    newline = '\r\n' if lines[labels[0]].endswith('\r\n') else '\n'
    block = MAIN_BEGIN_MARKER + '\n' + table + MAIN_END_MARKER + '\n\n'
    return ''.join(lines[:stop]) + newline + block.replace('\n', newline) + ''.join(lines[stop:])


def update_ablation_tables(document, results, paired_seeds, games=MAIN_GAMES):
    """Render all comparisons in memory before the caller writes the document."""
    for key, ablation in ABLATIONS.items():
        document = update_ablation_table(document, results[key], ablation)
    return update_main_ablation_table(document, results, paired_seeds, games)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--excel', type=Path, default=script_dir / 'converted_results.xlsx')
    parser.add_argument('--tex', type=Path, default=script_dir.parent.parent / 'iclr2027_conference.tex')
    parser.add_argument('--random-study', default=RANDOM_STUDY,
                        help='Study ID in the Paper Random Neighbor worksheet')
    parser.add_argument('--main-games', nargs='*', default=MAIN_GAMES, metavar='GAME',
                        help='Game rows in display order; overrides MAIN_GAMES for this run. '
                             'Pass no names to show only the appendix aggregate metrics.')
    args = parser.parse_args()

    with args.tex.open(encoding='utf-8', newline='') as source:
        document = source.read()
    results, paired_seeds = {}, {}
    for key, ablation in ABLATIONS.items():
        if key == 'random':
            results[key], paired_seeds[key] = load_random_neighbor_results(
                args.excel, study=args.random_study, include_seeds=True)
            continue
        results[key], paired_seeds[key] = load_results(
            args.excel, configs=ablation.configs, method_names=ablation.method_names,
            include_seeds=True,
        )
    document = update_ablation_tables(document, results, paired_seeds, args.main_games)
    with args.tex.open('w', encoding='utf-8', newline='') as destination:
        destination.write(document)
    print(f'Successfully updated {args.tex} with appendix and main-text ablation results.')


if __name__ == '__main__':
    main()
