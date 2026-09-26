"""Update appendix ablations and a compact main-text summary from Excel seed scores.

Usage: python STORM/results/update_tex_ablation.py [--excel PATH] [--tex PATH]
       [--main-games Frostbite Gopher KungFuMaster]
Edit MAIN_GAMES below to set the default game rows (in display order). The
--main-games option overrides that list for one run; --main-games alone shows
only metrics. Re-running regenerates the entire summary, including its layout.
Uses the same result loading, duplicate-result parsing, and aggregation as update_tex.py.
Each variant is paired independently with default FLASH on common training seeds.
The main summary shares Full FLASH for value/add only when all paired seed IDs
and baseline scores agree. Neighbor retrieval always keeps its own baseline.
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
# 본문에 표시할 게임을 원하는 순서로 지정하세요. CLI --main-games로도 변경 가능합니다.
MAIN_GAMES = ['Alien', 'Assault', 'BankHeist', 'ChopperCommand', 'CrazyClimber', 'Gopher', 'Jamesbond', 'MsPacman', 'Pong', 'Qbert']
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


def render_ablation_table(lines, results, ablation=NEIGHBOR):
    # Reuse the main-table calculation in memory for identical normalization,
    # rounding, and pooled per-seed IQM. Project only the two score columns.
    calculated_rows = main_table_rows(update_table(lines, results))
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
        r'The number of training seeds can differ across ablations '
        r'and from Table~\ref{tab:main_performance}. '
        r'A dash indicates no paired results. Aggregate metrics summarize games with '
        r'paired results using the Random/Human references in Table~\ref{tab:main_performance}. '
        r'Mean, Median, and Optimality Gap use human-normalized game means; IQM pools '
        r'the unrounded per-seed human-normalized scores. '
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
            if (len(scores) != 2 or not scores[0] or len(scores[0]) != len(scores[1])
                    or any(not math.isfinite(score) for method in scores for score in method)):
                raise ValueError(f'Invalid paired scores for {key}: {game}.')
        # Never pass the selected display games to the aggregation routine.
        cells[key] = {parts[0].strip(): parts
                      for _, parts, _ in main_table_rows(update_table(lines, results[key]))}

    shared = shared_trigger_baseline(results, paired_seeds)
    groups = [('Neighbor retrieval', [('neighbor', BASE_COLUMN, 'Full FLASH'),
                                      ('neighbor', OURS_COLUMN, 'No neighbor')])]
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
        r'\ref{tab:value_signal_ablation}, and \ref{tab:score_combination_ablation}). '
        r'Game scores use the paired training seeds of each appendix comparison. '
        r'Aggregate metrics match the corresponding appendix tables and use all '
        r'games with paired results for each comparison, independently of the game rows shown; '
        r'IQM pools per-seed human-normalized scores. '
    )
    if shared:
        caption += (r'Value and combination ablations share the same Full FLASH seeds '
                    r'and scores; neighbor retrieval has a separate baseline. ')
    else:
        caption += r'Full FLASH is shown separately where paired seeds or scores differ. '
    caption += (r'Bold indicates the best score within each comparison group '
                r'(lower for Optimality Gap), including ties.')
    table = [
        r'\begin{table}[!htbp]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}',
        r'\caption{' + caption + '}', r'\label{tab:ablation_main}',
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
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\FloatBarrier'])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--excel', type=Path, default=script_dir / 'converted_results.xlsx')
    parser.add_argument('--tex', type=Path, default=script_dir.parent.parent / 'iclr2027_conference.tex')
    parser.add_argument('--main-games', nargs='*', default=MAIN_GAMES, metavar='GAME',
                        help='Game rows in display order; overrides MAIN_GAMES for this run. '
                             'Pass no names to show only the appendix aggregate metrics.')
    args = parser.parse_args()

    with args.tex.open(encoding='utf-8', newline='') as source:
        document = source.read()
    results, paired_seeds = {}, {}
    for key, ablation in ABLATIONS.items():
        results[key], paired_seeds[key] = load_results(
            args.excel, configs=ablation.configs, method_names=ablation.method_names,
            include_seeds=True,
        )
        document = update_ablation_table(document, results[key], ablation)
    document = update_main_ablation_table(document, results, paired_seeds, args.main_games)
    with args.tex.open('w', encoding='utf-8', newline='') as destination:
        destination.write(document)
    print(f'Successfully updated {args.tex} with appendix and main-text ablation results.')


if __name__ == '__main__':
    main()
