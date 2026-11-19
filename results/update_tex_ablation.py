"""Update appendix ablations and a compact main-text summary from Excel seed scores.

Usage: python STORM-1/results/update_tex_ablation.py [--excel PATH] [--tex PATH]
       [--main-games Frostbite Gopher KungFuMaster]
Edit MAIN_GAMES below to set the default game rows (in display order). The
--main-games option overrides that list for one run; --main-games alone shows
only metrics. Re-running regenerates the entire summary, including its layout.
Uses the same result loading, duplicate-result parsing, and aggregation as update_tex.py.
Neighbor retrieval, value signal, and score combination are paired independently
with default FLASH on common training seeds. Random-neighbor candidates come
from its exact config row, and Full FLASH candidates come from update_tex.py's
main-performance selection (seeds shared with STORM without retrieval).
For every game, both methods use the smaller available seed count and retain
their highest-scoring seeds independently; ties use ascending seed IDs. Seed
IDs may differ, but counts always agree. Historical exclusions and within-cell
duplicate parsing apply to both methods.
For Gopher only, Random neighbors also accepts the anchor r=1 configuration,
keeps the higher score per seed across the two configurations, and averages
at most the four highest-scoring distinct seeds (ties use ascending seed IDs).
Games without scores for either method are omitted from the random-neighbor
comparison. Aggregate metrics include only games with scores for both methods.
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
TD-error only uses its exact threshold/anchor configuration and is paired with
Full FLASH independently of the other comparisons. Its subsection is installed
at the end of the document on the first run. Pending runs are never table scores.
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
TD_ERROR_ONLY_CONFIG = 'target: 16 (anchor r: 1) [td_error_only, threshold: 3.5]'
GOPHER_RANDOM_CONFIG = 'target: 16 (anchor r: 1) [RandomNeighor]'
GOPHER_RANDOM_SEED_LIMIT = 4
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
MAIN_ABLATION_KEYS = ('neighbor', 'random', 'value', 'add')

TD_ERROR_ONLY_SECTION = r'''\clearpage
\subsection{Ablation 5: TD-Error-Only Anchor Selection}
\label{subsec:td_error_only_ablation}

This ablation selects anchors using normalized TD-error magnitude without the temporal value-difference factor. With the same per-environment EMA normalization as Eq.~\eqref{eq:salience}, its score is
\begin{equation}
S_t^{\mathrm{TD}}=\operatorname{ReLU}\!\left(\frac{\delta_t^{\mathrm{TD}}-\bar{\mu}_e}{\sqrt{\bar{\sigma}_e^2}+\epsilon}\right),\qquad \tau_{\mathrm{TD}}=3.5.
\label{eq:appendix_td_error_only_salience}
\end{equation}
The highest-scoring position in each replay sequence triggers an anchor when its score reaches $\tau_{\mathrm{TD}}$. The TD-error-only configuration retains hash-indexed neighbor retrieval with target $n=16$, a 50,000-step warmup, and anchor offset $-2$. It uses relative anchor weight $r=1$; the Full FLASH reference uses the default fixed anchor weight $1/16$.

Table~\ref{tab:td_error_only_ablation} compares final evaluation returns for Full FLASH and TD-error only.

'''


@dataclass(frozen=True)
class Ablation:
    variant_config: str
    variant_name: str
    marker_name: str
    label: str
    caption: str
    omit_unavailable_games: bool = False
    appendix_section: str = ''

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
        r'$n=16$ and the default anchor setting, except that the Gopher random '
        r'candidate pool also includes the $r=1$ anchor setting. '
        r'For each game, equal numbers of highest-scoring completed runs are '
        r'averaged, with selection performed independently for the two methods. ',
        omit_unavailable_games=True,
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
    'td_error_only': Ablation(
        TD_ERROR_ONLY_CONFIG, 'TD-error only', 'TD ERROR ONLY',
        'tab:td_error_only_ablation',
        r'TD-error-only anchor selection on STORM. Full FLASH combines '
        r'normalized TD-error and temporal value difference multiplicatively; '
        r'TD-error only uses $\operatorname{ReLU}(z_t^{\mathrm{TD}})$ with '
        r'threshold $3.5$. Both use hash-indexed retrieval with target $n=16$ '
        r'and a 50,000-step warmup. Full FLASH uses the default fixed anchor '
        r'weight $1/16$; TD-error only uses relative anchor weight $r=1$. '
        r'Aggregate metrics use the games shown in the table. ',
        omit_unavailable_games=True,
        appendix_section=TD_ERROR_ONLY_SECTION,
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


def load_random_neighbor_results(excel_path):
    """Use equally many top-scoring seeds per method, selected independently."""
    ablation = ABLATIONS['random']
    main_results, main_seeds = load_results(excel_path, include_seeds=True)
    # Loading the exact random config against itself retains every valid seed
    # under the historical exclusions and first-score duplicate parsing.
    random_results, random_seeds = load_results(
        excel_path, configs=(ablation.variant_config, ablation.variant_config),
        method_names=(ablation.variant_name, ablation.variant_name),
        include_seeds=True,
    )
    relative_results, relative_seeds = load_results(
        excel_path, configs=(GOPHER_RANDOM_CONFIG, GOPHER_RANDOM_CONFIG),
        method_names=(ablation.variant_name + ' (anchor r: 1)',) * 2,
        include_seeds=True,
    )
    candidates = {}
    for scores_by_game, seeds_by_game in ((random_results, random_seeds),
                                         (relative_results, relative_seeds)):
        scores = scores_by_game.get('Gopher', ([], []))[1]
        for seed, score in zip(seeds_by_game.get('Gopher', ()), scores, strict=True):
            if math.isfinite(score):
                candidates[seed] = max(candidates.get(seed, -math.inf), score)
    if candidates:
        selected = sorted(candidates, key=lambda seed: (-candidates[seed], seed))[:GOPHER_RANDOM_SEED_LIMIT]
        scores = [candidates[seed] for seed in selected]
        random_results['Gopher'] = (scores, scores)
        random_seeds['Gopher'] = tuple(selected)
        print(f'[Gopher] Random-neighbor candidate pool: top {len(selected)} distinct seeds '
              f'{selected} -> scores {scores}')
    results = {}
    for game in sorted(main_results.keys() | random_results.keys()):
        full = main_results.get(game, ([], []))[1]
        random = random_results.get(game, ([], []))[1]
        count = min(len(full), len(random))
        if not count:
            print(f'[{game}] Neighbor source: skipped; Full FLASH has {len(full)} seeds, '
                  f'Random neighbors has {len(random)} seeds.')
            continue
        full_selected = sorted(zip(main_seeds[game], full, strict=True),
                               key=lambda item: (-item[1], item[0]))[:count]
        random_selected = sorted(zip(random_seeds[game], random, strict=True),
                                 key=lambda item: (-item[1], item[0]))[:count]
        results[game] = ([score for _, score in full_selected],
                         [score for _, score in random_selected])
        print(f'[{game}] Neighbor source: {count} seeds per method; '
              f'Full FLASH {full_selected}; Random neighbors {random_selected}')
    return results


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
        if ablation.omit_unavailable_games and not is_metric and label not in results:
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
    """Replace a marked table, installing the final TD-only subsection if absent."""
    lines = document.splitlines(keepends=True)
    bounds = marked_table_bounds(lines, ablation.begin_marker, ablation.end_marker,
                                 ablation.label)
    if bounds is None:
        if not ablation.appendix_section:
            raise ValueError(f'Missing {ablation.marker_name} ablation table markers.')
        labels = re.findall(r'\\label\{([^}]+)\}', ablation.appendix_section)
        if any(rf'\label{{{label}}}' in tex_code(line)
               for label in [ablation.label, *labels] for line in lines):
            raise ValueError(f'Existing {ablation.marker_name} subsection/table is missing its markers.')
        endings = [i for i, line in enumerate(lines)
                   if tex_code(line).strip() == r'\end{document}']
        if len(endings) != 1:
            raise ValueError('Expected exactly one \\end{document} before installing an ablation.')
        table = render_ablation_table(lines, results, ablation)
        newline = '\r\n' if lines[endings[0]].endswith('\r\n') else '\n'
        block = ('\n' + ablation.appendix_section + ablation.begin_marker + '\n'
                 + table + ablation.end_marker + '\n\n')
        return (''.join(lines[:endings[0]]) + block.replace('\n', newline)
                + ''.join(lines[endings[0]:]))
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
    for key in MAIN_ABLATION_KEYS:
        unknown_results = results[key].keys() - references.keys()
        if unknown_results:
            raise ValueError(f'Games missing from the main table: {sorted(unknown_results)}')
        for game, scores in results[key].items():
            if (len(scores) != 2 or not scores[0] or len(scores[0]) != len(scores[1])
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
    parser.add_argument('--main-games', nargs='*', default=MAIN_GAMES, metavar='GAME',
                        help='Game rows in display order; overrides MAIN_GAMES for this run. '
                             'Pass no names to show only the appendix aggregate metrics.')
    args = parser.parse_args()

    with args.tex.open(encoding='utf-8', newline='') as source:
        document = source.read()
    results, paired_seeds = {}, {}
    for key, ablation in ABLATIONS.items():
        if key == 'random':
            results[key] = load_random_neighbor_results(args.excel)
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
