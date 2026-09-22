"""Update the neighbor, value, and additive ablation tables from Excel seed scores.

Usage: python STORM/results/update_tex_ablation.py [--excel PATH] [--tex PATH]
Uses the same result loading, duplicate-result parsing, and aggregation as update_tex.py.
Each variant is paired independently with default FLASH on common training seeds.
Only the marked ablation tables are written; main performance scores are untouched.
Each table ends with a float barrier to keep it within its ablation subsection.
The target document must load the placeins package.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from update_tex import (
    BASE_COLUMN,
    OURS_COLUMN,
    load_results,
    main_table_rows,
    metric_name,
    update_table,
)


FLASH_CONFIG = 'target: 16 (anchor 미설정)'


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


def render_ablation_table(lines, results, ablation=NEIGHBOR):
    # Reuse the main-table calculation in memory for identical normalization,
    # rounding, and pooled per-seed IQM. Project only the two score columns.
    calculated_rows = main_table_rows(update_table(lines, results))
    games = [parts[0].strip() for _, parts, _ in calculated_rows
             if metric_name(parts[0]) is None]
    missing_references = set(results) - set(games)
    if missing_references:
        raise ValueError(f"Games missing from the main table: {sorted(missing_references)}")
    game_count = sum(game in results for game in games)
    seed_count = sum(len(results[game][0]) for game in games if game in results)

    table = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\small',
        r'\caption{' + ablation.caption +
        r'Each game mean is computed over $N$ paired training seeds shared by '
        r'both variants. The number of training seeds can differ across ablations '
        r'and from Table~\ref{tab:main_performance}. '
        f'The current comparison covers {game_count} games and {seed_count} seed pairs. '
        r'A dash indicates no paired results. Aggregate metrics summarize games with '
        r'paired results using the Random/Human references in Table~\ref{tab:main_performance}. '
        r'Mean, Median, and Optimality Gap use human-normalized game means; IQM pools '
        r'the unrounded per-seed human-normalized scores.}',
        rf'\label{{{ablation.label}}}',
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        f'Game & $N$ & {ablation.method_names[0]} & {ablation.method_names[1]} ' + r'\\',
        r'\midrule',
    ]
    metrics_started = False
    for _, parts, _ in calculated_rows:
        label = parts[0].strip()
        is_metric = metric_name(label) is not None
        if is_metric and not metrics_started:
            table.append(r'\midrule')
            metrics_started = True
        count = '' if is_metric else str(len(results[label][0]) if label in results else 0)
        full = parts[BASE_COLUMN].strip()
        variant = parts[OURS_COLUMN].strip()
        table.append(f'{label} & {count} & {full} & {variant} ' + r'\\')
    # Flush the table before the next subsection or Extended Related Work.
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\FloatBarrier'])
    return '\n'.join(table) + '\n'


def update_ablation_table(document, results, ablation=NEIGHBOR):
    """Replace exactly one marked table while preserving all surrounding text."""
    lines = document.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if line.strip() == ablation.begin_marker]
    ends = [i for i, line in enumerate(lines) if line.strip() == ablation.end_marker]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise ValueError(
            f'Expected exactly one ordered pair of {ablation.marker_name} ablation table markers.'
        )
    table = render_ablation_table(lines, results, ablation)
    return ''.join(lines[:begins[0] + 1]) + table + ''.join(lines[ends[0]:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--excel', type=Path, default=script_dir / 'converted_results.xlsx')
    parser.add_argument('--tex', type=Path, default=script_dir.parent.parent / 'iclr2027_conference.tex')
    args = parser.parse_args()

    document = args.tex.read_text(encoding='utf-8')
    for ablation in ABLATIONS.values():
        results = load_results(args.excel, configs=ablation.configs, method_names=ablation.method_names)
        document = update_ablation_table(document, results, ablation)
    args.tex.write_text(document, encoding='utf-8')
    print(f'Successfully updated {args.tex} with neighbor, value, and additive ablation results.')


if __name__ == '__main__':
    main()
