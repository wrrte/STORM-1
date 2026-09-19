"""Update the neighbor-retrieval ablation table from paired Excel seed scores.

Usage: python STORM-1/results/update_tex_ablation.py [--excel PATH] [--tex PATH]
Uses the same exclusions, duplicate-result parsing, and aggregation as update_tex.py.
Only the marked ablation table is written; main performance scores are untouched.
"""

import argparse
from pathlib import Path

from update_tex import (
    BASE_COLUMN,
    EXCLUDED_SEEDS,
    OURS_COLUMN,
    load_results,
    main_table_rows,
    metric_name,
    update_table,
)


CONFIGS = ('target: 16 (anchor 미설정)', 'target: 1 (anchor 미설정)')
METHOD_NAMES = ('Full FLASH', 'No neighbor')
BEGIN_MARKER = '% BEGIN AUTO NEIGHBOR RETRIEVAL ABLATION'
END_MARKER = '% END AUTO NEIGHBOR RETRIEVAL ABLATION'


def render_ablation_table(lines, results):
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
    exclusions = '; '.join(
        f"{game}: {', '.join(str(seed) for seed in sorted(seeds))}"
        for game, seeds in EXCLUDED_SEEDS.items()
    )

    table = [
        r'\begin{table}[!t]',
        r'\centering',
        r'\small',
        r'\caption{Neighbor retrieval ablation on STORM. Full FLASH uses retrieval '
        r'target $n=16$; No neighbor uses $n=1$ (anchor only), both with the default '
        r'anchor setting. Each game mean uses only training seeds with valid scores '
        r'for both variants; $N$ is the number of paired seeds. '
        f'Excluded training seeds are {exclusions}. '
        f'The current comparison covers {game_count} games and {seed_count} seed pairs. '
        r'A dash indicates no paired results. Aggregate metrics use only games with '
        r'paired results and the Random/Human references in Table~\ref{tab:main_performance}. '
        r'Mean, Median, and Optimality Gap use human-normalized game means; IQM pools '
        r'the unrounded per-seed human-normalized scores.}',
        r'\label{tab:neighbor_retrieval_ablation}',
        r'\begin{tabular}{lrrr}',
        r'\toprule',
        r'Game & $N$ & Full FLASH & No neighbor \\',
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
        no_neighbor = parts[OURS_COLUMN].strip()
        table.append(f'{label} & {count} & {full} & {no_neighbor} ' + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return '\n'.join(table) + '\n'


def update_ablation_table(document, results):
    """Replace exactly one marked table while preserving all surrounding text."""
    lines = document.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if line.strip() == BEGIN_MARKER]
    ends = [i for i, line in enumerate(lines) if line.strip() == END_MARKER]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise ValueError('Expected exactly one ordered pair of ablation table markers.')
    table = render_ablation_table(lines, results)
    return ''.join(lines[:begins[0] + 1]) + table + ''.join(lines[ends[0]:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--excel', type=Path, default=script_dir / 'converted_results.xlsx')
    parser.add_argument('--tex', type=Path, default=script_dir.parent.parent / 'iclr2027_conference.tex')
    args = parser.parse_args()

    document = args.tex.read_text(encoding='utf-8')
    results = load_results(args.excel, configs=CONFIGS, method_names=METHOD_NAMES)
    updated = update_ablation_table(document, results)
    args.tex.write_text(updated, encoding='utf-8')
    print(f'Successfully updated {args.tex} with Full FLASH / No neighbor results.')


if __name__ == '__main__':
    main()
