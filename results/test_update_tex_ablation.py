"""Regression checks for shared-game aggregate metrics in ablation tables."""

import copy
import re
import unittest
import warnings

import update_tex_ablation as updater


def fixture_lines():
    rows = [f'Game{i} & 0 & 100 & 13 & 17 & 4 & 100 & 110 & 10 '
            + r'\\' for i in range(26)]
    rows += [label + ' & - & - & 13 & 17 & 4 & 100 & 110 & 10 ' + r'\\'
             for label in (r'\#Superhuman', 'Mean', 'Median', 'IQM', 'Optimality Gap')]
    return [line + '\n' for line in [
        r'\begin{table}', r'\label{tab:main_performance}',
        r'\begin{tabular}{lrrrrrrrr}',
        r'Game & Random & Human & STORM & STORM+ours & $\Delta$ & DRAMA & DRAMA+ours & $\Delta$ \\',
        *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ]]


def table_cells(table):
    cells = {}
    for line in table.splitlines():
        label = line.partition('&')[0].strip()
        if not (re.fullmatch(r'Game\d+', label) or updater.metric_name(label)):
            continue
        cells[updater.metric_name(label) or label] = [
            re.sub(r'\\textbf\{([^{}]*)\}', r'\1', cell.strip())
            for cell in line.partition(r'\\')[0].split('&')[1:]
        ]
    return cells


class SharedGameMetricsTests(unittest.TestCase):
    def render_tables(self, random_results, displayed_games):
        results = {key: {} for key in updater.ABLATIONS}
        results['random'] = random_results
        seeds = {'value': {}, 'add': {}}
        lines = fixture_lines()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            appendix = updater.render_ablation_table(
                lines, random_results, updater.ABLATIONS['random'])
            summary = updater.render_main_ablation_table(lines, results, seeds, displayed_games)
        return table_cells(appendix), table_cells(summary)

    def test_one_sided_games_stay_visible_but_do_not_enter_any_metric(self):
        results = {
            'Game0': ([10., 40., 100.], [60., 20.]),
            'Game1': ([900.], []),
            'Game2': ([], [-300.]),
            'Game3': ([200.], [250.]),
        }
        original = copy.deepcopy(results)
        # Show only a one-sided game in the summary: metrics must still use
        # both shared games, including their unequal numbers of seed scores.
        appendix, summary = self.render_tables(results, ['Game1'])
        self.assertEqual(appendix['Game0'], ['50', '40'])
        self.assertEqual(appendix['Game1'], ['900', '-'])
        self.assertEqual(appendix['Game2'], ['-', '-300'])
        self.assertEqual(appendix['Game3'], ['200', '250'])
        self.assertEqual(summary['Game1'][2:4], ['900', '-'])
        expected = {
            '#Superhuman': ['1', '1'], 'Mean': ['1.250', '1.450'],
            'Median': ['1.250', '1.450'], 'IQM': ['0.700', '1.100'],
            'Optimality Gap': ['0.250', '0.300'],
        }
        for metric, cells in expected.items():
            with self.subTest(metric=metric):
                self.assertEqual(appendix[metric], cells)
                self.assertEqual(summary[metric][2:4], cells)
        self.assertEqual(results, original)

    def test_no_shared_games_produce_dashes_for_all_metrics(self):
        appendix, summary = self.render_tables({
            'Game0': ([200.], []), 'Game1': ([], [300.]),
        }, ['Game0', 'Game1'])
        self.assertEqual(appendix['Game0'], ['200', '-'])
        self.assertEqual(appendix['Game1'], ['-', '300'])
        for metric in ('#Superhuman', 'Mean', 'Median', 'IQM', 'Optimality Gap'):
            with self.subTest(metric=metric):
                self.assertEqual(appendix[metric], ['-', '-'])
                self.assertEqual(summary[metric][2:4], ['-', '-'])

    def test_zero_and_negative_scores_count_as_available_results(self):
        appendix, summary = self.render_tables({
            'Game0': ([0.], [0.]), 'Game1': ([-100.], [-50.]),
            'Game2': ([10000.], []),
        }, ['Game0'])
        self.assertEqual(appendix['Game0'], ['0', '0'])
        self.assertEqual(appendix['Game1'], ['-100', '-50'])
        for metric, cells in {
            '#Superhuman': ['0', '0'], 'Mean': ['-0.500', '-0.250'],
            'Median': ['-0.500', '-0.250'], 'IQM': ['-0.500', '-0.250'],
            'Optimality Gap': ['1.500', '1.250'],
        }.items():
            with self.subTest(metric=metric):
                self.assertEqual(appendix[metric], cells)
                self.assertEqual(summary[metric][2:4], cells)


if __name__ == '__main__':
    unittest.main()
