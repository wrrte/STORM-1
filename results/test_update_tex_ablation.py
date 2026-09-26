"""Regression checks for appendix parity and preservation of editable LaTeX prose.

Run: python -m unittest discover -s STORM/results -p test_update_tex_ablation.py
"""

from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

import update_tex_ablation as updater


# Keep synthetic fixtures independent of the user's editable MAIN_GAMES.
DISPLAY_GAMES = ['Frostbite', 'Gopher', 'Pong']
GAMES = DISPLAY_GAMES + [f'Game{i}' for i in range(23)]
METRICS = [r'\#Superhuman ($\uparrow$)', r'Mean ($\uparrow$)',
           r'Median ($\uparrow$)', r'IQM ($\uparrow$)', r'Optimality Gap ($\downarrow$)']
MARKERS = [(updater.MAIN_BEGIN_MARKER, updater.MAIN_END_MARKER)] + [
    (item.begin_marker, item.end_marker) for item in updater.ABLATIONS.values()
]


def placeholder(begin, end, label):
    return '\n'.join([begin, r'\begin{table}', rf'\label{{{label}}}',
                      r'\begin{tabular}{lr}', r'Old & - \\', r'\end{tabular}',
                      r'\end{table}', r'\FloatBarrier', end]) + '\n'


def document_fixture():
    lines = [r'\documentclass{article}', r'\begin{document}',
             'Introduction to be edited. 한글 본문과 공백.  ',
             r'\section{Experiments}', r'\begin{table}',
             r'\label{tab:main_performance}', r'\begin{tabular}{lllrrrrrr}',
             r'Game & Random & Human & STORM & STORM+ours & $\Delta$ & DRAMA & DRAMA+ours & $\Delta$ \\']
    lines += [f'{game} & 10 & 110 & 900 & 999 & 99 & 700 & 777 & 77 ' + r'\\'
              for game in GAMES]
    lines += [f'{metric} & 0 & 1 & 900 & 999 & 99 & 700 & 777 & 77 ' + r'\\'
              for metric in METRICS]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}',
              r'\subsection{Ablation Studies}', r'\label{subsec:ablation_main}',
              'Editable text immediately before the summary.']
    document = '\n'.join(lines) + '\n'
    document += placeholder(*MARKERS[0], 'tab:ablation_main')
    document += ('Editable text immediately after the summary.\n'
                 '\\subsection{Later analysis}\nUnrelated table follows.\n'
                 '\\begin{table}\n\\label{tab:unrelated}\n'
                 'Mean & 123 & 456 \\\\\n\\end{table}\n'
                 '\\section{Conclusion}\nConclusion prose.\n\\appendix\n')
    for item in updater.ABLATIONS.values():
        document += f'Appendix prose before {item.label}.\n'
        document += placeholder(item.begin_marker, item.end_marker, item.label)
        document += f'Appendix prose after {item.label}.\n'
    return document + '\\end{document}\n'


def block_pattern(begin, end):
    return (r'(?m)(^[ \t]*' + re.escape(begin) + r'[ \t]*\r?\n)'
            r'[\s\S]*?(^[ \t]*' + re.escape(end) + r'[ \t]*(?:\r?\n|$))')


def outside_tables(document):
    for begin, end in MARKERS:
        document = re.sub(block_pattern(begin, end), r'\1\2', document)
    return document


def table_rows(table):
    # Read physical variant rows, then expose game/metric cells for comparison
    # with the appendix. Split only the final row terminator: shortstack headers
    # contain their own LaTeX line breaks.
    if r'\label{tab:ablation_main}' in table:
        headers = None
        values = []
        for line in table.splitlines():
            if '&' not in line or line.lstrip().startswith('%'):
                continue
            cells = [cell.strip() for cell in line.rsplit(r'\\', 1)[0].split('&')]
            if cells[0] == 'Variant':
                headers = [
                    cell.removeprefix(r'\shortstack{').removesuffix('}').replace(r'\\', ' ')
                    if cell.startswith(r'\shortstack{') else cell
                    for cell in cells[1:]
                ]
            else:
                if headers is None or len(cells) != len(headers) + 1:
                    raise AssertionError(f'Malformed transposed row: {line}')
                values.append(cells[1:])
        if headers is None or not values:
            raise AssertionError('Missing transposed ablation header or variants')
        return dict(zip(headers, map(list, zip(*values))))
    rows = {}
    for line in table.splitlines():
        if '&' in line and not line.lstrip().startswith('%'):
            cells = [cell.strip() for cell in line.split(r'\\', 1)[0].split('&')]
            if cells[0] in GAMES or cells[0] in METRICS:
                rows[cells[0]] = cells[1:]
    return rows


class AblationUpdateTests(unittest.TestCase):
    def setUp(self):
        self.document = document_fixture()
        self.results = {
            'neighbor': {'Frostbite': ([10., 110., 210.], [60., 110., 160.]),
                         'Gopher': ([310.], [210.])},
            'value': {'Frostbite': ([30.04, 40.08], [15., 25.]),
                      'Gopher': ([110.], [120.])},
            'add': {'Frostbite': ([30.04], [45.]),
                    'Gopher': ([110.], [130.])},
        }
        self.seeds = {key: {game: tuple(range(len(scores[0])))
                           for game, scores in results.items()}
                      for key, results in self.results.items()}

    def update(self, document=None, games=DISPLAY_GAMES):
        document = self.document if document is None else document
        return updater.update_ablation_tables(document, self.results, self.seeds, games)

    def assert_metric_parity(self, games, shared=False):
        lines = self.document.splitlines(keepends=True)
        original_lines = lines.copy()
        main = table_rows(updater.render_main_ablation_table(
            lines, self.results, self.seeds, games))
        appendix = {key: table_rows(updater.render_ablation_table(lines, self.results[key], item))
                    for key, item in updater.ABLATIONS.items()}
        for label in [*games, *METRICS]:
            expected = appendix['neighbor'][label][1:] + appendix['value'][label][1:]
            expected += appendix['add'][label][2:] if shared else appendix['add'][label][1:]
            self.assertEqual(main[label], expected, label)
        self.assertEqual(lines, original_lines, 'Reference performance table was mutated')
        return main

    def test_main_layout_has_variant_rows_and_game_metric_columns(self):
        table = updater.render_main_ablation_table(
            self.document.splitlines(keepends=True), self.results, self.seeds, DISPLAY_GAMES)
        self.assertIn(r'\begin{tabular}{@{}lrrrrrrrr@{}}', table)
        self.assertIn('Variant & Frostbite & Gopher & Pong & ', table)
        self.assertIn(r'\shortstack{Optimality\\Gap\\($\downarrow$)}', table)
        self.assertIn(r'\resizebox{\linewidth}{!}{%', table)
        for title in ('Neighbor retrieval', 'Value signal', 'Combination'):
            self.assertIn(r'\multicolumn{9}{@{}l}{\textit{' + title + '}}', table)
        rows = [line for line in table.splitlines()
                if '&' in line and not line.startswith('Variant &')]
        self.assertEqual([line.split('&', 1)[0].strip() for line in rows],
                         ['Full FLASH', 'No neighbor', 'Full FLASH', 'Absolute value',
                          'Full FLASH', 'Additive'])
        self.assertTrue(all(line.count('&') == 8 for line in rows))

    def test_all_tables_reuse_one_calculation_per_comparison(self):
        with mock.patch.object(updater, 'update_table', wraps=updater.update_table) as calculate:
            updated = self.update()
        self.assertEqual(calculate.call_count, 3)
        main = table_rows(re.search(block_pattern(*MARKERS[0]), updated).group())
        for index, markers in enumerate(MARKERS[1:]):
            appendix = table_rows(re.search(block_pattern(*markers), updated).group())
            for label in [*DISPLAY_GAMES, *METRICS]:
                self.assertEqual(main[label][2 * index:2 * index + 2], appendix[label][1:])

    def test_partial_coverage_matches_all_appendix_cells_and_known_metrics(self):
        main = self.assert_metric_parity(DISPLAY_GAMES)
        expected = [('1', '1'), ('2.000', '1.500'), ('2.000', '1.500'),
                    ('1.500', '1.250'), ('0.000', '0.000')]
        for metric, values in zip(METRICS, expected):
            self.assertEqual(main[metric][:2], list(values))
        self.assertEqual(main['Pong'], ['-'] * 6)

    def test_metrics_do_not_depend_on_display_games(self):
        expected = self.assert_metric_parity(DISPLAY_GAMES)
        for games in ([], ['Gopher'], ['Pong', 'Frostbite'], list(reversed(GAMES))):
            with self.subTest(games=games):
                actual = self.assert_metric_parity(games)
                for metric in METRICS:
                    self.assertEqual(actual[metric], expected[metric])
                self.assertEqual([label for label in actual if label not in METRICS], games)

    def test_complete_coverage_matches_appendix(self):
        for key in self.results:
            for game in GAMES:
                self.results[key].setdefault(game, ([20.], [40.]))
                self.seeds[key].setdefault(game, (0,))
        self.assert_metric_parity(DISPLAY_GAMES)

    def test_empty_comparison_has_dashes_without_hiding_other_comparisons(self):
        self.results['neighbor'] = {}
        self.seeds['neighbor'] = {}
        main = self.assert_metric_parity([])
        for metric in METRICS:
            self.assertEqual(main[metric][:2], ['-', '-'])
            self.assertNotIn('-', main[metric][2:])

    def test_identical_baselines_share_rows(self):
        self.results['add'] = deepcopy(self.results['value'])
        self.results['add']['Frostbite'][1][0] += 50
        self.seeds['add'] = deepcopy(self.seeds['value'])
        self.assert_metric_parity(DISPLAY_GAMES, shared=True)

    def test_same_scores_with_different_seed_ids_keep_separate_baselines(self):
        self.results['add'] = deepcopy(self.results['value'])
        self.seeds['add'] = deepcopy(self.seeds['value'])
        self.seeds['add']['Frostbite'] = (100, 101)
        self.assert_metric_parity(DISPLAY_GAMES)

    def test_edited_prose_headings_and_section_labels_are_preserved_exactly(self):
        document = self.document.replace('Introduction to be edited.', 'Rewritten intro.\n' * 30)
        document = document.replace(r'\label{subsec:ablation_main}',
                                    r'\label{subsec:renamed_ablation} % renamed during revision')
        document = document.replace(r'\subsection{Ablation Studies}',
                                    r'\subsection*{Renamed heading}')
        document = document.replace(updater.MAIN_BEGIN_MARKER,
                                    '\\subsection{New subsection before table}\n'
                                    'Added discussion with \\ref{tab:main_performance}.\n'
                                    '% \\label{subsec:ablation_main}\n' + updater.MAIN_BEGIN_MARKER)
        document = document.replace(updater.MAIN_END_MARKER,
                                    updater.MAIN_END_MARKER + '\nNew discussion after table.\n')
        updated = self.update(document)
        self.assertEqual(outside_tables(document).encode(), outside_tables(updated).encode())
        self.assertNotEqual(document, updated)
        self.assertEqual(self.update(updated), updated)

    def test_first_install_accepts_inline_label_and_retains_all_original_text(self):
        document = re.sub(block_pattern(*MARKERS[0]), '', self.document)
        document = document.replace('\\subsection{Ablation Studies}\n\\label{subsec:ablation_main}',
                                    '\\subsection{Edited title}\\label{subsec:ablation_main} % note')
        updated = updater.update_main_ablation_table(document, self.results, self.seeds, DISPLAY_GAMES)
        inserted = '\n' + re.search(block_pattern(*MARKERS[0]), updated).group() + '\n'
        self.assertEqual(updated.replace(inserted, '', 1), document)
        self.assertEqual(
            updater.update_main_ablation_table(updated, self.results, self.seeds, DISPLAY_GAMES), updated)

    def test_existing_unmarked_summary_is_not_duplicated(self):
        document = self.document.replace(updater.MAIN_BEGIN_MARKER, '').replace(
            updater.MAIN_END_MARKER, '')
        with self.assertRaisesRegex(ValueError, 'missing its AUTO MAIN ABLATION markers'):
            self.update(document)

    def test_invalid_markers_and_oversized_blocks_are_rejected(self):
        for begin, end in MARKERS:
            mutations = {
                'missing_begin': self.document.replace(begin, '', 1),
                'missing_end': self.document.replace(end, '', 1),
                'duplicate_begin': self.document.replace(begin, begin + '\n' + begin, 1),
                'duplicate_end': self.document.replace(end, end + '\n' + end, 1),
                'reversed': self.document.replace(begin, 'TEMP').replace(end, begin).replace('TEMP', end),
                'prose_before': self.document.replace(begin, begin + '\nDo not erase this paragraph.'),
                'prose_after': self.document.replace(end, 'Do not erase this paragraph.\n' + end),
                'extra_table': self.document.replace(end, '\\begin{table}\nExtra\n\\end{table}\n' + end),
            }
            for name, document in mutations.items():
                with self.subTest(block=begin, mutation=name), self.assertRaises(ValueError):
                    self.update(document)

    def test_duplicate_table_labels_are_rejected(self):
        labels = ['tab:ablation_main'] + [item.label for item in updater.ABLATIONS.values()]
        for label in labels:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.update(self.document + rf'\label{{{label}}}' + '\n')

    def test_overlapping_markers_are_rejected(self):
        document = self.document.replace(updater.MAIN_END_MARKER + '\n', '')
        document = document.replace(updater.NEIGHBOR.end_marker,
                                    updater.MAIN_END_MARKER + '\n' + updater.NEIGHBOR.end_marker)
        with self.assertRaises(ValueError):
            self.update(document)

    def test_invalid_game_options_fail(self):
        for games in (['Unknown'], ['Pong', 'Pong']):
            with self.subTest(games=games), self.assertRaises(ValueError):
                self.update(games=games)

    def test_cli_excel_update_newline_preservation_idempotence_and_no_write_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tex = directory / 'paper.tex'
            excel = directory / 'results.xlsx'
            rows = [
                ['Frostbite', updater.FLASH_CONFIG, 10., 110., 210.],
                ['Frostbite', updater.ABLATIONS['neighbor'].variant_config, 60., 110., 160.],
                ['Frostbite', updater.ABLATIONS['value'].variant_config, 20., 80., None],
                ['Frostbite', updater.ABLATIONS['add'].variant_config, 30., None, 190.],
                ['Gopher', updater.FLASH_CONFIG, 310., None, None],
                ['Gopher', updater.ABLATIONS['neighbor'].variant_config, 210., None, None],
            ]
            pd.DataFrame(rows, columns=['Game', 'Config', 100, 101, 102]).set_index(
                ['Game', 'Config']).to_excel(excel, sheet_name='Results')
            command = [sys.executable, str(Path(updater.__file__).resolve()),
                       '--excel', str(excel), '--tex', str(tex), '--main-games']
            for newline in ('\n', '\r\n'):
                with self.subTest(newline=repr(newline)):
                    original = self.document.replace('\n', newline).encode()
                    tex.write_bytes(original)
                    run = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    updated = tex.read_bytes()
                    self.assertEqual(outside_tables(original.decode()).encode(),
                                     outside_tables(updated.decode()).encode())
                    if newline == '\r\n':
                        self.assertNotIn(b'\n', updated.replace(b'\r\n', b''))
                    main_block = re.search(block_pattern(*MARKERS[0]), updated.decode()).group()
                    main = table_rows(main_block)
                    self.assertEqual(list(main), METRICS)
                    for index, markers in enumerate(MARKERS[1:]):
                        appendix = table_rows(re.search(block_pattern(*markers), updated.decode()).group())
                        for metric in METRICS:
                            self.assertEqual(main[metric][2 * index:2 * index + 2], appendix[metric][1:])
                    repeated = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                    self.assertEqual(repeated.returncode, 0, repeated.stderr)
                    self.assertEqual(tex.read_bytes(), updated)
            for begin, end in MARKERS:
                broken = self.document.replace(end, '').encode()
                tex.write_bytes(broken)
                failed = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn('ValueError', failed.stderr)
                self.assertEqual(tex.read_bytes(), broken)
            tex.write_bytes(original)
            failed = subprocess.run(command + ['Unknown'], cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(tex.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
