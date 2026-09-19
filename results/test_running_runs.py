"""Offline checks for running-run export, workbook display, and score aggregation."""
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openpyxl import load_workbook


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, 'path', list(sys.path)):
        spec.loader.exec_module(module)
    return module


classifier = load_script('classify_wandb_runs')
converter = load_script('convert_csv_to_excel')
BASELINE = 'Retrieval 미사용'
TARGET = 'target: 16 (anchor 미설정)'


def make_run(run_id, seed, mode, state='running', score=None, hash_bits=10,
             created_at='2026-09-19T00:00:00Z'):
    suffix = '' if mode == 'Both' else ('_O' if mode else '_X')
    return SimpleNamespace(
        id=run_id, name=f'Alien_{run_id}_{seed}{suffix}', state=state,
        commit=None, created_at=created_at,
        summary={} if score is None else {'eval/episode_avg_return': score},
        config={'JointTrainAgent': {'Retrieval': {
            'enable': mode, 'target': 16, 'warmup_steps': 50000,
            'batch_size_reduction': 'retrieved', 'z_score_threshold': 3.5,
            'hash_bits': hash_bits,
        }}},
    )


class RunningRunsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        previous_cwd = Path.cwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, previous_cwd)

    def export(self, runs):
        with patch.object(classifier.wandb, 'Api') as api, \
                patch.object(classifier.os.path, 'exists', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            api.return_value.runs.return_value = runs
            classifier.main()
        with open('wandb_runs_classification.csv', newline='') as source:
            return {row['Run ID']: row for row in csv.DictReader(source)}

    def workbook_cells(self):
        with contextlib.redirect_stdout(io.StringIO()):
            converter.main()
        workbook = load_workbook('converted_results.xlsx')
        self.addCleanup(workbook.close)
        worksheet = workbook['Results']
        headers = [cell.value for cell in worksheet[1]]
        cells = {}
        game = None
        for row in worksheet.iter_rows(min_row=2):
            if row[0].value:
                game = str(row[0].value).split('\n')[0]
            for header, cell in zip(headers[2:], row[2:]):
                cells[game, row[1].value, header] = cell
        return cells

    def test_export_keeps_running_and_both_but_excludes_killed_and_finished_both(self):
        both = make_run('both', 5090, 'Both')
        suffix_both = make_run('suffix', 6000, 'Both')
        suffix_both.name += '_Both'
        rows = self.export([
            make_run('running', 710, False), both, suffix_both,
            make_run('killed', 6020, True, state='killed'),
            make_run('doneboth', 2010, 'Both', state='finished', score=999),
        ])
        self.assertEqual(len(rows), 37 + 3)
        self.assertEqual(rows['running']['State'], 'running')
        self.assertEqual(rows['running']['Eval Return'], 'N/A')
        self.assertEqual(rows['both']['Retrieval Enable'], 'Both')
        self.assertEqual(rows['both']['Seed'], '5090')
        self.assertEqual(rows['suffix']['Seed'], '6000')
        self.assertNotIn('killed', rows)
        self.assertNotIn('doneboth', rows)

    def test_workbook_marks_running_and_both_without_using_intermediate_scores(self):
        self.export([
            make_run('base1', 1, False, 'finished', 100, created_at='2026-08-01T00:00:00Z'),
            make_run('target1', 1, True, 'finished', 150),
            make_run('base2', 2, False, 'finished', 200),
            make_run('target2', 2, True, score=9999),
            make_run('base710', 710, False),
            make_run('old710', 710, True, 'finished', 80, created_at='2026-09-18T00:00:00Z'),
            make_run('target710', 710, True, hash_bits=11),
            make_run('both', 5090, 'Both'),
        ])
        cells = self.workbook_cells()
        for config, seed in ((BASELINE, 710), (TARGET, 2), (TARGET, 710),
                             (BASELINE, 5090), (TARGET, 5090)):
            with self.subTest(config=config, seed=seed):
                cell = cells['Alien', config, seed]
                self.assertIn('RUNNING', cell.value)
                self.assertEqual(cell.fill.fgColor.rgb[-6:], 'FFF2CC')
                self.assertEqual(cell.font.color.rgb[-6:], '9C6500')
                self.assertTrue(cell.font.bold)
        self.assertNotIn('9999', cells['Alien', TARGET, 2].value)
        self.assertIn('80.00', cells['Alien', TARGET, 710].value)
        self.assertEqual(cells['Alien', BASELINE, converter.PAIRED_MEAN_COLUMN].value, 100)
        self.assertEqual(cells['Alien', TARGET, converter.PAIRED_MEAN_COLUMN].value, 150)
        self.assertEqual(cells['Alien', BASELINE, 1].value, '100.00')
        self.assertIsNone(cells['Alien', BASELINE, 1].fill.fill_type)

    def test_finished_run_replaces_marker_and_restores_paired_mean(self):
        run = make_run('target', 2, True)
        baseline = make_run('baseline', 2, False, 'finished', 200)
        self.export([baseline, run])
        self.assertEqual(self.workbook_cells()['Alien', TARGET, 2].value, 'RUNNING')
        run.state = 'finished'
        run.summary['eval/episode_avg_return'] = 300
        rows = self.export([baseline, run])
        cells = self.workbook_cells()
        self.assertEqual(rows['target']['State'], 'finished')
        self.assertEqual(cells['Alien', TARGET, 2].value, '300.00')
        self.assertIsNone(cells['Alien', TARGET, 2].fill.fill_type)
        self.assertEqual(cells['Alien', TARGET, converter.PAIRED_MEAN_COLUMN].value, 300)


if __name__ == '__main__':
    unittest.main()
