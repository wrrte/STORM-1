"""Offline regression checks for the deleted target-1 run backup."""
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


spec = importlib.util.spec_from_file_location(
    'classify_wandb_runs', Path(__file__).with_name('classify_wandb_runs.py')
)
classifier = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', list(sys.path)):
    spec.loader.exec_module(classifier)


class ArchivedTargetOneTests(unittest.TestCase):
    def test_archive_is_complete_without_backup_file(self):
        rows = classifier.merge_archived_target_1_runs([])
        self.assertEqual(len(rows), 37)
        self.assertEqual(len({row['Run ID'] for row in rows}), 37)
        self.assertTrue(all(row['Retrieval Target'] == '1' for row in rows))
        self.assertTrue(all(row['Anchor Weight'] == 'N/A' for row in rows))
        frostbite = next(row for row in rows if row['Run ID'] == 'nmi7yn4w')
        self.assertEqual(frostbite['Eval Return'], '2119.5')
        self.assertEqual(frostbite['Seed'], '6000')
        self.assertEqual(frostbite['Created At'], '2026-09-03T05:10:47Z')

    def test_live_run_wins_and_new_same_seed_run_is_preserved(self):
        archived = classifier.merge_archived_target_1_runs([])[0]
        live = {**archived, 'Eval Return': 9999}
        new = {**archived, 'Run ID': 'new-run', 'Eval Return': 1234}
        baseline = {**new, 'Run ID': 'baseline', 'Retrieval Enable': False}
        rows = classifier.merge_archived_target_1_runs([live, new, baseline])
        self.assertEqual(len(rows), 39)
        by_id = {row['Run ID']: row for row in rows}
        self.assertEqual(by_id[live['Run ID']], live)
        self.assertEqual(by_id['new-run'], new)
        self.assertEqual(by_id['baseline'], baseline)
        self.assertEqual(classifier.merge_archived_target_1_runs(rows), rows)

    def test_export_includes_live_and_archived_runs(self):
        run = SimpleNamespace(
            id='new-run', name='Frostbite_new-run_6000_O', state='finished',
            commit=None, created_at='2026-09-18T00:00:00Z',
            summary={'eval/episode_avg_return': 1234},
            config={'JointTrainAgent': {'Retrieval': {
                'enable': True, 'target': 1, 'warmup_steps': 50000,
                'batch_size_reduction': 'retrieved', 'z_score_threshold': 3.5,
            }}},
        )
        for live_runs in ([], [run]):
            with self.subTest(live_runs=len(live_runs)), tempfile.TemporaryDirectory() as directory:
                previous_cwd = Path.cwd()
                try:
                    os.chdir(directory)
                    with patch.object(classifier.wandb, 'Api') as api, \
                            patch.object(classifier.os.path, 'exists', return_value=False), \
                            contextlib.redirect_stdout(io.StringIO()):
                        api.return_value.runs.return_value = live_runs
                        classifier.main()
                    with open('wandb_runs_classification.csv', newline='') as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 37 + len(live_runs))
                    by_id = {row['Run ID']: row for row in rows}
                    for archived in classifier.merge_archived_target_1_runs([]):
                        self.assertEqual(by_id[archived['Run ID']], archived)
                    if live_runs:
                        self.assertEqual(by_id['new-run']['Eval Return'], '1234')
                        self.assertEqual(by_id['new-run']['Retrieval Target'], '1')
                finally:
                    os.chdir(previous_cwd)


if __name__ == '__main__':
    unittest.main()
