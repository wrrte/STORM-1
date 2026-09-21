"""Offline state-machine and queue-dispatch checks; never starts training."""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from contextlib import redirect_stdout

import schedule_experiments as scheduler


NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)


def row(seed, score=None, kind='retrieval', state='finished', game='Gopher', run_id=None, **overrides):
    result = {
        'Run Name': f'{game}_{run_id or seed}_{seed}', 'Run ID': str(run_id or seed),
        'Seed': str(seed), 'State': state, 'Eval Return': str(score),
        'Retrieval Enable': 'True' if kind == 'retrieval' else 'False',
        'Created At': NOW.isoformat(), 'Warmup Steps': '50000', 'Retrieval Target': '16',
        'Batch Size Reduction': 'retrieved', 'Z Score Threshold': '3.5',
        'Hash Bits': '10', 'Anchor Weight': 'N/A', 'Value Signal': 'value_diff',
        'Score Combination': 'multiply', 'Pending Retrieval Configs': '[]',
        'Save Warmup': 'N/A', 'Save Warmup Requested': 'N/A',
    }
    result.update(overrides)
    result.update(game=game, seed=seed, score=scheduler.number(result['Eval Return']))
    result['kind'] = scheduler.row_kind(result)
    return result


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((scheduler.HERE / 'experiment_scheduler.json').read_text())
        self.config['lookahead_hours'] = 0
        for gpu, spec in self.config['gpus'].items():
            spec['count'] = int(gpu == 'pro6k')
        self.references = scheduler.literal_setting(scheduler.HERE / 'convert_csv_to_excel.py', 'REFERENCE_SCORES')
        self.rows = [row(10, 3000), row(1, 4000), row(10, 4000, kind='baseline', run_id='baseline10')]
        self.queues = dict.fromkeys(self.config['gpus'], '')

    def plan(self, rows=None, state=None, queues=None, excluded=None, **kwargs):
        return scheduler.plan(self.rows if rows is None else rows,
                              self.queues if queues is None else queues, state or {}, self.config,
                              self.references, excluded or {}, NOW, **kwargs)

    def test_two_configs_only_and_old_baseline_is_preserved(self):
        self.assertEqual(row(1, 1, kind='baseline', **{'Created At': '2020-01-01Z'})['kind'], 'baseline')
        for override in ({'Created At': '2026-08-23T00:00:00Z'}, {'Retrieval Target': '1'},
                         {'Value Signal': 'value'}, {'Score Combination': 'add'},
                         {'Anchor Weight': '0.5'}, {'Warmup Steps': '10000'}):
            with self.subTest(override=override):
                self.assertIsNone(row(1, 1, **override)['kind'])

    def test_hns_threshold_excludes_seeds_and_is_not_one_percent(self):
        rows = self.rows + [row(2000, 1)]
        state, report = self.plan(rows, excluded={'Gopher': {2000}})
        job = report['new_jobs'][0]
        self.assertEqual(job['reference_seed'], 10)
        self.assertAlmostEqual(job['threshold_score'], 5154.9)
        self.assertEqual(job['estimated_finish_hours'], 4.5)
        tokens = shlex.split(job['command'])
        self.assertEqual(tokens[tokens.index(scheduler.PREFIX + 'enable') + 1], "['retrieval']")
        self.assertEqual(tokens[tokens.index(scheduler.PREFIX + 'save_warmup') + 1], 'True')
        self.assertEqual(state['jobs'][0]['seed'], 10000)

    def test_finished_hash_10_is_preferred_as_in_workbook(self):
        older = row(10, 2000, run_id='older', **{'Created At': '2026-09-20T00:00:00Z'})
        newer = row(10, 1000, run_id='newer', **{'Hash Bits': '11'})
        scores = scheduler.latest_scores([older, newer], {})
        self.assertEqual(scores['Gopher', 'retrieval', 10]['score'], 2000)

    def test_pending_and_missing_result_do_not_duplicate(self):
        state, report = self.plan()
        queues = {**self.queues, 'pro6k': report['new_jobs'][0]['command']}
        state, report = self.plan(state=state, queues=queues)
        self.assertEqual(report['new_jobs'], [])
        state, report = self.plan(state=state)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(state['jobs'][0]['status'], 'awaiting_result')

    def test_success_resumes_identical_warmup_on_original_gpu_once(self):
        state, report = self.plan()
        seed = report['new_jobs'][0]['seed']
        result = row(seed, 5154.9, run_id='new', **{
            'Save Warmup Requested': 'True', 'Save Warmup': 'False',
            'Warmup Directory': f'/remote/STORM/ckpt/Gopher-{seed}_Shared/shared_warmup_50123'})
        rows = self.rows + [result]
        state, report = self.plan(rows, state)
        self.assertEqual(len(report['new_jobs']), 1)
        baseline = report['new_jobs'][0]
        self.assertEqual((baseline['kind'], baseline['gpu'], baseline['seed']), ('baseline', 'pro6k', seed))
        self.assertIn(f'ckpt/Gopher-{seed}_Shared/shared_warmup_50123', baseline['command'])
        self.assertNotIn('-config_path', baseline['command'])
        self.assertNotIn(scheduler.PREFIX + 'save_warmup', baseline['command'])
        queues = {**self.queues, 'pro6k': baseline['command']}
        state, report = self.plan(rows, state, queues)
        self.assertEqual(report['new_jobs'], [])
        rows.append(row(seed, 4400, kind='baseline', run_id='newbaseline'))
        state, report = self.plan(rows, state)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(state['jobs'][-1]['status'], 'complete')

    def test_rejection_reports_cleanup_and_replaces_seed(self):
        state, report = self.plan()
        seed = report['new_jobs'][0]['seed']
        state, report = self.plan(self.rows + [row(seed, 5154.8, run_id='failedscore')], state)
        self.assertEqual([j['kind'] for j in report['new_jobs']], ['retrieval'])
        self.assertEqual(report['new_jobs'][0]['seed'], seed + 1)
        self.assertEqual(report['cleanup'][0]['warmup'], f'ckpt/Gopher-{seed}_Shared')
        self.assertEqual(report['new_jobs'][0]['reference_score'], 3000)

    def test_live_score_never_promotes_and_shared_phase_reserves_seed(self):
        state, report = self.plan()
        seed = report['new_jobs'][0]['seed']
        for overrides in ({}, {'Retrieval Enable': '["retrieval"]'}):
            with self.subTest(overrides=overrides):
                state, report = self.plan(self.rows + [row(seed, 100000, state='running', **overrides)], state)
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(state['jobs'][0]['status'], 'running')

    def test_existing_queued_rerun_does_not_reuse_stale_finished_score(self):
        queues = {**self.queues, 'A6000': scheduler.make_command('Gopher', 6040, 'retrieval')}
        rows = self.rows + [row(6040, 10000, run_id='old')]
        state, report = self.plan(rows, queues=queues)
        self.assertEqual(report['new_jobs'], [])
        state, report = self.plan(rows, state)
        self.assertEqual(state['jobs'][0]['status'], 'awaiting_result')
        self.assertFalse(state['jobs'][0].get('accepted', False))

    def test_all_gpu_slots_get_work_before_second_wave(self):
        self.config = json.loads((scheduler.HERE / 'experiment_scheduler.json').read_text())
        self.config['max_new_jobs'] = 7
        state, report = self.plan()
        counts = {gpu: sum(j['gpu'] == gpu for j in report['new_jobs']) for gpu in self.config['gpus']}
        self.assertEqual(counts, {'pro6k': 1, '3090': 1, 'A6000': 4, 'titan': 1})
        self.assertEqual(len({j['seed'] for j in state['jobs']}), 7)

    def test_other_configs_consume_capacity_but_do_not_supply_scores(self):
        self.config['gpus']['titan']['count'] = 1
        queue = scheduler.make_command('Krull', 1, 'retrieval').replace("['retrieval']", "['value', 'add']")
        state, report = self.plan(queues={**self.queues, 'pro6k': queue})
        self.assertEqual(report['gpu_available_hours_before']['pro6k'], [7.5])
        self.assertEqual(report['new_jobs'][0]['gpu'], 'titan')
        self.assertEqual({job['game'] for job in state['jobs']}, {'Gopher'})

    def test_cleanup_is_suppressed_while_another_variant_uses_seed(self):
        state, report = self.plan()
        seed = report['new_jobs'][0]['seed']
        rows = self.rows + [row(seed, 1000), row(seed, state='running', run_id='ablation', **{'Value Signal': 'value'})]
        _, report = self.plan(rows, state)
        self.assertEqual(report['cleanup'], [])

    def test_failed_run_or_manual_failure_releases_replacement(self):
        state, _ = self.plan()
        job = state['jobs'][0]
        state, report = self.plan(state=state, fail_jobs=[job['id']])
        self.assertEqual(state['jobs'][0]['status'], 'failed')
        self.assertEqual(report['new_jobs'][0]['seed'], job['seed'] + 1)
        self.assertEqual(len(report['cleanup']), 1)

    def test_old_baseline_score_cannot_complete_new_followup(self):
        state, _ = self.plan()
        seed = state['jobs'][0]['seed']
        rows = self.rows + [row(seed, 6000), row(seed, 3000, kind='baseline', run_id='oldbaseline')]
        state, report = self.plan(rows, state)
        self.assertEqual(report['new_jobs'][0]['kind'], 'baseline')
        state, report = self.plan(rows, state)
        self.assertEqual(state['jobs'][-1]['status'], 'awaiting_result')

    def test_baseline_crash_retries_same_warmup_without_repeating_target(self):
        state, _ = self.plan()
        seed = state['jobs'][0]['seed']
        rows = self.rows + [row(seed, 6000)]
        state, _ = self.plan(rows, state)
        rows.append(row(seed, state='crashed', kind='baseline', run_id='crashedbaseline'))
        state, report = self.plan(rows, state)
        self.assertEqual([j['kind'] for j in report['new_jobs']], ['baseline'])
        self.assertEqual(state['jobs'][-1]['attempt'], 2)
        self.assertEqual(len(state['jobs']), 2)
        state, report = self.plan(rows, state)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(state['jobs'][-1]['status'], 'awaiting_result')

    def test_dispatch_preserves_queue_without_newline_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(self.config))
            csv_path = root / 'runs.csv'
            with csv_path.open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            queue = root / 'job_queue_pro6k.txt'
            old = scheduler.make_command('Alien', 1, 'retrieval')
            queue.write_text(old)
            state_path = root / 'state.json'
            args = ['--config', str(config_path), '--csv', str(csv_path),
                    '--queue-dir', str(root), '--state', str(state_path)]
            with redirect_stdout(io.StringIO()):
                report = scheduler.main(args + ['--dry-run'])
                self.assertFalse(state_path.exists())
                self.assertEqual(queue.read_text(), old)
                report = scheduler.main(args)
                self.assertEqual(len(report['new_jobs']), 1)
                first = queue.read_text()
                self.assertEqual(first.splitlines()[0], old)
                report = scheduler.main(args)
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(queue.read_text(), first)
                self.assertEqual(len(json.loads(state_path.read_text())['jobs']), 1)


if __name__ == '__main__':
    unittest.main()
