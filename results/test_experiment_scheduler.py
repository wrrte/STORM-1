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
        self.config['successful_pairs_per_game'] = 1
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
        self.assertEqual(state['jobs'][0]['seed'], 2)

    def test_finished_hash_10_is_preferred_as_in_workbook(self):
        older = row(10, 2000, run_id='older', **{'Created At': '2026-09-20T00:00:00Z'})
        newer = row(10, 1000, run_id='newer', **{'Hash Bits': '11'})
        scores = scheduler.latest_scores([older, newer], {})
        self.assertEqual(scores['Gopher', 'retrieval', 10]['score'], 2000)

    def test_pending_queue_is_idempotent_and_missing_command_is_replanned(self):
        state, report = self.plan()
        job = report['new_jobs'][0]
        queues = {**self.queues, 'pro6k': job['command']}
        state, report = self.plan(state=state, queues=queues)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['removed_jobs'], [])
        state, report = self.plan(state=state)
        self.assertEqual(report['removed_jobs'], [job['id']])
        self.assertEqual([item['id'] for item in report['new_jobs']], [job['id']])
        self.assertEqual(state['jobs'][0]['status'], 'pending')

    def test_queue_sync_replans_with_current_exclusions_and_cleans_legacy_waits(self):
        original, _ = self.plan()
        job = original['jobs'][0]
        for status in ('pending', 'awaiting_result'):
            with self.subTest(status=status):
                job['status'] = status
                new_state, report = self.plan(state=original, excluded={'Gopher': {10}})
                replacement = report['new_jobs'][0]
                self.assertEqual(replacement['id'], job['id'])
                self.assertEqual(replacement['reference_seed'], 1)
                self.assertAlmostEqual(replacement['threshold_score'], 6154.9)
                self.assertEqual(report['removed_jobs'], [job['id']])
                self.assertEqual(report['gpu_available_hours_before']['pro6k'], [0])
                self.assertEqual(report['cleanup'], [])
                self.assertEqual(len(new_state['jobs']), 1)
                self.assertEqual(job['reference_seed'], 10)
                self.assertEqual(job['status'], status)
        _, report = self.plan(state=original, excluded={'Gopher': {job['seed']}})
        self.assertEqual(report['new_jobs'][0]['seed'], 710)

    def test_queue_sync_preserves_csv_execution_and_results(self):
        state, _ = self.plan()
        job = state['jobs'][0]
        for status, overrides in (
                ('running', {}), ('finished', {}), ('failed', {}),
                ('running', {'Retrieval Enable': "['retrieval']"}),
                ('finished', {'Retrieval Enable': "['retrieval']"})):
            rows = self.rows + [row(job['seed'], 6000, state=status, **overrides)]
            with self.subTest(status=status, overrides=overrides):
                updated, report = self.plan(rows, state)
                self.assertEqual(report['removed_jobs'], [])
                self.assertTrue(updated['jobs'][0]['execution_observed'])

    def test_queue_sync_remembers_execution_when_later_csv_omits_it(self):
        state, _ = self.plan()
        job = state['jobs'][0]
        state, _ = self.plan(self.rows + [row(job['seed'], state='running')], state)
        for _ in range(2):
            state, report = self.plan(state=state)
            self.assertEqual(report['removed_jobs'], [])
            self.assertEqual(report['new_jobs'], [])
            self.assertEqual(state['jobs'][0]['status'], 'awaiting_result')
        # A running status from an older ledger is also execution evidence.
        state['jobs'][0].pop('execution_observed')
        state['jobs'][0]['status'] = 'running'
        state, report = self.plan(state=state)
        self.assertEqual(report['removed_jobs'], [])
        self.assertTrue(state['jobs'][0]['execution_observed'])

    def test_queue_sync_replans_baseline_and_preserves_completed_source(self):
        state, _ = self.plan()
        seed = state['jobs'][0]['seed']
        rows = self.rows + [row(seed, 6000, **{
            'Warmup Directory': f'ckpt/Gopher-{seed}_Shared/shared_warmup_50123'})]
        state, report = self.plan(rows, state)
        baseline = report['new_jobs'][0]
        completed = state['jobs'][0].copy()
        state, report = self.plan(rows, state)
        self.assertEqual(report['removed_jobs'], [baseline['id']])
        self.assertEqual(state['jobs'][0], completed)
        self.assertEqual(len(report['new_jobs']), 1)
        self.assertEqual(report['new_jobs'][0]['command'], baseline['command'])
        self.assertEqual(report['cleanup'], [])

    def test_queue_sync_only_removes_deleted_entries_and_releases_capacity(self):
        self.config['lookahead_hours'] = 18
        self.config['successful_pairs_per_game'] = None
        state, _ = self.plan()
        kept = state['jobs'][::2]
        deleted = state['jobs'][1::2]
        queues = {**self.queues, 'pro6k': '\n'.join(job['command'] for job in kept)}
        updated, report = self.plan(state=state, queues=queues,
                                    excluded={'Gopher': {deleted[0]['seed']}})
        self.assertEqual(set(report['removed_jobs']), {job['id'] for job in deleted})
        self.assertEqual(report['gpu_available_hours_before']['pro6k'], [9])
        self.assertEqual(report['gpu_available_hours_after']['pro6k'], [18])
        self.assertEqual(len(report['new_jobs']), 2)
        by_id = {job['id']: job for job in updated['jobs']}
        self.assertEqual(len(by_id), 4)
        for job in kept:
            self.assertEqual(by_id[job['id']], job)
        self.assertNotIn(deleted[0]['id'], by_id)

    def test_queue_sync_removes_obsolete_job_without_recreating_it(self):
        state, _ = self.plan()
        job_id = state['jobs'][0]['id']
        # Current scores no longer meet the anomaly criteria.
        rows = [row(10, 100000), row(1, 100000)]
        state, report = self.plan(rows, state)
        self.assertEqual(state['jobs'], [])
        self.assertEqual(report['removed_jobs'], [job_id])
        self.assertEqual(report['new_jobs'], [])

    def test_queue_sync_handles_replacing_a_command_with_another_variant(self):
        state, _ = self.plan()
        job = state['jobs'][0]
        queues = {**self.queues, 'pro6k': job['command'].replace("['retrieval']", "['value']")}
        state, report = self.plan(state=state, queues=queues)
        self.assertEqual(report['removed_jobs'], [job['id']])
        self.assertEqual(report['new_jobs'][0]['seed'], 710)
        self.assertEqual(report['gpu_available_hours_before']['pro6k'], [4.5])

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
        self.assertEqual(report['new_jobs'][0]['seed'], 710)
        self.assertEqual(report['cleanup'][0]['warmup'], f'ckpt/Gopher-{seed}_Shared')
        self.assertEqual(report['new_jobs'][0]['reference_score'], 3000)

    def test_excluded_external_results_are_not_adopted_or_reused(self):
        excluded = {'Gopher': {2}}
        for score in (1000, 6000):
            with self.subTest(score=score):
                rows = self.rows + [row(2, score, **{'Save Warmup Requested': 'True'})]
                state, report = self.plan(rows, excluded=excluded)
                self.assertEqual([job['seed'] for job in state['jobs']], [710])
                self.assertEqual([(job['seed'], job['kind']) for job in report['new_jobs']],
                                 [(710, 'retrieval')])
                self.assertEqual(report['cleanup'], [])

    def test_excluded_ledger_results_are_silent_and_remain_reserved(self):
        for score, status in ((1000, 'finished'), (6000, 'finished'), (None, 'failed')):
            with self.subTest(score=score, status=status):
                state, _ = self.plan()
                job = state['jobs'][0]
                excluded = {'Gopher': {job['seed']}}
                rows = self.rows + [row(job['seed'], score, state=status, run_id='result')]
                # Cover both newly observed results and already recorded results.
                for _ in range(2):
                    state, report = self.plan(rows, state, excluded=excluded)
                    self.assertIn(job['id'], [item['id'] for item in state['jobs']])
                    self.assertTrue(all(item['seed'] != job['seed'] for item in report['new_jobs']))
                    self.assertEqual(report['cleanup'], [])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        scheduler.print_report(report, state, True, excluded)
                    self.assertNotIn(job['id'], output.getvalue())
                    self.assertNotIn('warmup 삭제 필요', output.getvalue())

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
        state, report = self.plan(rows, state, queues)
        self.assertEqual(state['jobs'][0]['status'], 'pending')
        self.assertFalse(state['jobs'][0].get('accepted', False))
        state, report = self.plan(rows, state)
        self.assertEqual(report['removed_jobs'], ['Gopher:6040:retrieval'])
        self.assertFalse(any(job['seed'] == 6040 for job in state['jobs']))

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
        self.assertEqual(report['new_jobs'][0]['seed'], 710)
        self.assertEqual(len(report['cleanup']), 1)

    def test_old_baseline_score_cannot_complete_new_followup(self):
        state, _ = self.plan()
        seed = state['jobs'][0]['seed']
        rows = self.rows + [row(seed, 6000), row(seed, 3000, kind='baseline', run_id='oldbaseline')]
        state, report = self.plan(rows, state)
        self.assertEqual(report['new_jobs'][0]['kind'], 'baseline')
        baseline = report['new_jobs'][0]
        queues = {**self.queues, 'pro6k': baseline['command']}
        state, report = self.plan(rows, state, queues)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(state['jobs'][-1]['status'], 'pending')
        state, report = self.plan(rows, state)
        self.assertEqual(report['removed_jobs'], [baseline['id']])
        self.assertEqual(report['new_jobs'][0]['kind'], 'baseline')
        self.assertEqual(state['jobs'][-1]['status'], 'pending')

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
        queues = {**self.queues, 'pro6k': report['new_jobs'][0]['command']}
        state, report = self.plan(rows, state, queues)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(state['jobs'][-1]['status'], 'pending')

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
                # Simulate removing only the generated command before execution.
                queue.write_text(old)
                saved_state = state_path.read_text()
                job_id = json.loads(saved_state)['jobs'][0]['id']
                report = scheduler.main(args + ['--dry-run'])
                self.assertEqual(report['removed_jobs'], [job_id])
                self.assertEqual(len(report['new_jobs']), 1)
                self.assertEqual(state_path.read_text(), saved_state)
                self.assertEqual(queue.read_text(), old)
                scheduler.main(args)
                self.assertEqual(queue.read_text(), first)
                report = scheduler.main(args)
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(queue.read_text(), first)

    def test_anomaly_mean_and_low_seed_boundaries(self):
        self.references = {'Gopher': (0, 100, 100)}
        for values, expected in [
                ([80], True),                         # Exactly 0.2 below paper.
                ([80.00000001], False),
                ([40, 150], True),                     # Mean only 0.05 below, one seed 0.6 below.
                ([40.00000001, 150], False),
                ([40, 160], False),                    # Mean equals paper.
                ([40, 170], False)]:                   # Mean above paper.
            for kind in ('retrieval', 'baseline'):
                with self.subTest(values=values, kind=kind):
                    rows = [row(seed, score, kind=kind) for seed, score in zip((1, 10), values)]
                    _, report = self.plan(rows)
                    self.assertEqual('Gopher' in report['games'], expected)

    def test_excluded_seed_and_unrelated_config_do_not_trigger_anomaly(self):
        self.references = {'Gopher': (0, 100, 100)}
        rows = [row(1, 40), row(10, 150), row(2, -1000, **{'Value Signal': 'value'})]
        _, report = self.plan(rows, excluded={'Gopher': {1}})
        self.assertNotIn('Gopher', report['games'])
        # Both halves of the second condition belong to the same config.
        rows = [row(1, 95), row(1, 40, kind='baseline'), row(10, 170, kind='baseline')]
        _, report = self.plan(rows)
        self.assertNotIn('Gopher', report['games'])

    def test_each_gpu_gets_eighteen_hours_without_old_sixteen_job_cap(self):
        self.config = json.loads((scheduler.HERE / 'experiment_scheduler.json').read_text())
        state, report = self.plan()
        self.assertEqual(len(report['new_jobs']), 22)
        self.assertEqual(report['gpu_available_hours_after'], {
            'pro6k': [18], '3090': [28], 'A6000': [18, 18, 18, 18], 'titan': [20]})
        self.assertEqual(report['coverage_shortfall_hours'], {})
        pairs = [(job['game'], job['seed']) for job in report['new_jobs']]
        self.assertEqual(len(pairs), len(set(pairs)))
        queues = {gpu: '\n'.join(j['command'] for j in report['new_jobs'] if j['gpu'] == gpu)
                  for gpu in self.config['gpus']}
        _, repeated = self.plan(state=state, queues=queues)
        self.assertEqual(repeated['new_jobs'], [])

    def test_gpu_expansion_and_uneven_existing_work_are_filled_individually(self):
        self.config = json.loads((scheduler.HERE / 'experiment_scheduler.json').read_text())
        self.config['gpus']['A6000']['count'] = 5
        self.config['gpus']['pro6k']['count'] = 2
        # One long command can leave other A6000 devices completely idle.
        long_job = scheduler.make_command('Alien', 6000, 'retrieval').replace(
            "['retrieval']", "['retrieval', 'value', 'add', 'target1', 'baseline']")
        _, report = self.plan(queues={**self.queues, 'A6000': long_job})
        self.assertEqual(report['gpu_available_hours_before']['A6000'], [19, 0, 0, 0, 0])
        for hours in report['gpu_available_hours_after'].values():
            self.assertTrue(all(hour >= 18 for hour in hours))
        self.assertEqual(report['gpu_available_hours_after']['A6000'], [19, 18, 18, 18, 18])

    def test_full_queues_do_not_get_an_unnecessary_first_trial(self):
        self.config['lookahead_hours'] = 18
        queues = {**self.queues, 'pro6k': '\n'.join(
            scheduler.make_command('Alien', seed, 'retrieval') for seed in (1, 2, 10, 710))}
        _, report = self.plan(queues=queues)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['gpu_available_hours_after']['pro6k'], [18])

    def test_eighteen_hour_fill_continues_after_a_success_if_game_is_still_low(self):
        state, _ = self.plan()
        seed = state['jobs'][0]['seed']
        self.config['lookahead_hours'] = 18
        self.config['successful_pairs_per_game'] = None
        _, report = self.plan(self.rows + [row(seed, 6000)], state)
        self.assertEqual(report['new_jobs'][0]['kind'], 'baseline')
        self.assertTrue(any(j['kind'] == 'retrieval' for j in report['new_jobs']))
        self.assertGreaterEqual(report['gpu_available_hours_after']['pro6k'][0], 18)

    def test_gpu_seed_rules_extend_and_reuse_seeds_between_different_games(self):
        cases = [
            ('3090', [2000, 2010], 2020),
            ('A6000', list(range(6000, 6100, 10)), 6100),
            ('pro6k', [1, 2, 10, 710, 1710, 2710, 3710], 4710),
            ('titan', [9999, 9998, 9997, 9996], 9995),
        ]
        for gpu, used, expected in cases:
            with self.subTest(gpu=gpu):
                for name, spec in self.config['gpus'].items():
                    spec['count'] = int(name == gpu)
                rows = [row(seed, 3000) for seed in used]
                rows.append(row(expected, 1000, game='Alien'))
                _, report = self.plan(rows, games=['Gopher'])
                self.assertEqual([(j['gpu'], j['seed']) for j in report['new_jobs']], [(gpu, expected)])
                self.assertEqual(scheduler.gpu_for_seed(expected, self.config, [], []), gpu)

    def test_unrun_historical_seed_precedes_extended_seed(self):
        for name, spec in self.config['gpus'].items():
            spec['count'] = int(name == '3090')
        _, report = self.plan([row(2000, 3000), row(2010, 1000, game='Alien')], games=['Gopher'])
        self.assertEqual(report['new_jobs'][0]['seed'], 2010)

    def test_excluded_and_reserved_seed_are_skipped_per_game(self):
        for name, spec in self.config['gpus'].items():
            spec['count'] = int(name == '3090')
        state, report = self.plan([row(2000, 3000), row(2010, 3000)], excluded={'Gopher': {2020}})
        self.assertEqual(report['new_jobs'][0]['seed'], 2030)
        state, report = self.plan([row(2000, 3000), row(2010, 3000)], state,
                                 excluded={'Gopher': {2020}}, fail_jobs=['Gopher:2030:retrieval'])
        self.assertEqual(report['new_jobs'][0]['seed'], 2040)

    def test_legacy_dispatched_seed_keeps_its_gpu_after_rule_change(self):
        state, _ = self.plan()
        job = state['jobs'][0]
        job.update(seed=10000, id='Gopher:10000:retrieval', warmup='ckpt/Gopher-10000_Shared')
        rows = self.rows + [row(10000, 6000)]
        _, report = self.plan(rows, state)
        self.assertEqual(report['new_jobs'][0]['gpu'], 'pro6k')
        self.assertEqual(report['new_jobs'][0]['seed'], 10000)

    def test_optional_cap_reports_unfilled_capacity(self):
        self.config['lookahead_hours'] = 18
        self.config['max_new_jobs'] = 1
        _, report = self.plan()
        self.assertEqual(report['coverage_shortfall_hours'], {'pro6k': [13.5]})
        self.assertTrue(any('max_new_jobs' in notice for notice in report['notices']))


class MissingSeedTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((scheduler.HERE / 'experiment_scheduler.json').read_text())
        self.config['lookahead_hours'] = 0
        for gpu, spec in self.config['gpus'].items():
            spec['count'] = int(gpu == 'pro6k')
        self.references = {'Gopher': (0, 100, 100)}
        self.rows = self.pairs((1, 2, 10))
        self.queues = dict.fromkeys(self.config['gpus'], '')

    @staticmethod
    def pairs(seeds, game='Gopher'):
        return [row(seed, 100, game=game, kind=kind, run_id=f'{game}-{seed}-{kind}')
                for seed in seeds for kind in ('baseline', 'retrieval')]

    def plan(self, rows=None, state=None, queues=None, excluded=None, **kwargs):
        return scheduler.plan(self.rows if rows is None else rows,
                              self.queues if queues is None else queues, state or {}, self.config,
                              self.references, excluded or {}, NOW, **kwargs)

    def test_report_counts_only_valid_common_seeds_and_excludes_anomalies(self):
        self.references.update({game: (0, 100, 100) for game in ('Alien', 'Amidar', 'Assault')})
        rows = self.pairs((1, 2)) + [
            row(1, 101, run_id='duplicate'), row(10, 100),
            row(710, 100), row(710, 100, kind='baseline', state='failed'),
            row(1710, 100), row(1710, 100, kind='baseline', state='running'),
            row(2710, 100, kind='baseline'), row(2710, 100, **{'Retrieval Target': '1'}),
            row(1, 50, game='Alien', kind='baseline'),
        ] + self.pairs((1, 2, 10, 710), game='Amidar')
        _, report = self.plan(rows, excluded={'Gopher': {2}})
        self.assertEqual(report['insufficient_seeds'], {
            'Gopher': {'count': 1, 'missing': 3, 'seeds': [1]},
            'Assault': {'count': 0, 'missing': 4, 'seeds': []},
        })
        self.assertIn('Alien', report['games'])
        self.assertFalse(any(job['kind'] == 'paired' for job in report['new_jobs']))

    def test_shortage_output_is_red_on_terminal_and_plain_when_redirected(self):
        state, report = self.plan()
        for tty in (False, True):
            with self.subTest(tty=tty):
                output = io.StringIO()
                output.isatty = lambda: tty
                with redirect_stdout(output):
                    scheduler.print_report(report, state, True, {})
                text = output.getvalue()
                self.assertIn('main performance 공통 시드 4개 미만: 1개 게임 (이상 게임 제외)', text)
                self.assertIn('Gopher: 3개 / 4개 (1개 부족, 시드: [1, 2, 10])', text)
                self.assertEqual('\033[31m' in text, tty)
                if tty:
                    self.assertIn('\033[31m  Gopher:', text)

    def test_fill_dispatches_both_without_save_warmup_and_is_idempotent(self):
        state, report = self.plan(fill_missing_seeds=True)
        self.assertEqual(len(report['new_jobs']), 1)
        job = report['new_jobs'][0]
        parsed = scheduler.parse_command(job['command'], job['gpu'])
        self.assertEqual(parsed['names'], ['retrieval', 'baseline'])
        self.assertNotIn(scheduler.PREFIX + 'save_warmup', job['command'])
        self.assertFalse(parsed['resume'])
        self.assertEqual(job['estimated_finish_hours'], 6)
        queues = {**self.queues, 'pro6k': job['command']}
        state, report = self.plan(state=state, queues=queues, fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['removed_jobs'], [])
        self.assertEqual(state['jobs'][0]['status'], 'pending')
        # Deleting an unstarted command releases the pair reservation as well.
        _, report = self.plan(state=state, fill_missing_seeds=True)
        self.assertEqual(report['removed_jobs'], [job['id']])
        self.assertEqual([j['id'] for j in report['new_jobs']], [job['id']])

    def test_zero_seeds_need_no_comparison_score_and_stop_at_four(self):
        _, report = self.plan(rows=[], fill_missing_seeds=True)
        self.assertEqual(len(report['new_jobs']), 4)
        self.assertEqual({job['kind'] for job in report['new_jobs']}, {'paired'})
        _, report = self.plan(rows=self.pairs((1, 2, 10, 710)), fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['insufficient_seeds'], {})

    def test_external_pending_pairs_count_once_and_excluded_seeds_do_not_count(self):
        command = scheduler.make_command('Gopher', 710, 'paired')
        queues = {**self.queues, 'pro6k': command + '\n' + command}
        _, report = self.plan(queues=queues, fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        _, report = self.plan(queues=queues, excluded={'Gopher': {710}}, fill_missing_seeds=True)
        self.assertEqual([job['seed'] for job in report['new_jobs']], [1710])
        shared = row(710, state='running', **{'Retrieval Enable': "['retrieval', 'baseline']"})
        _, report = self.plan(rows=self.rows + [shared], fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        baseline = scheduler.make_command('Gopher', 710, 'baseline', 'ckpt/source_Shared')
        _, report = self.plan(rows=self.rows + [row(710, 100)],
                              queues={**self.queues, 'pro6k': baseline}, fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        # A retrieval-only queue does not guarantee a completed pair.
        _, report = self.plan(queues={**self.queues, 'pro6k': command.replace(
            "['retrieval', 'baseline']", "['retrieval']")}, fill_missing_seeds=True)
        self.assertEqual([job['kind'] for job in report['new_jobs']], ['paired'])

    def test_pair_waits_for_both_results_without_threshold_or_followup(self):
        state, report = self.plan(fill_missing_seeds=True)
        job = report['new_jobs'][0]
        rows = self.rows + [row(job['seed'], 100, run_id='new-retrieval')]
        state, report = self.plan(rows=rows, state=state, fill_missing_seeds=True)
        self.assertEqual(state['jobs'][0]['status'], 'awaiting_result')
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['gpu_available_hours_before']['pro6k'], [6])
        rows.append(row(job['seed'], kind='baseline', state='running', run_id='new-baseline'))
        state, report = self.plan(rows=rows, state=state, fill_missing_seeds=True)
        self.assertEqual(state['jobs'][0]['status'], 'running')
        with self.assertRaisesRegex(ValueError, '아직 큐/CSV에 실행 중'):
            self.plan(rows=rows, state=state, fail_jobs=[job['id']])
        rows[-1] = row(job['seed'], 100, kind='baseline', run_id='new-baseline')
        # Reconciliation also works when the fill flag is omitted next time.
        state, report = self.plan(rows=rows, state=state)
        self.assertEqual(state['jobs'][0]['status'], 'complete')
        self.assertEqual(state['jobs'][0]['scores'], {'retrieval': 100, 'baseline': 100})
        self.assertEqual(report['insufficient_seeds'], {})
        self.assertEqual(report['new_jobs'], [])
        self.assertEqual(report['cleanup'], [])

    def test_failed_pair_replaced_without_saved_warmup_cleanup(self):
        state, report = self.plan(fill_missing_seeds=True)
        job = report['new_jobs'][0]
        for kind in ('retrieval', 'baseline'):
            with self.subTest(kind=kind):
                rows = self.rows + [row(job['seed'], state='crashed', kind=kind, run_id='failed')]
                updated, report = self.plan(rows=rows, state=state, fill_missing_seeds=True)
                self.assertEqual(updated['jobs'][0]['status'], 'failed')
                self.assertEqual([j['seed'] for j in report['new_jobs']], [1710])
                self.assertEqual(report['cleanup'], [])

    def test_fill_respects_gpu_horizon_and_job_cap(self):
        self.config['lookahead_hours'] = 18
        _, report = self.plan(rows=[], fill_missing_seeds=True)
        self.assertEqual(len(report['new_jobs']), 3)
        self.assertEqual(report['gpu_available_hours_after']['pro6k'], [18])
        self.config['max_new_jobs'] = 1
        _, report = self.plan(rows=[], fill_missing_seeds=True)
        self.assertEqual(len(report['new_jobs']), 1)
        self.assertEqual(report['coverage_shortfall_hours']['pro6k'], [12])

    def test_fill_uses_game_filter_and_never_adds_pairs_for_anomalies(self):
        self.references['Alien'] = (0, 100, 100)
        rows = self.rows + self.pairs((1, 2, 10), game='Alien')
        _, report = self.plan(rows=rows, games=['Gopher'], fill_missing_seeds=True)
        self.assertEqual([(job['game'], job['kind']) for job in report['new_jobs']], [('Gopher', 'paired')])
        rows = self.rows + [row(1, 50, game='Alien')]
        _, report = self.plan(rows=rows, fill_missing_seeds=True)
        self.assertNotIn('Alien', report['insufficient_seeds'])
        self.assertEqual([(job['game'], job['kind']) for job in report['new_jobs']],
                         [('Gopher', 'paired'), ('Alien', 'retrieval')])

    def test_reuse_hns_boundary_and_single_branch_commands_in_both_directions(self):
        boundary = 100 - 100 * scheduler.REUSE_HNS_GAP
        for existing_kind in ('baseline', 'retrieval'):
            missing_kind = 'retrieval' if existing_kind == 'baseline' else 'baseline'
            for score, reusable in ((boundary - 0.00001, False), (boundary, False), (boundary + 0.00001, True),
                                    (100, True), (150, True)):
                with self.subTest(existing_kind=existing_kind, score=score):
                    rows = self.rows + [row(710, score, kind=existing_kind)]
                    _, report = self.plan(rows=rows, fill_missing_seeds=True)
                    self.assertEqual(len(report['new_jobs']), 1)
                    job = report['new_jobs'][0]
                    parsed = scheduler.parse_command(job['command'], job['gpu'])
                    self.assertEqual(job['seed'], 710 if reusable else 1710)
                    self.assertEqual(parsed['names'], [missing_kind] if reusable else ['retrieval', 'baseline'])
                    self.assertNotIn(scheduler.PREFIX + 'save_warmup', job['command'])
                    self.assertFalse(parsed['resume'])
                    hours = (4.5 if missing_kind == 'retrieval' else 3) if reusable else 6
                    self.assertEqual(job['estimated_finish_hours'], hours)

    def test_reuse_priority_groups_and_order_with_only_one_seed_missing(self):
        candidates = [
            row(710, 130), row(1710, 120),
            row(2710, 110, kind='baseline'), row(3710, 125, kind='baseline'),
            row(4710, 100), row(5710, 95),
            row(6710, 95, kind='baseline'), row(7710, 100, kind='baseline'),
        ]
        # Removing each winner checks the actual dispatch order across all four
        # groups, including equality with the paired mean and paper score.
        for index in range(len(candidates)):
            remaining = candidates[index:]
            with self.subTest(index=index):
                _, report = self.plan(rows=self.rows + list(reversed(remaining)), fill_missing_seeds=True)
                ranked = report['reusable_seeds']['Gopher']
                self.assertEqual([item['seed'] for item in ranked], [sample['seed'] for sample in remaining])
                self.assertEqual([item['priority'] for item in ranked], [1, 1, 2, 2, 3, 3, 4, 4][index:])
                self.assertEqual([job['seed'] for job in report['new_jobs']], [remaining[0]['seed']])

    def test_priority_mean_uses_only_nonexcluded_common_seeds(self):
        rows = self.rows + [row(710, 110), row(1710, 1000), row(2710, 101, kind='baseline')]
        for sample in rows:
            if sample['seed'] == 2 and sample['kind'] == 'retrieval':
                sample['score'] = 1000
        _, report = self.plan(rows=rows, excluded={'Gopher': {2}}, fill_missing_seeds=True)
        ranked = report['reusable_seeds']['Gopher']
        self.assertEqual([item['seed'] for item in ranked], [1710, 710, 2710])
        self.assertEqual([item['priority'] for item in ranked], [1, 1, 2])
        self.assertTrue(all(item['main_retrieval_mean'] == 100 for item in ranked))
        self.assertEqual([job['seed'] for job in report['new_jobs']], [1710, 710])

    def test_no_common_seed_mean_places_retrieval_in_third_group(self):
        rows = [row(710, 150), row(1710, 101, kind='baseline')]
        _, report = self.plan(rows=rows, fill_missing_seeds=True)
        ranked = report['reusable_seeds']['Gopher']
        self.assertEqual([(item['seed'], item['priority']) for item in ranked], [(1710, 2), (710, 3)])
        self.assertTrue(all(item['main_retrieval_mean'] is None for item in ranked))
        self.assertEqual([job['seed'] for job in report['new_jobs'][:2]], [1710, 710])
        self.assertEqual([job['names'] for job in report['new_jobs']],
                         [['retrieval'], ['baseline'], ['retrieval', 'baseline'], ['retrieval', 'baseline']])

    def test_reused_single_branch_queue_running_and_completion_are_idempotent(self):
        for existing_kind in ('baseline', 'retrieval'):
            with self.subTest(existing_kind=existing_kind):
                missing_kind = 'retrieval' if existing_kind == 'baseline' else 'baseline'
                rows = self.rows + [row(710, 100, kind=existing_kind, run_id='existing')]
                state, report = self.plan(rows=rows, fill_missing_seeds=True)
                job = report['new_jobs'][0]
                queues = {**self.queues, 'pro6k': job['command']}
                state, report = self.plan(rows=rows, state=state, queues=queues, fill_missing_seeds=True)
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(state['jobs'][0]['status'], 'pending')
                self.assertTrue(report['reusable_seeds']['Gopher'][0]['reserved'])
                # The pre-existing opposite result is not evidence that the new
                # command ran if the command is removed before execution.
                _, removed = self.plan(rows=rows, state=state, fill_missing_seeds=True)
                self.assertEqual(removed['removed_jobs'], [job['id']])
                running = row(710, kind=missing_kind, state='running', run_id='new')
                state, report = self.plan(rows=rows + [running], state=state, fill_missing_seeds=True)
                self.assertEqual(state['jobs'][0]['status'], 'running')
                self.assertEqual(report['new_jobs'], [])
                state, report = self.plan(rows=rows, state=state, fill_missing_seeds=True)
                self.assertEqual(state['jobs'][0]['status'], 'awaiting_result')
                self.assertEqual(report['new_jobs'], [])
                expected_hours = 4.5 if missing_kind == 'retrieval' else 3
                self.assertEqual(report['gpu_available_hours_before']['pro6k'], [expected_hours])
                completed = row(710, 100, kind=missing_kind, run_id='new')
                state, report = self.plan(rows=rows + [completed], state=state)
                self.assertEqual(state['jobs'][0]['status'], 'complete')
                self.assertEqual(state['jobs'][0]['scores'], {missing_kind: 100})
                self.assertEqual(report['insufficient_seeds'], {})
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(report['cleanup'], [])

    def test_failed_missing_branch_retries_only_that_branch(self):
        rows = self.rows + [row(710, 100, kind='baseline', run_id='existing')]
        state, _ = self.plan(rows=rows, fill_missing_seeds=True)
        rows.append(row(710, state='crashed', run_id='failed'))
        state, report = self.plan(rows=rows, state=state, fill_missing_seeds=True)
        self.assertEqual([(job['seed'], job['names'], job['attempt']) for job in report['new_jobs']],
                         [(710, ['retrieval'], 2)])
        self.assertIn('failed', state['jobs'][0]['before_ids'])
        self.assertEqual(report['cleanup'], [])

    def test_breakout_reuse_score_hns_and_missing_kind_are_printed_in_red(self):
        self.references = {'Breakout': (1.7, 30.5, 16)}
        rows = [row(seed, 16, game='Breakout', kind=kind, run_id=f'{seed}-{kind}')
                for seed in (10, 2000, 3710) for kind in ('baseline', 'retrieval')]
        rows.append(row(2, 14.35, game='Breakout', kind='baseline'))
        state, report = self.plan(rows=rows, fill_missing_seeds=True)
        self.assertEqual([(job['seed'], job['names']) for job in report['new_jobs']], [(2, ['retrieval'])])
        item = report['reusable_seeds']['Breakout'][0]
        self.assertAlmostEqual(item['hns'], (14.35 - 1.7) / 28.8)
        self.assertAlmostEqual(item['hns_gap'], (16 - 14.35) / 28.8)
        output = io.StringIO()
        output.isatty = lambda: True
        with redirect_stdout(output):
            scheduler.print_report(report, state, True, {})
        text = output.getvalue()
        self.assertIn('\033[31m    재사용 가능 (우선순위 4)', text)
        self.assertIn('seed 2: 기존 baseline 점수=14.35, HNS=0.4392', text)
        self.assertIn('ΔHNS=+0.0573', text)
        self.assertIn('retrieval만 평가', text)

    def test_unavailable_priority_seed_waits_without_lower_priority_replacement(self):
        self.config['gpus']['pro6k']['count'] = 0
        self.config['gpus']['A6000']['count'] = 1
        rows = self.rows + [row(710, 130), row(6000, 110, kind='baseline')]
        _, report = self.plan(rows=rows, fill_missing_seeds=True)
        self.assertEqual(report['new_jobs'], [])
        self.assertTrue(any('Gopher seed 710: baseline만' in notice for notice in report['notices']))

    def test_cli_fill_dry_run_and_dispatch_use_only_temporary_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, csv_path, state_path = (root / name for name in ('config.json', 'runs.csv', 'state.json'))
            config_path.write_text(json.dumps(self.config))
            # Use real reference scores so Gopher is not anomalous through main().
            rows = self.pairs((1, 2, 10))
            for sample in rows:
                sample['Eval Return'] = '10000'
            with csv_path.open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            args = ['--config', str(config_path), '--csv', str(csv_path), '--queue-dir', str(root),
                    '--state', str(state_path), '--games', 'Gopher', '--fill-missing-seeds']
            queue = root / 'job_queue_pro6k.txt'
            with redirect_stdout(io.StringIO()):
                report = scheduler.main(args + ['--dry-run'])
                self.assertEqual([j['kind'] for j in report['new_jobs']], ['paired'])
                self.assertFalse(state_path.exists())
                self.assertFalse(queue.exists())
                scheduler.main(args)
                commands = queue.read_text()
                self.assertEqual(len(commands.splitlines()), 1)
                report = scheduler.main(args)
                self.assertEqual(report['new_jobs'], [])
                self.assertEqual(queue.read_text(), commands)


if __name__ == '__main__':
    unittest.main()
