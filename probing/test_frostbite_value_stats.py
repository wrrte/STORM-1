"""Offline tests: no rollouts, checkpoints or GPU."""
import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch
import eval_frostbite_value as evaluation
import tempfile
import unittest
from pathlib import Path
import numpy as np
from frostbite_value_stats import make_blocks, correlations, paired_bootstrap, analyze_values

class ValueStatisticsTests(unittest.TestCase):
    def test_blocks(self):
        self.assertEqual(make_blocks(np.zeros(30), np.repeat([0,1,2], [10,8,12])), [(0,10),(18,30)])
        self.assertEqual(make_blocks(np.zeros(70), np.zeros(70)), [(0,64)])

    def test_correlations(self):
        scores = correlations([np.arange(17), -np.arange(17), np.ones(17)])
        np.testing.assert_allclose(scores[:2], [[1,1],[-1,-1]])
        self.assertTrue(np.isnan(scores[2]).all())

    def test_paired_unequal_clusters(self):
        baseline = np.array([0.,0.,0.,1.])
        result = paired_bootstrap(baseline, baseline+.1, [0,0,0,1], 1000, 7)
        np.testing.assert_allclose(result['means'], [.25,.35,.1])
        np.testing.assert_allclose(result['ci95'][2], [.1,.1], atol=1e-12)
        np.testing.assert_allclose(result['ci95'][0], [0,1])
        self.assertIsNone(paired_bootstrap(baseline, baseline, [0]*4)['ci95'])

    def test_undefined_pairs(self):
        result = paired_bootstrap([0,np.nan,1], [.5,1,np.nan], [0,1,2])
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['means'], [0.,.5,.5])
        self.assertIsNone(result['ci95'])

    def test_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            baseline = np.array([np.arange(17), -np.arange(17), np.ones(17), np.arange(17)])
            flash = np.array([np.arange(17), np.arange(17), np.arange(17), -np.arange(17)])
            np.savez(output/'paired_values.npz', baseline=baseline, flash=flash,
                     frame_id=np.arange(4), episode_step=np.arange(4),
                     episode_id=np.array([0,0,1,1]), block_id=np.arange(4))
            stats = analyze_values(output, 100, 7)
            self.assertEqual(stats['metrics']['spearman']['excluded_pairs'], 1)
            self.assertAlmostEqual(stats['metrics']['spearman']['improved_fraction'], 1/3)
            for name in ('statistics.json','report_ko.md','per_context_correlations.csv',
                         'per_stage_values.csv','global_profile.csv',
                         'correlation_distributions.png','correlation_distributions.pdf'):
                self.assertTrue((output/name).is_file(), name)

    def analyze_single(self, values, episodes, repetitions=1000):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            np.savez(output/'model_values.npz', values=values, run_name=np.array('td-only'),
                     frame_id=np.arange(len(values)), episode_step=np.arange(len(values)),
                     episode_id=np.array(episodes), block_id=np.arange(len(values)))
            result = analyze_values(output, repetitions, 7)
            for name in ('statistics.json', 'report_ko.md', 'per_context_correlations.csv',
                         'per_stage_values.csv', 'global_profile.csv',
                         'correlation_distributions.png', 'correlation_distributions.pdf'):
                self.assertTrue((output/name).is_file(), name)
            self.assertNotIn('FLASH', (output/'report_ko.md').read_text())
            return result

    def test_single_distinguishes_aggregation_orders_and_weights_contexts(self):
        stages = np.arange(17)
        values = np.array([stages, stages, stages, -2*stages])
        result = self.analyze_single(values, [0, 0, 0, 1])
        for metric in ('spearman', 'kendall'):
            stats = result['metrics'][metric]
            self.assertAlmostEqual(stats['mean'], .5)
            self.assertAlmostEqual(result['global_profile_correlations'][metric], 1.)
            np.testing.assert_allclose(stats['ci95'], [-1, 1])
            np.testing.assert_allclose(result['global_profile_ci95'][metric], [-1, 1])
        reference = paired_bootstrap([1., 1., 1., -1.], [1., 1., 1., -1.], [0, 0, 0, 1], 1000, 7)
        np.testing.assert_allclose(result['metrics']['spearman']['ci95'], reference['ci95'][0])

    def test_single_undefined_contexts_and_constant_profile_draws(self):
        stages = np.arange(17)
        result = self.analyze_single(np.array([stages, -stages, np.ones(17), stages]), [0, 0, 1, 1])
        for metric in ('spearman', 'kendall'):
            self.assertEqual(result['metrics'][metric]['undefined_contexts'], 1)
            self.assertEqual(result['metrics'][metric]['n'], 3)
            self.assertAlmostEqual(result['metrics'][metric]['mean'], 1/3)
            self.assertGreater(result['global_profile_undefined_bootstrap_repetitions'][metric], 0)
            np.testing.assert_allclose(result['global_profile_ci95'][metric], [1, 1])

    def test_single_constant_values_have_no_defined_scores(self):
        result = self.analyze_single(np.ones((4, 17)), [0, 0, 1, 1], repetitions=100)
        for metric in ('spearman', 'kendall'):
            self.assertIsNone(result['metrics'][metric]['mean'])
            self.assertIsNone(result['metrics'][metric]['ci95'])
            self.assertIsNone(result['global_profile_correlations'][metric])
            self.assertIsNone(result['global_profile_ci95'][metric])
            self.assertEqual(result['global_profile_undefined_bootstrap_repetitions'][metric], 100)

    def test_single_one_episode_has_scores_but_no_ci(self):
        result = self.analyze_single(np.tile(np.arange(17), (3, 1)), [0, 0, 0], repetitions=100)
        for metric in ('spearman', 'kendall'):
            self.assertAlmostEqual(result['metrics'][metric]['mean'], 1.)
            self.assertAlmostEqual(result['global_profile_correlations'][metric], 1.)
            self.assertIsNone(result['metrics'][metric]['ci95'])
            self.assertIsNone(result['global_profile_ci95'][metric])

    def test_single_global_profile_ci_recomputes_profile_on_each_resample(self):
        stages = np.arange(17)
        values = np.array([stages, stages[::-1]**2, stages**2, stages[::-1]])
        episodes = np.array([0, 0, 1, 2])
        result = self.analyze_single(values, episodes, repetitions=100)
        rng = np.random.default_rng(7)
        draws = []
        for _ in range(100):
            selected = rng.integers(0, 3, 3)
            rows = np.concatenate([values[episodes == episode] for episode in selected])
            draws.append(correlations(rows.mean(axis=0)[None])[0])
        draws = np.array(draws)
        for j, metric in enumerate(('spearman', 'kendall')):
            expected = np.quantile(draws[np.isfinite(draws[:, j]), j], [.025, .975])
            np.testing.assert_allclose(result['global_profile_ci95'][metric], expected)

class ValueEvaluationCLITests(unittest.TestCase):
    def evaluate_saved_contexts(self, paired):
        runs = ['baseline', 'candidate'] if paired else ['td-only']
        flags = ['--baseline-run', runs[0], '--flash-run', runs[1]] if paired else ['--run', runs[0]]
        values = np.tile(np.arange(17), (4, 1))
        with tempfile.TemporaryDirectory() as temp:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp)
                for run in runs:
                    checkpoint_dir = Path('ckpt') / run
                    checkpoint_dir.mkdir(parents=True)
                    for kind in ('world_model', 'agent'):
                        (checkpoint_dir / f'{kind}_100000.pth').touch()
                np.savez('shared.npz', obs=np.zeros((20, 64, 64, 3), dtype=np.uint8),
                         ram=np.zeros((20, 128), dtype=np.uint8), action=np.zeros(20, dtype=int),
                         episode_id=np.repeat([0, 1], 10), episode_step=np.tile(np.arange(10), 2),
                         episode_end=np.tile([False]*9 + [True], 2),
                         collection_run=np.array('original-collector'))
                with patch.object(sys, 'argv', ['eval_frostbite_value.py', *flags,
                        '--data', 'shared.npz', '--output', 'result']), \
                     patch.dict(sys.modules, {'utils': SimpleNamespace(load_config=lambda _: object())}), \
                     patch.object(evaluation.torch.cuda, 'is_available', return_value=True), \
                     patch.object(evaluation.torch.cuda, 'empty_cache'), \
                     patch.object(evaluation.cv2, 'imread', return_value=np.zeros((7, 16, 3), dtype=np.uint8)), \
                     patch.object(evaluation, 'seed_all'), \
                     patch.object(evaluation, 'load_models', return_value=(object(), object())) as loader, \
                     patch.object(evaluation, 'collect_episodes_data') as collector, \
                     patch.object(evaluation, 'evaluate_values', return_value=values) as evaluator, \
                     patch.object(evaluation, 'analyze_values') as analyzer, redirect_stdout(io.StringIO()):
                    evaluation.main()
                self.assertEqual([call.args[0] for call in loader.call_args_list], runs)
                self.assertEqual(evaluator.call_count, len(runs))
                collector.assert_not_called()
                analyzer.assert_called_once_with(Path('result'), 10000, 2027, 'episode')
                archive = 'paired_values.npz' if paired else 'model_values.npz'
                with np.load(Path('result') / archive) as saved:
                    np.testing.assert_array_equal(saved['baseline' if paired else 'values'], values)
                    np.testing.assert_array_equal(saved['episode_id'], [0, 0, 1, 1])
                protocol = json.loads(Path('result/protocol.json').read_text())
                self.assertEqual(protocol['evaluation_mode'], 'paired' if paired else 'single')
                self.assertEqual(protocol['collection_run'], 'original-collector')
                self.assertFalse((Path('result') / ('model_values.npz' if paired else 'paired_values.npz')).exists())
            finally:
                os.chdir(original_cwd)

    def test_single_loads_only_requested_model_and_reuses_contexts(self):
        self.evaluate_saved_contexts(paired=False)

    def test_paired_evaluation_still_loads_both_models(self):
        self.evaluate_saved_contexts(paired=True)

    def test_rejects_mixed_single_and_paired_flags(self):
        with patch.object(sys, 'argv', ['eval_frostbite_value.py', '--run', 'td-only',
                                       '--baseline-run', 'baseline']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                evaluation.main()
        self.assertEqual(raised.exception.code, 2)



if __name__ == '__main__':
    unittest.main()
