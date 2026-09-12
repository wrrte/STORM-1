"""Offline tests: no rollouts, checkpoints or GPU."""
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

if __name__ == '__main__':
    unittest.main()
