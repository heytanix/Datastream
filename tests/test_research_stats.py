"""Statistics helpers used by the experiments: permutation test, Holm correction, AUC."""
import unittest

import numpy as np

from datastream.research.exp_paired import holm, sign_flip_p
from datastream.research.gen import auc


class SignFlip(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(sign_flip_p(np.array([1.0, 1.0, 1.0])), 2 / 8)      # only the two all-same-sign patterns are as extreme
        self.assertAlmostEqual(sign_flip_p(np.array([1.0, -1.0])), 1.0)
        self.assertEqual(sign_flip_p(np.zeros(5)), 1.0)
        self.assertEqual(sign_flip_p(np.array([])), 1.0)
        self.assertAlmostEqual(sign_flip_p(np.ones(10)), 2 / 1024)                 # smallest attainable p with 10 seeds

    def test_null_rejection_rate_is_controlled(self):
        rng = np.random.default_rng(0)
        p = [sign_flip_p(rng.normal(0, 1, 8)) for _ in range(400)]
        self.assertLess(np.mean(np.array(p) <= 0.05), 0.08)                        # exact test: no more than ~5% (+ MC error)

    def test_detects_a_real_shift(self):
        rng = np.random.default_rng(1)
        self.assertLess(sign_flip_p(rng.normal(2.0, 0.5, 10)), 0.01)


class HolmTests(unittest.TestCase):
    def test_holm_is_monotone_bounded_and_not_smaller_than_raw(self):
        ps = [0.01, 0.04, 0.03, 0.5]
        adj = holm(ps)
        self.assertTrue(all(a >= p - 1e-12 for a, p in zip(adj, ps)))
        self.assertTrue(all(0 <= a <= 1 for a in adj))
        order = np.argsort(ps)
        self.assertTrue(all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(len(ps) - 1)))
        self.assertAlmostEqual(adj[0], 0.04)                                       # 4 * 0.01
        self.assertEqual(holm([0.2]), [0.2])


class AucTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(auc(np.array([2, 3]), np.array([0, 1])), 1.0)
        self.assertEqual(auc(np.array([0, 1]), np.array([2, 3])), 0.0)
        self.assertAlmostEqual(auc(np.array([1, 1]), np.array([1, 1])), 0.5)        # ties count half
        self.assertTrue(np.isnan(auc(np.array([]), np.array([1.0]))))


if __name__ == "__main__":
    unittest.main()
