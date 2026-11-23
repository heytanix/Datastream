"""M3: divergences, tail term, standardised temporal statistics, Fisher-z correlations."""
import math
import unittest

import numpy as np

from dejastream.mvfp import extract_feature_fp
from dejastream.research.exp_dist_time import ar, fpw, shape_family, temporal_family
from dejastream.research.gen import auc
from dejastream.similarity import (DIST_METRICS, SimilarityConfig, compare, energy_distance, js_divergence, psi, wasserstein1)


def sk(x):
    return np.quantile(x, np.linspace(0, 1, 21))


class Divergences(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.a, self.b = sk(r.normal(0, 1, 200)), sk(r.normal(0.7, 1.3, 200))

    def test_symmetry_properties(self):
        self.assertAlmostEqual(psi(self.a, self.b, "pooled"), psi(self.b, self.a, "pooled"), places=9)
        self.assertNotAlmostEqual(psi(self.a, self.b, "a_deciles"), psi(self.b, self.a, "a_deciles"), places=3)
        self.assertAlmostEqual(js_divergence(self.a, self.b), js_divergence(self.b, self.a), places=9)
        self.assertAlmostEqual(energy_distance(self.a, self.b), energy_distance(self.b, self.a), places=9)
        self.assertAlmostEqual(wasserstein1(self.a, self.b), wasserstein1(self.b, self.a), places=9)

    def test_identity_and_bounds(self):
        self.assertAlmostEqual(psi(self.a, self.a, "pooled"), 0.0, places=9)
        self.assertAlmostEqual(js_divergence(self.a, self.a), 0.0, places=9)
        self.assertAlmostEqual(energy_distance(self.a, self.a), 0.0, places=9)
        self.assertLessEqual(js_divergence(self.a, self.b + 100), math.log(2) + 1e-9)

    def test_constant_and_sparse_sketches_are_finite(self):
        c = np.full(21, 2.0)
        for f in (lambda x, y: psi(x, y, "pooled"), lambda x, y: psi(x, y, "a_deciles"), js_divergence, energy_distance, wasserstein1):
            for x, y in ((c, c), (c, c + 1), (c, self.a), (self.a, c)):
                self.assertTrue(np.isfinite(f(x, y)))
        self.assertAlmostEqual(psi(c, c, "pooled"), 0.0, places=9)

    def test_all_metric_options_bounded_and_configurable(self):
        r = np.random.default_rng(1)
        x, y = fpw(shape_family("normal", 80, r)), fpw(shape_family("bimodal", 80, r))
        vals = {}
        for m in DIST_METRICS:
            d = compare(x, y, SimilarityConfig(dist_metric=m))
            self.assertTrue(0.0 <= d.components["dist"] < 1.0, m)
            vals[m] = d.components["dist"]
        self.assertGreater(vals["psi"], vals["wasserstein"])            # PSI sees the shape change W1 averages away

    def test_psi_family_catches_shape_that_wasserstein_misses(self):
        r = np.random.default_rng(2)
        same = [compare(fpw(shape_family("normal", 80, r)), fpw(shape_family("normal", 80, r)), SimilarityConfig(dist_metric="wasserstein")).components["dist"] for _ in range(30)]
        bi = [compare(fpw(shape_family("normal", 80, r)), fpw(shape_family("bimodal", 80, r)), SimilarityConfig(dist_metric="psi")).components["dist"] for _ in range(30)]
        self.assertGreater(np.mean(bi), np.quantile(same, 0.9))


class TailTerm(unittest.TestCase):
    def test_tail_term_sees_outlier_shape_with_unchanged_core(self):
        r = np.random.default_rng(3)
        off, on = SimilarityConfig(tail_weight=0.0), SimilarityConfig(tail_weight=1.0)
        a = [fpw(shape_family("normal", 120, r)) for _ in range(20)]
        b = [fpw(shape_family("heavy_tail_core", 120, r)) for _ in range(20)]
        d0 = np.mean([compare(x, y, off).components["stat"] for x, y in zip(a, b)])
        d1 = np.mean([compare(x, y, on).components["stat"] for x, y in zip(a, b)])
        self.assertGreater(d1, d0)

    def test_tail_term_is_off_by_default_and_same_concept_stays_small(self):
        self.assertEqual(SimilarityConfig().tail_weight, 0.0)
        r = np.random.default_rng(4)
        on = SimilarityConfig(tail_weight=1.0)
        same = [compare(fpw(shape_family("normal", 100, r)), fpw(shape_family("normal", 100, r)), on).components["stat"] for _ in range(30)]
        self.assertLess(np.quantile(same, 0.95), 0.25)


class Temporal(unittest.TestCase):
    cfg = SimilarityConfig(time_norm="standardised")

    def test_standardised_separates_the_temporal_tasks(self):
        r = np.random.default_rng(5)
        for ref, other in (("iid", "ar0.9"), ("sine12", "sine25"), ("trend_up", "trend_down")):
            same = [compare(fpw(temporal_family(ref, 120, r)), fpw(temporal_family(ref, 120, r)), self.cfg).components["time"] for _ in range(20)]
            diff = [compare(fpw(temporal_family(ref, 120, r)), fpw(temporal_family(other, 120, r)), self.cfg).components["time"] for _ in range(20)]
            self.assertGreater(auc(diff, same), 0.9, (ref, other))      # rank separation, as measured in the experiment

    def test_insufficient_observations_are_unavailable_never_zero(self):
        r = np.random.default_rng(6)
        d = compare(fpw(r.normal(0, 1, (8, 3))), fpw(r.normal(0, 1, (8, 3))), self.cfg)
        self.assertIsNone(d.components["time"])

    def test_irregular_timestamps_exclude_lag_statistics_but_not_everything(self):
        r = np.random.default_rng(7)
        t = np.cumsum(r.exponential(1.0, 150) ** 2)
        a = extract_feature_fp(r.normal(0, 1, (150, 2)), t=t)
        b = extract_feature_fp(r.normal(0, 1, (150, 2)), t=t)
        self.assertTrue(np.isnan(a.temporal["acf1"]).all())
        d = compare(a, b, self.cfg)
        self.assertTrue(d.components["time"] is not None and np.isfinite(d.components["time"]))

    def test_each_statistic_is_transformed_to_comparable_units(self):
        from dejastream.similarity import _time_transform
        for key, v in (("acf1", np.array([0.2, 0.9, np.nan])), ("volatility", np.array([0.5, 2.0, np.nan])), ("abrupt", np.array([0.0, 0.05, np.nan]))):
            out = _time_transform(key, v)
            self.assertTrue(np.isfinite(out[:2]).all() and np.isnan(out[2]))


class Correlation(unittest.TestCase):
    def draw(self, c, d, n, r, miss=0.0):
        Z = r.normal(0, 1, (n, d))
        if c:
            Z = Z @ np.linalg.cholesky((1 - c) * np.eye(d) + c * np.ones((d, d))).T
        if miss:
            Z[r.random(Z.shape) < miss] = np.nan
        return fpw(Z)

    def test_both_transforms_detect_changed_dependence(self):
        r = np.random.default_rng(8)
        w = {"stat": 0, "dist": 1, "time": 0, "pred": 0}
        for tr in ("abs", "fisher"):
            cfg = SimilarityConfig(corr_transform=tr, weights=w)
            same = [compare(self.draw(0, 6, 80, r), self.draw(0, 6, 80, r), cfg).components["dist"] for _ in range(20)]
            diff = [compare(self.draw(0, 6, 80, r), self.draw(0.6, 6, 80, r), cfg).components["dist"] for _ in range(20)]
            self.assertGreater(np.mean(diff), np.mean(same) + 0.1, tr)

    def test_missing_values_and_constant_features(self):
        r = np.random.default_rng(9)
        X = r.normal(0, 1, (60, 5))
        X[:, 1] = 4.0
        X[r.random(X.shape) < 0.1] = np.nan
        d = compare(fpw(X), fpw(X.copy()), SimilarityConfig(corr_transform="fisher", ess_mode="per_stat"))
        self.assertTrue(np.isfinite(d.combined))

    def test_pair_cap_is_deterministic_for_high_dimensions(self):
        r = np.random.default_rng(10)
        a, b = self.draw(0, 40, 100, r), self.draw(0.4, 40, 100, r)           # 780 pairs
        cfg = SimilarityConfig(max_corr_pairs=100)
        self.assertEqual(compare(a, b, cfg).combined, compare(a, b, cfg).combined)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            SimilarityConfig(corr_transform="spearman-z")


if __name__ == "__main__":
    unittest.main()
