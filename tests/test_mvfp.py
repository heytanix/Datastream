"""Fingerprint stability, edge cases, discrimination, label independence."""
import inspect
import json
import unittest

import numpy as np

from datastream.matcher import ConceptRecord
from datastream.models import LogReg
from datastream.mvfp import (FeatureFingerprint, PredictiveAccumulator, extract_feature_fp, migrate_v1, predictive_fp)
from datastream.similarity import SimilarityConfig, compare, psi, wasserstein1
from datastream.stream import CONCEPTS


def window(name, n, seed, noise=0.0):
    c = CONCEPTS[name]
    r = np.random.default_rng(seed)
    X = r.normal(c.mu, c.sigma, (n, 4))
    if noise:
        X = X + r.normal(0, noise, X.shape)
    return X


def fp(X, **kw):
    return extract_feature_fp(X, t=np.arange(len(X)), **kw)


class Stability(unittest.TestCase):
    def test_same_concept_multiple_windows(self):
        ds = [compare(fp(window("A", 100, s)), fp(window("A", 100, s + 50))).combined for s in range(20)]
        self.assertLess(max(ds), 0.25)

    def test_noisy_same_concept(self):
        ds = [compare(fp(window("B", 100, s, 0.3)), fp(window("B", 100, s + 50, 0.3))).combined for s in range(20)]
        self.assertLess(np.mean(ds), 0.2)

    def test_different_window_sizes(self):
        ds = [compare(fp(window("A", 40, s)), fp(window("A", 300, s + 50))).combined for s in range(20)]
        self.assertLess(np.mean(ds), 0.15)

    def test_different_concepts_far_apart_even_with_small_windows(self):
        for other in "BCD":
            self.assertGreater(compare(fp(window("A", 40, 1)), fp(window(other, 40, 2))).combined, 0.4)

    def test_outlier_does_not_dominate(self):
        X = window("A", 100, 3)
        Xo = X.copy()
        Xo[5, 0] = 1e6
        self.assertLess(compare(fp(X), fp(Xo)).combined, 0.25)

    def test_missing_values(self):
        X = window("A", 200, 4)
        Xm = X.copy()
        Xm[np.random.default_rng(0).random(X.shape) < 0.15] = np.nan
        f = fp(Xm)
        self.assertTrue(np.all(np.isfinite(f.statistics["median"])))
        self.assertAlmostEqual(float(f.statistics["missing"].mean()), 0.15, delta=0.04)
        self.assertLess(compare(fp(X), f).combined, 0.4)

    def test_constant_feature(self):
        X = window("A", 100, 5)
        X[:, 2] = 3.0
        f = fp(X)
        self.assertTrue(np.isnan(f.statistics["skew"][2]))          # unavailable, not zero
        self.assertTrue(np.isnan(f.temporal["trend"][2]))
        self.assertTrue(np.isnan(f.distributions["corr"][1]))       # pair (0,2) involves the constant column
        self.assertAlmostEqual(compare(f, fp(X)).combined, 0.0, places=6)
        X2 = X.copy()
        X2[:, 2] = 9.0
        self.assertGreater(compare(f, fp(X2)).components["stat"], 0.1)   # constant column moved: detected, no crash

    def test_deterministic(self):
        X = window("C", 100, 6)
        self.assertEqual(json.dumps(fp(X).to_dict(), sort_keys=True), json.dumps(fp(X).to_dict(), sort_keys=True))

    def test_serialisation_roundtrip(self):
        f = fp(window("D", 100, 7))
        g = FeatureFingerprint.from_dict(json.loads(json.dumps(f.to_dict())))
        self.assertAlmostEqual(compare(f, g).combined, 0.0, places=9)
        self.assertEqual(g.schema_version, "2.0")

    def test_unknown_schema_rejected(self):
        d = fp(window("A", 50, 1)).to_dict()
        d["schema_version"] = "1.0"
        with self.assertRaises(ValueError):
            FeatureFingerprint.from_dict(d)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            extract_feature_fp(np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            extract_feature_fp(np.zeros((5, 3)), t=np.array([0, 1, np.nan, 3, 4.0]))
        with self.assertRaises(ValueError):
            extract_feature_fp(np.zeros((5, 3)), categorical=[7])


class Categorical(unittest.TestCase):
    def make(self, probs, n=300, seed=0):
        r = np.random.default_rng(seed)
        cat = r.choice(len(probs), n, p=probs).astype(float)
        cat[r.random(n) < 0.05] = np.nan
        return np.column_stack([r.normal(0, 1, n), cat])

    def test_frequencies_and_missing(self):
        f = extract_feature_fp(self.make([0.7, 0.2, 0.1]), categorical=[1])
        freq = f.categorical["freq"][0]
        self.assertAlmostEqual(sum(freq.values()), 1.0)
        self.assertGreater(freq[0], freq[2])
        self.assertEqual(f.categorical["n_unique"][0], 3)
        self.assertAlmostEqual(float(f.categorical["missing"][0]), 0.05, delta=0.04)

    def test_category_shift_and_unknown_category(self):
        a = extract_feature_fp(self.make([0.7, 0.2, 0.1]), categorical=[1])
        same = extract_feature_fp(self.make([0.7, 0.2, 0.1], seed=3), categorical=[1])
        shifted = extract_feature_fp(self.make([0.1, 0.2, 0.7], seed=4), categorical=[1])
        unseen = np.column_stack([np.random.default_rng(1).normal(0, 1, 300), np.full(300, 9.0)])
        self.assertLess(compare(a, same).components["stat"], 0.2)
        self.assertGreater(compare(a, shifted).components["stat"], 0.4)
        self.assertGreater(compare(a, extract_feature_fp(unseen, categorical=[1])).components["stat"], 0.5)


class Temporal(unittest.TestCase):
    def test_insufficient_observations_are_unavailable_not_zero(self):
        f = extract_feature_fp(window("A", 8, 1), t=np.arange(8))
        self.assertTrue(all(np.isnan(v).all() for v in f.temporal.values()))
        self.assertFalse(f.view_available("time"))
        d = compare(f, extract_feature_fp(window("A", 8, 2), t=np.arange(8)))
        self.assertIsNone(d.components["time"])

    def test_row_order_follows_timestamps(self):
        X = window("A", 120, 2)
        t = np.arange(120.0)
        perm = np.random.default_rng(0).permutation(120)
        a, b = extract_feature_fp(X, t=t), extract_feature_fp(X[perm], t=t[perm])
        for k in a.temporal:
            np.testing.assert_allclose(a.temporal[k], b.temporal[k], equal_nan=True)

    def test_irregular_timestamps_drop_lag_statistics_only(self):
        X = window("A", 120, 2)
        t = np.cumsum(np.random.default_rng(1).exponential(1.0, 120) ** 2)
        f = extract_feature_fp(X, t=t)
        self.assertTrue(np.isnan(f.temporal["acf1"]).all())
        self.assertTrue(np.isnan(f.temporal["season"]).all())
        self.assertTrue(np.isfinite(f.temporal["volatility"]).any())

    def test_autocorrelation_and_trend_are_detected(self):
        r = np.random.default_rng(0)
        ar = np.zeros((300, 1))
        for i in range(1, 300):
            ar[i] = 0.9 * ar[i - 1] + r.normal(0, 0.4)
        f = extract_feature_fp(ar, t=np.arange(300))
        self.assertGreater(f.temporal["acf1"][0], 0.7)
        ramp = extract_feature_fp(np.arange(200.0)[:, None] + r.normal(0, 3, (200, 1)), t=np.arange(200))
        self.assertGreater(ramp.temporal["trend"][0], 0.9)
        sine = extract_feature_fp(np.sin(np.arange(300) * 2 * np.pi / 12)[:, None] + r.normal(0, .2, (300, 1)), t=np.arange(300))
        self.assertGreater(sine.temporal["season"][0], 0.6)


class Discrimination(unittest.TestCase):
    def test_same_mean_and_std_different_distribution(self):
        r = np.random.default_rng(1)
        norm = fp(r.normal(0, 1, (300, 2)))
        uni = fp(r.uniform(-np.sqrt(3), np.sqrt(3), (300, 2)))
        d = compare(norm, uni)
        self.assertGreater(d.components["dist"], d.components["stat"])
        self.assertGreater(d.components["dist"], 0.25)
        same = compare(norm, fp(r.normal(0, 1, (300, 2))))
        self.assertLess(same.components["dist"], 0.15)

    def test_same_distribution_different_temporal_pattern(self):
        r = np.random.default_rng(2)
        iid = r.normal(0, 1, (300, 2))
        ar = np.zeros((300, 2))
        for i in range(1, 300):
            ar[i] = 0.9 * ar[i - 1] + r.normal(0, np.sqrt(1 - 0.81), 2)
        d = compare(fp(iid), fp(ar))
        self.assertGreater(d.components["time"], 0.4)
        self.assertLess(d.components["stat"], 0.3)

    def test_similar_x_different_y_is_visible_only_in_predictive_view(self):
        rng = np.random.default_rng(3)
        w = np.array([1.0, 1.0, 0, 0])
        X1, X2 = window("A", 400, 10), window("A", 400, 11)
        y1, y2 = (X1 @ w > 0).astype(int), (X2 @ w <= 0).astype(int)     # same p(x), flipped labelling
        self.assertLess(compare(fp(X1), fp(X2)).combined, 0.15)           # X alone cannot tell them apart
        m = LogReg(4).fit(X1, y1)
        stored = predictive_fp(y1, [m.predict(x) for x in X1], [m.proba(x) for x in X1])
        same_obs = predictive_fp(y1[:100], [m.predict(x) for x in X1[:100]], [m.proba(x) for x in X1[:100]])
        diff_obs = predictive_fp(y2[:100], [m.predict(x) for x in X2[:100]], [m.proba(x) for x in X2[:100]])
        a = compare(fp(X1), fp(X2), pred=(stored, same_obs)).components["pred"]
        b = compare(fp(X1), fp(X2), pred=(stored, diff_obs)).components["pred"]
        self.assertLess(a, 0.3)
        self.assertGreater(b, 0.8)

    def test_overlapping_concepts_rank_correctly(self):
        a = window("A", 200, 1)
        near = a + np.array([0.6, 0, 0, 0])
        far = window("B", 200, 2)
        d_near, d_far = compare(fp(a), fp(near + 0.0)).combined, compare(fp(a), fp(far)).combined
        self.assertLess(d_near, d_far)


class LabelIndependence(unittest.TestCase):
    def test_feature_fingerprint_api_takes_no_labels(self):
        self.assertNotIn("y", inspect.signature(extract_feature_fp).parameters)
        self.assertNotIn("y", inspect.signature(compare).parameters)

    def test_feature_fingerprint_is_the_same_whatever_the_labels_are(self):
        X = window("A", 100, 1)
        before = fp(X).to_dict()
        _ = predictive_fp(np.zeros(100, int), np.ones(100, int))      # building F_XY must not touch F_X
        self.assertEqual(before, fp(X).to_dict())

    def test_missing_labels_are_not_zero_error(self):
        self.assertIsNone(predictive_fp([None, None, np.nan], [0, 1, 1]))
        pf = predictive_fp([1, None, 0, None], [1, 1, 1, 0])
        self.assertEqual(pf.n_labeled, 2)
        self.assertAlmostEqual(pf.error_rate, 0.5)

    def test_delayed_labels_update_predictive_fingerprint(self):
        acc = PredictiveAccumulator()
        for i in range(30):
            acc.record(i, y_pred=i % 2, proba=0.9)
        self.assertEqual(acc.n_labeled, 0)
        self.assertEqual(acc.n_unlabeled, 30)
        self.assertIsNone(acc.fingerprint())
        for i in range(20):
            acc.reveal(i, i % 2)                                      # labels arrive late, 20 of 30
        pf = acc.fingerprint()
        self.assertEqual(pf.n_labeled, 20)
        self.assertEqual(pf.error_rate, 0.0)
        for i in range(20, 30):
            acc.reveal(i, 1 - i % 2)
        self.assertAlmostEqual(acc.fingerprint().error_rate, 10 / 30)
        self.assertFalse(acc.reveal("unknown", 1))

    def test_predictive_fingerprint_components(self):
        r = np.random.default_rng(0)
        y = r.integers(0, 2, 400)
        p = np.clip(y * 0.8 + 0.1 + r.normal(0, 0.1, 400), 0.01, 0.99)
        pf = predictive_fp(y, (p > 0.5).astype(int), p)
        self.assertLess(pf.error_rate, 0.05)
        self.assertIsNotNone(pf.calibration_error)
        self.assertEqual(pf.residual_quantiles.shape, (5,))
        self.assertTrue(np.isfinite(pf.class_recall).all())
        self.assertEqual(pf.confusion.shape, (2, 2))


class Metrics(unittest.TestCase):
    def test_psi_and_wasserstein_basic(self):
        q = np.linspace(-2, 2, 21)
        self.assertAlmostEqual(wasserstein1(q, q), 0.0)
        self.assertAlmostEqual(psi(q, q), 0.0, places=6)
        self.assertGreater(psi(q, q + 3), 1.0)
        c = np.full(21, 2.0)
        self.assertTrue(np.isfinite(psi(c, c)) and np.isfinite(psi(c, c + 1)))        # constant sketches

    def test_metric_is_configurable_and_bounded(self):
        a, b = fp(window("A", 100, 1)), fp(window("B", 100, 2))
        for metric in ("wasserstein", "psi", "both"):
            d = compare(a, b, SimilarityConfig(dist_metric=metric))
            self.assertTrue(0.0 <= d.combined <= 1.0 and 0.0 <= d.similarity <= 1.0)
        with self.assertRaises(ValueError):
            SimilarityConfig(dist_metric="nope")

    def test_nothing_available_is_maximal_distance_not_perfect_similarity(self):
        a = extract_feature_fp(np.full((20, 2), np.nan))
        d = compare(a, a)
        self.assertEqual(d.combined, 1.0)
        self.assertEqual(d.completeness, 0.0)

    def test_missing_component_renormalises_weights(self):
        a, b = fp(window("A", 100, 1)), fp(window("B", 100, 2))
        d = compare(a, b)
        manual = (0.4 * d.components["stat"] + 0.3 * d.components["dist"] + 0.1 * d.components["time"]) / 0.8
        self.assertAlmostEqual(d.combined, manual, places=6)

    def test_exp_similarity_option(self):
        a, b = fp(window("A", 100, 1)), fp(window("A", 100, 2))
        lin = compare(a, b, SimilarityConfig(similarity="linear"))
        ex = compare(a, b, SimilarityConfig(similarity="exp"))
        self.assertAlmostEqual(lin.similarity, 1 - lin.combined)
        self.assertAlmostEqual(ex.similarity, np.exp(-3 * ex.combined))


class Migration(unittest.TestCase):
    def test_v1_migration_keeps_only_label_free_entries_and_reports_low_completeness(self):
        vec = np.concatenate([[1, 2, 3, 4], [1, 1, 1, 1], [9, 9, 9, 9], [0.5, 0.5, 0.5, 0.5], [0.5]])
        f = migrate_v1(vec, n=200)
        np.testing.assert_allclose(f.statistics["mean"], [1, 2, 3, 4])
        self.assertIn("class_gap", f.meta["dropped_label_dims"])
        self.assertLess(f.completeness(), 0.5)
        rec = ConceptRecord.from_dict({"id": 3, "fingerprint": vec.tolist(), "n": 200, "uses": 2, "created_t": 10})
        self.assertEqual(rec.concept_id, "concept_003")
        self.assertEqual(rec.meta["migrated_from"], "1.0")
        with self.assertRaises(ValueError):
            migrate_v1(vec[:5], 10)


if __name__ == "__main__":
    unittest.main()
