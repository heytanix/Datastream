"""M5: variability tracking, conformal thresholds, similarity/evidence/confidence split, calibrator, explanations."""
import unittest

import numpy as np

from dejastream.matcher import (ConceptRepository, Decision, MatchCalibrator, MatchConfig, VariabilityTracker, conformal_quantile,
                                match_features)
from dejastream.mvfp import extract_feature_fp
from dejastream.research.gen import Spec, sample

Z4, O4 = (0.0,) * 4, (1.0,) * 4


def fp(spec, n, rng):
    return extract_feature_fp(sample(spec, n, rng), t=np.arange(n))


class Conformal(unittest.TestCase):
    def test_quantile_and_conservative_fallback(self):
        v = np.arange(1, 100) / 100.0                         # 99 values
        self.assertAlmostEqual(conformal_quantile(v, 0.9), 0.90)           # ceil(100*0.9)=90th smallest
        self.assertAlmostEqual(conformal_quantile(np.array([0.1, 0.2, 0.3]), 0.99), 0.3 * 1.25)   # cannot certify: widen
        with self.assertRaises(ValueError):
            conformal_quantile(np.array([]), 0.9)

    def test_empirical_coverage_on_exchangeable_data(self):
        rng = np.random.default_rng(0)
        hits = []
        for _ in range(300):
            cal, new = rng.normal(size=60), rng.normal()
            hits.append(new <= conformal_quantile(cal, 0.9))
        self.assertAlmostEqual(np.mean(hits), 0.9, delta=0.05)


class Tracker(unittest.TestCase):
    def test_summary_and_size_filter(self):
        t = VariabilityTracker()
        for n, d in ((40, 0.10), (40, 0.20), (100, 0.05), (100, 0.07)):
            t.add(n, d)
        self.assertEqual(t.summary()["n_obs"], 4)
        self.assertEqual(sorted(t.values(40, 1.5)), [0.10, 0.20])
        self.assertEqual(sorted(t.values(100, 1.5)), [0.05, 0.07])
        self.assertEqual(VariabilityTracker().summary(), {"n_obs": 0})


class Thresholds(unittest.TestCase):
    def repo(self, mode, **kw):
        rng = np.random.default_rng(1)
        r = ConceptRepository(MatchConfig(threshold_mode=mode, **kw))
        rec = r.add(fp(Spec("a", Z4, O4), 300, rng), "a")
        return r, rec, rng

    def test_legacy_mode_matches_the_old_formula(self):
        r, rec, rng = self.repo("legacy")
        self.assertEqual(r.thresholds(rec.concept_id, 60)["thr_cand"], 0.32)
        rec.intra = 0.2
        th = r.thresholds(rec.concept_id, 60)
        self.assertAlmostEqual(th["thr_cand"], 0.5)
        self.assertAlmostEqual(th["thr_high"], max(0.18, 2.5 * 0.2 * 0.6))

    def test_empirical_falls_back_when_there_is_no_data(self):
        r, rec, _ = self.repo("empirical")
        self.assertEqual(r.thresholds(rec.concept_id, 60)["source"], "legacy_fallback")

    def test_empirical_uses_concept_then_pool_and_orders_thresholds(self):
        r, rec, rng = self.repo("empirical", min_obs=5)
        other = r.add(fp(Spec("b", (4.0,) * 4, O4), 300, rng), "b")
        for _ in range(30):
            r.observe_same(rec.concept_id, fp(Spec("a", Z4, O4), 60, rng))
        th = r.thresholds(rec.concept_id, 60)
        self.assertEqual(th["source"], "concept")
        self.assertLessEqual(th["thr_high"], th["thr_cand"])
        self.assertGreaterEqual(th["thr_cand"], r.match_cfg.t_min)
        young = r.thresholds(other.concept_id, 60)                 # no own observations: borrow the pool
        self.assertEqual(young["source"], "pool")
        self.assertGreater(young["n_obs"], 5)

    def test_observe_same_records_but_does_not_merge(self):
        r, rec, rng = self.repo("empirical")
        before = rec.fp.to_dict()
        n_before = rec.sample_count
        d = r.observe_same(rec.concept_id, fp(Spec("a", Z4, O4), 60, rng))
        self.assertEqual(rec.fp.to_dict(), before)
        self.assertEqual(rec.sample_count, n_before)
        self.assertEqual(len(rec.var.obs), 1)
        self.assertEqual(len(r.pool), 1)
        self.assertGreaterEqual(d, 0.0)
        r.update(rec.concept_id, fp(Spec("a", Z4, O4), 60, rng), t=1)
        self.assertEqual(len(rec.var.obs), 2)                       # update() also feeds the tracker

    def test_config_validation(self):
        for kw in ({"threshold_mode": "x"}, {"alpha_cand": 0.0}, {"alpha_high": 1.0}, {"l1_repr": "x"}, {"top_k_mode": "x"}):
            with self.assertRaises(ValueError):
                MatchConfig(**kw)

    def test_variability_survives_serialisation(self):
        r, rec, rng = self.repo("empirical")
        for _ in range(6):
            r.observe_same(rec.concept_id, fp(Spec("a", Z4, O4), 60, rng))
        r2 = ConceptRepository(MatchConfig(threshold_mode="empirical"))
        r2.load_dict(r.to_dict())
        self.assertEqual(len(r2.records[rec.concept_id].var.obs), 6)


class Result(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.repo = ConceptRepository()
        self.repo.add(fp(Spec("a", Z4, O4), 300, self.rng), "a")

    def test_three_quantities_are_separate_and_bounded(self):
        res = self.repo.match(fp(Spec("a", Z4, O4), 80, self.rng))
        self.assertIsNotNone(res.similarity)
        for v in (res.similarity, res.evidence_quality, res.confidence):
            self.assertTrue(0.0 <= v <= 1.0)
        self.assertIsNone(res.margin)                               # a single stored concept has no runner-up
        self.assertTrue(0.0 <= res.margin_norm <= 1.0)
        self.assertEqual(set(res.confidence_terms), {"closeness", "separation", "evidence", "variability", "predictive"})
        self.assertIsNone(res.match_probability)

    def test_duplicates_give_a_small_margin(self):
        self.repo.add(fp(Spec("a", Z4, O4), 300, self.rng), "a2")
        res = self.repo.match(fp(Spec("a", Z4, O4), 80, self.rng))
        self.assertLess(res.margin, 0.06)
        self.assertEqual(res.decision, Decision.CANDIDATE_MATCH)

    def test_explanation_structure(self):
        self.repo.add(fp(Spec("a", Z4, O4), 300, self.rng), "a2")
        ex = self.repo.match(fp(Spec("a", Z4, O4), 40, self.rng)).explanation()
        for key in ("decision", "concept_id", "distance", "similarity", "margin", "evidence_quality", "confidence_score", "match_probability",
                    "completeness", "available_views", "missing_views", "alternatives_rejected", "reason", "additional_evidence_needed"):
            self.assertIn(key, ex)
        self.assertIn("pred", ex["missing_views"])
        self.assertTrue(any("labelled" in s for s in ex["additional_evidence_needed"]))
        self.assertTrue(any("ambiguous" in " ".join(a["reasons"]) for a in ex["alternatives_rejected"]))

    def test_new_concept_and_insufficient_explanations(self):
        far = self.repo.match(fp(Spec("z", (9.0,) * 4, O4), 80, self.rng)).explanation()
        self.assertEqual(far["decision"], "NEW_CONCEPT")
        few = self.repo.match(fp(Spec("a", Z4, O4), 12, self.rng)).explanation()
        self.assertEqual(few["decision"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(few["concept_id"])

    def test_unscored_concepts_are_reported(self):
        rng = np.random.default_rng(3)
        repo = ConceptRepository(MatchConfig(top_k=1))
        for i in range(4):
            repo.add(fp(Spec("c", (i * 3.0,) * 4, O4), 200, rng), f"m{i}")
        ex = repo.match(fp(Spec("q", (3.0,) * 4, O4), 80, rng)).explanation()
        self.assertTrue(any("not scored" in " ".join(a["reasons"]) for a in ex["alternatives_rejected"]))


class Calibrator(unittest.TestCase):
    def data(self, n, rng):
        X = rng.normal(size=(n, 6))
        p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1])))
        return X, (rng.random(n) < p).astype(float)

    def test_fit_improves_brier_on_heldout_and_is_monotone(self):
        rng = np.random.default_rng(4)
        X, y = self.data(600, rng)
        Xt, yt = self.data(600, rng)
        m = MatchCalibrator().fit(X, y)
        p = m.predict(Xt)
        self.assertLess(np.mean((p - yt) ** 2), np.mean((y.mean() - yt) ** 2))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        base = np.zeros((2, 6)); base[1, 0] = 2.0
        self.assertLess(m.predict(base)[0], m.predict(base)[1])

    def test_refuses_bad_calibration_data(self):
        rng = np.random.default_rng(5)
        with self.assertRaises(ValueError):
            MatchCalibrator().fit(rng.normal(size=(10, 6)), np.array([0, 1] * 5))
        with self.assertRaises(ValueError):
            MatchCalibrator().fit(rng.normal(size=(50, 6)), np.ones(50))
        with self.assertRaises(RuntimeError):
            MatchCalibrator().predict(np.zeros(6))

    def test_probability_only_with_a_fitted_calibrator(self):
        rng = np.random.default_rng(6)
        repo = ConceptRepository()
        repo.add(fp(Spec("a", Z4, O4), 300, rng), "a")
        q = fp(Spec("a", Z4, O4), 80, rng)
        self.assertIsNone(repo.match(q).match_probability)
        X, y = Calibrator.data(self, 100, rng)
        repo.calibrator = MatchCalibrator().fit(X, y)
        res = repo.match(q)
        self.assertTrue(0.0 <= res.match_probability <= 1.0)
        self.assertEqual(match_features(res).shape, (6,))


if __name__ == "__main__":
    unittest.main()
