"""M2: effective sample size, empirical same-concept profiles, and a regression freeze of schema-2.0 defaults."""
import unittest
from dataclasses import replace

import numpy as np

from datastream.mvfp import extract_feature_fp
from datastream.noise import EssConfig, build_profile, effective_n, effective_n_corr, lrv_factor, nu_vector
from datastream.research.gen import Spec, base_spec, sample
from datastream.similarity import SimilarityConfig, compare
from datastream.stream import CONCEPTS


def w(name, n, seed):
    c = CONCEPTS[name]
    return extract_feature_fp(np.random.default_rng(seed).normal(c.mu, c.sigma, (n, 4)), t=np.arange(n))


class Regression(unittest.TestCase):
    def test_schema_2_0_defaults_are_frozen(self):
        for a, b, n, D, st, di in (("A", "A", 100, 0.081037, 0.066639, 0.127246), ("A", "B", 100, 0.537976, 0.615944, 0.613344),
                                   ("A", "C", 60, 0.600712, 0.670782, 0.707522)):
            d = compare(w(a, n, 1), w(b, n, 2))
            self.assertAlmostEqual(d.combined, D, places=5)
            self.assertAlmostEqual(d.components["stat"], st, places=5)
            self.assertAlmostEqual(d.components["dist"], di, places=5)
            self.assertAlmostEqual(d.components["time"], 0.0, places=6)


class EffectiveN(unittest.TestCase):
    def test_bounds_and_degenerate_inputs(self):
        cfg = EssConfig()
        for kind in ("mean", "quantile", "scale", "acf", "rate"):
            for rho in (np.nan, None, -0.99, -0.5, 0.0, 0.5, 0.99, 5.0):
                for n in (1, 3, 10, 100, 2000):
                    e = effective_n(n, rho, kind, cfg)
                    self.assertTrue(np.isfinite(e) and e > 0)
                    self.assertLessEqual(e, n / cfg.f_min + 1e-9)
                    self.assertGreaterEqual(e, min(cfg.n_min, n) - 1e-9)
        with self.assertRaises(ValueError):
            effective_n(0, 0.1)
        with self.assertRaises(ValueError):
            lrv_factor(0.5, "nope", 50)

    def test_direction_by_statistic(self):
        n, rho = 200, 0.8
        mean, quant, scale, acf = (effective_n(n, rho, k) for k in ("mean", "quantile", "scale", "acf"))
        self.assertLess(mean, n)
        self.assertLess(quant, n)
        self.assertGreater(quant, mean)                    # quantiles lose less information than the mean
        self.assertGreater(acf, n)                         # the lag-1 estimate gets *more* precise under dependence
        self.assertEqual(effective_n(n, rho, "rate"), n)
        self.assertGreater(effective_n(n, -0.6, "mean"), n)   # negative autocorrelation helps the mean
        self.assertEqual(effective_n(n, np.nan, "mean"), n)   # constant feature / unavailable rho: no discount

    def test_ar1_prediction_matches_simulation_for_the_mean(self):
        rng = np.random.default_rng(0)
        rho, n = 0.7, 120
        means = []
        for _ in range(500):
            z = np.empty(n)
            z[0] = rng.normal()
            e = rng.normal(size=n) * np.sqrt(1 - rho ** 2)
            for i in range(1, n):
                z[i] = rho * z[i - 1] + e[i]
            means.append(z.mean())
        observed_n_eff = 1.0 / np.var(means)
        predicted = n / lrv_factor(rho, "mean", n)
        self.assertAlmostEqual(observed_n_eff / predicted, 1.0, delta=0.25)

    def test_different_window_sizes_and_feature_specific_rho(self):
        nu = nu_vector(40, np.array([0.0, 0.9]), 300, np.array([0.0, 0.9]), "mean")
        self.assertGreater(nu[1], nu[0])                   # the autocorrelated feature is noisier
        self.assertGreater(nu[0], np.sqrt(1 / 300))        # the smaller window dominates
        self.assertTrue(np.isfinite(effective_n_corr(50, np.nan, 0.5)))

    def test_per_stat_mode_runs_and_stays_bounded(self):
        cfg = SimilarityConfig(ess_mode="per_stat")
        for name in ("A", "B"):
            d = compare(w("A", 60, 3), w(name, 60, 4), cfg)
            self.assertTrue(0.0 <= d.combined <= 1.0)
        self.assertLess(compare(w("A", 100, 1), w("A", 100, 2), cfg).combined, 0.25)


class Profile(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.mk = lambda X, t: extract_feature_fp(X, t=t)
        self.cfg0 = SimilarityConfig(noise_mode="none")
        self.cmp = lambda a, b: compare(a, b, self.cfg0)

    def test_disjoint_profile_and_block_bootstrap_fallback(self):
        ref = sample(base_spec(4), 800, self.rng)
        p = build_profile(ref, 60, 60, self.mk, self.cmp, B=12, rng=self.rng)
        self.assertEqual(p.method, "disjoint")
        self.assertEqual(len(p.same["stat"]), 12)
        short = build_profile(ref[:90], 60, 60, self.mk, self.cmp, B=8, rng=self.rng)
        self.assertEqual(short.method, "block_bootstrap")

    def test_profile_separates_noise_from_real_change(self):
        ref = sample(base_spec(4), 900, self.rng)
        prof = build_profile(ref, 80, 80, self.mk, self.cmp, B=30, rng=self.rng)
        cfg = SimilarityConfig(noise_mode="profile")
        same = [compare(self.mk(sample(base_spec(4), 80, self.rng), np.arange(80)), self.mk(sample(base_spec(4), 80, self.rng), np.arange(80)), cfg, profile=prof) for _ in range(30)]
        shifted = Spec("s", (1.0,) * 4, (1.0,) * 4)
        diff = [compare(self.mk(sample(base_spec(4), 80, self.rng), np.arange(80)), self.mk(sample(shifted, 80, self.rng), np.arange(80)), cfg, profile=prof) for _ in range(30)]
        self.assertLess(np.mean([d.combined for d in same]), 0.1)
        self.assertGreater(np.mean([d.combined for d in diff]), 0.25)
        self.assertEqual(diff[0].noise["stat"], "exceeds_noise")
        self.assertEqual(diff[0].noise_source, "profile")
        self.assertIn(same[0].noise["stat"], ("within_noise", "exceeds_noise"))

    def test_pvalue_is_monotone_and_power_in_unit_interval(self):
        ref = sample(base_spec(4), 600, self.rng)
        prof = build_profile(ref, 60, 60, self.mk, self.cmp, B=20, rng=self.rng)
        self.assertGreaterEqual(prof.pvalue("stat", 0.0), prof.pvalue("stat", 0.9))
        self.assertTrue(0.0 <= prof.power("stat") <= 1.0)
        self.assertIsNone(prof.quantile("pred", 0.9))

    def test_analytic_mode_never_claims_statistical_significance(self):
        a, b = w("A", 100, 1), w("B", 100, 2)
        for d in (compare(a, b), compare(a, w("A", 100, 3))):
            self.assertNotIn("exceeds_noise", d.noise.values())
        self.assertIn(compare(a, b).noise["stat"], ("above_floor", "unresolvable"))

    def test_missing_profile_falls_back_to_analytic_and_says_so(self):
        d = compare(w("A", 60, 1), w("A", 60, 2), SimilarityConfig(noise_mode="profile"))
        self.assertIn(d.noise_source, ("none", "analytic"))

    def test_unresolvable_window_is_flagged_not_silently_matched(self):
        rng = np.random.default_rng(1)
        sp = base_spec(2, rho=0.9)
        ref = sample(sp, 600, rng)
        prof = build_profile(ref, 20, 20, self.mk, self.cmp, B=24, rng=rng)
        cfg = SimilarityConfig(noise_mode="profile", min_power=0.8)
        d = compare(self.mk(sample(sp, 20, rng), np.arange(20)), self.mk(sample(sp, 20, rng), np.arange(20)), cfg, profile=prof)
        self.assertIn("unresolvable", d.noise.values())


class ConfigValidation(unittest.TestCase):
    def test_bad_options_rejected(self):
        for kw in ({"ess_mode": "x"}, {"noise_mode": "x"}, {"psi_binning": "x"}, {"time_norm": "x"}, {"corr_transform": "x"},
                   {"group_agg": "x"}, {"weight_mode": "x"}, {"dist_metric": "x"}):
            with self.assertRaises(ValueError):
                SimilarityConfig(**kw)


if __name__ == "__main__":
    unittest.main()
