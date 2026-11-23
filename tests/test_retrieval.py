"""M4: Level-1 index and representations, adaptive top-k, fallback scan, aggregation strategies, weight modes."""
import unittest

import numpy as np

from datastream.matcher import ConceptRepository, Decision, MatchConfig
from datastream.mvfp import extract_feature_fp
from datastream.research.gen import Spec, sample
from datastream.research.presets import improved_similarity
from datastream.similarity import SimilarityConfig, _agg_groups, compare


def fp(spec, n, rng, **kw):
    return extract_feature_fp(sample(spec, n, rng), t=np.arange(n), **kw)


Z4, O4 = (0.0,) * 4, (1.0,) * 4


class Index(unittest.TestCase):
    def test_index_follows_mutations(self):
        rng = np.random.default_rng(0)
        repo = ConceptRepository(MatchConfig(max_concepts=3))
        a = repo.add(fp(Spec("a", Z4, O4), 200, rng), "a")
        q = fp(Spec("q", (5.0,) * 4, O4), 80, rng)
        self.assertEqual(repo.level1(q), [a.concept_id])
        b = repo.add(fp(Spec("b", (5.0,) * 4, O4), 200, rng), "b")
        self.assertEqual(repo.level1(q)[0], b.concept_id)                       # new concept visible immediately
        repo.update(b.concept_id, fp(Spec("b2", (5.0,) * 4, O4), 80, rng), t=1)
        for i in range(4):
            repo.add(fp(Spec("x", (9.0 + i,) * 4, O4), 100, rng), f"x{i}", protect=b.concept_id)
        self.assertLessEqual(len(repo), 3)
        self.assertTrue(all(c in repo for c in repo.level1(q)))                 # evicted concepts never returned

    def test_all_representations_handle_missing_constant_categorical_and_all_nan(self):
        rng = np.random.default_rng(1)
        for rep in ("basic", "shape", "shape_corr", "full"):
            repo = ConceptRepository(MatchConfig(l1_repr=rep))
            X = rng.normal(0, 1, (120, 5))
            X[:, 2] = 3.0
            X[:, 4] = rng.integers(0, 3, 120)
            X[rng.random(X.shape[0]) < 0.1, 0] = np.nan
            repo.add(extract_feature_fp(X, t=np.arange(120), categorical=[4]), "c")
            q = extract_feature_fp(X[:60], t=np.arange(60), categorical=[4])
            self.assertEqual(len(repo.level1(q)), 1, rep)
            nan_q = extract_feature_fp(np.full((60, 5), np.nan), categorical=[4])
            self.assertEqual(repo.match(nan_q).decision, Decision.INSUFFICIENT_EVIDENCE, rep)

    def test_unknown_representation_rejected(self):
        rng = np.random.default_rng(2)
        f = fp(Spec("a", Z4, O4), 60, rng)
        with self.assertRaises(ValueError):
            f.l1_unit("nope")


class TopK(unittest.TestCase):
    def build(self, **kw):
        rng = np.random.default_rng(3)
        repo = ConceptRepository(MatchConfig(**kw))
        for i in range(12):
            repo.add(fp(Spec("c", tuple([float(i * 6)] * 4), O4), 200, rng), f"m{i}")
        return repo, rng

    def test_adaptive_k_is_bounded_and_small_for_distinctive_queries(self):
        repo, rng = self.build(top_k_mode="adaptive", k_min=2, k_max=6)
        sl = repo.level1(fp(Spec("q", (12.0,) * 4, O4), 80, rng))
        self.assertEqual(len(sl), 2)                                           # well separated: k_min suffices
        self.assertEqual(repo.records[sl[0]].model_id, "m2")

    def test_adaptive_k_grows_when_ambiguous_and_stops_at_k_max(self):
        rng = np.random.default_rng(4)
        repo = ConceptRepository(MatchConfig(top_k_mode="adaptive", k_min=2, k_max=5))
        for i in range(9):
            repo.add(fp(Spec("c", tuple([0.02 * i] * 4), O4), 200, rng), f"m{i}")   # nine near-duplicates
        sl = repo.level1(fp(Spec("q", Z4, O4), 80, rng))
        self.assertEqual(len(sl), 5)

    def test_fixed_mode_returns_top_k(self):
        repo, rng = self.build(top_k=4)
        self.assertEqual(len(repo.level1(fp(Spec("q", Z4, O4), 80, rng))), 4)


class Fallback(unittest.TestCase):
    def test_full_scan_rescues_a_concept_that_level1_hid(self):
        """Level 1 is blind to temporal behaviour; with a time-weighted Level 2 the decoy is rejected as NEW unless
        the fallback scan finds the right concept. Search seeds for a case where Level 1 really picked the decoy."""
        sim = improved_similarity(weights={"stat": 0.3, "dist": 0.2, "time": 0.5, "pred": 0.2})
        core, ar = Spec("core", Z4, O4), Spec("ar", Z4, O4, rho=0.9)
        found = None
        for seed in range(60):
            res = {}
            for expand in (False, True):
                repo = ConceptRepository(MatchConfig(top_k=1, expand_on_new=expand), sim)
                repo.add(fp(core, 300, np.random.default_rng(seed)), "core")
                repo.add(fp(ar, 300, np.random.default_rng(seed + 1000)), "ar")
                res[expand] = repo.match(fp(ar, 100, np.random.default_rng(seed + 2000)))
            if res[False].best_id == "concept_001" and res[False].decision == Decision.NEW_CONCEPT:
                found = res
                break
        self.assertIsNotNone(found, "no seed produced a Level-1 miss; the scenario is not exercising the fallback")
        self.assertEqual(found[True].best_id, "concept_002")                    # the fallback scan found the right concept
        self.assertNotEqual(found[True].decision, Decision.NEW_CONCEPT)
        self.assertEqual(len(found[True].ranking), 2)

    def test_fallback_is_skipped_for_large_repositories(self):
        rng = np.random.default_rng(9)
        repo = ConceptRepository(MatchConfig(top_k=1, expand_on_new=True, max_full_scan=2))
        for i in range(4):
            repo.add(fp(Spec("c", tuple([float(i * 5)] * 4), O4), 150, rng), f"m{i}")
        r = repo.match(fp(Spec("q", (50.0,) * 4, O4), 80, rng))
        self.assertEqual(r.decision, Decision.NEW_CONCEPT)
        self.assertEqual(len(r.ranking), 1)


class Aggregation(unittest.TestCase):
    v = np.array([0.1, 0.1, 0.2, 0.9, np.nan])

    def test_strategies(self):
        cfg = lambda **k: SimilarityConfig(**k)
        self.assertAlmostEqual(_agg_groups(self.v, cfg(group_agg="mean")), 0.325)
        self.assertAlmostEqual(_agg_groups(self.v, cfg(group_agg="median")), 0.15)
        self.assertAlmostEqual(_agg_groups(self.v, cfg(group_agg="mean_worst")), 0.7 * 0.325 + 0.3 * 0.9)
        self.assertAlmostEqual(_agg_groups(np.arange(10) / 10, cfg(group_agg="trimmed", trim=0.2)), 0.45)
        self.assertAlmostEqual(_agg_groups(self.v, cfg(group_agg="quantile", group_quantile=1.0)), 0.9)
        self.assertAlmostEqual(_agg_groups(self.v, cfg(group_agg="weighted", group_weights={"a": 3.0}), names=["a", "b", "c", "d", "e"]),
                               (3 * 0.1 + 0.1 + 0.2 + 0.9) / 6)
        self.assertTrue(np.isnan(_agg_groups(np.array([np.nan]), cfg())))

    def test_unstable_irrelevant_feature_cannot_dominate_with_mean(self):
        rng = np.random.default_rng(10)
        def win(shift):
            X = rng.normal(0, 1, (80, 10))
            X[:, 0] = rng.standard_cauchy(80) * np.exp(rng.normal(0, 1))
            X[:, 1:] += shift
            return extract_feature_fp(X, t=np.arange(80))
        q = {}
        for name in ("mean_worst", "mean"):
            cfg = SimilarityConfig(group_agg=name)
            q[name] = np.quantile([compare(win(0), win(0), cfg).components["stat"] for _ in range(40)], 0.95)
        self.assertLess(q["mean"], q["mean_worst"])

    def test_feature_groups_are_respected(self):
        rng = np.random.default_rng(11)
        a = extract_feature_fp(rng.normal(0, 1, (100, 4)), t=np.arange(100))
        X = rng.normal(0, 1, (100, 4)); X[:, 3] += 3
        b = extract_feature_fp(X, t=np.arange(100))
        d = compare(a, b, SimilarityConfig(feature_groups=((0, 1, 2), (3,)), group_agg="mean"))
        self.assertEqual(set(d.groups), {"0+1+2", "3"})
        self.assertGreater(d.groups["3"], d.groups["0+1+2"])


class WeightModes(unittest.TestCase):
    def test_fixed_vs_availability_when_a_view_is_missing(self):
        rng = np.random.default_rng(12)
        a, b = (extract_feature_fp(rng.normal(0, 1, (8, 3)), t=np.arange(8)) for _ in range(2))      # too short for temporal
        av = compare(a, b, SimilarityConfig(weight_mode="availability"))
        fx = compare(a, b, SimilarityConfig(weight_mode="fixed", missing_prior=0.5))
        self.assertIsNone(av.components["time"])
        self.assertGreater(fx.combined, av.combined)                  # a missing view is not read as perfect similarity
        self.assertAlmostEqual(av.completeness, fx.completeness)

    def test_identical_when_all_views_present(self):
        rng = np.random.default_rng(13)
        a, b = (extract_feature_fp(rng.normal(0, 1, (100, 3)), t=np.arange(100)) for _ in range(2))
        self.assertAlmostEqual(compare(a, b, SimilarityConfig(weight_mode="fixed")).combined, compare(a, b).combined, places=9)

    def test_calibrated_mode_is_accepted_and_tagged(self):
        cfg = SimilarityConfig(weight_mode="calibrated", weights={"stat": .4, "dist": .2, "time": .4, "pred": .2}, weights_source="calibrated")
        self.assertEqual(cfg.weights_source, "calibrated")


if __name__ == "__main__":
    unittest.main()
