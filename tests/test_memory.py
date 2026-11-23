"""M7: mergeable summaries, versioned records, backward compatibility, eviction policies."""
import json
import math
import unittest

import numpy as np

from dejastream.matcher import ConceptRecord, ConceptRepository, MatchConfig
from dejastream.memory import ConceptSummaries, Moments, QuantileSketch
from dejastream.mvfp import extract_feature_fp
from dejastream.research.gen import Spec, sample

Z4, O4 = (0.0,) * 4, (1.0,) * 4


def fpx(X):
    return extract_feature_fp(X, t=np.arange(len(X)))


class MomentsTests(unittest.TestCase):
    def test_merge_equals_pooled_statistics_exactly_even_with_missing_values(self):
        rng = np.random.default_rng(0)
        ws = [rng.normal(m, s, (n, 3)) for m, s, n in ((0, 1, 120), (5, 1, 40), (2, 3, 80))]
        ws[1][::7, 0] = np.nan
        mom = Moments.from_window(ws[0])
        for w in ws[1:]:
            mom.merge(Moments.from_window(w))
        allx = np.vstack(ws)
        for j in range(3):
            v = allx[np.isfinite(allx[:, j]), j]
            self.assertAlmostEqual(mom.mean[j], v.mean(), places=10)
            self.assertAlmostEqual(math.sqrt(mom.m2[j] / mom.n[j]), v.std(), places=10)
            self.assertAlmostEqual(math.sqrt(mom.n[j]) * mom.m3[j] / mom.m2[j] ** 1.5, np.mean(((v - v.mean()) / v.std()) ** 3), places=9)
        self.assertEqual(mom.rows, len(allx))

    def test_merge_order_does_not_matter(self):
        rng = np.random.default_rng(1)
        ws = [rng.normal(rng.normal(), 1, (60, 2)) for _ in range(5)]
        a, b = Moments.from_window(ws[0]), Moments.from_window(ws[-1])
        for w in ws[1:]:
            a.merge(Moments.from_window(w))
        for w in ws[-2::-1]:
            b.merge(Moments.from_window(w))
        np.testing.assert_allclose(a.mean, b.mean, atol=1e-10)
        np.testing.assert_allclose(a.m2, b.m2, rtol=1e-10)
        np.testing.assert_allclose(a.m3, b.m3, rtol=1e-8, atol=1e-8)

    def test_empty_and_constant_windows(self):
        m = Moments.from_window(np.full((10, 2), np.nan)).merge(Moments.from_window(np.full((10, 2), 4.0)))
        np.testing.assert_allclose(m.mean, 4.0)
        np.testing.assert_allclose(m.m2, 0.0)


class SketchTests(unittest.TestCase):
    def test_accuracy_after_many_merges_and_bounded_size(self):
        rng = np.random.default_rng(2)
        parts = [rng.normal(rng.normal(0, 3), rng.uniform(0.5, 2), rng.integers(50, 200)) for _ in range(60)]
        sk = QuantileSketch.from_values(parts[0])
        for p in parts[1:]:
            sk.merge(QuantileSketch.from_values(p))
        allp = np.concatenate(parts)
        ps = np.array([.01, .05, .1, .25, .5, .75, .9, .95, .99])
        err = np.abs(sk.quantiles(ps) - np.quantile(allp, ps)).max() / (np.quantile(allp, .99) - np.quantile(allp, .01))
        self.assertLess(err, 0.01)
        self.assertLess(len(sk.means), 80)                                      # bounded, not one centroid per observation
        self.assertEqual(sk.min, allp.min())
        self.assertEqual(sk.max, allp.max())
        self.assertEqual(sk.n, len(allp))
        naive = np.mean([np.quantile(p, ps) for p in parts], axis=0)
        self.assertLess(err, np.abs(naive - np.quantile(allp, ps)).max() / (np.quantile(allp, .99) - np.quantile(allp, .01)))

    def test_degenerate_inputs_and_serialisation(self):
        self.assertTrue(np.isnan(QuantileSketch.from_values(np.array([np.nan, np.inf])).quantiles(np.array([.5]))).all())
        c = QuantileSketch.from_values(np.full(100, 3.0))
        self.assertTrue(np.allclose(c.quantiles(np.array([0.0, .5, 1.0])), 3.0))
        s = QuantileSketch.from_values(np.random.default_rng(3).normal(size=300))
        s2 = QuantileSketch.from_dict(json.loads(json.dumps(s.to_dict())))
        np.testing.assert_allclose(s.quantiles(np.array([.1, .5, .9])), s2.quantiles(np.array([.1, .5, .9])))
        self.assertEqual(QuantileSketch().merge(QuantileSketch()).n, 0)

    def test_from_quantiles_is_a_usable_approximation(self):
        x = np.random.default_rng(4).normal(0, 1, 400)
        s = QuantileSketch.from_quantiles(np.quantile(x, np.linspace(0, 1, 21)), 400)
        self.assertAlmostEqual(float(s.quantiles(np.array([.5]))[0]), float(np.median(x)), delta=0.1)


class SummariesTests(unittest.TestCase):
    def test_the_audit_example_is_now_correct(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 1, (300, 1)), rng.normal(5, 1, (100, 1))
        sa, sb = (ConceptSummaries.from_window(x, fpx(x)) for x in (a, b))
        sa.merge_window(sb)
        fp = sa.compose()
        both = np.vstack([a, b])[:, 0]
        self.assertAlmostEqual(fp.statistics["std"][0], both.std(), places=9)
        for key, p in (("q10", .1), ("median", .5), ("q90", .9)):
            self.assertAlmostEqual(fp.statistics[key][0], np.quantile(both, p), delta=0.1)

    def test_categorical_and_missing_are_exact(self):
        rng = np.random.default_rng(6)
        mk = lambda p, n: np.column_stack([rng.normal(0, 1, n), np.where(rng.random(n) < 0.1, np.nan, rng.choice(3, n, p=p))])
        a, b = mk([.7, .2, .1], 200), mk([.1, .2, .7], 100)
        sa, sb = (ConceptSummaries.from_window(x, extract_feature_fp(x, t=np.arange(len(x)), categorical=[1])) for x in (a, b))
        sa.merge_window(sb)
        fp = sa.compose()
        both = np.vstack([a, b])[:, 1]
        ok = np.isfinite(both)
        for c in (0, 1, 2):
            self.assertAlmostEqual(fp.categorical["freq"][0].get(c, 0.0), np.mean(both[ok] == c), places=10)
        self.assertAlmostEqual(float(fp.categorical["missing"][0]), 1 - ok.mean(), places=10)

    def test_temporal_statistics_are_not_averaged(self):
        rng = np.random.default_rng(7)
        iid = [sample(Spec("a", Z4, O4), 100, rng) for _ in range(5)]
        ar = [sample(Spec("b", Z4, O4, rho=0.9), 100, rng) for _ in range(2)]
        windows = iid + ar
        s = ConceptSummaries.from_window(windows[0], fpx(windows[0]))
        for w in windows[1:]:
            s.merge_window(ConceptSummaries.from_window(w, fpx(w)))
        composed = float(np.nanmean(s.compose().temporal["acf1"]))
        averaged = float(np.mean([np.nanmean(fpx(w).temporal["acf1"]) for w in windows]))
        self.assertGreater(averaged, 0.2)                          # what averaging would have produced
        self.assertLess(abs(composed), 0.15)                       # the median over windows ignores the two outliers

    def test_bounded_memory_after_many_windows(self):
        rng = np.random.default_rng(8)
        w0 = sample(Spec("a", Z4, O4), 100, rng)
        s = ConceptSummaries.from_window(w0, fpx(w0))
        sizes = []
        for i in range(120):
            w = sample(Spec("a", Z4, O4), 100, rng)
            s.merge_window(ConceptSummaries.from_window(w, fpx(w)))
            sizes.append(s.nbytes())
        self.assertLessEqual(len(s.history), 20)
        self.assertLess(max(sizes[60:]), 1.1 * min(sizes[60:]) + 1500)           # plateaus
        self.assertEqual(s.n_windows, 121)

    def test_serialisation_roundtrip_preserves_the_composed_fingerprint(self):
        rng = np.random.default_rng(9)
        w = sample(Spec("a", Z4, O4), 120, rng)
        s = ConceptSummaries.from_window(w, fpx(w))
        s2 = ConceptSummaries.from_dict(json.loads(json.dumps(s.to_dict())))
        a, b = s.compose(), s2.compose()
        np.testing.assert_allclose(a.statistics["std"], b.statistics["std"])
        np.testing.assert_allclose(a.distributions["quantiles"], b.distributions["quantiles"])

    def test_incompatible_layouts_are_rejected(self):
        rng = np.random.default_rng(10)
        a, b = rng.normal(size=(60, 3)), rng.normal(size=(60, 4))
        sa, sb = ConceptSummaries.from_window(a, fpx(a)), ConceptSummaries.from_window(b, fpx(b))
        with self.assertRaises(ValueError):
            sa.merge_window(sb)


class MergeableRepository(unittest.TestCase):
    def repo(self, **kw):
        rng = np.random.default_rng(11)
        r = ConceptRepository(MatchConfig(merge_mode="mergeable", **kw))
        X = sample(Spec("a", Z4, O4), 200, rng)
        rec = r.add(fpx(X), "m1", t=3, X=X)
        return r, rec, rng

    def test_update_recomputes_instead_of_averaging(self):
        r, rec, rng = self.repo()
        Xb = sample(Spec("b", (5.0,) * 4, O4), 100, rng)
        r.update(rec.concept_id, fpx(Xb), t=5, X=Xb)
        self.assertGreater(rec.fp.statistics["std"][0], 1.8)                      # pooled std of a bimodal mixture, not ~1.0
        self.assertEqual(rec.update_count, 1)
        self.assertTrue(rec.meta["summaries_exact"])

    def test_versioning_fields(self):
        r, rec, rng = self.repo()
        self.assertEqual(rec.model_versions[0]["model_id"], "m1")
        r.replace_model(rec.concept_id, "m2")
        self.assertEqual([v["model_id"] for v in rec.model_versions], ["m1", "m2"])
        d = r.to_dict()
        self.assertEqual(d["schema_version"], "2.1")
        self.assertEqual(d["concepts"][0]["schema_version"], "2.1")
        r2 = ConceptRepository(MatchConfig(merge_mode="mergeable"))
        r2.load_dict(json.loads(json.dumps(d)))
        rec2 = r2.records[rec.concept_id]
        self.assertEqual(rec2.model_versions, rec.model_versions)
        np.testing.assert_allclose(rec2.summaries.compose().statistics["std"], rec.summaries.compose().statistics["std"])

    def test_legacy_mode_dump_stays_schema_2_0(self):
        rng = np.random.default_rng(12)
        r = ConceptRepository()
        r.add(fpx(sample(Spec("a", Z4, O4), 100, rng)), "m")
        self.assertEqual(r.to_dict()["schema_version"], "2.0")

    def test_a_schema_2_0_dump_loads_and_upgrades_on_first_update(self):
        rng = np.random.default_rng(13)
        old = ConceptRepository()
        X = sample(Spec("a", Z4, O4), 200, rng)
        old.add(fpx(X), "m")
        dump = json.loads(json.dumps(old.to_dict()))
        new = ConceptRepository(MatchConfig(merge_mode="mergeable"))
        new.load_dict(dump)
        rec = next(iter(new.records.values()))
        self.assertIsNone(rec.summaries)
        w = sample(Spec("a", Z4, O4), 80, rng)
        new.update(rec.concept_id, fpx(w), t=1, X=w)
        self.assertIsNotNone(rec.summaries)
        self.assertFalse(rec.meta["summaries_exact"])                             # honest: the base was reconstructed
        self.assertEqual(new.to_dict()["schema_version"], "2.1")

    def test_v1_migrated_concept_can_be_updated_in_mergeable_mode(self):
        vec = np.concatenate([[1, 2, 3, 4], [1, 1, 1, 1], [9, 9, 9, 9], [.5] * 4, [.5]])
        rec = ConceptRecord.from_dict({"id": 1, "fingerprint": vec.tolist(), "n": 100})
        r = ConceptRepository(MatchConfig(merge_mode="mergeable"))
        r.records[rec.concept_id] = rec
        rng = np.random.default_rng(14)
        w = sample(Spec("a", (1.0, 2.0, 3.0, 4.0), O4), 100, rng)
        r.update(rec.concept_id, fpx(w), t=1, X=w)
        self.assertTrue(np.isfinite(rec.fp.statistics["std"]).all())


class Eviction(unittest.TestCase):
    def fill(self, **kw):
        rng = np.random.default_rng(15)
        r = ConceptRepository(MatchConfig(max_concepts=3, **kw))
        ids = []
        for i in range(3):
            ids.append(r.add(fpx(sample(Spec("c", (i * 4.0,) * 4, O4), 80, rng)), f"m{i}", t=i).concept_id)
        return r, ids, rng

    def add_new(self, r, rng, t, protect=None):
        return r.add(fpx(sample(Spec("n", (50.0 + t,) * 4, O4), 80, rng)), f"n{t}", t=t, protect=protect)

    def test_lru_evicts_the_oldest_even_if_frequently_used(self):
        r, ids, rng = self.fill(eviction="lru")
        r.records[ids[0]].uses = 50                                               # old but frequently recurring
        self.add_new(r, rng, 10)
        self.assertNotIn(ids[0], r)

    def test_score_policy_keeps_the_frequently_recurring_old_concept(self):
        r, ids, rng = self.fill(eviction="score")
        r.records[ids[0]].uses = 50
        self.add_new(r, rng, 10)
        self.assertIn(ids[0], r)
        self.assertEqual(len(r), 3)

    def test_score_policy_uses_model_performance(self):
        r, ids, rng = self.fill(eviction="score", evict_weights=(0.0, 0.0, 1.0))
        for i, acc in zip(ids, (0.9, 0.2, 0.9)):
            r.record_performance(i, acc)
        self.add_new(r, rng, 10)
        self.assertNotIn(ids[1], r)                                               # the concept with the worst model goes first

    def test_grace_period_protects_young_concepts(self):
        r, ids, rng = self.fill(eviction="least_used", grace_updates=2)
        young = r.records[ids[2]]
        for rec in (r.records[ids[0]], r.records[ids[1]]):
            rec.update_count = 5
        young.update_count = 0
        young.uses = 0                                                            # would normally be the first victim
        self.add_new(r, rng, 10)
        self.assertIn(ids[2], r)

    def test_protected_concept_is_never_evicted_and_cap_holds(self):
        r, ids, rng = self.fill(eviction="score")
        keep = ids[0]
        for t in range(10, 20):
            self.add_new(r, rng, t, protect=keep)
        self.assertIn(keep, r)
        self.assertEqual(len(r), 3)

    def test_a_newly_added_concept_survives_even_when_everything_else_is_used_more(self):
        """Regression: with protect=<old id> the NEW concept used to be evictable, leaving the engine with a dangling active id."""
        r, ids, rng = self.fill(eviction="legacy")
        for i in ids:
            r.records[i].uses = 9
        new = r.add(fpx(sample(Spec("n", (80.0,) * 4, O4), 80, rng)), "new", t=50, protect=ids[0])
        self.assertIn(new.concept_id, r)
        self.assertIn(ids[0], r)
        self.assertEqual(len(r), 3)

    def test_engine_never_references_an_evicted_concept_with_a_tiny_cap(self):
        from dejastream.mv_engine import MVEngine
        from dejastream.scenarios import abrupt_recurring
        e = MVEngine(match=MatchConfig(max_concepts=2))
        for t, x, y, _ in abrupt_recurring():
            e.process(t, x, y)
            if e.active_id is not None:
                self.assertIn(e.active_id, e.repo)
        self.assertLessEqual(len(e.repo), 2)

    def test_config_validation(self):
        for kw in ({"merge_mode": "x"}, {"eviction": "x"}, {"max_concepts": 0}, {"evict_weights": (1.0, 1.0)}, {"grace_updates": -1}):
            with self.assertRaises(ValueError):
                MatchConfig(**kw)


if __name__ == "__main__":
    unittest.main()
