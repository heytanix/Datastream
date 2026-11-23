"""Hierarchical matching, confidence, repository behaviour."""
import unittest

import numpy as np

from datastream.matcher import ConceptRepository, Decision, MatchConfig
from datastream.mvfp import extract_feature_fp
from datastream.stream import CONCEPTS


def fp(name, n, seed, shift=0.0):
    c = CONCEPTS[name]
    X = np.random.default_rng(seed).normal(np.array(c.mu) + shift, c.sigma, (n, 4))
    return extract_feature_fp(X, t=np.arange(n))


def repo_abc(n=300):
    r = ConceptRepository()
    for i, nm in enumerate("ABC"):
        r.add(fp(nm, n, i), f"model_{i}")
    return r


class Matching(unittest.TestCase):
    def test_recurring_concept_is_retrieved(self):
        r = repo_abc()
        for nm, cid in zip("ABC", ("concept_001", "concept_002", "concept_003")):
            res = r.match(fp(nm, 80, 50))
            self.assertIn(res.decision, (Decision.HIGH_CONFIDENCE_MATCH, Decision.CANDIDATE_MATCH))
            self.assertEqual(res.best_id, cid)
            self.assertEqual(res.ranking[0].concept_id, cid)

    def test_unseen_concept_is_new(self):
        res = repo_abc().match(fp("D", 80, 7))
        self.assertEqual(res.decision, Decision.NEW_CONCEPT)
        self.assertIn("too far", res.reason)

    def test_false_match_rejected_for_similar_but_shifted(self):
        res = repo_abc().match(fp("A", 100, 8, shift=1.5))
        self.assertEqual(res.decision, Decision.NEW_CONCEPT)

    def test_insufficient_evidence(self):
        res = repo_abc().match(fp("A", 15, 3))
        self.assertEqual(res.decision, Decision.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(res.best_id)
        allnan = extract_feature_fp(np.full((100, 4), np.nan))
        self.assertEqual(repo_abc().match(allnan).decision, Decision.INSUFFICIENT_EVIDENCE)

    def test_empty_repository_gives_new_concept(self):
        self.assertEqual(ConceptRepository().match(fp("A", 80, 1)).decision, Decision.NEW_CONCEPT)

    def test_ambiguous_candidates_are_not_high_confidence(self):
        r = ConceptRepository()
        r.add(fp("A", 300, 1), "m1")
        r.add(fp("A", 300, 2, shift=0.05), "m2")             # near-duplicate concept
        res = r.match(fp("A", 100, 3))
        self.assertEqual(res.decision, Decision.CANDIDATE_MATCH)
        self.assertIn("ambiguous", res.reason)
        self.assertLess(res.ranking[1].D - res.ranking[0].D, 0.06)

    def test_confidence_is_separate_from_similarity(self):
        r = repo_abc()
        big, small = r.match(fp("A", 100, 5)), r.match(fp("A", 32, 5))
        self.assertGreater(big.confidence, small.confidence)           # fewer samples -> less confidence
        self.assertGreater(small.similarity, 0.75)                      # raw similarity is high either way
        self.assertLess(small.confidence, 0.7 * big.confidence)         # ... but confidence reflects the thin evidence
        for res in (big, small):
            self.assertTrue(0.0 <= res.confidence <= 1.0)

    def test_result_exposes_debug_information(self):
        d = repo_abc().match(fp("B", 80, 9)).as_dict()
        for key in ("decision", "best", "similarity", "components", "confidence", "reason", "ranking"):
            self.assertIn(key, d)
        self.assertEqual(set(d["components"]), {"stat", "dist", "time", "pred"})
        self.assertIsNone(d["components"]["pred"])                      # no predictive evidence supplied

    def test_thresholds_are_configurable(self):
        r = ConceptRepository(match=MatchConfig(t_high=0.001, t_cand=0.002))
        r.add(fp("A", 300, 1), "m")
        self.assertEqual(r.match(fp("A", 100, 2)).decision, Decision.NEW_CONCEPT)

    def test_exclude_set(self):
        r = repo_abc()
        res = r.match(fp("A", 80, 4), exclude={"concept_001"})
        self.assertNotEqual(res.best_id, "concept_001")


class Repository(unittest.TestCase):
    def test_bounded_growth_and_protection(self):
        r = ConceptRepository(match=MatchConfig(max_concepts=3))
        for i in range(6):
            r.add(fp("A", 50, i, shift=i), f"m{i}", t=i, protect="concept_001")
        self.assertEqual(len(r), 3)
        self.assertIn("concept_001", r)

    def test_update_merges_and_tracks_variability(self):
        r = ConceptRepository()
        rec = r.add(fp("A", 100, 1), "m")
        d = r.update(rec.concept_id, fp("A", 100, 2), t=10, count_use=True)
        self.assertLess(d, 0.25)
        self.assertEqual(rec.sample_count, 200)
        self.assertEqual(rec.uses, 2)
        self.assertIsNotNone(rec.intra)

    def test_concept_and_model_identity_are_separate(self):
        r = ConceptRepository()
        rec = r.add(fp("A", 100, 1), "model_001")
        r.replace_model(rec.concept_id, "model_002")
        self.assertEqual(rec.concept_id, "concept_001")
        self.assertEqual(rec.model_id, "model_002")
        self.assertEqual(rec.model_history, ["model_001", "model_002"])
        self.assertIsNone(rec.pred)

    def test_serialisation_roundtrip_preserves_matching(self):
        r = repo_abc()
        r2 = ConceptRepository()
        r2.load_dict(r.to_dict())
        self.assertEqual(len(r2), 3)
        self.assertEqual(r2.match(fp("C", 80, 11)).best_id, "concept_003")
        self.assertEqual(r2.to_dict()["schema_version"], "2.0")

    def test_repository_holds_no_model_objects(self):
        d = repo_abc().to_dict()
        self.assertNotIn("weights", str(d).lower())
        self.assertTrue(all(c["model_id"] for c in d["concepts"]))

    def test_memory_per_concept_is_small(self):
        self.assertLess(repo_abc().memory_per_concept(), 5000)


if __name__ == "__main__":
    unittest.main()
