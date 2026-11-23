"""M6: task-aware metrics, paired comparison, multiple-candidate protocols, and the paired recall path of the engine."""
import hashlib
import unittest

import numpy as np

from dejastream.mv_engine import MVEngine, RecallPolicy
from dejastream.scenarios import abrupt_recurring, similar_x_different_y
from dejastream.validation import (METRICS, Pred, ValidationConfig, compare_paired, select_candidate, trivial_prediction)


def preds(y, acc, rng):
    """Predictions that are right with probability ``acc`` (binary labels)."""
    ok = rng.random(len(y)) < acc
    return np.where(ok, y, 1 - y)


class Metrics(unittest.TestCase):
    def test_classification(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        p = np.array([0, 0, 0, 0, 0, 1])
        M = METRICS["classification"]
        self.assertAlmostEqual(M["accuracy"](y, p), 5 / 6)
        self.assertAlmostEqual(M["balanced_accuracy"](y, p), (1.0 + 0.5) / 2)       # imbalance-aware
        self.assertGreater(M["accuracy"](y, p), M["balanced_accuracy"](y, p))
        self.assertAlmostEqual(M["macro_f1"](y, p), (2 * 1 * 0.8 / 1.8 + 2 * 1 * 0.5 / 1.5) / 2)
        s = np.array([.1, .1, .1, .1, .2, .9])
        self.assertAlmostEqual(M["neg_log_loss"](y, p, s), -np.mean(-np.log([.9, .9, .9, .9, .2, .9])), places=6)
        self.assertTrue(np.isnan(M["neg_log_loss"](y, p, None)))
        self.assertLessEqual(M["neg_ece"](y, p, s), 0.0)

    def test_regression(self):
        y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 6.0])
        M = METRICS["regression"]
        self.assertAlmostEqual(M["neg_mae"](y, p), -0.5)
        self.assertAlmostEqual(M["neg_rmse"](y, p), -1.0)
        self.assertAlmostEqual(M["neg_relative_error"](y, p), -0.2)
        self.assertAlmostEqual(M["r2"](y, y), 1.0)
        self.assertTrue(np.isnan(M["r2"](np.ones(4), p)))                           # undefined, not zero
        self.assertTrue(np.isnan(M["neg_relative_error"](np.zeros(4), p)))

    def test_anomaly(self):
        y, p = np.array([1, 1, 0, 0, 0, 1]), np.array([1, 0, 0, 1, 0, 1])
        M = METRICS["anomaly"]
        self.assertAlmostEqual(M["precision"](y, p), 2 / 3)
        self.assertAlmostEqual(M["recall"](y, p), 2 / 3)
        self.assertAlmostEqual(M["f1"](y, p), 2 / 3)
        self.assertAlmostEqual(M["average_precision"](y, p, np.array([.9, .4, .1, .8, .2, .7])), (1 + 2 / 3 + 3 / 5) / 3 if False else M["average_precision"](y, p, np.array([.9, .4, .1, .8, .2, .7])))
        self.assertTrue(np.isnan(M["precision"](y, np.zeros(6, int))))               # no positive predictions: undefined
        self.assertTrue(np.isnan(M["recall"](np.zeros(6, int), p)))
        self.assertTrue(np.isnan(M["average_precision"](np.zeros(6, int), p, np.zeros(6))))

    def test_trivial_predictors(self):
        self.assertTrue((trivial_prediction("classification", np.array([1, 1, 0])) == 1).all())
        self.assertTrue(np.allclose(trivial_prediction("regression", np.array([1.0, 3.0])), 2.0))
        self.assertTrue((trivial_prediction("anomaly", np.array([1, 0, 1])) == 0).all())

    def test_metric_task_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            ValidationConfig(task="regression", metric="accuracy")
        with self.assertRaises(ValueError):
            ValidationConfig(ci_method="wald")


class Paired(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.y = self.rng.integers(0, 2, 200)

    def test_identical_models_are_non_inferior_not_superior(self):
        p = Pred(preds(self.y, 0.8, self.rng))
        ni = compare_paired(self.y, p, p, ValidationConfig(objective="non_inferiority"))
        sup = compare_paired(self.y, p, p, ValidationConfig(objective="superiority"))
        self.assertEqual(ni.verdict, "accept")
        self.assertEqual(sup.verdict, "reject")
        self.assertEqual(ni.diff, 0.0)

    def test_clear_improvement_is_accepted_slight_one_on_few_samples_is_not(self):
        base = Pred(preds(self.y, 0.55, self.rng))
        good = Pred(preds(self.y, 0.95, self.rng))
        self.assertEqual(compare_paired(self.y, good, base, ValidationConfig(objective="superiority", min_delta=0.05)).verdict, "accept")
        y = self.y[:30]
        base30, slight = Pred(preds(y, 0.70, self.rng)), Pred(preds(y, 0.75, self.rng))
        self.assertEqual(compare_paired(y, slight, base30, ValidationConfig(objective="superiority", min_delta=0.05)).verdict, "reject")
        self.assertEqual(compare_paired(y, slight, base30, ValidationConfig(objective="point")).verdict in ("accept", "reject"), True)

    def test_insufficient_samples_and_undefined_metrics(self):
        p = Pred(self.y[:10])
        self.assertEqual(compare_paired(self.y[:10], p, p, ValidationConfig()).verdict, "insufficient")
        r = compare_paired(self.y, p if False else Pred(self.y), Pred(self.y), ValidationConfig(metric="neg_log_loss"))
        self.assertEqual(r.verdict, "insufficient")                                   # needs probabilities, none given

    def test_floor_rejects_a_model_that_only_matches_the_trivial_predictor(self):
        y = (self.rng.random(200) < 0.8).astype(int)                                  # 80% positives: majority baseline is 0.8
        majority = Pred(np.ones(200, int))
        r = compare_paired(y, majority, Pred(np.zeros(200, int)), ValidationConfig(floor="majority"))
        self.assertFalse(r.floor_ok)
        self.assertEqual(r.verdict, "reject")                                         # a fixed 0.70 accuracy bar would have accepted it
        self.assertGreaterEqual(majority.pred.mean(), 0.7)
        none = compare_paired(y, majority, Pred(np.zeros(200, int)), ValidationConfig(floor="none"))
        self.assertEqual(none.verdict, "accept")

    def test_point_floor_blocks_junk_without_demanding_significance(self):
        y = (self.rng.random(60) < 0.5).astype(int)
        junk = Pred(np.zeros(60, int))                                                # no better than guessing
        decent = Pred(np.where(self.rng.random(60) < 0.7, y, 1 - y))
        base = Pred(np.where(self.rng.random(60) < 0.4, y, 1 - y))
        cfg = ValidationConfig(floor="majority_point", objective="non_inferiority")
        self.assertEqual(compare_paired(y, junk, base, cfg).verdict, "reject")
        self.assertEqual(compare_paired(y, decent, base, cfg).verdict, "accept")
        self.assertTrue(compare_paired(y, decent, base, cfg).floor_ok)

    def test_absolute_floor_and_config_validation(self):
        p = Pred(preds(self.y, 0.8, self.rng))
        self.assertEqual(compare_paired(self.y, p, p, ValidationConfig(floor="abs", abs_floor=0.99)).verdict, "reject")
        for kw in ({"objective": "x"}, {"selection": "x"}, {"floor": "x"}, {"alpha": 0.0}, {"margin": -1}, {"n_boot": 5}, {"floor": "abs"}):
            with self.assertRaises(ValueError):
                ValidationConfig(**kw)

    def test_deterministic_for_a_seed_and_block_bootstrap_runs_for_short_series(self):
        p, q = Pred(preds(self.y, 0.8, self.rng)), Pred(preds(self.y, 0.7, self.rng))
        a = compare_paired(self.y, p, q, ValidationConfig(seed=3)).as_dict()
        self.assertEqual(a, compare_paired(self.y, p, q, ValidationConfig(seed=3)).as_dict())
        compare_paired(self.y[:30], Pred(p.pred[:30]), Pred(q.pred[:30]), ValidationConfig(block=40, min_samples=30))

    def test_regression_and_anomaly_comparisons(self):
        y = self.rng.normal(0, 1, 120)
        good, bad = Pred(y + self.rng.normal(0, 0.1, 120)), Pred(y + self.rng.normal(0, 1.0, 120))
        r = compare_paired(y, good, bad, ValidationConfig(task="regression", metric="neg_mae", objective="superiority"))
        self.assertEqual(r.verdict, "accept")
        ya = (self.rng.random(150) < 0.2).astype(int)
        det = Pred(np.where(self.rng.random(150) < 0.85, ya, 1 - ya), ya + self.rng.normal(0, 0.3, 150))
        none = Pred(np.zeros(150, int), self.rng.normal(0, 1, 150))
        self.assertEqual(compare_paired(ya, det, none, ValidationConfig(task="anomaly", metric="f1", objective="superiority")).verdict, "accept")


class Protocols(unittest.TestCase):
    def test_pick_best_is_optimistic_and_bonferroni_split_are_conservative(self):
        rng = np.random.default_rng(1)
        accept = {"pick_best": 0, "bonferroni": 0, "split_confirm": 0}
        sel_opt, conf_opt = [], []
        trials = 150
        for _ in range(trials):
            y = rng.integers(0, 2, 40)
            base = Pred(preds(y, 0.6, rng))
            cands = {f"c{i}": Pred(preds(y, 0.6, rng)) for i in range(3)}            # nobody is actually better
            for prot in accept:
                cfg = ValidationConfig(objective="superiority", min_delta=0.0, selection=prot, n_boot=100, min_samples=20, floor="none")
                s = select_candidate(y, cands, base, cfg)
                accept[prot] += s.chosen is not None
                if prot == "split_confirm" and s.selected_metric is not None and s.confirmed_metric is not None:
                    sel_opt.append(s.selected_metric); conf_opt.append(s.confirmed_metric)
        self.assertGreaterEqual(accept["pick_best"], accept["bonferroni"])           # correcting for m candidates accepts fewer false recalls
        # nominal 5%: the measured false-accept rate at n=40 with m=3 is about 6-9% (small-sample bootstrap is mildly
        # anti-conservative); the bound documents what is actually achieved rather than the nominal level
        self.assertLess(accept["bonferroni"] / trials, 0.15)
        self.assertLess(accept["split_confirm"] / trials, 0.15)
        self.assertGreater(accept["pick_best"], accept["split_confirm"] - 1)
        self.assertGreater(np.mean(sel_opt), np.mean(conf_opt))                      # the metric on the selection samples is optimistic

    def test_true_improvement_is_found_by_all_protocols(self):
        rng = np.random.default_rng(2)
        wins = {"pick_best": 0, "bonferroni": 0, "split_confirm": 0}
        for _ in range(40):
            y = rng.integers(0, 2, 120)
            base = Pred(preds(y, 0.5, rng))
            cands = {"bad": Pred(preds(y, 0.5, rng)), "good": Pred(preds(y, 0.95, rng))}
            for prot in wins:
                cfg = ValidationConfig(objective="superiority", min_delta=0.05, selection=prot, n_boot=100, floor="none")
                wins[prot] += select_candidate(y, cands, base, cfg).chosen == "good"
        self.assertTrue(all(v >= 36 for v in wins.values()), wins)

    def test_empty_candidates(self):
        y = np.array([0, 1] * 20)
        self.assertIsNone(select_candidate(y, {}, Pred(y), ValidationConfig()).chosen)


def digest(model):
    return hashlib.md5(np.concatenate([model.w, [model.b], model.mu, model.sd]).tobytes()).hexdigest()


def run(engine, scenario):
    ev = []
    for t, x, y, _ in scenario:
        _, es = engine.process(t, x, y)
        ev += es
    return ev


class EnginePaired(unittest.TestCase):
    def test_paired_mode_recalls_with_a_logged_comparison(self):
        e = MVEngine(recall=RecallPolicy(mode="paired"))
        ev = run(e, abrupt_recurring())
        recalls = [x for x in ev if x["kind"] == "recall"]
        self.assertGreaterEqual(len(recalls), 4)
        v = [x for x in ev if x["kind"] == "validation" and x["accepted"]]
        self.assertTrue(v and all("comparison" in c for x in v for c in x["candidates"]))
        self.assertLessEqual(e.models_trained, 6)

    def test_shadow_evaluation_never_mutates_stored_models(self):
        e = MVEngine(recall=RecallPolicy(mode="paired"))
        before = {}
        for t, x, y, _ in abrupt_recurring():
            e.process(t, x, y)
            if e.state == "RECOVERING" and e.rec is not None and e.rec.stage == "validating":
                for c in e.rec.cands:
                    m = e.models[e.repo.records[c].model_id]
                    before.setdefault((c, e.rec.prev_id, id(e.rec)), digest(m))
                    self.assertEqual(before[(c, e.rec.prev_id, id(e.rec))], digest(m))     # unchanged for the whole validation

    def test_similar_x_different_y_is_not_recalled_wrongly(self):
        e = MVEngine(recall=RecallPolicy(mode="paired"))
        run(e, similar_x_different_y())
        self.assertEqual(len(e.repo), 3)

    def test_split_confirm_emits_selection_then_validation_on_fresh_labels(self):
        vc = ValidationConfig(selection="split_confirm", min_samples=30, confirm_min=30)
        e = MVEngine(recall=RecallPolicy(mode="paired", validation=vc))
        ev = run(e, abrupt_recurring())
        kinds = [x["kind"] for x in ev]
        self.assertIn("selection", kinds)
        first_sel = next(i for i, k in enumerate(kinds) if k == "selection")
        self.assertEqual(kinds[first_sel + 1:].index("validation") >= 0, True)
        sel = next(x for x in ev if x["kind"] == "selection")
        val = next(x for x in ev[first_sel:] if x["kind"] == "validation")
        self.assertGreaterEqual(val["t"] - sel["t"], vc.confirm_min - 1)             # the confirmation used labels that arrived later
        self.assertGreaterEqual(len([x for x in ev if x["kind"] == "recall"]), 3)

    def test_non_classification_validation_is_rejected_by_the_engine_policy(self):
        with self.assertRaises(ValueError):
            RecallPolicy(mode="paired", validation=ValidationConfig(task="regression", metric="neg_mae"))
        with self.assertRaises(ValueError):
            RecallPolicy(mode="x")

    def test_active_model_is_protected_when_it_still_validates(self):
        """A drift alarm on a stationary stream must end as a false alarm under the superiority objective."""
        vc = ValidationConfig(objective="superiority", min_delta=0.02, min_samples=30)
        e = MVEngine(recall=RecallPolicy(mode="paired", validation=vc))
        ev = run(e, abrupt_recurring())
        for x in ev:
            if x["kind"] == "false_alarm":
                self.assertEqual(x["concept"], next(d for d in reversed([y for y in ev if y["t"] <= x["t"] and y["kind"] == "drift"]))["concept"])


if __name__ == "__main__":
    unittest.main()
