"""Safe model recall: validation, provisional activation, rollback, delayed/missing labels."""
import unittest

import numpy as np

from datastream.mv_engine import MVEngine, RecallPolicy
from datastream.scenarios import _segments, abrupt_recurring, similar_x_different_y
from datastream.stream import CONCEPTS, Concept


def run(engine, scenario, drop_labels_after=None):
    ev, correct = [], 0
    for t, x, y, _ in scenario:
        yy = None if (drop_labels_after is not None and t >= drop_labels_after) else y
        p, es = engine.process(t, x, yy)
        correct += int(p == y)
        ev += es
    return ev, correct / len(scenario)


def kinds(ev, *k):
    return [e for e in ev if e["kind"] in k]


class Recall(unittest.TestCase):
    def test_successful_recall_after_shadow_validation(self):
        e = MVEngine()
        ev, acc = run(e, abrupt_recurring())
        self.assertEqual(e.models_trained, 4)
        self.assertGreaterEqual(len(kinds(ev, "recall")), 6)
        v = kinds(ev, "validation")
        self.assertTrue(v and all("candidates" in x and "baseline_acc" in x for x in v))
        self.assertTrue(all(x["accepted"] for x in v if x["t"] > 4000 and x["accepted"]))
        self.assertGreater(acc, 0.9)

    def test_every_recall_has_a_preceding_accepted_validation(self):
        ev, _ = run(MVEngine(), abrupt_recurring())
        for i, e in enumerate(ev):
            if e["kind"] == "recall":
                prev = [x for x in ev[:i] if x["kind"] == "validation"][-1]
                self.assertTrue(prev["accepted"])
                self.assertGreaterEqual(e["val_acc"], 0.70)

    def test_repeated_transitions_do_not_grow_repository(self):
        e = MVEngine()
        run(e, abrupt_recurring())
        self.assertEqual(len(e.repo), 4)
        self.assertEqual(len(e.models), 4)

    def test_failed_validation_triggers_rollback_and_new_concept(self):
        """A B Ap: after B the data looks like A again but the labelling is flipped."""
        a = CONCEPTS["A"]
        cs = {"A": a, "B": CONCEPTS["B"], "Ap": Concept("Ap", a.mu, a.sigma, tuple(-v for v in a.w))}
        sc = _segments("t", ["A", "B", "Ap"], 1500, 3, concepts=cs)
        e = MVEngine(label_delay=60, buffer_min=60, recall=RecallPolicy(val_min=80))
        ev, _ = run(e, sc)
        prov, roll = kinds(ev, "provisional"), kinds(ev, "rollback")
        self.assertTrue(prov, "HIGH-confidence candidate should be activated provisionally while labels are late")
        self.assertTrue(roll, "validation must fail and roll back")
        self.assertEqual(roll[0]["concept"], 2)                         # back to the previous concept (B = concept_002)
        self.assertLess(prov[0]["t"], roll[0]["t"])
        rejected = [x for x in kinds(ev, "validation") if not x["accepted"]]
        self.assertTrue(rejected)
        self.assertEqual(e.models_trained, 3)                           # A, B, and a new model for Ap
        self.assertEqual(len(e.repo), 3)
        self.assertNotIn(1, [x["concept"] for x in kinds(ev, "recall")])  # the wrong recall was never accepted

    def test_similar_x_different_y_keeps_two_concepts(self):
        e = MVEngine()
        ev, acc = run(e, similar_x_different_y())
        self.assertEqual(len(e.repo), 3)                                 # A, Ap and B
        self.assertGreater(acc, 0.88)

    def test_insufficient_validation_samples_never_activate(self):
        """No labels after warm-up: features alone must not cause a model to be recalled."""
        e = MVEngine()
        ev, _ = run(e, abrupt_recurring(), drop_labels_after=3200)       # concepts A,B,C seen, labels then vanish
        late = [x for x in ev if x["t"] > 3300]
        self.assertTrue(kinds(late, "match"), "X-only evidence is still used for matching")
        self.assertFalse(kinds(late, "recall", "new_concept", "provisional"))
        self.assertTrue(kinds(late, "recovery_timeout"), "recovery must end instead of waiting for labels forever")

    def test_provisional_activation_can_be_disabled(self):
        e = MVEngine(label_delay=60, recall=RecallPolicy(provisional=False))
        ev, _ = run(e, abrupt_recurring())
        self.assertFalse(kinds(ev, "provisional"))
        self.assertFalse(kinds(ev, "rollback"))

    def test_acceptance_criteria_are_configurable(self):
        strict = MVEngine(recall=RecallPolicy(min_acc=1.01))      # unreachable bar
        ev, _ = run(strict, abrupt_recurring())
        self.assertFalse(kinds(ev, "recall"))                            # nothing can pass an unreachable bar
        self.assertGreater(strict.models_trained, 4)

    def test_delayed_labels_still_recall(self):
        e = MVEngine(label_delay=40)
        ev, acc = run(e, abrupt_recurring())
        self.assertGreaterEqual(len(kinds(ev, "recall")), 5)
        self.assertLessEqual(e.models_trained, 5)
        self.assertGreater(acc, 0.85)

    def test_feature_shift_detects_drift_without_any_labels(self):
        """Same labelling rule, shifted inputs, labels never arrive: the label-free detector must still fire."""
        a = CONCEPTS["A"]
        cs = {"A": a, "A2": Concept("A2", tuple(v + 3 for v in a.mu), a.sigma, a.w)}
        sc = _segments("t", ["A", "A2"], 1000, 1, concepts=cs)
        e = MVEngine(warmup=150)
        ev = []
        for t, x, y, _ in sc:
            _, es = e.process(t, x, y if t < 300 else None)               # labels vanish after warm-up
            ev += es
        d = kinds(ev, "drift")
        self.assertTrue(d and d[0]["why"] == "feature_shift")
        self.assertLess(d[0]["t"] - 1000, 150)

    def test_log_and_events_are_json_serialisable(self):
        import json
        e = MVEngine()
        ev, _ = run(e, abrupt_recurring())
        json.dumps(ev)
        json.dumps(e.dash_info())
        json.dumps(e.snapshot())


if __name__ == "__main__":
    unittest.main()
