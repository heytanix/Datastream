"""Reliability: rollback state, false alarms, label timing, missing labels, bad records, model lifecycle, worker failures."""
import os
import tempfile
import time
import unittest

import numpy as np

from datastream.matcher import MatchConfig
from datastream.mv_engine import MVEngine, RecallPolicy
from datastream.scenarios import _segments, abrupt_recurring
from datastream.stream import CONCEPTS, Concept


def kinds(ev, *k):
    return [e for e in ev if e["kind"] in k]


def flipped_a():
    """A B A' where A' has A's inputs but the opposite labelling rule: the fingerprint says A, the labels say no."""
    a = CONCEPTS["A"]
    cs = {"A": a, "B": CONCEPTS["B"], "Ap": Concept("Ap", a.mu, a.sigma, tuple(-v for v in a.w))}
    return _segments("t", ["A", "B", "Ap"], 1500, 3, concepts=cs)


class RollbackState(unittest.TestCase):
    def test_rollback_restores_the_previous_model_object(self):
        e = MVEngine(label_delay=60, buffer_min=60, train_min=200, recall=RecallPolicy(val_min=80))
        prev = provisional = None
        checked = False
        for t, x, y, _ in flipped_a():
            _, ev = e.process(t, x, y)
            if kinds(ev, "provisional"):
                prev, provisional = e.rec.prev_model, e.model
                self.assertIsNot(provisional, prev)
            if kinds(ev, "rollback"):
                self.assertFalse(kinds(ev, "new_concept"), "train_min keeps training out of the rollback step")
                self.assertIs(e.model, prev, "rollback must serve the previous model again")
                self.assertIsNot(e.model, provisional)
                checked = True
                break
        self.assertTrue(checked, "the scenario must produce a provisional recall that is rolled back")


class FalseAlarm(unittest.TestCase):
    def test_label_noise_burst_is_a_false_alarm_not_a_new_concept(self):
        """A burst of flipped labels inside concept A triggers the error detector; the data never changed."""
        rng = np.random.default_rng(3)
        a = CONCEPTS["A"]
        mu, sg, w = (np.array(v, float) for v in (a.mu, a.sigma, a.w))
        e = MVEngine(recall=RecallPolicy(val_min=150))
        events = []
        for t in range(4000):
            x = rng.normal(mu, sg)
            y = int((x - mu) @ w > 0)
            if 1500 <= t < 1525:
                y = 1 - y
            _, ev = e.process(t, x, y)
            events += ev
        drifts = [d for d in kinds(events, "drift") if d["t"] >= 1500]
        self.assertTrue(drifts and drifts[0]["why"] == "error")
        fa = kinds(events, "false_alarm")
        self.assertTrue(fa, "the unchanged concept must be confirmed as a false alarm")
        self.assertEqual(fa[0]["concept"], 1)
        self.assertEqual(e.models_trained, 1)
        self.assertEqual(len(e.repo), 1)
        self.assertFalse(kinds(events, "recall", "new_concept")[1:])     # only the warm-up concept


class LabelTiming(unittest.TestCase):
    def test_label_is_released_exactly_label_delay_samples_later(self):
        e = MVEngine(warmup=10, label_delay=5)
        for t in range(14):
            e.process(t, np.zeros(4), t % 2)
            self.assertEqual(e.state, "WARMUP", f"t={t}: only {max(0, t - 4)} labels are due")
        e.process(14, np.zeros(4), 0)                                   # labels of t=0..9 are now due
        self.assertEqual(e.state, "STABLE")


class MissingLabels(unittest.TestCase):
    def test_recovery_times_out_and_buffers_stay_bounded(self):
        e = MVEngine(max_recovery=600, max_buffer=300)
        ev = []
        for t, x, y, _ in abrupt_recurring():
            _, es = e.process(t, x, y if t < 3200 else None)
            ev += es
            if e.rec is not None:
                self.assertLessEqual(len(e.rec.X), 300)
                self.assertLessEqual(len(e.rec.lab), 300)
        late = [x for x in ev if x["t"] > 3300]
        to = kinds(late, "recovery_timeout")
        self.assertTrue(to, "recovery must end when labels never arrive")
        self.assertTrue(all(x["samples"] == 600 and x["labels"] == 0 for x in to))
        self.assertFalse(kinds(late, "recall", "new_concept"), "without labels nothing may be recalled or trained")
        self.assertEqual(e.timeouts, len(to))

    def test_timeout_rolls_back_an_unvalidated_provisional_model(self):
        e = MVEngine(label_delay=60, buffer_min=60, train_min=200, recall=RecallPolicy(val_min=80), max_recovery=300)
        for t, x, y, _ in flipped_a():
            _, ev = e.process(t, x, y if t < 3000 else None)             # labels stop just before A' starts
            if kinds(ev, "provisional"):
                prev = e.rec.prev_model
            if kinds(ev, "recovery_timeout") and t > 3000:
                self.assertIs(e.model, prev)
                self.assertTrue(kinds(ev, "rollback"))
                return
        self.fail("expected a provisional model followed by a timeout")

    def test_warm_up_buffer_is_bounded_without_labels(self):
        e = MVEngine(max_buffer=400)
        for t in range(3000):
            e.process(t, np.zeros(4), None)
        self.assertEqual(e.state, "WARMUP")
        self.assertLessEqual(len(e._wu_X), 400)

    def test_max_buffer_must_hold_the_recovery_minimums(self):
        with self.assertRaises(ValueError):
            MVEngine(max_buffer=50)


class BadRecords(unittest.TestCase):
    def test_duplicates_and_replays_are_not_learned_twice(self):
        e = MVEngine()
        sc = list(abrupt_recurring())
        for t, x, y, _ in sc[:400]:
            e.process(t, x, y)
        n = e.detector.n
        for t, x, y, _ in sc[350:400]:                                  # replay 50 seen records
            pred, ev = e.process(t, x, y)
            self.assertIn(pred, (0, 1))
            self.assertEqual(ev, [])
        self.assertEqual(e.rejected, 50)
        self.assertEqual(e.detector.n, n)

    def test_out_of_order_records_do_not_stall_label_release(self):
        e = MVEngine(warmup=20, label_delay=5)
        e.process(100, np.zeros(4), 1)
        for t in range(99, 50, -1):                                      # older timestamps arrive late
            e.process(t, np.zeros(4), 0)
        self.assertEqual(e.rejected, 49)
        for t in range(101, 130):
            e.process(t, np.zeros(4), t % 2)
        self.assertEqual(e.state, "STABLE", "labels must keep flowing after out-of-order records")


class ModelLifecycle(unittest.TestCase):
    def test_models_of_evicted_concepts_are_released(self):
        e = MVEngine(match=MatchConfig(max_concepts=2))
        for t, x, y, _ in abrupt_recurring():
            e.process(t, x, y)
            self.assertLessEqual(len(e.models), len(e.repo) + 1)          # +1: a model that is serving right now
        live = {r.model_id for r in e.repo.records.values()}
        self.assertTrue(set(e.models) - live <= {k for k, m in e.models.items() if m is e.model})


class WorkerFailure(unittest.TestCase):
    def test_a_failing_worker_is_reported_and_the_others_keep_running(self):
        from datastream.pipeline import Pipeline
        with tempfile.TemporaryDirectory() as d:
            p = Pipeline(rate=100_000, loops=1, seg_len=300, db_path=os.path.join(d, "t.db"))
            eng = p.engines["datastream_mv"]
            real = eng.process

            def broken(t, x, y):
                if t >= 500:
                    raise RuntimeError("injected fault")
                return real(t, x, y)

            eng.process = broken
            with self.assertLogs("datastream.pipeline", level="ERROR") as logs:
                p.start()
                deadline = time.time() + 30
                while time.time() < deadline:
                    h = p.snapshot()["health"]
                    if h["proc-datastream_mv"]["status"] == "failed" and p.snapshot()["summary"]["static"]["seen"] >= 3000:
                        break
                    time.sleep(0.2)
                h = p.snapshot()["health"]
                p.close()
            self.assertEqual(h["proc-datastream_mv"]["status"], "failed")
            self.assertGreaterEqual(h["proc-datastream_mv"]["errors"], 100)
            self.assertIn("injected fault", h["proc-datastream_mv"]["last_error"])
            self.assertFalse(h["proc-datastream_mv"]["alive"])
            self.assertEqual(h["proc-static"]["status"], "ok")
            self.assertTrue(h["proc-static"]["alive"])
            self.assertTrue(any("injected fault" in m for m in logs.output))
            p.store.db.close()


if __name__ == "__main__":
    unittest.main()
