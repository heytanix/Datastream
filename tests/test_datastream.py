import unittest

import numpy as np

from datastream.bench import run
from datastream.bus import Bus
from datastream.fingerprint import fingerprint
from datastream.stream import CONCEPTS


class BusTests(unittest.TestCase):
    def test_groups_read_independently(self):
        bus = Bus()
        bus.create_topic("t")
        for i in range(5):
            bus.publish("t", i)
        a, b = bus.consumer("a", ["t"]), bus.consumer("b", ["t"])
        self.assertEqual([r.value for r in a.poll(0.01)], [0, 1, 2, 3, 4])
        self.assertEqual([r.value for r in b.poll(0.01)], [0, 1, 2, 3, 4])
        self.assertEqual(a.poll(0.01), [])
        self.assertEqual(bus.stats()["lag"]["a"]["t"], 0)

    def test_sender_drops_instead_of_blocking(self):
        bus = Bus()
        bus.create_topic("t")
        s = bus.sender(capacity=1)
        s._stop.set()                       # stop the drain thread so the queue fills
        s.thread.join()
        results = [s.send("t", i) for i in range(5)]
        self.assertEqual(results.count(False), 4)
        self.assertEqual(bus.dropped, 4)

    def test_retention_trims_log(self):
        bus = Bus()
        bus.create_topic("t", retention=10)
        for i in range(25):
            bus.publish("t", i)
        recs = bus.consumer("g", ["t"]).poll(0.01)
        self.assertEqual(recs[0].value, 15)


class FingerprintTests(unittest.TestCase):
    def test_same_concept_closer_than_different(self):
        def window(name, seed):
            c = CONCEPTS[name]
            rng = np.random.default_rng(seed)
            X = rng.normal(c.mu, c.sigma, (200, 4))
            y = (((X - c.mu) @ np.array(c.w)) > 0).astype(int)
            return fingerprint(X, y)
        a1, a2, b = window("A", 1), window("A", 2), window("B", 3)
        self.assertLess(np.linalg.norm(a1 - a2), np.linalg.norm(a1 - b))


class EndToEnd(unittest.TestCase):
    def test_datastream_recalls_instead_of_retraining(self):
        res, events, _ = run(out=None, verbose=False)
        s = res["strategies"]
        self.assertEqual(s["datastream"]["detected_changes"], s["datastream"]["true_changes"])
        self.assertEqual(s["datastream"]["models_trained"], 4)          # A, B, C, D once each
        self.assertGreaterEqual(s["datastream"]["recalls"], 6)
        self.assertGreater(s["datastream"]["post_drift_300_acc"], s["retrain"]["post_drift_300_acc"])
        self.assertGreater(s["datastream"]["accuracy"], s["static"]["accuracy"] + 0.2)


if __name__ == "__main__":
    unittest.main()
