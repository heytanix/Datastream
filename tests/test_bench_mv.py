"""Benchmark plumbing, scenario ground truth, CSV loader."""
import json
import os
import tempfile
import unittest

import numpy as np

from datastream.bench_mv import run, run_one, to_markdown
from datastream.datasets import load_csv_stream
from datastream.scenarios import SCENARIOS, abrupt_recurring, similar_x_different_y


class Scenarios(unittest.TestCase):
    def test_all_scenarios_are_deterministic_and_well_formed(self):
        for name, f in SCENARIOS.items():
            a, b = f(seed=3), f(seed=3)
            np.testing.assert_array_equal(a.X, b.X)
            self.assertEqual(len(a.truth), len(a))
            self.assertTrue(set(np.unique(a.y)) <= {0, 1}, name)
            if name == "feature_availability":                         # missing values are the point of this scenario
                b = np.array([c == "B" for c in a.truth])
                self.assertTrue(np.isnan(a.X[b, 3]).all() and not np.isnan(a.X[b, :3]).any())
                self.assertTrue(np.isnan(a.X[np.array([c == "C" for c in a.truth])]).mean() > 0.2)
                self.assertFalse(np.isnan(a.X[np.array([c == "A" for c in a.truth])]).any())
            else:
                self.assertFalse(np.isnan(a.X).any(), name)
        self.assertFalse(np.array_equal(abrupt_recurring(seed=1).X, abrupt_recurring(seed=2).X))

    def test_change_points_match_truth(self):
        s = abrupt_recurring()
        for t, c in s.change_points:
            self.assertEqual(s.truth[t], c)

    def test_similar_x_scenario_has_identical_inputs_distribution(self):
        s = similar_x_different_y()
        a = s.X[[i for i, c in enumerate(s.truth) if c == "A"]]
        ap = s.X[[i for i, c in enumerate(s.truth) if c == "Ap"]]
        self.assertAlmostEqual(a.mean(), ap.mean(), delta=0.1)


class Bench(unittest.TestCase):
    def test_run_one_reports_all_metrics_for_every_strategy(self):
        d = abrupt_recurring(seed=1)
        mv = run_one(d, "datastream_mv")
        for k in ("accuracy", "post_drift_acc", "detection_delay", "models_trained", "recalls", "retrieval_precision",
                  "retrieval_recall", "false_match_rate", "missed_match_rate", "recall_success_rate", "fp_ms",
                  "bytes_per_concept", "us_per_sample"):
            self.assertIn(k, mv)
            self.assertIsNotNone(mv[k], k)
        self.assertEqual(mv["retrieval_precision"], 1.0)
        static = run_one(d, "static")
        self.assertLess(static["accuracy"], mv["accuracy"] - 0.2)

    def test_run_writes_json_and_markdown_without_hardcoded_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "b.json")
            r = run(seeds=(1, 2), scenarios=["similar_x_different_y"], out=out, verbose=False)
            with open(out) as fh:
                self.assertEqual(json.load(fh)["seeds"], [1, 2])
            self.assertTrue(os.path.exists(out.replace(".json", ".md")))
            cell = r["scenarios"]["similar_x_different_y"]["datastream_mv"]["accuracy"]
            self.assertEqual(cell["n"], 2)
            self.assertGreater(cell["std"] + 1, 0)
            self.assertIn("±", to_markdown(r))


class Loader(unittest.TestCase):
    def test_csv_with_categorical_missing_and_timestamps(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("ts,day,price,class\n3,mon,1.5,UP\n1,tue,?,DOWN\n2,mon,2.5,UP\n")
            path = f.name
        try:
            X, y, t, cat, names = load_csv_stream(path, "class", categorical=("day",), time_column="ts")
            self.assertEqual(names, ["day", "price"])
            self.assertEqual(cat, (0,))
            np.testing.assert_array_equal(t, [1, 2, 3])                  # sorted by timestamp
            self.assertTrue(np.isnan(X[0, 1]))                           # missing value kept as NaN
            self.assertEqual(sorted(set(y)), [0, 1])
            with self.assertRaises(ValueError):
                load_csv_stream(path, "nope")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()


class RealDatasetLoaders(unittest.TestCase):
    """The loaders are exercised on tiny files in the documented layouts only; no real data is bundled."""

    def test_intel_berkeley_layout(self):
        from datastream.datasets import load_intel_berkeley
        rows = []
        for i in range(60):
            rows.append(f"2004-02-28 00:{i % 60:02d}:16.0 {i + 1} {1 + i % 2} {19.0 + 0.1 * i:.4f} 37.1 45.0 2.7")
        rows.insert(5, "2004-02-28 corrupt line")                              # truncated record, as in the original file
        rows.insert(9, "2004-02-28 00:00:00.0 99 1 abc 37.1 45.0 2.7")          # unparseable value
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(rows))
            path = f.name
        try:
            X, y, t, cat, names = load_intel_berkeley(path, mote_id=1, horizon=5)
            self.assertEqual(X.shape[1], 4)
            self.assertEqual(names[0], "temperature")
            self.assertTrue((np.diff(t) > 0).all())                              # chronological
            self.assertEqual(int((y == -1).sum()), 5)                             # last `horizon` rows are unlabelled
            self.assertTrue((y[:-5] == 1).all())                                  # temperature rises in this toy file
            with self.assertRaises(ValueError):
                load_intel_berkeley(path, mote_id=7)
        finally:
            os.unlink(path)

    def test_uci_electricity_layout(self):
        from datastream.datasets import load_uci_electricity
        header = '"";"MT_001";"MT_002";"MT_003"'
        comma = lambda v: str(v).replace(".", ",")                                  # the file uses a decimal comma
        rows = [f"2011-01-01 {i // 4:02d}:{(i % 4) * 15:02d}:00;{comma(1 + i * 0.5)};{comma(2.5 + i)};{comma(i / 10)}" for i in range(12)]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join([header] + rows))
            path = f.name
        try:
            X, y, t, cat, names = load_uci_electricity(path, clients=(0, 1))
            self.assertEqual(X.shape, (12, 3))
            self.assertAlmostEqual(X[1, 0], 1.5)                                  # decimal comma parsed
            self.assertEqual(t[1], 15.0)
            self.assertEqual(names, ["client_0", "client_1", "hour"])
            self.assertEqual(int(y[-1]), -1)
        finally:
            os.unlink(path)

    def test_replay_stream_is_labelled_as_constructed_recurrence(self):
        from datastream.datasets import replay_stream
        X, y = np.arange(40.0)[:, None], np.arange(40) % 2
        Xr, yr, truth = replay_stream(X, y, [(0, 10), (10, 25)], [0, 1, 0])
        self.assertEqual(len(Xr), 35)
        self.assertEqual(truth[:10], [0] * 10)
        np.testing.assert_array_equal(Xr[:10], Xr[25:])                           # the recurrence is an exact replay


class ArffLoader(unittest.TestCase):
    def test_numeric_nominal_missing_and_quoting(self):
        from datastream.datasets import load_arff
        text = """% comment
@relation toy
@attribute date numeric
@attribute day {1,2,3}
@attribute airport {'AAA','BBB'}
@attribute price numeric
@attribute class {UP,DOWN}
@data
0,2,'AAA',0.5,UP
0.1, 3 ,BBB,?,DOWN
0.2,1,'AAA',0.7,UP
"""
        with tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            X, y, names, nominal = load_arff(path, "class", drop=("date",))
            self.assertEqual(names, ["day", "airport", "price"])
            self.assertEqual(nominal, ["day", "airport"])
            np.testing.assert_array_equal(X[:, 0], [1, 2, 0])                  # codes in declaration order
            np.testing.assert_array_equal(X[:, 1], [0, 1, 0])
            self.assertTrue(np.isnan(X[1, 2]))
            np.testing.assert_array_equal(y, [0, 1, 0])                        # UP=0, DOWN=1 (declaration order)
            self.assertEqual(len(load_arff(path, "class", max_rows=2)[1]), 2)
            with self.assertRaises(ValueError):
                load_arff(path, "nope")
        finally:
            os.unlink(path)


class ArffStreamTests(unittest.TestCase):
    def test_streams_rows_drops_high_cardinality_nominals_and_loops(self):
        from datastream.datasets import ArffStream
        rows = "\n".join(f"{i % 3},{i * 0.5},'C{i % 40}',{'UP' if i % 2 == 0 else 'DOWN'}" for i in range(80))
        text = "@relation t\n@attribute day {0,1,2}\n@attribute v numeric\n@attribute code {" + ",".join(f"'C{i}'" for i in range(40)) + "}\n@attribute class {UP,DOWN}\n@data\n" + rows + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".arff", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            s = ArffStream(path, "class", limit=50, loops=2)
            self.assertEqual(s.names, ["day", "v"])                              # the 40-level nominal column is left out
            self.assertEqual((s.n_features, len(s)), (2, 50))
            out = list(s)
            self.assertEqual(len(out), 100)                                      # two passes
            self.assertEqual(out[0][3], "data")
            self.assertEqual(ArffStream(path, "class", features=["code", "v"]).names, ["code", "v"])
        finally:
            os.unlink(path)
