"""Load a real-world CSV (e.g. Electricity, Airlines) as a stream. Numeric and categorical columns, missing values.

No dataset is bundled or downloaded. Obtain a CSV yourself (OpenML 'electricity' / 'airlines') and point
``load_csv_stream`` at it. The loader is tested on a small synthetic CSV only.
"""
from __future__ import annotations

import csv

import numpy as np


def load_csv_stream(path: str, target: str, categorical: tuple[str, ...] = (), time_column: str | None = None,
                    max_rows: int | None = None):
    """Return (X, y, t, cat_idx, feature_names). Missing cells ('', '?', 'NA', 'nan') become NaN.

    Categorical columns are integer-coded in order of first appearance. The target is coded the same way
    (so binary targets become 0/1). Rows keep file order, or timestamp order if ``time_column`` is given.
    """
    missing = {"", "?", "na", "nan", "null"}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if target not in (rd.fieldnames or []):
            raise ValueError(f"target column {target!r} not in {rd.fieldnames}")
        feats = [c for c in rd.fieldnames if c not in (target, time_column)]
        rows = []
        for i, row in enumerate(rd):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    if not rows:
        raise ValueError("empty CSV")
    codes: dict[str, dict[str, int]] = {c: {} for c in list(categorical) + [target]}

    def cell(col, raw):
        raw = raw.strip()
        if col in codes:
            if raw.lower() in missing:
                return np.nan
            return float(codes[col].setdefault(raw, len(codes[col])))
        return np.nan if raw.lower() in missing else float(raw)

    X = np.array([[cell(c, r[c]) for c in feats] for r in rows], float)
    y_raw = [cell(target, r[target]) for r in rows]
    if any(np.isnan(v) for v in y_raw):
        y = np.array([-1 if np.isnan(v) else int(v) for v in y_raw])       # -1 = unlabelled
    else:
        y = np.array([int(v) for v in y_raw])
    t = np.array([float(r[time_column]) for r in rows]) if time_column else np.arange(len(rows), dtype=float)
    if time_column:
        order = np.argsort(t, kind="stable")
        X, y, t = X[order], y[order], t[order]
    return X, y, t, tuple(i for i, c in enumerate(feats) if c in categorical), feats


# -------------------------------------------------------------------------------------------------
# Loaders for the real-world streams named in the research plan. NO data is bundled or downloaded; these loaders were
# written from the published file formats and are tested ONLY on tiny synthetic files in the same layout.
#   Electricity (ELEC2) / Airlines : OpenML CSV exports -> use ``load_csv_stream`` (target "class" / "Delay").
#   Intel Berkeley sensor data     : data.txt (whitespace separated: date time epoch moteid temperature humidity light voltage)
#   UCI Electricity Load Diagrams  : LD2011_2014.txt (';' separated, decimal comma, first column = timestamp, one column per client)
# Neither sensor dataset has a class label or a ground-truth concept. The labels below are DERIVED (documented per loader) so
# the classification engine can run; they are not evidence that natural recurring concepts exist in the data.
# -------------------------------------------------------------------------------------------------
def load_intel_berkeley(path: str, mote_id: int | None = None, max_rows: int | None = None, horizon: int = 30):
    """Return (X, y, t, cat_idx, names) from the Intel Berkeley lab ``data.txt``.

    Features: temperature, humidity, light, voltage (NaN where a field is missing/corrupt). Timestamps: epoch counter.
    DERIVED label: 1 if temperature ``horizon`` readings ahead is higher than now, else 0 (the last ``horizon`` rows get -1 = unlabelled).
    """
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:                      # truncated records exist in the original file
                continue
            try:
                mote, epoch = int(p[3]), float(p[2])
                vals = [float(v) for v in p[4:8]]
            except ValueError:
                continue
            if mote_id is not None and mote != mote_id:
                continue
            rows.append((epoch, vals))
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError("no usable rows")
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    X = np.array([r[1] for r in rows], float)
    temp = X[:, 0]
    y = np.full(len(rows), -1, int)
    y[:-horizon] = (temp[horizon:] > temp[:-horizon]).astype(int)
    return X, y, t, (), ["temperature", "humidity", "light", "voltage"]


def load_uci_electricity(path: str, clients: tuple[int, ...] = (0, 1, 2, 3), max_rows: int | None = None):
    """Return (X, y, t, cat_idx, names) from UCI ``LD2011_2014.txt`` (``;`` separated, decimal comma).

    Features: the chosen client columns (kW per 15 min), plus hour-of-day. DERIVED label: 1 if the total demand of the
    next interval is higher than the current one. Timestamps are minutes since the first row.
    """
    import datetime as dt
    X, tt = [], []
    with open(path) as f:
        header = f.readline()
        if ";" not in header:
            raise ValueError("expected a ';' separated file")
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            p = line.strip().strip('"').split(";")
            ts = dt.datetime.fromisoformat(p[0].strip('"'))
            vals = [float(p[1 + c].replace(",", ".")) if p[1 + c].strip() else np.nan for c in clients]
            X.append(vals + [float(ts.hour)])
            tt.append(ts)
    if len(X) < 3:
        raise ValueError("too few rows")
    X = np.array(X, float)
    t = np.array([(x - tt[0]).total_seconds() / 60.0 for x in tt])
    total = np.nansum(X[:, :-1], axis=1)
    y = np.full(len(X), -1, int)
    y[:-1] = (total[1:] > total[:-1]).astype(int)
    return X, y, t, (), [f"client_{c}" for c in clients] + ["hour"]


def replay_stream(X: np.ndarray, y: np.ndarray, blocks: list[tuple[int, int]], order: list[int]):
    """CONSTRUCTED recurrence: concatenate chronological blocks of a real stream in a chosen order (e.g. [0, 1, 0]).

    Each block keeps its internal chronological order, but the repetition is an experimental construction, not a naturally
    recurring concept; results must be described as such. Returns (X, y, truth) with ``truth[i]`` = block index.
    """
    parts = [(X[a:b], y[a:b], [k] * (b - a)) for k in order for a, b in [blocks[k]]]
    return (np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), [k for p in parts for k in p[2]])


def load_arff(path: str, target: str, drop: tuple[str, ...] = (), max_rows: int | None = None):
    """Minimal ARFF reader (numeric and nominal attributes; '?' = missing). Returns (X, y, names, nominal).

    Nominal attributes become integer codes in declaration order (so the codes of ``DayOfWeek`` {1..7} are 0..6, and the codes of a
    high-cardinality attribute such as an airport carry no ordering meaning). The target must be nominal; its codes are 0/1/...
    ``nominal`` lists which kept columns are nominal.
    """
    attrs: list[tuple[str, list[str] | None]] = []
    rows: list[list[str]] = []
    in_data = False
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                body = line.split(None, 1)[1].strip()
                if body.startswith(("'", '"')):
                    q = body[0]
                    end = body.index(q, 1)
                    name, rest = body[1:end], body[end + 1:].strip()
                else:
                    name, rest = body.split(None, 1)
                if rest.startswith("{"):
                    vals = [v.strip().strip("'\"") for v in rest.strip("{}").split(",")]
                    attrs.append((name, vals))
                else:
                    attrs.append((name, None))
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                rows.append([c.strip().strip("'\"") for c in line.split(",")])
                if max_rows and len(rows) >= max_rows:
                    break
    names = [a for a, _ in attrs]
    if target not in names:
        raise ValueError(f"target {target!r} not in {names}")
    maps = [({v: i for i, v in enumerate(vals)} if vals else None) for _, vals in attrs]
    def conv(raw, m):
        if raw == "?":
            return np.nan
        return float(m[raw]) if m is not None else float(raw)
    M = np.array([[conv(r[i], maps[i]) for i in range(len(attrs))] for r in rows], float)
    ti = names.index(target)
    keep = [i for i, n in enumerate(names) if n != target and n not in drop]
    y = M[:, ti].astype(int)
    return M[:, keep], y, [names[i] for i in keep], [names[i] for i in keep if maps[i] is not None]


class ArffStream:
    """Iterate an ARFF file as a stream of (t, x, y, "data"). Nominal columns with more than ``max_levels`` observed values
    (e.g. airport codes, whose integer codes carry no ordering) are left out unless ``features`` names them explicitly."""

    def __init__(self, path: str, target: str, limit: int | None = None, features: list[str] | None = None, drop: tuple[str, ...] = (),
                 loops: int = 0, max_levels: int = 30):
        X, y, names, nominal = load_arff(path, target, drop=drop, max_rows=limit)
        if features:
            keep = [names.index(f) for f in features]
        else:
            keep = [i for i, n in enumerate(names) if not (n in nominal and len(np.unique(X[:, i][np.isfinite(X[:, i])])) > max_levels)]
        self.X, self.y, self.names, self.loops = X[:, keep], y, [names[i] for i in keep], loops
        self.n_features = self.X.shape[1]
        self.change_points = []

    def __len__(self) -> int:
        return len(self.y)

    def __iter__(self):
        t, k = 0, 0
        while self.loops == 0 or k < self.loops:
            for i in range(len(self.y)):
                yield t, self.X[i], int(self.y[i]), "data"
                t += 1
            k += 1
