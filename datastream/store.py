"""Layer 5: storage. SQLite stands in for Postgres: events, metrics and the concept repository."""
from __future__ import annotations

import json
import sqlite3
import threading


class Store:
    def __init__(self, path: str = "results/datastream.db", fresh: bool = True):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        if fresh:
            for tbl in ("events", "metrics", "concepts"):
                self.db.execute(f"DROP TABLE IF EXISTS {tbl}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events(t INT, strategy TEXT, kind TEXT, concept INT, dist REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS metrics(t INT, strategy TEXT, acc REAL, models INT);
            CREATE TABLE IF NOT EXISTS concepts(strategy TEXT, id INT, created_t INT, uses INT, d_in REAL,
                                                fingerprint TEXT, PRIMARY KEY(strategy, id));
        """)

    def add_event(self, e: dict):
        with self.lock:
            self.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?)",
                            (e["t"], e["strategy"], e["kind"], e.get("concept"), e.get("dist"), json.dumps(e)))

    def add_metric(self, m: dict):
        with self.lock:
            self.db.execute("INSERT INTO metrics VALUES(?,?,?,?)", (m["t"], m["strategy"], m["acc"], m["models"]))

    def save_concepts(self, strategy: str, concepts: list[dict]):
        with self.lock:
            for c in concepts:
                self.db.execute("INSERT OR REPLACE INTO concepts VALUES(?,?,?,?,?,?)",
                                (strategy, c["id"], c["created_t"], c["uses"], c["d_in"], json.dumps(c["z"])))

    def commit(self):
        with self.lock:
            self.db.commit()

    def query(self, sql: str, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()
