"""Layer 2: in-process, Kafka-style message bus.

Topics are append-only logs with offsets. Consumers belong to groups and each group
keeps its own committed offset per topic, so any number of independent readers can
replay the same topic at their own speed. Producers go through a ``Sender`` that
queues records and drops them (counted) instead of blocking the caller.
"""
from __future__ import annotations

import queue
import threading
from collections import namedtuple

Record = namedtuple("Record", "topic offset value")


class Topic:
    def __init__(self, name: str, retention: int = 200_000):
        self.name = name
        self.retention = retention
        self.base = 0                  # offset of log[0]
        self.log: list = []

    @property
    def end(self) -> int:
        return self.base + len(self.log)

    def append(self, value) -> int:
        self.log.append(value)
        if len(self.log) > self.retention:
            cut = len(self.log) - self.retention
            del self.log[:cut]
            self.base += cut
        return self.end - 1


class Bus:
    def __init__(self):
        self.topics: dict[str, Topic] = {}
        self.committed: dict[tuple[str, str], int] = {}   # (group, topic) -> next offset
        self.cond = threading.Condition()
        self.dropped = 0
        self.sent = 0

    def create_topic(self, name: str, retention: int = 200_000) -> Topic:
        with self.cond:
            return self.topics.setdefault(name, Topic(name, retention))

    def publish(self, topic: str, value) -> int:
        with self.cond:
            off = self.topics[topic].append(value)
            self.sent += 1
            self.cond.notify_all()
        return off

    def consumer(self, group: str, topics: list[str]) -> "Consumer":
        return Consumer(self, group, topics)

    def sender(self, capacity: int = 10_000) -> "Sender":
        return Sender(self, capacity)

    def stats(self) -> dict:
        with self.cond:
            topics = {n: {"end": t.end, "base": t.base} for n, t in self.topics.items()}
            groups: dict[str, dict] = {}
            for (g, tn), off in self.committed.items():
                groups.setdefault(g, {})[tn] = max(0, self.topics[tn].end - off)
        return {"topics": topics, "lag": groups, "sent": self.sent, "dropped": self.dropped}


class Consumer:
    def __init__(self, bus: Bus, group: str, topics: list[str]):
        self.bus, self.group, self.topics = bus, group, list(topics)
        with bus.cond:
            for t in self.topics:
                bus.committed.setdefault((group, t), bus.topics[t].base)

    def poll(self, timeout: float = 0.2, max_records: int = 500) -> list[Record]:
        bus = self.bus
        with bus.cond:
            out = self._collect(max_records)
            if not out:
                bus.cond.wait(timeout)
                out = self._collect(max_records)
        return out

    def _collect(self, limit: int) -> list[Record]:
        out: list[Record] = []
        for tn in self.topics:
            topic = self.bus.topics[tn]
            key = (self.group, tn)
            off = max(self.bus.committed[key], topic.base)     # skip records lost to retention
            take = min(limit - len(out), topic.end - off)
            if take > 0:
                chunk = topic.log[off - topic.base: off - topic.base + take]
                out.extend(Record(tn, off + i, v) for i, v in enumerate(chunk))
                self.bus.committed[key] = off + take
            if len(out) >= limit:
                break
        return out


class Sender:
    """producer.Sender: bounded queue + background thread; drops instead of blocking."""

    def __init__(self, bus: Bus, capacity: int):
        self.bus = bus
        self.q: queue.Queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="bus-sender")
        self.thread.start()

    def send(self, topic: str, value) -> bool:
        try:
            self.q.put_nowait((topic, value))
            return True
        except queue.Full:
            with self.bus.cond:
                self.bus.dropped += 1
            return False

    def _run(self):
        while not self._stop.is_set() or not self.q.empty():
            try:
                topic, value = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.bus.publish(topic, value)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=2)
