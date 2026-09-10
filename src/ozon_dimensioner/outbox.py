"""Durable, idempotent outbox. Delivery is at-least-once."""

import json
import sqlite3
from urllib.request import Request, urlopen


class Outbox:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS outbox (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL,
            delivered INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0)""")
        self.db.commit()

    def close(self):
        self.db.close()

    def enqueue(self, payload):
        identifier = payload["measurement_id"]
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO outbox(id,payload) VALUES (?,?)",
                (identifier, encoded),
            )
            previous = self.db.execute(
                "SELECT payload FROM outbox WHERE id=?", (identifier,)
            ).fetchone()[0]
            if previous != encoded:
                raise ValueError("An existing measurement_id has different content")

    def pending(self):
        return self.db.execute(
            "SELECT COUNT(*) FROM outbox WHERE delivered=0"
        ).fetchone()[0]

    def flush(self, sender):
        delivered = 0
        for identifier, encoded in self.db.execute(
            "SELECT id,payload FROM outbox WHERE delivered=0 ORDER BY rowid"
        ).fetchall():
            with self.db:
                self.db.execute(
                    "UPDATE outbox SET attempts=attempts+1 WHERE id=?", (identifier,)
                )
            try:
                acknowledged = sender(json.loads(encoded), identifier)
            except (OSError, TimeoutError):
                continue
            if acknowledged:
                with self.db:
                    self.db.execute(
                        "UPDATE outbox SET delivered=1 WHERE id=?", (identifier,)
                    )
                delivered += 1
        return delivered


def http_sender(url, timeout=2.0):
    def send(payload, identifier):
        req = Request(
            url,
            data=json.dumps(payload, allow_nan=False).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "Idempotency-Key": identifier},
        )
        with urlopen(req, timeout=timeout) as response:
            return 200 <= response.status < 300

    return send
