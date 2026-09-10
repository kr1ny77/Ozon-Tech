import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from ozon_dimensioner.outbox import Outbox, http_sender


def test_retry_and_restart(tmp_path):
    path = tmp_path / "outbox.sqlite"
    b = Outbox(path)
    b.enqueue({"measurement_id": "a", "status": "ok"})

    def fail(*args):
        raise TimeoutError()

    assert b.flush(fail) == 0 and b.pending() == 1
    b.close()
    b = Outbox(path)
    assert b.flush(lambda p, k: True) == 1
    assert b.pending() == 0
    assert b.flush(lambda p, k: True) == 0
    b.close()


def test_duplicate_and_conflict(tmp_path):
    b = Outbox(tmp_path / "outbox.sqlite")
    p = {"measurement_id": "a", "status": "ok"}
    b.enqueue(p)
    b.enqueue(p)
    assert b.pending() == 1
    with pytest.raises(ValueError):
        b.enqueue({**p, "status": "review"})
    assert b.flush(lambda payload, k: payload == p) == 1
    b.close()


def test_http_roundtrip(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.headers["Idempotency-Key"], p))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    b = Outbox(tmp_path / "outbox.sqlite")
    try:
        p = {"measurement_id": "http-1", "status": "review"}
        b.enqueue(p)
        assert (
            b.flush(http_sender(f"http://127.0.0.1:{server.server_port}/measurements"))
            == 1
        )
        assert received == [("http-1", p)]
    finally:
        b.close()
        server.shutdown()
        server.server_close()
        thread.join()
