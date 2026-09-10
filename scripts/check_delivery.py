"""Exercise the actual WMS demo receiver with every saved measurement."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from ozon_dimensioner.outbox import Outbox, http_sender

ROOT = Path(__file__).resolve().parents[1]


def run():
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/wms_demo_server.py"), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        line = server.stdout.readline().strip()
        assert line.startswith("WMS demo: http://127.0.0.1:"), line
        url = line.removeprefix("WMS demo: ")
        payloads = json.loads((ROOT / "results/measurements.json").read_text())
        with tempfile.TemporaryDirectory(prefix="ozon-wms-") as temp:
            outbox = Outbox(Path(temp) / "queue.sqlite")
            try:
                for payload in payloads:
                    outbox.enqueue(payload)
                first = {
                    "delivered": outbox.flush(http_sender(url)),
                    "pending": outbox.pending(),
                }
                second = {
                    "delivered": outbox.flush(http_sender(url)),
                    "pending": outbox.pending(),
                }
            finally:
                outbox.close()

        def post(payload):
            return urlopen(
                Request(
                    url,
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Idempotency-Key": payload["measurement_id"],
                    },
                    method="POST",
                ),
                timeout=2,
            )

        with post(payloads[0]) as response:
            duplicate = json.load(response)["duplicate"]
        try:
            with post({**payloads[0], "status": "conflict"}):
                raise AssertionError("Conflicting content accepted")
        except HTTPError as error:
            conflict_status = error.code
        assert first == {"delivered": len(payloads), "pending": 0}
        assert second == {"delivered": 0, "pending": 0}
        assert duplicate and conflict_status == 409
        result = {
            "first_delivery": first,
            "repeat_delivery": second,
            "receiver_duplicate_confirmed": duplicate,
            "conflict_status": conflict_status,
            "receiver": "scripts/wms_demo_server.py on ephemeral localhost port",
            "passed": True,
        }
        (ROOT / "results/integration_demo.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))
    finally:
        server.terminate()
        server.wait(timeout=5)
        server.stdout.close()


if __name__ == "__main__":
    run()
