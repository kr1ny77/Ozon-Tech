"""Local in-memory WMS receiver for exercising idempotent delivery."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/measurements":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 1_000_000:
                    raise ValueError()
                payload = json.loads(self.rfile.read(size))
                key = self.headers.get("Idempotency-Key")
                if not key or key != payload["measurement_id"]:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
                return
            if key in received and received[key] != payload:
                self.send_error(409)
                return
            duplicate = key in received
            received[key] = payload
            body = json.dumps({"measurement_id": key, "duplicate": duplicate}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"WMS demo: http://127.0.0.1:{server.server_port}/measurements", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
