"""Layer 6: dashboard. Stdlib HTTP server; the page polls /api/state."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pipeline import Pipeline

WEB = os.path.join(os.path.dirname(__file__), "web")


def make_handler(pipe: Pipeline):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                self._send(json.dumps(pipe.snapshot(), default=float).encode(), "application/json")
            elif self.path in ("/", "/index.html"):
                with open(os.path.join(WEB, "index.html"), "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            else:
                self.send_error(404)
    return H


def serve(rate: int = 600, port: int = 8000, seed: int = 7, loops: int = 0, arff: str | None = None, target: str | None = None,
          limit: int | None = None, features: list[str] | None = None):
    stream, nf = None, 4
    if arff:
        from .datasets import ArffStream
        stream = ArffStream(arff, target or "class", limit=limit, features=features, loops=loops)
        nf = stream.n_features
        print(f"data source: {arff}  target={target}  rows={len(stream)}  features={stream.names}")
    pipe = Pipeline(rate=rate, seed=seed, loops=loops, stream=stream, n_features=nf)
    pipe.start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(pipe))
    print(f"DejaStream dashboard on http://127.0.0.1:{port}  ({rate} samples/s)  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()
