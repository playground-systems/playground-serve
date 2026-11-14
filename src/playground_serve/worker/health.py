"""Runpod load-balancer health responder: GET /ping on PORT_HEALTH answers 204 until /tmp/rs_ready exists, then 200.

It runs on the base image's own python3 from the first second of the container, before the openpi
environment is unpacked, so the load balancer sees "initializing" rather than a refused connection.
"""

import http.server
import os


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        code = 404 if self.path != "/ping" else 200 if os.path.exists("/tmp/rs_ready") else 204
        self.send_response(code)
        if code != 204:
            self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT_HEALTH", "8001"))), H).serve_forever()
