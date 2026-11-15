"""The direct port (`pgs up MODEL --direct`): the worker's server also listens on a TCP port that Runpod maps to the
worker's public address, so a client's requests skip Runpod's gateway. From the US east coast to US-CA-2 a 19.7 kB
request and its reply took 77 ms on that path against 91 ms through the gateway (docs/BENCHMARKS.md).

Standard library and the openssl command only. `setup()` runs once per boot: it reads the address Runpod gave the port
(RUNPOD_PUBLIC_IP, RUNPOD_TCP_PORT_<port>), makes a new self-signed certificate and a random token, and returns what
the metadata advertises: {"host", "port", "cert", "token"}. The metadata reaches a client only through the gateway,
which checks the Runpod API key, so: the client trusts exactly this certificate (TLS without a public CA) and sends
the token in the upgrade's Authorization header; `allowed` checks it. The certificate, its key and the token die with
the worker.
"""

from __future__ import annotations

import hmac
import os
import secrets
import ssl
import subprocess
import tempfile

PORT_ENV = "RS_DIRECT"  # the container port of the direct listener, set by `pgs up --direct`


def address(environ=os.environ):
    """(public host, public port, container port) of the direct port, or None when it is off or Runpod mapped none."""
    port = environ.get(PORT_ENV) or ""
    host, public = environ.get("RUNPOD_PUBLIC_IP"), environ.get(f"RUNPOD_TCP_PORT_{port}") or ""
    if not (port.isdigit() and host and public.isdigit()):
        return None
    return host, int(public), int(port)


def certificate():
    """(PEM text of a new self-signed certificate, a server-side TLS context holding it and its key). The key is
    written to a temporary directory only for the moment the context loads it."""
    with tempfile.TemporaryDirectory() as tmp:
        key, cert = os.path.join(tmp, "key.pem"), os.path.join(tmp, "cert.pem")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                        "-days", "365", "-subj", "/CN=playground-serve"], check=True, capture_output=True, timeout=60)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        with open(cert) as f:
            return f.read(), context


def setup(environ=os.environ):
    """(what the metadata advertises, the TLS context, the container port), or None when the direct port is off."""
    where = address(environ)
    if where is None:
        return None
    pem, context = certificate()
    return {"host": where[0], "port": where[1], "cert": pem, "token": secrets.token_urlsafe(32)}, context, where[2]


def allowed(header, token: str) -> bool:
    """Whether an upgrade's Authorization header carries the token. websockets 13 decodes header bytes outside ASCII
    as surrogates; surrogateescape turns them back into bytes, so such a header is refused instead of raising."""
    return hmac.compare_digest((header or "").encode("utf-8", "surrogateescape"), f"Bearer {token}".encode())
