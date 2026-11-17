"""The worker's direct port (worker/direct.py and its use in worker/serve.py), without a GPU and without Runpod.

The unit tests cover the certificate, its pinning and the token check. The other tests run the bundled serve.py on the
CPU with RS_DIRECT set, against stand-ins for JAX, openpi's policy and nvidia-smi: the fake policy answers with the
observation's first state value. A small TCP forwarder plays the public port that Runpod maps to the container port,
and the real client connects to the worker's plain port as it would through the gateway.
"""

import contextlib
import hashlib
import json
import os
import pathlib
import queue
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect as ws_connect

from playground_serve import bundle, example_obs, models
from playground_serve.client import connect
from playground_serve.client.codec import unpackb
from playground_serve.worker import direct

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command")
TESTS = pathlib.Path(__file__).resolve().parent


def pinned(pem: str) -> ssl.SSLContext:
    """A client TLS context that trusts exactly `pem` and checks no host name, as the client builds it."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.load_verify_locations(cadata=pem)
    return context


def fingerprint(pem: str) -> str:
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()


def handshake(server: ssl.SSLContext, client: ssl.SSLContext) -> bytes:
    """TLS between the two contexts over a socket pair; the server certificate (DER) the client accepted."""
    a, b = socket.socketpair()
    a.settimeout(10)
    b.settimeout(10)

    def serve():
        with contextlib.suppress(OSError):  # the client refused the certificate
            server.wrap_socket(a, server_side=True).close()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        with client.wrap_socket(b) as tls:
            return tls.getpeercert(binary_form=True)
    finally:
        b.close()
        thread.join()
        a.close()


def test_certificate_is_new_at_every_boot_and_works_only_where_pinned():
    pem, context = direct.certificate()
    other, _ = direct.certificate()
    assert pem.startswith("-----BEGIN CERTIFICATE-----") and "PRIVATE KEY" not in pem  # what the metadata carries
    assert fingerprint(pem) != fingerprint(other)
    assert hashlib.sha256(handshake(context, pinned(pem))).hexdigest() == fingerprint(pem)
    with pytest.raises(ssl.SSLCertVerificationError):
        handshake(context, pinned(other))
    public = ssl.create_default_context()
    public.check_hostname = False
    with pytest.raises(ssl.SSLCertVerificationError):  # no public authority signed it
        handshake(context, public)


def test_token_check_accepts_only_the_bearer_token():
    env = {"RS_DIRECT": "8002", "RUNPOD_PUBLIC_IP": "127.0.0.1", "RUNPOD_TCP_PORT_8002": "48245"}
    token = direct.setup(env)[0]["token"]
    assert len(token) >= 43 and token != direct.setup(env)[0]["token"]  # 32 random bytes, new at every boot
    assert direct.allowed(f"Bearer {token}", token)
    for header in (None, "", "Bearer", "Bearer ", token, f"bearer {token}", f"Basic {token}", f"Bearer  {token}",
                   f"Bearer {token[:-1]}", f"Bearer {token}x", f"Bearer {token} ", "Bearer x"):
        assert not direct.allowed(header, token), header
    # Bytes outside ASCII: websockets 13 decodes them as surrogates, later versions as Latin-1. Refused, not an error.
    for header in (f"Bearer {token[:-1]}\udcff", f"Bearer {token[:-1]}\xff"):
        assert not direct.allowed(header, token)


class PortMap:
    """Runpod's public port mapped to the worker's container port. mode "open" forwards each connection; "refuse"
    closes it at once; "drop" accepts it and never answers, as a network that drops the packets."""

    def __init__(self, target: int):
        self.target, self.mode, self.connections = target, "open", 0
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._listener.settimeout(0.2)
        self.port = self._listener.getsockname()[1]
        self._held, self._closed = [], False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.connections += 1
            if self.mode == "refuse":
                conn.close()
            elif self.mode == "drop":
                self._held.append(conn)
            else:
                upstream = socket.create_connection(("127.0.0.1", self.target))
                for src, dst in ((conn, upstream), (upstream, conn)):
                    threading.Thread(target=_pipe, args=(src, dst), daemon=True).start()

    def close(self):
        self._closed = True
        self._listener.close()
        for conn in self._held:
            conn.close()


def _pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            with contextlib.suppress(OSError):
                s.shutdown(socket.SHUT_RDWR)


# Stand-ins: serve.py imports jax, openpi's server and policies, and scripts/serve_policy.py from the openpi checkout.
STUBS = {
    "jax/__init__.py": '__version__ = "stub"\n\n\ndef devices():\n    return ["cpu"]\n',
    "openpi/__init__.py": "",
    "openpi/serving/__init__.py": "",
    "openpi/policies/__init__.py": '''"""openpi's robot input transforms and examples, for warmup.py: LIBERO only."""
import types

import numpy as np


class LiberoInputs:
    pass


def make_libero_example():
    return {"observation/state": np.random.rand(8), "prompt": "do something",
            "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)}


libero_policy = types.SimpleNamespace(LiberoInputs=LiberoInputs, make_libero_example=make_libero_example)
droid_policy = types.SimpleNamespace(DroidInputs=type("DroidInputs", (), {}), make_droid_example=None)
aloha_policy = types.SimpleNamespace(AlohaInputs=type("AlohaInputs", (), {}), make_aloha_example=None)
''',
}
FAKE_SERVE_POLICY = '''"""Stand-in for openpi's scripts/serve_policy.py: a policy that needs both images, decoded, and answers with the first
state value in every action."""
import dataclasses
import types

import numpy as np

from openpi.policies import libero_policy


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    port: int = 8000
    policy: Checkpoint = None


class _Policy:
    metadata = {"policy": "fake"}
    _input_transform = types.SimpleNamespace(transforms=[libero_policy.LiberoInputs()])

    def infer(self, obs):
        for key in ("observation/image", "observation/wrist_image"):
            assert obs[key].dtype == np.uint8 and obs[key].shape == (224, 224, 3), key
        return {"actions": np.full((10, 7), np.asarray(obs["observation/state"])[0], np.float32)}


def create_policy(args):
    return _Policy()
'''


@pytest.fixture(scope="module")
def observations():
    return [example_obs(seed) for seed in range(4)]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    """The bundled serve.py with the direct port on, ready. Yields the gateway URL, the PortMap, the log records by
    phase, the output lines, and `read`, which takes in the output since."""
    tmp = tmp_path_factory.mktemp("worker")
    rs, checkout, stubs, bin_dir = tmp / "rs", tmp / "openpi", tmp / "stubs", tmp / "bin"
    for name, data in bundle.unpack(bundle.endpoint_env(models.get("pi05_libero"), "test")["RS_FILES"]).items():
        (rs / name).parent.mkdir(parents=True, exist_ok=True)
        (rs / name).write_bytes(data)
    stub_files = {**STUBS, "openpi/serving/websocket_policy_server.py":
                  (TESTS / "client" / "openpi_websocket_policy_server.py").read_text()}
    for name, text in {**{f"stubs/{k}": v for k, v in stub_files.items()},
                       "openpi/scripts/serve_policy.py": FAKE_SERVE_POLICY,
                       "bin/nvidia-smi": '#!/bin/sh\necho "NVIDIA GeForce RTX 4090, GPU-test, 595.91.07, 24564 MiB"\n'
                       }.items():
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp / name).write_text(text)
    (bin_dir / "nvidia-smi").chmod(0o755)
    gateway, container = _free_port(), _free_port()
    mapped = PortMap(container)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RS_", "RUNPOD_", "JAX_"))}
    env.update(RS_MODEL="pi05_libero", RS_OPENPI_CONFIG="pi05_libero", RS_CHECKPOINT="gs://test/pi05_libero",
               RS_RELEASE_ID="test", PORT=str(gateway), RS_DIRECT=str(container), RUNPOD_PUBLIC_IP="127.0.0.1",
               **{f"RUNPOD_TCP_PORT_{container}": str(mapped.port)}, PYTHONPATH=str(stubs),
               PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}")
    had_ready = os.path.exists("/tmp/rs_ready")  # serve.py touches it, as on a worker
    proc = subprocess.Popen([sys.executable, "-u", str(rs / "serve.py")], cwd=checkout, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    log, output, deadline = {}, [], time.time() + 120

    def read():
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            output.append(line)
            if line.startswith('{"rs"'):  # print writes the line's end separately: a stderr line may follow at once
                record = json.JSONDecoder().raw_decode(line)[0]
                log[record["rs"]] = record

    try:
        while "selftest_ws" not in log and time.time() < deadline and proc.poll() is None:
            time.sleep(0.05)
            read()
        assert "selftest_ws" in log, "".join(output[-40:])
        yield f"ws://127.0.0.1:{gateway}", mapped, log, output, read
    finally:
        proc.kill()
        proc.wait()
        mapped.close()
        if not had_ready and os.path.exists("/tmp/rs_ready"):
            os.remove("/tmp/rs_ready")


def test_worker_hands_the_direct_port_over_through_the_gateway_socket(worker):
    url, mapped, log, output, read = worker
    with ws_connect(url + "/ws", compression=None, max_size=None, open_timeout=10) as ws:  # no token on this port
        info = unpackb(ws.recv(timeout=10))["playground_serve"]["direct"]
    assert set(info) == {"host", "port", "cert", "token"}
    assert (info["host"], info["port"]) == ("127.0.0.1", mapped.port)  # the public port, not the container port
    with socket.create_connection(("127.0.0.1", mapped.port), timeout=10) as raw:
        with pinned(info["cert"]).wrap_socket(raw) as tls:  # the port serves the certificate the metadata carries
            assert hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest() == fingerprint(info["cert"])
    read()
    assert log["direct"]["port"] == mapped.port and log["direct"]["container_port"] != mapped.port
    selftest = log["selftest_ws"]["metadata"]  # the worker logs its metadata without the direct map
    assert "'playground_serve': {'jpeg': True, 'noop': True, 'hot': True, 'path': 'stock'" in selftest
    assert "direct" not in selftest


@pytest.mark.parametrize("token", ["missing", "wrong", "right"])
def test_worker_direct_port_lets_in_only_the_token(worker, token):
    url, mapped, *_ = worker
    with ws_connect(url, open_timeout=10) as ws:
        info = unpackb(ws.recv(timeout=10))["playground_serve"]["direct"]
    target = f"wss://127.0.0.1:{mapped.port}/ws"
    headers = {"missing": None, "wrong": {"Authorization": "Bearer not-the-token"},
               "right": {"Authorization": f"Bearer {info['token']}"}}[token]
    if token == "right":
        with ws_connect(target, ssl=pinned(info["cert"]), additional_headers=headers, open_timeout=10) as ws:
            assert unpackb(ws.recv(timeout=10))["playground_serve"]["direct"]["port"] == mapped.port
    else:
        with pytest.raises(InvalidStatus) as e:
            ws_connect(target, ssl=pinned(info["cert"]), additional_headers=headers, open_timeout=10)
        assert e.value.response.status_code == 401


@pytest.mark.parametrize("hot", [False, True])
def test_round_trip_through_the_direct_port(worker, observations, hot):
    url, mapped, log, output, read = worker
    read()
    before, seen = mapped.connections, len(output)
    with connect(url, hot=hot) as client:
        assert client._direct is not None and client._aux is not None  # split requests follow the direct path
        for obs in observations:
            actions = client.infer(obs)["actions"]
            assert client.timing["path"] == "direct" and actions.shape == (10, 7)
            assert actions[0, 0] == pytest.approx(obs["observation/state"][0])  # this observation, both images in it
    assert mapped.connections - before == 2  # the direct connection and its second socket
    read()
    opened = [line for line in output[seen:] if '"rs": "ws_open"' in line]
    assert len(opened) == 3 and all(("rs_hot=1" in line) == hot for line in opened)  # gateway, direct, second socket


@pytest.mark.parametrize("mode", ["refuse", "drop"])
def test_client_stays_on_the_gateway_when_the_direct_port_is_unreachable(worker, observations, mode, caplog):
    url, mapped, *_ = worker
    mapped.mode = mode
    try:
        t = time.monotonic()
        with connect(url, open_timeout=2) as client:
            assert time.monotonic() - t < 6 and client._direct is None and "no direct connection" in caplog.text
            for obs in observations[:2]:
                actions = client.infer(obs)["actions"]
                assert client.timing["path"] == "gateway"
                assert actions[0, 0] == pytest.approx(obs["observation/state"][0])
    finally:
        mapped.mode = "open"


def test_worker_log_never_holds_the_token(worker):
    url, mapped, log, output, read = worker  # after the tests above, when the module runs whole
    with ws_connect(url, open_timeout=10) as ws:
        token = unpackb(ws.recv(timeout=10))["playground_serve"]["direct"]["token"]
    read()
    assert "ready" in log and not any(token in line for line in output)


def test_worker_rejects_object_array_without_crashing(worker):
    import msgpack

    url, *_ = worker
    with ws_connect(url, open_timeout=10) as ws:
        ws.recv(timeout=10)
        ws.send(msgpack.packb({"observation/state": {b"__ndarray__": True, b"dtype": "O",
                                                    b"shape": [1], b"data": bytes(8)}}))
        assert "unsupported wire dtype" in ws.recv(timeout=10)
    # The same worker remains available to other clients after rejecting the payload.
    with connect(url, keepalive=None) as client:
        assert client.infer(example_obs())["actions"].shape == (10, 7)
