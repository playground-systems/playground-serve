"""Launcher tests against an in-memory fake of the Runpod API (no network, no real models/bundle modules)."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

import msgpack
import pytest

from playground_serve import cli, state
from playground_serve.launcher import RELEASE, RELEASE_KEY, Launcher, LauncherError, candidates, release_id
from playground_serve.runpod_api import ApiError

NAME = re.compile(r"playground-serve-o[0-9a-f]{8}-pi05-libero-[0-9a-f]{6}")
DEFAULT = {"min": 0, "max": 1, "idleTimeout": 300}
WARM = {"min": 1, "max": 1, "idleTimeout": 300}
DOWN = {"min": 0, "max": 1, "idleTimeout": 5}


@dataclass(frozen=True)
class FakeSpec:  # same fields as playground_serve.models.ModelSpec
    name: str = "pi05_libero"
    openpi_config: str = "pi05_libero"
    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    volume_gb: int = 40
    min_container_ram_gb: int = 20
    gpu_pools: list = field(default_factory=lambda: ["ADA_24", "AMPERE_48"])
    image: str = "runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404"


FILES = ["files-v1"]
START = ["bash", "-c", "unpack RS_FILES && exec bash /rs/boot.sh"]  # like playground_serve.bundle.START_COMMAND


def make_env(spec, release):  # same signature as playground_serve.bundle.endpoint_env
    return {"RS_FILES": FILES[0], "PORT": "8000", "PORT_HEALTH": "8001", "RS_RELEASE": release,
            "OPENPI_DATA_HOME": "/runpod-volume/openpi_data"}


def get_spec(name):
    if name != "pi05_libero":
        raise KeyError(name)
    return FakeSpec()


def released(env):  # the env map `up` sends: with the release id of the rest
    return {**env, RELEASE_KEY: release_id(env)}


GPUS = [
    {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "pool": "ADA_24", "minRam": 31,
     "price": {"serverless": 1.34},
     "dataCenters": [{"id": "US-TX-3", "availability": "LOW"}, {"id": "US-IL-1", "availability": "HIGH"},
                     {"id": "EU-NOVOL", "availability": "HIGH"}]},
    {"id": "SMALL-RAM", "name": "small", "pool": "ADA_24", "minRam": 15, "price": {"serverless": 1.34},
     "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"}]},
    {"id": "NVIDIA A40", "name": "A40", "pool": "AMPERE_48", "minRam": 50, "price": {"serverless": 1.22},
     "dataCenters": [{"id": "CA-MTL-1", "availability": "HIGH"}, {"id": "EU-SE-1", "availability": "NONE"}]},
    {"id": "NVIDIA H100", "name": "H100", "pool": "ADA_80_PRO", "minRam": 188, "price": {"serverless": 5.89},
     "dataCenters": [{"id": "US-IL-1", "availability": "HIGH"}]},
]
DCS = [{"id": "US-IL-1", "networkVolumeTypes": ["STANDARD"]},
       {"id": "US-TX-3", "networkVolumeTypes": ["STANDARD", "HIGH_PERFORMANCE"]},
       {"id": "EU-NOVOL", "networkVolumeTypes": []},
       {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"]},
       {"id": "CA-MTL-1", "networkVolumeTypes": ["HIGH_PERFORMANCE"]},
       {"id": "EU-SE-1", "networkVolumeTypes": ["STANDARD"]}]


class FakeWs:  # first the worker's metadata frame (`meta`), then an echo of each message, as a probe does
    socket = type("Sock", (), {"setsockopt": lambda self, *a: None})()

    def __init__(self, meta=None):
        self._msg = msgpack.packb(meta or {})

    def send(self, msg):
        self._msg = msg

    def recv(self, timeout=None):
        return self._msg

    def close(self):
        pass


class FakeApi:
    def __init__(self):
        self.gpus, self.dcs = copy.deepcopy(GPUS), copy.deepcopy(DCS)
        self.endpoints, self.volumes, self.calls = {}, {}, []
        self.refuse = set()  # (pool, dc) pairs whose endpoint create is refused
        self.refuse_volume_dcs = set()
        self.endpoint_errors = []  # raised, in order, by the next create_endpoint calls
        self.gateway = []  # status codes returned by gateway_status, then 200
        self.logs = []
        self.woken = set()  # endpoints whose worker a gateway request started (min 0)
        self.rtt = {}  # data center -> round trip (ms) its probe measures; None or missing: no answer
        self.served = {}  # endpoint -> the release id its running worker reports (default: its env's); max 0 clears it
        self.n = 0

    def _id(self, p):
        self.n += 1
        return f"{p}{self.n}"

    def list_endpoints(self):
        return copy.deepcopy(list(self.endpoints.values()))

    def list_volumes(self):
        return copy.deepcopy(list(self.volumes.values()))

    def get_endpoint(self, eid):
        if eid not in self.endpoints:
            raise ApiError(404, "GET", f"/serverless/{eid}", "endpoint not found")
        return copy.deepcopy(self.endpoints[eid])

    def get_volume(self, vid):
        if vid not in self.volumes:
            raise ApiError(404, "GET", f"/network-volumes/{vid}", "network volume not found")
        return copy.deepcopy(self.volumes[vid])

    def serverless_gpus(self, min_cuda=None):
        return copy.deepcopy(self.gpus)

    def datacenters(self):
        return copy.deepcopy(self.dcs)

    def serverless_cpus(self):  # serverless CPU workers offered in US-IL-1 only: elsewhere a probe runs on a GPU
        return [{"id": "cpu3c", "dataCenters": [{"id": "US-IL-1", "availability": "HIGH"},
                                                {"id": "US-TX-3", "availability": "NONE"}]}]

    def create_volume(self, body):
        self.calls.append(("create_volume", copy.deepcopy(body)))
        if body["dataCenter"] in self.refuse_volume_dcs:
            raise ApiError(400, "POST", "/network-volumes", "data center is out of storage")
        vol = {"id": self._id("vol"), "name": body["name"], "size": body["size"], "dataCenter": body["dataCenter"],
               "type": body.get("type", "STANDARD")}
        self.volumes[vol["id"]] = vol
        return copy.deepcopy(vol)

    def delete_volume(self, vid):
        self.calls.append(("delete_volume", vid))
        del self.volumes[vid]

    def create_endpoint(self, body):
        if "-probe-" in body["name"]:  # region probes: recorded apart from the model's resources
            self.calls.append(("probe_create", body["dataCenterIds"][0]))
            if sum(e["workers"]["max"] for e in self.endpoints.values()) + body["workers"]["max"] > 5:
                raise ApiError(400, "POST", "/serverless", "max workers across all endpoints must not exceed the "
                               "account's quota of 5")
            eid = f"probe{len(self.probes())}"  # a counter of its own: the model's resource ids stay as they were
            self.endpoints[eid] = {**copy.deepcopy(body), "id": eid}
            return copy.deepcopy(self.endpoints[eid])
        self.calls.append(("create_endpoint", copy.deepcopy(body)))
        if self.endpoint_errors:
            raise self.endpoint_errors.pop(0)
        if (body["gpu"]["pools"][0], body["dataCenterIds"][0]) in self.refuse:
            raise ApiError(400, "POST", "/serverless", "no capacity")
        eid = self._id("ep")
        ep = {**copy.deepcopy(body), "id": eid, "createdAt": f"2026-11-13T00:00:{self.n:02d}Z",
              "requestUrls": {"base": f"https://{eid}.api.runpod.ai", "health": f"https://{eid}.api.runpod.ai/ping"}}
        self.endpoints[eid] = ep
        return copy.deepcopy(ep)

    def update_endpoint(self, eid, body):
        self.calls.append(("update_endpoint", eid, copy.deepcopy(body)))
        self.endpoints[eid].update(copy.deepcopy(body))  # like Runpod: env (and every field sent) is replaced whole
        if (body.get("workers") or {}).get("max") == 0:  # every worker stops
            self.served.pop(eid, None)
            self.woken.discard(eid)
        return copy.deepcopy(self.endpoints[eid])

    def delete_endpoint(self, eid):
        self.calls.append(("probe_delete" if "-probe-" in self.endpoints[eid]["name"] else "delete_endpoint", eid))
        del self.endpoints[eid]

    def websocket(self, url, timeout=10.0):
        eid = url.split("//")[1].split(".")[0]
        release = self.served.get(eid, (self.endpoints[eid].get("env") or {}).get(RELEASE_KEY))
        return FakeWs({"playground_serve": {"jpeg": True, "release": release}})

    def measure(self, api, url, until, **kw):  # Launcher(measure=...): the round trip of the probe at `url`
        return self.rtt.get(self.endpoints[url.split("//")[1].split(".")[0]]["dataCenterIds"][0])

    def probes(self):
        return [c for c in self.calls if c[0].startswith("probe_")]

    def endpoint_workers(self, eid):
        if self.endpoints[eid]["workers"]["max"] == 0:
            return {"workers": []}
        up = self.endpoints[eid]["workers"]["min"] > 0 or eid in self.woken
        extra = [{"id": "w2", "status": "IDLE", "startedAt": "2026-11-13T00:00:09Z", "isStale": False}]
        return {"workers": [{"id": "w1", "status": "RUNNING", "startedAt": "2026-11-13T00:00:00Z",
                             "isStale": True}] + extra if up else extra}  # Runpod lists extra workers beyond max

    def worker_log_lines(self, eid, wid, since=None, wait=3.0, tail=300):
        self.calls.append(("logs", wid))
        return [self._boot_release(eid, line) for line in (self.logs.pop(0) if self.logs else [])]

    def _boot_release(self, eid, line):  # boot.sh names the release its env set in the boot line
        if line == json.dumps({"rs": "boot"}):
            return json.dumps({"rs": "boot", "release_id": self.served.get(eid, self.endpoints[eid]["env"][RELEASE_KEY])})
        return line

    def gateway_status(self, url, timeout=10.0):
        code = self.gateway.pop(0) if self.gateway else 200
        if code not in (401, 403):  # a request the gateway accepts starts a worker
            self.woken.add(url.split("//")[1].split(".")[0])
        return code

    def creates(self):
        return [c for c in self.calls if c[0].startswith(("create", "update", "delete"))]


@pytest.fixture
def env(tmp_path):
    api, out, t = FakeApi(), [], [0.0]

    def launcher(confirm=None, path=tmp_path / "state.json"):
        return Launcher(api, path, get_spec=get_spec, make_env=make_env, start_command=START,
                        model_names=lambda: ["pi05_libero"], out=out.append, confirm=confirm,
                        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], poll=5,
                        measure=api.measure, now=lambda: 1_800_000_000 + t[0], here=lambda: None)

    FILES[0] = "files-v1"
    return api, launcher, out, tmp_path / "state.json"


def test_candidates_follow_pool_order_then_stock_and_need_volumes_and_ram():
    cands = candidates(GPUS, DCS, FakeSpec())
    assert [(c.pool, c.dc) for c in cands] == [("ADA_24", "US-IL-1"), ("ADA_24", "US-TX-3"), ("AMPERE_48", "CA-MTL-1")]
    assert cands[0].excluded == ["SMALL-RAM"] and cands[0].price == 1.34 and cands[2].price == 1.22
    assert [(c.pool, c.dc) for c in candidates(GPUS, DCS, FakeSpec(), only_dc="CA-MTL-1")] == [("AMPERE_48", "CA-MTL-1")]


def test_up_creates_volume_then_endpoint_and_prints_prices_first(env):
    api, launcher, out, path = env
    result = launcher().up("pi05_libero")
    kinds = [c[0] for c in api.creates()]
    assert kinds == ["create_volume", "create_endpoint"]
    vol_body, ep_body = api.creates()[0][1], api.creates()[1][1]
    assert vol_body["dataCenter"] == "US-IL-1" and vol_body["size"] == 40 and vol_body["type"] == "STANDARD"
    assert NAME.fullmatch(vol_body["name"]) and NAME.fullmatch(ep_body["name"])
    assert ep_body["type"] == "LOAD_BALANCER" and ep_body["dataCenterIds"] == ["US-IL-1"]
    assert ep_body["gpu"] == {"pools": ["ADA_24"], "count": 1, "minCudaVersion": "12.8", "excludedTypes": ["SMALL-RAM"]}
    assert ep_body["env"] == released(make_env(None, "0.1.0")) and ep_body["ports"] == ["8000/http", "8001/http"]
    assert ep_body["entrypoint"] == ["bash", "-c"] and ep_body["cmd"] == [START[2]]
    assert ep_body["workers"] == DEFAULT and ep_body["scaling"] == {"type": "REQUEST_COUNT", "requestCount": 4}
    assert ep_body["flashboot"] == "OFF" and vol_body["name"][:26] == ep_body["name"][:26]  # same owner id
    assert ep_body["mounts"] == {"network": [{"volumeId": result["volume"]["id"]}]}
    text = "\n".join(out)
    assert text.index("1.34 USD/hr") < text.index("Created volume") < text.index("Created endpoint")
    assert text.index("about 2.80 USD/month") < text.index("Created volume")
    assert "First start on this volume" in text
    assert text.index("Ready after") < text.index("now saving the environment and checkpoint to the volume")
    assert result["url"] == f"wss://{result['endpoint']['id']}.api.runpod.ai/ws" and f"Websocket: {result['url']}" in text
    assert 'from playground_serve import connect\npolicy = connect("pi05_libero")\nactions = policy.act(obs)' in text
    assert "only while a worker runs" in text and "stops by itself 5 min after the last request" in text
    assert [c[0] for c in api.creates()] == ["create_volume", "create_endpoint"]  # no workers PATCH after ready
    saved = state.load(path)
    assert saved["models"]["pi05_libero"]["endpoint"]["id"] == result["endpoint"]["id"]
    assert saved["models"]["pi05_libero"]["volume"]["ready"] is True and re.fullmatch(r"o[0-9a-f]{8}", saved["owner"])
    assert saved["owner"] in ep_body["name"]


def test_up_twice_creates_nothing_the_second_time(env):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero")
    n = len(api.creates())
    out.clear()
    second = launcher().up("pi05_libero")
    assert api.creates()[n:] == []
    assert second["endpoint"]["id"] == first["endpoint"]["id"] and second["url"] == first["url"]
    assert not any("First start" in line or "saving the environment" in line for line in out)


def test_refused_endpoint_moves_on_and_drops_the_fresh_volume(env):
    api, launcher, out, _ = env
    api.refuse.add(("ADA_24", "US-IL-1"))
    result = launcher().up("pi05_libero")
    steps = [(c[0], c[1]["dataCenter"] if c[0] == "create_volume" else
              c[1]["dataCenterIds"][0] if c[0] == "create_endpoint" else c[1]) for c in api.creates()]
    assert steps == [("create_volume", "US-IL-1"), ("create_endpoint", "US-IL-1"), ("delete_volume", "vol1"),
                     ("create_volume", "US-TX-3"), ("create_endpoint", "US-TX-3")]
    assert list(api.volumes) == [result["volume"]["id"]] and result["volume"]["dataCenter"] == "US-TX-3"


def test_refused_volume_and_pool_fall_through_to_next_pool(env):
    api, launcher, out, _ = env
    api.refuse_volume_dcs.update({"US-IL-1", "US-TX-3"})
    result = launcher().up("pi05_libero")
    assert result["endpoint"]["gpu"]["pools"] == ["AMPERE_48"] and result["volume"]["type"] == "HIGH_PERFORMANCE"
    assert "1.22 USD/hr" in "\n".join(out)


def test_every_candidate_refused_leaves_nothing_behind(env):
    api, launcher, _, path = env
    api.refuse.update({("ADA_24", "US-IL-1"), ("ADA_24", "US-TX-3"), ("AMPERE_48", "CA-MTL-1")})
    with pytest.raises(LauncherError, match="refused every candidate"):
        launcher().up("pi05_libero")
    assert api.volumes == {} and api.endpoints == {} and state.load(path)["models"] == {}


def test_no_stock_anywhere_creates_nothing(env):
    api, launcher, _, _ = env
    for g in api.gpus:
        for d in g["dataCenters"]:
            d["availability"] = "NONE"
    with pytest.raises(LauncherError, match="No serverless stock"):
        launcher().up("pi05_libero")
    assert api.creates() == []


def test_template_name_collision_retries_with_a_new_suffix(env):
    api, launcher, out, _ = env
    api.endpoint_errors.append(ApiError(400, "POST", "/serverless", "endpoint template names must be unique. check "
                                        "that you do not already have an endpoint with a similar template."))
    launcher().up("pi05_libero")
    names = [c[1]["name"] for c in api.creates() if c[0] == "create_endpoint"]
    assert len(names) == 2 and names[0] != names[1] and all(NAME.fullmatch(n) for n in names)
    assert len(api.volumes) == 1 and len(api.endpoints) == 1


def test_create_lost_in_transit_is_found_by_name_not_created_twice(env):
    api, launcher, _, _ = env
    real = api.create_endpoint

    def flaky(body):
        real(body)
        raise ApiError(0, "POST", "/serverless", "network error: timed out")

    api.create_endpoint = flaky
    launcher().up("pi05_libero")
    assert len(api.endpoints) == 1


def test_down_then_up_sends_the_complete_env_map_and_workers_object(env):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero")
    eid = first["endpoint"]["id"]
    launcher().down("pi05_libero")
    assert api.calls[-1] == ("update_endpoint", eid, {"workers": DOWN})
    assert any("volume" in line and "kept" in line for line in out)
    FILES[0] = "files-v2"  # new worker files
    api.endpoints[eid]["env"].pop("OPENPI_DATA_HOME")  # and an env edited outside the tool
    launcher().up("pi05_libero")
    patches = [c[2] for c in api.calls if c[0] == "update_endpoint"][-2:]
    assert patches[0]["env"] == released(make_env(None, "0.1.0"))  # every key, not just the changed one
    assert "workers" not in patches[0]  # the readiness requests wake the worker; the workers object once it serves
    assert patches[1] == {"workers": DEFAULT}
    assert api.endpoints[eid]["env"]["OPENPI_DATA_HOME"] == "/runpod-volume/openpi_data"
    assert len([c for c in api.calls if c[0] == "create_endpoint"]) == 1



def test_after_a_release_change_up_waits_for_a_worker_of_the_new_release(env):
    # Runpod kept the old worker after the update (a client held it) and routed new connections to it: `up` reported
    # ready on the old release. Now the worker's metadata must name the release `up` just set; one that does not is
    # stopped (workers max 0) and a worker of the new release starts.
    api, launcher, out, _ = env
    eid = launcher().up("pi05_libero")["endpoint"]["id"]
    old = api.endpoints[eid]["env"][RELEASE_KEY]
    api.served[eid] = old
    FILES[0] = "files-v2"  # an upgrade: new worker files
    api.calls.clear()
    out.clear()
    launcher().up("pi05_libero")
    new = api.endpoints[eid]["env"][RELEASE_KEY]
    patches = [c[2] for c in api.calls if c[0] == "update_endpoint"]
    assert new != old and patches[0]["env"][RELEASE_KEY] == new
    assert patches[1:] == [{"workers": {**DEFAULT, "max": 0}}, {"workers": DEFAULT}]
    text = "\n".join(out)
    assert f"runs another release ({old}), not this one ({new})" in text
    assert text.index("runs another release") < text.index("Ready after")
    api.calls.clear()
    launcher().up("pi05_libero")  # the new release serves: nothing to stop
    assert [c for c in api.calls if c[0] == "update_endpoint"] == []


def test_without_gateway_access_the_boot_log_must_name_the_new_release(env):
    api, launcher, out, _ = env
    eid = launcher().up("pi05_libero")["endpoint"]["id"]
    api.served[eid] = api.endpoints[eid]["env"][RELEASE_KEY]
    api.gateway = [403]
    ready = [json.dumps({"rs": p}) for p in ("boot", "unpack_done", "ready")]
    api.logs = [ready, ready]
    api.calls.clear()
    FILES[0] = "files-v2"
    launcher().up("pi05_libero")
    patches = [c[2] for c in api.calls if c[0] == "update_endpoint"]
    assert patches[0]["env"]["RS_FILES"] == "files-v2"
    assert patches[1:] == [{"workers": WARM}, {"workers": {**DEFAULT, "max": 0}}, {"workers": WARM},
                           {"workers": DEFAULT}]
    assert "runs another release" in "\n".join(out) and sum(c[0] == "logs" for c in api.calls) == 2

def test_up_switches_pool_in_the_volume_data_center_when_stock_is_gone(env):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero")
    launcher().down()
    api.gpus[0]["dataCenters"] = [{"id": "US-IL-1", "availability": "NONE"}]
    api.gpus[2]["dataCenters"].append({"id": "US-IL-1", "availability": "LOW"})
    launcher().up("pi05_libero")
    patch = [c for c in api.calls if c[0] == "update_endpoint"][-2][2]
    assert patch["gpu"]["pools"] == ["AMPERE_48"] and api.endpoints[first["endpoint"]["id"]]["dataCenterIds"] == ["US-IL-1"]


def test_gpu_option_uses_that_pool_and_a_later_up_keeps_it(env):
    api, launcher, out, _ = env
    api.gpus.append({"id": "NVIDIA H200", "name": "H200 SXM", "pool": "HOPPER_141", "minRam": 188,
                     "price": {"serverless": 6.99}, "dataCenters": [{"id": "US-TX-3", "availability": "LOW"}]})
    first = launcher().up("pi05_libero", gpu="HOPPER_141", dc="US-TX-3")
    body = api.creates()[1][1]
    assert body["gpu"]["pools"] == ["HOPPER_141"] and body["dataCenterIds"] == ["US-TX-3"]
    assert "6.99 USD/hr" in "\n".join(out)
    launcher().down()
    launcher().up("pi05_libero")  # without --gpu: stays on the pool it was created with
    assert api.endpoints[first["endpoint"]["id"]]["gpu"]["pools"] == ["HOPPER_141"]
    assert not any("gpu" in c[2] for c in api.calls if c[0] == "update_endpoint")
    assert "No serverless stock" not in "\n".join(out)


def test_direct_option_adds_the_port_and_env_and_a_later_up_keeps_them(env):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero", direct=True)
    body = api.creates()[1][1]
    assert body["ports"] == ["8000/http", "8001/http", "8002/tcp"]
    assert body["env"] == released({**make_env(FakeSpec(), RELEASE), "RS_DIRECT": "8002"})
    assert any(line.startswith("Direct port: on") for line in out)
    launcher().down()
    calls = len(api.calls)
    launcher().up("pi05_libero")  # without --direct: the endpoint keeps its port and its release
    ep = api.endpoints[first["endpoint"]["id"]]
    assert ep["ports"][-1] == "8002/tcp" and ep["env"]["RS_DIRECT"] == "8002"
    assert not any("env" in c[2] or "ports" in c[2] for c in api.calls[calls:] if c[0] == "update_endpoint")


def test_without_direct_the_endpoint_is_as_before(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    body = api.creates()[1][1]
    assert body["ports"] == ["8000/http", "8001/http"] and "RS_DIRECT" not in body["env"]
    assert not any(line.startswith("Direct port") for line in out)


def test_endpoint_deleted_outside_the_tool_is_forgotten_and_recreated_on_the_same_volume(env):
    api, launcher, out, path = env
    first = launcher().up("pi05_libero")
    del api.endpoints[first["endpoint"]["id"]]
    out.clear()
    second = launcher().up("pi05_libero")
    assert any("no longer on the account" in line for line in out)
    assert second["endpoint"]["id"] != first["endpoint"]["id"]
    assert second["volume"]["id"] == first["volume"]["id"] and len(api.volumes) == 1
    assert state.load(path)["models"]["pi05_libero"]["endpoint"]["id"] == second["endpoint"]["id"]
    assert not any("First start" in line for line in out)  # the volume was already built


def test_lost_state_file_means_not_yours_until_adopted(env):
    """A new state has a new owner id: the old endpoint is not adopted by name any more; --adopt takes it over."""
    api, launcher, out, path = env
    first = launcher().up("pi05_libero")
    path.unlink()
    n = len(api.creates())
    with pytest.raises(LauncherError, match="--adopt.*--new"):
        launcher().up("pi05_libero")
    assert api.creates()[n:] == []
    second = launcher().up("pi05_libero", adopt=True)
    assert second["endpoint"]["id"] == first["endpoint"]["id"] and second["volume"]["id"] == first["volume"]["id"]
    assert [c[0] for c in api.creates()].count("create_endpoint") == 1
    saved = state.load(path)
    assert saved["models"]["pi05_libero"]["endpoint"]["id"] == first["endpoint"]["id"]
    assert saved["owner"] not in first["endpoint"]["name"]  # adopted by id, not renamed


def test_wait_without_gateway_access_follows_the_worker_log(env):
    api, launcher, out, _ = env
    api.gateway = [403]
    phase = lambda p: json.dumps({"rs": p})
    api.logs = [[phase("boot"), phase("build_start")], [phase("boot"), phase("build_start"), phase("build_done"),
                phase("imports_done")], [phase("boot"), phase("imports_done"), phase("weights_loaded"), phase("ready")]]
    launcher().up("pi05_libero")
    text = "\n".join(out)
    assert "cannot call the endpoint (gateway HTTP 403)" in text
    assert "building the Python environment" in text and "loading the checkpoint" in text
    assert re.search(r"\[\d+:\d\d\]", text) and "Ready after" in text


def test_gateway_426_from_the_websocket_server_counts_as_ready(env):
    api, launcher, out, _ = env
    api.gateway = [502, 0, 426]
    launcher().up("pi05_libero")
    assert "Ready after" in "\n".join(out)
    assert {c[1] for c in api.calls if c[0] == "logs"} == {"w1"}  # the busy worker, not the extra IDLE one


def test_without_gateway_a_resumed_container_is_ready_from_its_earlier_log(env):
    api, launcher, out, _ = env
    api.gateway = [403]
    resumed = [json.dumps({"rs": p}) for p in ("boot", "imports_done", "ready", "ws_open", "ws_close")] + ["426"]
    api.logs = [[json.dumps({"rs": "boot"}), json.dumps({"rs": "ready"}), json.dumps({"rs": "boot"})], resumed]
    launcher().up("pi05_libero")
    assert "Ready after" in "\n".join(out)
    assert sum(c[0] == "logs" for c in api.calls) == 2  # the first read ended in a new boot: not ready yet


def test_phase_stays_when_build_output_hides_the_phase_lines(env):
    api, launcher, out, _ = env
    api.gateway = [502, 502, 502]
    api.logs = [[json.dumps({"rs": "boot"}), json.dumps({"rs": "build_start"})], ["uv sync output"] * 5]
    launcher().up("pi05_libero")
    text = "\n".join(out)
    assert "building the Python environment" in text and "waiting for the server's first log line" not in text


def test_wait_keeps_one_held_gateway_request_in_flight(env):
    """The gateway holds /healthz while the worker starts: short attempts back to back (it may not hand a held one
    over), and the worker log, a blocking read, only every 15 s."""
    api, launcher, out, _ = env
    held = []
    lch = launcher()

    def gateway_status(url, timeout=10.0):
        held.append((lch.clock(), timeout))
        if len(held) < 7:
            lch.sleep(timeout)  # no worker ready: the gateway holds the request until the client gives up
            return 0
        return 200

    api.gateway_status = gateway_status
    lch.up("pi05_libero")
    assert [t for t, _ in held] == [0, 3, 6, 9, 12, 15, 18] and {h for _, h in held} == {3}
    assert sum(c[0] == "logs" for c in api.calls) == 2  # at 3 and 18 s
    assert "Ready after 0:18" in "\n".join(out)


def test_wait_retries_a_quick_refusal_after_a_second(env):
    api, launcher, out, _ = env
    api.gateway = [502] * 40
    lch = launcher()
    lch.up("pi05_libero")
    assert "Ready after 0:40" in "\n".join(out)
    assert sum(c[0] == "logs" for c in api.calls) == 3  # at 1, 16 and 31 s


def test_wait_times_out_and_scales_to_zero(env):
    api, launcher, out, _ = env
    api.gateway = [502] * 10_000
    with pytest.raises(LauncherError, match="Scaled the endpoint to zero"):
        launcher().up("pi05_libero")
    assert api.calls[-1][2] == {"workers": DOWN}


@pytest.mark.parametrize("warm", [False, True])
def test_up_after_down_sends_workers_only_once_the_gateway_answers(env, warm):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero")
    eid = first["endpoint"]["id"]
    launcher().down("pi05_libero")
    api.calls.clear()
    api.gateway = [0, 0, 200]
    launcher().up("pi05_libero", warm=warm)
    assert [c for c in api.calls if c[0] == "update_endpoint"] == [("update_endpoint", eid,
                                                                     {"workers": WARM if warm else DEFAULT})]
    assert api.gateway == [] and api.endpoints[eid]["workers"]["min"] == (1 if warm else 0)


@pytest.mark.parametrize("warm", [False, True])
def test_up_after_down_without_gateway_access_sets_min_1_at_once(env, warm):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero")
    launcher().down("pi05_libero")
    api.calls.clear()
    api.gateway = [403]
    api.logs = [[json.dumps({"rs": p}) for p in ("boot", "unpack_done", "ready")]]
    launcher().up("pi05_libero", warm=warm)
    first_update = next(i for i, c in enumerate(api.calls) if c[0] == "update_endpoint")
    first_logs = next(i for i, c in enumerate(api.calls) if c[0] == "logs")
    assert first_update < first_logs  # min 1 before reading the log: no request can wake the worker
    patches = [c[2] for c in api.calls if c[0] == "update_endpoint"]
    assert patches == ([{"workers": WARM}] if warm else [{"workers": WARM}, {"workers": DEFAULT}])  # then min 0
    assert api.endpoints[first["endpoint"]["id"]]["workers"] == (WARM if warm else DEFAULT)


def test_down_waits_until_the_first_start_has_saved_to_the_volume(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    out.clear()
    boot = [json.dumps({"rs": p}) for p in ("boot", "build_start", "ready", "ws_open", "fill_start")]
    api.logs = [boot, boot + [json.dumps({"rs": "fill_env_done"})], boot + [json.dumps({"rs": "fill_done"})]]
    launcher().down("pi05_libero")
    text = "\n".join(out)
    assert "still saving the environment and checkpoint to the volume" in text
    assert sum(c[0] == "logs" for c in api.calls) == 3
    assert api.calls[-1][0] == "update_endpoint" and api.calls[-1][2]["workers"]["min"] == 0


def test_down_does_not_wait_when_nothing_is_being_saved(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    api.calls.clear()
    api.logs = [[json.dumps({"rs": p}) for p in ("boot", "unpack_done", "ready")]]
    launcher().down("pi05_libero")
    assert [c[0] for c in api.calls] == ["logs", "update_endpoint"]


def test_down_ctrl_c_stops_waiting_and_scales_down(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    api.logs = [[json.dumps({"rs": p}) for p in ("boot", "ready", "fill_start")]] * 3

    def interrupt(s):
        raise KeyboardInterrupt

    lch = launcher()
    lch.sleep = interrupt
    lch.down("pi05_libero")
    assert "Not waiting." in "\n".join(out)
    assert api.calls[-1][0] == "update_endpoint" and api.calls[-1][2]["workers"]["min"] == 0


def test_status_reports_workers_and_price(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    out.clear()
    launcher().status()
    text = "\n".join(out)
    assert "RUNNING, IDLE" in text and "now about 1.34 USD/hr" in text and "2.80 USD/month" in text
    assert "only while a worker runs; a worker stops by itself 5 min after its last client" in text


def test_delete_asks_and_removes_endpoint_and_volume(env):
    api, launcher, out, path = env
    launcher().up("pi05_libero")
    launcher(confirm=lambda q: False).delete("pi05_libero", volume=True)
    assert len(api.endpoints) == 1 and len(api.volumes) == 1
    asked = []
    launcher(confirm=lambda q: asked.append(q) or True).delete("pi05_libero", volume=True)
    assert "[y/N]" in asked[0] and api.endpoints == {} and api.volumes == {}
    assert state.load(path)["models"] == {}


def test_delete_without_a_terminal_says_to_pass_yes(env, monkeypatch, capsys):
    import io

    api, launcher, out, path = env
    launcher().up("pi05_libero")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # a script or CI job: no terminal
    with pytest.raises(LauncherError, match=r"Delete endpoint playground-serve-.*\? pgs asks before it deletes, and "
                                            r"this is not a terminal: .* pass --yes .* Nothing deleted\."):
        launcher().delete("pi05_libero", volume=True)
    assert len(api.endpoints) == 1 and len(api.volumes) == 1
    monkeypatch.setattr(state, "default_path", lambda: path)
    monkeypatch.setattr(cli, "RunpodApi", lambda key: api)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    assert cli.main(["delete", "pi05_libero", "--volume"]) == 1
    err = capsys.readouterr().err
    assert "pass --yes" in err and "Traceback" not in err and len(api.endpoints) == 1
    assert cli.main(["delete", "pi05_libero", "--volume", "--yes"]) == 0
    assert api.endpoints == {} and api.volumes == {}


def test_closed_input_on_a_terminal_is_a_no(monkeypatch):
    from playground_serve import launcher as launcher_module

    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda q: (_ for _ in ()).throw(EOFError))
    assert launcher_module.ask("Delete x? [y/N] ") is False


def test_delete_without_flag_keeps_the_volume(env):
    api, launcher, out, _ = env
    launcher().up("pi05_libero")
    launcher(confirm=lambda q: True).delete("pi05_libero")
    assert api.endpoints == {} and len(api.volumes) == 1
    assert any("kept" in line for line in out)


def test_real_models_and_bundle_fit_the_launcher(tmp_path):
    from playground_serve import bundle, models

    api = FakeApi()
    result = Launcher(api, tmp_path / "state.json", out=lambda s: None, sleep=lambda s: None).up("pi05_libero")
    body = [c[1] for c in api.calls if c[0] == "create_endpoint"][0]
    assert body["entrypoint"] + body["cmd"] == list(bundle.START_COMMAND)
    assert body["env"] == released(bundle.endpoint_env(models.get("pi05_libero"), "0.1.0"))
    assert body["image"] == models.get("pi05_libero").image and body["gpu"]["minCudaVersion"] == "12.8"
    assert NAME.fullmatch(result["endpoint"]["name"])


def test_cli_never_prints_the_key(monkeypatch, tmp_path, capsys):
    secret = "test-key-0123456789"

    class Refusing(FakeApi):
        def __init__(self, key):
            super().__init__()

        def list_endpoints(self):
            raise ApiError(403, "GET", "/serverless", "this API key does not grant serverless:read")

    monkeypatch.setenv("RUNPOD_API_KEY", secret)
    monkeypatch.setattr(cli, "RunpodApi", Refusing)
    monkeypatch.setattr(cli.state, "default_path", lambda: tmp_path / "state.json")
    assert cli.main(["status"]) == 1
    captured = capsys.readouterr()
    assert "serverless:read" in captured.err and secret not in captured.out + captured.err


def test_cli_without_a_key_explains_what_to_do(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert cli.main(["status"]) == 1
    assert "RUNPOD_API_KEY" in capsys.readouterr().err


# ----- status only reads; reconciliation forgets only on a 404


def snapshot(path):
    st = path.stat()
    return path.read_bytes(), st.st_mtime_ns, st.st_ino


@pytest.mark.parametrize("case", ["normal", "list_fails", "partial_list", "deleted", "other_owner", "workers_fail"])
def test_status_never_writes_the_state_file(env, case):
    api, launcher, out, path = env
    first = launcher().up("pi05_libero")
    eid = first["endpoint"]["id"]
    before = snapshot(path)
    if case == "list_fails":
        api.list_endpoints = lambda: (_ for _ in ()).throw(ApiError(500, "GET", "/serverless", "boom"))
    elif case == "partial_list":
        api.list_endpoints = lambda: []
    elif case == "deleted":
        del api.endpoints[eid]
    elif case == "other_owner":
        api.endpoints["x9"] = {**copy.deepcopy(api.endpoints[eid]), "id": "x9",
                               "name": "playground-serve-o00000000-pi05-libero-aaaaaa"}
    elif case == "workers_fail":
        api.endpoint_workers = lambda e: (_ for _ in ()).throw(ApiError(502, "GET", "/workers", "bad gateway"))
    out.clear()
    try:
        launcher().status()
    except ApiError:
        assert case in ("list_fails", "workers_fail")
    assert snapshot(path) == before
    text = "\n".join(out)
    if case in ("partial_list", "deleted"):
        assert f"({eid}) from this state file is not in the account listing" in text
    if case == "other_owner":
        assert "not yours: endpoint playground-serve-o00000000-pi05-libero-aaaaaa (x9)" in text
        assert "created by owner o00000000" in text and text.count("now about") == 1  # only yours priced
    assert [c for c in api.creates() if c[0] != "create_volume" and c[0] != "create_endpoint"] == []


def test_status_without_a_state_file_creates_none(env):
    api, launcher, out, path = env
    api.endpoints["x9"] = {"id": "x9", "name": "playground-serve-o00000000-pi05-libero-aaaaaa", "workers": DEFAULT}
    launcher().status()
    assert not path.exists() and not path.with_suffix(".tmp").exists()
    assert any("not yours" in line for line in out)


@pytest.mark.parametrize("kind", ["endpoint", "volume"])
def test_a_partial_listing_does_not_drop_a_state_entry(env, kind):
    api, launcher, out, path = env
    first = launcher().up("pi05_libero")
    n, before = len(api.creates()), state.load(path)
    if kind == "endpoint":
        api.list_endpoints = lambda: []
    else:
        api.list_volumes = lambda: []
    second = launcher().up("pi05_libero")
    assert api.creates()[n:] == [] and second["endpoint"]["id"] == first["endpoint"]["id"]
    assert state.load(path) == before
    launcher().down("pi05_libero")  # down finds it by id too
    assert api.calls[-1] == ("update_endpoint", first["endpoint"]["id"], {"workers": DOWN})
    assert state.load(path)["models"] == before["models"]


@pytest.mark.parametrize("command", ["up", "down", "delete"])
def test_a_failed_read_by_id_keeps_the_entry_and_stops(env, command):
    api, launcher, out, path = env
    launcher().up("pi05_libero")
    before = snapshot(path)
    n = len(api.creates())
    api.list_endpoints = lambda: []
    api.get_endpoint = lambda eid: (_ for _ in ()).throw(ApiError(503, "GET", f"/serverless/{eid}", "unavailable"))
    with pytest.raises(ApiError):
        if command == "delete":
            launcher(confirm=lambda q: True).delete("pi05_libero", volume=True)
        else:
            getattr(launcher(), command)("pi05_libero")
    assert snapshot(path) == before and api.creates()[n:] == []


# ----- owners


def other_owner_launcher(env, name="other"):
    api, launcher, out, path = env
    return lambda confirm=None: launcher(confirm=confirm, path=path.parent / name / "state.json")


def test_another_owner_is_refused_then_new_creates_a_separate_one(env):
    api, launcher, out, path = env
    a = launcher().up("pi05_libero")
    a_ep, a_vol = copy.deepcopy(api.endpoints[a["endpoint"]["id"]]), a["volume"]["id"]
    b = other_owner_launcher(env)
    b_path = path.parent / "other" / "state.json"
    n = len(api.creates())
    with pytest.raises(LauncherError) as e:
        b().up("pi05_libero")
    msg = str(e.value)
    assert a["endpoint"]["id"] in msg and "--adopt" in msg and "--new" in msg and "does not change it" in msg
    assert api.creates()[n:] == [] and api.endpoints[a["endpoint"]["id"]] == a_ep and not b_path.exists()
    out.clear()
    b().status()
    assert f"not yours: endpoint {a['endpoint']['name']}" in "\n".join(out) and not b_path.exists()
    b().down("pi05_libero")
    b().down()
    with pytest.raises(LauncherError, match="nothing of yours to delete.*--force"):
        b(confirm=lambda q: True).delete("pi05_libero", volume=True)
    assert api.creates()[n:] == [] and api.endpoints[a["endpoint"]["id"]] == a_ep
    mine = b().up("pi05_libero", new=True)
    assert mine["endpoint"]["id"] != a["endpoint"]["id"] and mine["volume"]["id"] != a_vol
    b_owner = state.load(b_path)["owner"]
    assert b_owner != state.load(path)["owner"] and b_owner in mine["endpoint"]["name"] + mine["volume"]["name"]
    assert api.endpoints[a["endpoint"]["id"]] == a_ep  # still untouched
    b().down()
    assert api.endpoints[a["endpoint"]["id"]] == a_ep and api.endpoints[mine["endpoint"]["id"]]["workers"] == DOWN
    second = b().up("pi05_libero")  # now B has its own: no refusal, no create
    assert second["endpoint"]["id"] == mine["endpoint"]["id"]
    out.clear()
    launcher().status()  # A sees B's as not yours
    assert f"not yours: endpoint {mine['endpoint']['name']}" in "\n".join(out)


def test_delete_force_deletes_another_owners_endpoint_after_asking(env):
    api, launcher, out, path = env
    a = launcher().up("pi05_libero")
    b = other_owner_launcher(env)
    asked = []
    b(confirm=lambda q: asked.append(q) or False).delete("pi05_libero", force=True)
    assert "not yours: created by owner" in asked[0] and a["endpoint"]["id"] in api.endpoints
    b(confirm=lambda q: True).delete("pi05_libero", volume=True, force=True)
    assert api.endpoints == {} and api.volumes == {}


def test_adopt_takes_over_by_id(env):
    api, launcher, out, path = env
    a = launcher().up("pi05_libero")
    c = launcher(path=path.parent / "c" / "state.json").up("pi05_libero", new=True)
    b = other_owner_launcher(env)
    with pytest.raises(LauncherError, match=r"--adopt ID"):
        b().up("pi05_libero")
    with pytest.raises(LauncherError, match="nope is not one of"):
        b().up("pi05_libero", adopt="nope")
    got = b().up("pi05_libero", adopt=a["endpoint"]["id"])
    assert got["endpoint"]["id"] == a["endpoint"]["id"] and got["volume"]["id"] == a["volume"]["id"]
    rec = state.load(path.parent / "other" / "state.json")["models"]["pi05_libero"]
    assert rec["endpoint"]["id"] == a["endpoint"]["id"] and rec["volume"]["id"] == a["volume"]["id"]
    assert c["endpoint"]["id"] in api.endpoints and len(api.endpoints) == 2


def test_resources_from_before_owner_ids(env):
    """Unowned names are not yours unless this state file holds their ids (a state file from before owner ids)."""
    api, launcher, out, path = env
    legacy = launcher().up("pi05_libero")
    eid, vid = legacy["endpoint"]["id"], legacy["volume"]["id"]
    api.endpoints[eid]["name"] = "playground-serve-pi05-libero-abcdef"
    api.volumes[vid]["name"] = "playground-serve-pi05-libero-123456"
    old_state = state.load(path)
    old_state.pop("owner")
    for m in old_state["models"].values():
        m["endpoint"]["name"], m["volume"]["name"] = api.endpoints[eid]["name"], api.volumes[vid]["name"]
    stranger = other_owner_launcher(env)
    with pytest.raises(LauncherError, match="created before owner ids"):
        stranger().up("pi05_libero")
    out.clear()
    stranger().status()
    assert "not yours: endpoint playground-serve-pi05-libero-abcdef" in "\n".join(out)
    state.save(path, old_state)  # the creator's state file from before owner ids holds the ids
    n = len(api.creates())
    again = launcher().up("pi05_libero")
    assert again["endpoint"]["id"] == eid and api.creates()[n:] == []
    out.clear()
    launcher().status()
    assert "not yours" not in "\n".join(out) and "abcdef" in "\n".join(out)
    launcher().down("pi05_libero")
    assert api.endpoints[eid]["workers"] == DOWN


# ----- per-project state, default and --warm workers, CLI flags


def test_per_project_state_location(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("PLAYGROUND_SERVE_HOME", raising=False)
    assert state.default_path() == tmp_path / "xdg" / "playground-serve" / "state.json"
    monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path / "proj"))
    assert state.default_path() == tmp_path / "proj" / "state.json"
    assert state.key_path() == tmp_path / "xdg" / "playground-serve" / "key"  # the key stays per user


def test_cli_two_projects_on_one_account(monkeypatch, tmp_path, capsys):
    api = FakeApi()
    monkeypatch.setattr(cli, "RunpodApi", lambda key: api)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-0123456789")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    pgs = lambda home, *argv: (monkeypatch.setenv("PLAYGROUND_SERVE_HOME", str(tmp_path / home)), cli.main(list(argv)))[1]
    assert pgs("a", "up", "pi05_libero") == 0
    capsys.readouterr()
    assert pgs("b", "up", "pi05_libero") == 1
    assert "--adopt" in capsys.readouterr().err and len(api.endpoints) == 1
    assert pgs("b", "up", "pi05_libero", "--new", "--warm", "--idle", "10") == 0
    assert len(api.endpoints) == 2 and sorted(e["workers"]["min"] for e in api.endpoints.values()) == [0, 1]
    assert pgs("b", "status") == 0 and "not yours" in capsys.readouterr().out
    owners = {state.load(tmp_path / h / "state.json")["owner"] for h in "ab"}
    assert len(owners) == 2 and not (tmp_path / "xdg" / "playground-serve" / "state.json").exists()
    assert {e["workers"]["idleTimeout"] for e in api.endpoints.values()} == {300, 600}


@pytest.mark.parametrize("warm,idle", [(False, 5), (True, 5), (False, 10)])
def test_default_and_warm_worker_settings(env, warm, idle):
    api, launcher, out, _ = env
    launcher().up("pi05_libero", warm=warm, idle_min=idle)
    body = next(c[1] for c in api.calls if c[0] == "create_endpoint")
    assert body["workers"] == {"min": int(warm), "max": 1, "idleTimeout": idle * 60}
    text = "\n".join(out)
    if warm:
        assert "1 worker is kept running (--warm): 1.34 USD/hr with or without clients, until `pgs down" in text
    else:
        assert f"stops by itself {idle} min after the last request if no client connects" in text


def test_switching_warm_to_default_lets_the_kept_worker_stop_at_once(env):
    api, launcher, out, _ = env
    first = launcher().up("pi05_libero", warm=True)
    eid = first["endpoint"]["id"]
    api.calls.clear()
    api.gateway = [200]
    launcher().up("pi05_libero")
    patches = [c[2] for c in api.calls if c[0] == "update_endpoint"]
    assert patches == [{"workers": DEFAULT}] and api.endpoints[eid]["workers"] == DEFAULT
    out.clear()
    launcher().status()
    assert "min 0 max 1, idle timeout 5 min" in "\n".join(out)


def test_cli_flags():
    p = cli.parser()
    a = p.parse_args(["up", "pi05_libero"])
    assert (a.warm, a.idle, a.adopt, a.new) == (False, 5, None, False)
    assert p.parse_args(["up", "pi05_libero", "--adopt"]).adopt is True
    assert p.parse_args(["up", "pi05_libero", "--adopt", "ep123"]).adopt == "ep123"
    assert p.parse_args(["delete", "pi05_libero", "--force", "--volume"]).force is True
    assert a.gpu is None and p.parse_args(["up", "pi05_libero", "--gpu", "H200"]).gpu == "H200"
    with pytest.raises(SystemExit):
        p.parse_args(["up", "pi05_libero", "--adopt", "--new"])
    with pytest.raises(SystemExit):
        cli.main(["up", "pi05_libero", "--idle", "0"])
