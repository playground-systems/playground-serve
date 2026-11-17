"""`pgs up` creates a new endpoint at once, in the data center a probe of the last day measured nearest, else the one
nearest to this machine's time zone (region.guess), else the one with most stock, or where `--region` says; the probe
runs while the worker boots. Probes stay within the account's worker quota and never stay behind; `--stock-path` turns
the fast model path off. In-memory fake of the Runpod API."""

from __future__ import annotations

import threading
import time

import pytest

from playground_serve import cli, region, state
from playground_serve.launcher import RELEASE, Launcher, LauncherError
from playground_serve.runpod_api import ApiError
from test_launcher import START, FakeApi, get_spec, make_env, released

DAY = 24 * 3600
NEW_YORK, DALLAS = (40.71, -74.01), (32.78, -96.80)


@pytest.fixture
def env(tmp_path):
    api, out, t = FakeApi(), [], [0.0]

    def launcher(confirm=None, measure=None, here=None):
        return Launcher(api, tmp_path / "state.json", get_spec=get_spec, make_env=make_env, start_command=START,
                        model_names=lambda: ["pi05_libero"], out=out.append, confirm=confirm,
                        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0], poll=5,
                        measure=measure or api.measure, now=lambda: 1_800_000_000 + t[0], here=lambda: here)

    return api, launcher, out, tmp_path / "state.json", t


def model_dc(api):
    return next(c[1]["dataCenterIds"][0] for c in api.calls if c[0] == "create_endpoint")


def probes_left(api):
    return [e for e in api.endpoints.values() if "-probe-" in e["name"]]


def test_up_creates_at_once_probes_while_the_worker_boots_and_the_next_create_uses_it(env):
    api, launcher, out, path, _ = env
    api.rtt = {"US-IL-1": 46.0, "US-TX-3": 21.0}  # US-IL-1 has more stock, US-TX-3 is nearer
    result = launcher().up("pi05_libero")
    kinds = [c[0] for c in api.calls if c[0] in ("create_endpoint", "probe_create")]
    assert kinds[0] == "create_endpoint" and model_dc(api) == "US-IL-1"  # nothing measured before the create
    assert sorted(c[1] for c in api.probes() if c[0] == "probe_create") == ["US-IL-1", "US-TX-3"]
    assert len([c for c in api.probes() if c[0] == "probe_delete"]) == 2 and probes_left(api) == []
    text = "\n".join(out)
    assert "Data center: US-IL-1, round trip from here not measured yet; most stock" in text
    assert "measured while the worker started: US-TX-3 21.0 ms, US-IL-1 46.0 ms." in text
    assert text.index("Ready after") < text.index("US-TX-3 is nearer than US-IL-1 (21.0 against 46.0 ms): "
                                                  "`pgs delete pi05_libero --volume`, then `pgs up pi05_libero`")
    saved = state.load(path)["probes"]
    assert {dc: v["ms"] for dc, v in saved.items()} == {"US-IL-1": 46.0, "US-TX-3": 21.0}
    assert result["volume"]["dataCenter"] == "US-IL-1"
    launcher(confirm=lambda q: True).delete("pi05_libero", volume=True)
    api.calls.clear()
    out.clear()
    launcher().up("pi05_libero")  # the measurement decides now
    assert model_dc(api) == "US-TX-3" and api.probes() == []
    assert "Data center: US-TX-3, 21.0 ms round trip from here (probe)" in "\n".join(out)


def test_without_a_measurement_the_time_zone_guess_decides_and_stock_breaks_ties(tmp_path):
    gpus = [{"id": "g", "name": "RTX 4090", "pool": "ADA_24", "minRam": 31, "price": {"serverless": 1.34},
             "dataCenters": [{"id": d, "availability": a} for d, a in
                             (("US-IL-1", "HIGH"), ("US-TX-3", "LOW"), ("US-NC-1", "MEDIUM"), ("CA-MTL-3", "LOW"),
                              ("EU-RO-1", "HIGH"))]}]
    dcs = [{"id": d, "networkVolumeTypes": ["STANDARD"]} for d in ("US-IL-1", "US-TX-3", "US-NC-1", "CA-MTL-3",
                                                                  "EU-RO-1")]
    from test_launcher import FakeSpec
    from playground_serve.launcher import candidates

    cands = candidates(gpus, dcs, FakeSpec())
    order = lambda here: [c.dc for c in Launcher(FakeApi(), tmp_path / "s.json", here=lambda: here,  # noqa: E731
                                                  out=lambda s: None)._order(cands, [], "o00000000")[0]]
    assert order(None) == ["EU-RO-1", "US-IL-1", "US-NC-1", "CA-MTL-3", "US-TX-3"]  # stock, then name
    # New York: Montreal (530 km) and North Carolina (690 km) both under 1000 km, more stock first; Chicago next
    assert order(NEW_YORK)[:3] == ["US-NC-1", "CA-MTL-3", "US-IL-1"] and order(NEW_YORK)[-1] == "EU-RO-1"
    assert order(DALLAS)[0] == "US-TX-3"  # nearest wins over more stock elsewhere
    assert order((45.0, 25.0))[0] == "EU-RO-1"


def test_time_zone_location_comes_from_the_local_zone_table(tmp_path):
    tab = tmp_path / "zone.tab"
    tab.write_text("# comment\nUS\t+404251-0740023\tAmerica/New_York\tEastern (most areas)\n"
                   "DE\t+5230+01322\tEurope/Berlin\n")
    ny = region.here("America/New_York", tabs=(str(tmp_path / "missing.tab"), str(tab)))
    assert ny == pytest.approx((40.714, -74.006), abs=1e-3)
    assert region.here("Europe/Berlin", tabs=(str(tab),)) == pytest.approx((52.5, 13.367), abs=1e-3)
    assert region.here("Etc/UTC", tabs=(str(tab),)) is None and region.here("X", tabs=()) is None
    assert region.band(ny, "US-NC-1") == 0 and region.band(ny, "US-IL-1") == 1 and region.band(ny, "EU-RO-1") == 7
    assert region.band(ny, "XX-YY-1") == 99 and region.band(None, "US-NC-1") == 0


def test_probe_names_carry_the_owner_and_the_body_is_a_small_cpu_endpoint(env):
    api, launcher, _, path, _ = env
    bodies = []
    real = api.create_endpoint
    api.create_endpoint = lambda body: bodies.append(body) or real(body)
    launcher().up("pi05_libero")
    probes = {b["dataCenterIds"][0]: b for b in bodies if "-probe-" in b["name"]}
    owner = state.load(path)["owner"]
    assert all(region.is_probe(b["name"], owner) and not region.is_probe(b["name"], "o00000000") for b in probes.values())
    assert probes["US-IL-1"]["cpu"] == [{"id": "cpu3c", "vcpuCount": 2}] and "gpu" not in probes["US-IL-1"]
    # US-TX-3 offers no serverless CPU workers: the probe runs on the cheapest GPU pool with stock there
    assert probes["US-TX-3"]["gpu"] == {"pools": ["ADA_24"], "count": 1} and "cpu" not in probes["US-TX-3"]
    assert all(b["workers"]["max"] == 1 and b["image"] == region.PROBE_IMAGE for b in probes.values())


def test_cheapest_gpu_pool_with_stock_and_cpu_data_centers():
    from test_launcher import GPUS

    assert region.gpu_pool(GPUS, "US-IL-1") == ("ADA_24", 1.34)  # the H100 there costs more
    assert region.gpu_pool(GPUS, "CA-MTL-1") == ("AMPERE_48", 1.22)
    assert region.gpu_pool(GPUS, "EU-SE-1") is None  # listed, no stock
    cpus = [{"id": "cpu3c", "dataCenters": [{"id": "A", "availability": "LOW"}, {"id": "B", "availability": "NONE"}]},
            {"id": "cpu5c", "dataCenters": [{"id": "C", "availability": "HIGH"}]}]
    assert region.cpu_dcs(cpus) == {"A"}


def test_probe_results_are_cached_for_a_day(env):
    api, launcher, _, _, t = env
    api.rtt = {"US-IL-1": 46.0, "US-TX-3": 21.0}
    launcher().up("pi05_libero")
    launcher(confirm=lambda q: True).delete("pi05_libero", volume=True)
    api.calls.clear()
    api.rtt = {}  # a probe now would measure nothing
    launcher().up("pi05_libero")
    assert api.probes() == [] and model_dc(api) == "US-TX-3"  # from the cache
    launcher(confirm=lambda q: True).delete("pi05_libero", volume=True)
    api.calls.clear()
    t[0] += DAY
    launcher().up("pi05_libero")
    assert len(api.probes()) == 4 and model_dc(api) == "US-IL-1"  # expired: probed again, nothing answered: stock


def test_unanswered_probes_fall_back_to_stock_order(env):
    api, launcher, out, path, _ = env
    api.rtt = {"US-TX-3": None}
    launcher().up("pi05_libero")
    assert model_dc(api) == "US-IL-1" and probes_left(api) == []
    assert state.load(path).get("probes") == {}  # nothing measured, nothing cached: the next `up` probes again
    assert "Data center: US-IL-1, round trip from here not measured yet; most stock" in "\n".join(out)


def test_region_override_skips_the_probe(env):
    api, launcher, out, _, _ = env
    api.rtt = {"US-IL-1": 10.0}
    launcher().up("pi05_libero", dc="US-TX-3")
    assert api.probes() == [] and model_dc(api) == "US-TX-3"


def test_region_without_stock_or_volumes_creates_nothing(env):
    api, launcher, _, _, _ = env
    with pytest.raises(LauncherError, match="EU-NOVOL.*--region"):
        launcher().up("pi05_libero", dc="EU-NOVOL")
    assert api.creates() == [] and api.probes() == []


def test_region_cannot_move_an_existing_volume(env):
    api, launcher, _, _, _ = env
    launcher().up("pi05_libero")
    n = len(api.creates())
    with pytest.raises(LauncherError, match="already in US-IL-1.*delete pi05_libero --volume"):
        launcher().up("pi05_libero", dc="US-TX-3")
    assert len(api.creates()) == n
    launcher().up("pi05_libero", dc="US-IL-1")  # the same data center is fine


@pytest.mark.parametrize("others,probed", [(2, 2), (3, 1), (4, 0), (5, 0)])
def test_probes_stay_within_the_account_worker_quota(env, others, probed):
    api, launcher, out, _, _ = env
    for i in range(others):  # other projects' endpoints, one max worker each (5 in all is the quota)
        api.endpoints[f"x{i}"] = {"id": f"x{i}", "name": f"rs-other-{i}", "workers": {"min": 0, "max": 1}}
    api.rtt = {"US-IL-1": 46.0, "US-TX-3": 21.0}
    peak, real = [0], api.create_endpoint

    def create(body):
        ep = real(body)
        peak[0] = max(peak[0], len(probes_left(api)))
        return ep

    api.create_endpoint = create
    if others == 5:
        api.refuse.update({("ADA_24", "US-IL-1"), ("ADA_24", "US-TX-3"), ("AMPERE_48", "CA-MTL-1")})
        with pytest.raises(LauncherError):
            launcher().up("pi05_libero")
    else:
        launcher().up("pi05_libero")
    assert len([c for c in api.probes() if c[0] == "probe_create"]) == probed and peak[0] <= probed
    assert probes_left(api) == []
    if probed == 1:  # the nearest first: here the stock order
        assert [c[1] for c in api.probes() if c[0] == "probe_create"] == ["US-IL-1"] and model_dc(api) == "US-IL-1"


def test_a_refused_probe_is_not_measured_and_nothing_stays(env):
    api, launcher, out, _, _ = env
    real = api.create_endpoint

    def create(body):
        if "-probe-us-tx-3-" in body["name"]:
            raise ApiError(400, "POST", "/serverless", "no CPU capacity in this data center")
        return real(body)

    api.create_endpoint = create
    api.rtt = {"US-IL-1": 46.0, "US-TX-3": 21.0}
    launcher().up("pi05_libero")
    assert model_dc(api) == "US-IL-1" and probes_left(api) == [] and "US-TX-3: probe not created" in "\n".join(out)


def test_probes_are_deleted_when_measuring_fails_or_is_interrupted(env):
    api, launcher, _, _, _ = env

    def broken(*a, **k):
        raise RuntimeError("websocket library exploded")

    launcher(measure=broken).up("pi05_libero")
    assert probes_left(api) == [] and model_dc(api) == "US-IL-1"
    launcher(confirm=lambda q: True).delete("pi05_libero", volume=True)
    api.calls.clear()

    measuring = threading.Event()

    def slow(api_, url, until, **k):  # a probe still waiting for its worker when Ctrl-C comes
        measuring.set()
        while until():
            time.sleep(0.01)
        return None

    def ctrl_c(url, timeout=10.0):
        measuring.wait(5)
        raise KeyboardInterrupt

    api.gateway_status = ctrl_c
    with pytest.raises(KeyboardInterrupt):
        launcher(measure=slow).up("pi05_libero")
    assert probes_left(api) == [] and len(api.probes()) >= 2
    assert len([c for c in api.probes() if c[0] == "probe_delete"]) == len([c for c in api.probes()
                                                                           if c[0] == "probe_create"])


def test_a_probe_create_lost_in_transit_is_found_and_deleted(env):
    api, launcher, _, _, _ = env
    real = api.create_endpoint

    def lossy(body):
        ep = real(body)
        if "-probe-" in body["name"]:
            raise ApiError(0, "POST", "/serverless", "network error: timed out")
        return ep

    api.create_endpoint = lossy
    launcher().up("pi05_libero")
    assert probes_left(api) == []


def test_a_probe_left_by_an_interrupted_run_is_deleted_by_the_next_up(env):
    api, launcher, out, path, _ = env
    owner = "o1234abcd"
    state.save(path, {"owner": owner, "models": {}})
    mine = region.probe_name(owner, "US-IL-1")
    theirs = region.probe_name("o9999ffff", "US-IL-1")
    for eid, name in (("old1", mine), ("old2", theirs)):
        api.endpoints[eid] = {"id": eid, "name": name, "workers": {"min": 0, "max": 1}}
    launcher().up("pi05_libero")
    assert "old1" not in api.endpoints and "old2" in api.endpoints  # another owner's probe is not ours to delete
    assert "left by an interrupted `pgs up`" in "\n".join(out)


def test_model_endpoint_create_waits_out_a_quota_refusal(env):
    api, launcher, out, _, _ = env
    api.endpoint_errors.append(ApiError(400, "POST", "/serverless", "max workers across all endpoints must not "
                                        "exceed the account's quota of 5"))
    result = launcher().up("pi05_libero")
    creates = [c[1]["dataCenterIds"][0] for c in api.calls if c[0] == "create_endpoint"]
    assert creates == ["US-IL-1", "US-IL-1"] and result["endpoint"]["dataCenterIds"] == ["US-IL-1"]
    assert "worker quota is full; retrying" in "\n".join(out)


def test_stock_path_turns_the_fast_path_off_and_back_on(tmp_path):
    from playground_serve import bundle, models

    api = FakeApi()
    make = lambda: Launcher(api, tmp_path / "state.json", out=lambda s: None, sleep=lambda s: None,  # noqa: E731
                            measure=api.measure)
    result = make().up("pi05_libero", stock_path=True)
    eid = result["endpoint"]["id"]
    want = bundle.endpoint_env(models.get("pi05_libero"), RELEASE)
    assert want["RS_FAST"] == "auto" and api.endpoints[eid]["env"] == released({**want, "RS_FAST": "off"})
    make().up("pi05_libero")
    assert api.endpoints[eid]["env"] == released(want)  # a new release: the fast path again


def test_cli_region_and_stock_path_flags():
    a = cli.parser().parse_args(["up", "pi05_libero", "--region", "US-NC-1", "--stock-path"])
    assert (a.region, a.stock_path) == ("US-NC-1", True)
    a = cli.parser().parse_args(["up", "pi05_libero"])
    assert (a.region, a.stock_path) == (None, False)
