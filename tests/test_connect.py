"""`playground_serve.connect`: target resolution (URL, model via state, PLAYGROUND_SERVE_URL, model via the Runpod
API, a key that may not read the account), key resolution, both server metadata keys, and `act`."""

from __future__ import annotations

import urllib.request

import numpy as np
import pytest

from playground_serve import resolve, state
from playground_serve.client.mock_server import Config, MockServer
from playground_serve.resolve import URL_ENV, EndpointNotFound, resolve_key, resolve_url
from playground_serve.runpod_api import ApiError

KEY = "test-key-0123456789"


def endpoint(eid: str, name: str, created: str = "2026-11-13T00:00:00Z") -> dict:
    return {"id": eid, "name": name, "createdAt": created,
            "requestUrls": {"base": f"https://{eid}.api.runpod.ai", "health": f"https://{eid}.api.runpod.ai/ping"}}


class FakeApi:
    endpoints: list = []
    keys: list = []

    def __init__(self, key):
        FakeApi.keys.append(key)

    def list_endpoints(self):
        return list(FakeApi.endpoints)


class RefusingApi:
    """The account API as a key that may only call endpoints sees it: every read is refused."""

    made = 0
    status = 403

    def __init__(self, key):
        RefusingApi.made += 1

    def list_endpoints(self):
        raise ApiError(RefusingApi.status, "GET", "/serverless", "you do not have permission to perform this action")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty config dir and no RUNPOD_API_KEY or PLAYGROUND_SERVE_URL, whatever this machine has."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv(URL_ENV, raising=False)
    FakeApi.endpoints, FakeApi.keys = [], []
    RefusingApi.made, RefusingApi.status = 0, 403
    return tmp_path / "playground-serve"


def write_state(models: dict) -> None:
    state.save(state.default_path(), {"models": {m: {"endpoint": {"id": i, "name": f"playground-serve-x-{i}",
                                                                  "url": f"wss://{i}.api.runpod.ai/ws"}}
                                                 for m, i in models.items()}})


def test_url_target_is_used_as_given(home):
    assert resolve_url("ws://127.0.0.1:8000", None, api_factory=FakeApi) == "ws://127.0.0.1:8000"
    assert resolve_url("wss://abc.api.runpod.ai/ws", KEY, api_factory=FakeApi) == "wss://abc.api.runpod.ai/ws"
    assert FakeApi.keys == []


def test_model_from_the_state_file_needs_no_api_call(home):
    write_state({"pi05_libero": "ep1", "pi05_droid": "ep2"})
    assert resolve_url("pi05_libero", None, api_factory=FakeApi) == "wss://ep1.api.runpod.ai/ws"
    assert FakeApi.keys == []


def test_no_target_with_one_endpoint_in_state(home):
    write_state({"pi05_libero": "ep1"})
    assert resolve_url(None, None, api_factory=FakeApi) == "wss://ep1.api.runpod.ai/ws"


def test_no_target_with_several_endpoints_is_ambiguous(home):
    write_state({"pi05_libero": "ep1", "pi05_droid": "ep2"})
    with pytest.raises(EndpointNotFound, match=r'several endpoints are up \(pi05_droid, pi05_libero\)'):
        resolve_url(None, KEY, api_factory=FakeApi)


def test_model_found_on_the_account_by_managed_name(home):
    FakeApi.endpoints = [endpoint("old1", "playground-serve-pi05-libero-aaaaaa", "2026-11-13T01:00:00Z"),
                         endpoint("new1", "playground-serve-pi05-libero-bbbbbb", "2026-11-13T02:00:00Z"),
                         endpoint("drd1", "playground-serve-pi05-droid-cccccc"),
                         endpoint("other1", "my-other-endpoint")]  # not managed: ignored
    assert resolve_url("pi05_libero", KEY, api_factory=FakeApi) == "wss://new1.api.runpod.ai/ws"
    assert FakeApi.keys == [KEY]


def test_account_lookup_reads_owner_names_and_prefers_the_callers_own(home):
    FakeApi.endpoints = [endpoint("theirs", "playground-serve-o11111111-pi05-libero-aaaaaa", "2026-11-13T03:00:00Z"),
                         endpoint("mine", "playground-serve-o22222222-pi05-libero-bbbbbb", "2026-11-13T01:00:00Z"),
                         endpoint("drd1", "playground-serve-o22222222-pi05-droid-cccccc")]
    assert resolve_url("pi05_libero", KEY, api_factory=FakeApi) == "wss://theirs.api.runpod.ai/ws"  # no owner: newest
    state.save(state.default_path(), {"owner": "o22222222", "models": {}})
    assert resolve_url("pi05_libero", KEY, api_factory=FakeApi) == "wss://mine.api.runpod.ai/ws"
    with pytest.raises(EndpointNotFound, match=r"several endpoints are up \(pi05-droid, pi05-libero\)"):
        resolve_url(None, KEY, api_factory=FakeApi)


def test_no_target_on_the_account(home):
    FakeApi.endpoints = [endpoint("e1", "playground-serve-pi05-libero-aaaaaa"), endpoint("other1", "my-other-endpoint")]
    assert resolve_url(None, KEY, api_factory=FakeApi) == "wss://e1.api.runpod.ai/ws"
    FakeApi.endpoints.append(endpoint("e2", "playground-serve-pi05-droid-bbbbbb"))
    with pytest.raises(EndpointNotFound, match="several endpoints are up"):
        resolve_url(None, KEY, api_factory=FakeApi)


def test_no_endpoint_says_to_run_up_first(home):
    FakeApi.endpoints = [endpoint("other1", "my-other-endpoint"), endpoint("d1", "playground-serve-pi05-droid-cccccc")]
    with pytest.raises(EndpointNotFound, match=r"no Playground Serve endpoint for pi05_libero.*run `pgs up pi05_libero` first"):
        resolve_url("pi05_libero", KEY, api_factory=FakeApi)
    with pytest.raises(EndpointNotFound, match=r"no Runpod API key.*pgs up.*PLAYGROUND_SERVE_URL"):
        resolve_url("pi05_libero", None, api_factory=FakeApi)


@pytest.mark.parametrize("code", [401, 403])
def test_a_key_that_may_not_read_the_account_is_told_to_pass_the_url(home, code):
    RefusingApi.status = code
    with pytest.raises(EndpointNotFound, match=rf"may not list the account's endpoints \(HTTP {code}.*pgs up.*"
                                               r'connect\("wss://<endpoint-id>.api.runpod.ai/ws"\).*PLAYGROUND_SERVE_URL'):
        resolve_url("pi05_libero", KEY, api_factory=RefusingApi)
    assert RefusingApi.made == 1


def test_url_state_and_env_need_no_account_read(home, monkeypatch):
    assert resolve_url("wss://abc.api.runpod.ai/ws", KEY, api_factory=RefusingApi) == "wss://abc.api.runpod.ai/ws"
    monkeypatch.setenv(URL_ENV, " wss://env1.api.runpod.ai/ws\n")
    assert resolve_url("pi05_libero", KEY, api_factory=RefusingApi) == "wss://env1.api.runpod.ai/ws"
    assert resolve_url(None, KEY, api_factory=RefusingApi) == "wss://env1.api.runpod.ai/ws"
    write_state({"pi05_libero": "ep1", "pi05_droid": "ep2"})  # the state file comes first, for the names it holds
    assert resolve_url("pi05_libero", KEY, api_factory=RefusingApi) == "wss://ep1.api.runpod.ai/ws"
    assert resolve_url("pi0_aloha_sim", KEY, api_factory=RefusingApi) == "wss://env1.api.runpod.ai/ws"
    assert resolve_url(None, KEY, api_factory=RefusingApi) == "wss://env1.api.runpod.ai/ws"  # two in state: ambiguous
    assert RefusingApi.made == 0
    monkeypatch.setenv(URL_ENV, "https://env1.api.runpod.ai")
    with pytest.raises(ValueError, match="PLAYGROUND_SERVE_URL must be a ws:// or wss:// URL"):
        resolve_url("pi0_aloha_sim", KEY, api_factory=RefusingApi)


def test_connect_by_url_or_env_makes_no_http_call(home, monkeypatch, observations):
    def no_http(*args, **kwargs):
        raise AssertionError("connect made an HTTP call")

    monkeypatch.setattr(urllib.request, "urlopen", no_http)
    with MockServer() as mock:
        for target, env in ((mock.url, None), ("pi05_libero", mock.url)):
            if env:
                monkeypatch.setenv(URL_ENV, env)
            policy = resolve.connect(target, api_key=KEY, keepalive=None)
            try:
                assert policy.act(observations[0]).shape == (10, 7)
            finally:
                policy.close()


def test_key_order_argument_key_file_env_default_file(home, tmp_path, monkeypatch):
    assert resolve_key() is None
    home.mkdir(parents=True)
    (home / "key").write_text("from-default-file\n")
    assert resolve_key() == "from-default-file"
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert resolve_key() == "from-env"
    (tmp_path / "other.key").write_text("from-key-file\n")
    assert resolve_key(key_file=tmp_path / "other.key") == "from-key-file"
    assert resolve_key("from-argument", tmp_path / "other.key") == "from-argument"


# "robo_serve" is the pre-rename wire key, still accepted for compatibility.
@pytest.mark.parametrize("feature_key", ["playground_serve", "robo_serve"])
def test_both_metadata_keys_enable_the_features(home, feature_key, observations):
    with MockServer(Config(feature_key=feature_key)) as mock:
        policy = resolve.connect(mock.url, keepalive=None)
        try:
            assert policy.features == {"jpeg": True, "noop": True} and policy.mode == "jpeg"
            policy.act(observations[0])
            assert policy.timing["payload_bytes"] < 40_000  # jpeg went out
        finally:
            policy.close()


def test_connect_by_model_name_and_act_returns_the_action_array(home, observations, monkeypatch):
    with MockServer() as mock:
        state.save(state.default_path(), {"models": {"pi05_libero": {"endpoint": {"id": "x", "name": "n",
                                                                                   "url": mock.url}}}})
        monkeypatch.setenv("RUNPOD_API_KEY", KEY)
        from playground_serve import connect

        policy = connect("pi05_libero", keepalive=None)
        try:
            actions = policy.act(observations[0])
            assert isinstance(actions, np.ndarray) and actions.shape == (10, 7)
            reply = policy.infer(observations[0])
            assert set(reply) >= {"actions", "server_timing"}  # infer keeps openpi's whole reply
            assert policy._headers == {"Authorization": f"Bearer {KEY}"}
        finally:
            policy.close()


def test_package_exports():
    import playground_serve

    assert {"connect", "ActionQueue", "status", "Status"} <= set(playground_serve.__all__)
    assert playground_serve.ActionQueue.__module__ == "playground_serve.client.action_queue"
    assert playground_serve.connect is resolve.connect
    assert playground_serve.status is resolve.status and playground_serve.Status is resolve.Status


@pytest.fixture(scope="session")
def observations():
    d = np.load(__import__("pathlib").Path(__file__).parent / "client" / "data" / "libero_frames.npz")
    return [{"observation/image": d["image"][i], "observation/wrist_image": d["wrist_image"][i],
             "observation/state": d["state"][i], "prompt": str(d["prompt"])} for i in range(len(d["image"]))]
