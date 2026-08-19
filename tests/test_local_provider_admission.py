"""`synth_mlx_rl` admission: one shared validator, three call sites.

The validator is the security boundary for a local MLX proxy. It fails closed:
loopback over http is admitted unconditionally, exactly matching the
`_validate_responses_endpoint` precedent; everything else — https, and the
Docker host alias — must be named in SYNTH_MLX_RL_ALLOWED_ENDPOINTS.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.local_provider import (
    ALLOWLIST_ENV,
    CHAT_COMPLETIONS,
    PROVIDER_ID,
    RESPONSES,
    local_endpoint,
    normalize_api_family,
    validate_local_endpoint,
)
from synth_containers.platform.react import CRAFTAX_REACT_SYSTEM_PROMPT, OpenRouterReAct, PolicyConfigError
from synth_containers.platform.runtimes import healthbench


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def _react_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model": "Qwen/Qwen3.5-0.8B",
        "effort": "medium",
        "max_tokens": 1024,
        "context_token_budget": 16000,
        "compact_at": 0.7,
        "keep_recent_messages": 8,
        "keep_recent_frames": 2,
        "observation_mode": "text",
        "provider": PROVIDER_ID,
        "api_family": CHAT_COMPLETIONS,
        "base_url": "http://127.0.0.1:8765/v1",
        "api_key_env": "SYNTH_MLX_RL_API_KEY",
        "parse_retries": 0,
        "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT,
    }
    config.update(overrides)
    return config


class _FakeHTTPResponse:
    def __init__(self, text: str, content_type: str) -> None:
        self._data = text.encode("utf-8")
        self._offset = 0
        self.headers = {"Content-Type": content_type}

    def read(self, amount: int | None = None) -> bytes:
        if amount is None:
            chunk, self._offset = self._data[self._offset :], len(self._data)
            return chunk
        chunk = self._data[self._offset : self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _stub_urlopen(monkeypatch, captured: dict, text: str, *, content_type: str) -> None:
    import urllib.request

    from synth_containers.platform import react as react_module

    def fake_urlopen(request, timeout=None):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(text, content_type)

    monkeypatch.setattr(react_module.urllib.request, "urlopen", fake_urlopen)
    del urllib


# --- the validator itself ---------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
@pytest.mark.parametrize("family", [CHAT_COMPLETIONS, RESPONSES])
def test_loopback_is_admitted_over_http(host: str, family: str) -> None:
    authority = f"[{host}]" if host == "::1" else host
    suffix = "/chat/completions" if family == CHAT_COMPLETIONS else "/responses"
    validate_local_endpoint(f"http://{authority}:8765/v1{suffix}", api_family=family)
    assert local_endpoint(f"http://{authority}:8765/v1", api_family=family) == (
        f"http://{authority}:8765/v1{suffix}"
    )


@pytest.mark.parametrize("family", [CHAT_COMPLETIONS, RESPONSES])
def test_docker_host_alias_is_refused_unless_allowlisted(monkeypatch, family: str) -> None:
    """`host.docker.internal` resolves to the host from inside a container. Blanket
    admission would hand a policy config a probe across every port on the machine, so
    it goes through the allowlist where the port is named."""
    suffix = "/chat/completions" if family == CHAT_COMPLETIONS else "/responses"
    endpoint = f"http://host.docker.internal:8787/v1{suffix}"

    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint, api_family=family)

    monkeypatch.setenv(ALLOWLIST_ENV, "http://host.docker.internal:8787")
    validate_local_endpoint(endpoint, api_family=family)

    # The allowlist names an origin including its port: a different port stays refused.
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(
            f"http://host.docker.internal:5432/v1{suffix}", api_family=family
        )


def test_a_public_http_origin_is_refused(monkeypatch) -> None:
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("http://exfil.example/v1/chat/completions")
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("http://10.0.0.5:8765/v1/chat/completions")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:password@127.0.0.1:8765/v1/chat/completions",
        "http://user@127.0.0.1:8765/v1/chat/completions",
        "http://token:@localhost:8765/v1/chat/completions",
    ],
)
def test_a_userinfo_bearing_url_is_refused(endpoint: str) -> None:
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8765/v1/chat/completions?api_key=secret",
        "http://127.0.0.1:8765/v1/chat/completions#fragment",
        "file:///etc/passwd",
        "ftp://127.0.0.1/v1/chat/completions",
        "127.0.0.1:8765/v1/chat/completions",
        "",
    ],
)
def test_query_fragment_and_non_http_schemes_are_refused(endpoint: str) -> None:
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint)


def test_the_path_must_match_the_declared_family() -> None:
    validate_local_endpoint("http://127.0.0.1:8765/v1/responses", api_family=RESPONSES)
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("http://127.0.0.1:8765/v1/responses", api_family=CHAT_COMPLETIONS)
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("http://127.0.0.1:8765/v1/chat/completions", api_family=RESPONSES)


def test_an_allowlisted_origin_is_admitted_only_when_the_env_var_names_it(monkeypatch) -> None:
    endpoint = "http://mlx-host.internal:9000/v1/chat/completions"
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint)

    # A different origin in the list does not admit this one.
    monkeypatch.setenv(ALLOWLIST_ENV, "http://other-host.internal:9000")
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint)

    # The port is part of the origin.
    monkeypatch.setenv(ALLOWLIST_ENV, "http://mlx-host.internal:9001")
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint(endpoint)

    monkeypatch.setenv(ALLOWLIST_ENV, "http://a.example:1, http://mlx-host.internal:9000/ ,")
    validate_local_endpoint(endpoint)


def test_https_is_unchanged_from_the_responses_precedent(monkeypatch) -> None:
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    # https never gets the loopback exemption — the precedent only exempts http.
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("https://127.0.0.1:8765/v1/chat/completions")
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        validate_local_endpoint("https://mlx.example/v1/chat/completions")
    monkeypatch.setenv(ALLOWLIST_ENV, "https://mlx.example")
    validate_local_endpoint("https://mlx.example/v1/chat/completions")


def test_api_family_defaults_to_chat_completions_and_refuses_unknown_names() -> None:
    assert normalize_api_family(None) == CHAT_COMPLETIONS
    assert normalize_api_family("") == CHAT_COMPLETIONS
    assert normalize_api_family("Responses") == RESPONSES
    with pytest.raises(RuntimeError, match="synth_mlx_rl_api_family_unsupported"):
        normalize_api_family("completions")
    with pytest.raises(RuntimeError, match="synth_mlx_rl_api_family_unsupported"):
        normalize_api_family(7)


def test_a_base_url_that_already_names_the_route_is_not_doubled() -> None:
    assert local_endpoint("http://127.0.0.1:8765/v1/responses", api_family=RESPONSES) == (
        "http://127.0.0.1:8765/v1/responses"
    )
    with pytest.raises(RuntimeError, match="synth_mlx_rl_base_url_missing"):
        local_endpoint("")


# --- call site 1: banking77 -------------------------------------------------


def _banking77_client() -> TestClient:
    return TestClient(create_compat_app("banking77_classify"))


def _run_banking77(client: TestClient, rollout_id: str, config_id: str) -> dict:
    prepared = client.post(
        "/rollouts/prepare", json={"rollout_id": rollout_id, "telemetry": TELEMETRY}
    )
    assert prepared.status_code == 200, prepared.text
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "classify", "config": config_id},
        },
    )
    assert started.status_code == 200, started.text
    return started.json()


def _banking77_error_code(client: TestClient, rollout_id: str) -> str | None:
    events = client.get(f"/rollouts/{rollout_id}/events", params={"after": 0}).json()["events"]
    closed = next(row for row in events if row["kind"] == "span.policy.closed")
    return closed["payload"].get("error_code")


@pytest.mark.parametrize("family", [CHAT_COMPLETIONS, RESPONSES])
def test_banking77_admits_the_local_provider_on_both_families(monkeypatch, family) -> None:
    from synth_containers.platform.banking77_world import load_row

    gold = load_row("heldout", 0)
    assert gold is not None
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            if family == RESPONSES:
                return json.dumps({"output_text": gold.label}).encode()
            return json.dumps({"choices": [{"message": {"content": gold.label}}]}).encode()

    def fake_urlopen(request, *, timeout):
        del timeout
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = _banking77_client()
    config: dict[str, object] = {
        "provider": PROVIDER_ID,
        "model": "Qwen/Qwen3.5-0.8B",
        "api_family": family,
        "base_url": "http://127.0.0.1:8765/v1",
    }
    if family == RESPONSES:
        config["responses_api_key"] = "rollout-scoped-token"
        config["responses_idempotency_key"] = "roll-a-generation-1"
    registered = client.post(
        "/policy-configs",
        json={"config_id": f"mlx_{family}", "harness": "classify", "config": config},
    )
    assert registered.status_code == 200, registered.text
    started = _run_banking77(client, f"b77_mlx_{family}", f"mlx_{family}")
    assert started["status"] == "completed"
    expected = "/responses" if family == RESPONSES else "/chat/completions"
    assert captured["url"] == f"http://127.0.0.1:8765/v1{expected}"
    scored = client.post(
        "/reward", json={"rollout_id": f"b77_mlx_{family}", "mode": "terminal"}
    ).json()
    assert scored["reward"] == 1.0


@pytest.mark.parametrize(
    ("base_url", "family"),
    [
        ("http://exfil.example/v1", CHAT_COMPLETIONS),
        ("http://user:pass@127.0.0.1:8765/v1", CHAT_COMPLETIONS),
        ("http://exfil.example/v1", RESPONSES),
    ],
)
def test_banking77_refuses_a_bad_local_endpoint_before_the_network(
    monkeypatch, base_url, family
) -> None:
    def unexpected(*_args, **_kwargs):  # pragma: no cover - the point is it is never reached
        raise AssertionError("network must not run")

    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    client = _banking77_client()
    client.post(
        "/policy-configs",
        json={
            "config_id": "mlx_refused",
            "harness": "classify",
            "config": {
                "provider": PROVIDER_ID,
                "model": "Qwen/Qwen3.5-0.8B",
                "api_family": family,
                "base_url": base_url,
                "responses_api_key": "token",
                "responses_idempotency_key": "key",
            },
        },
    )
    started = _run_banking77(client, "b77_mlx_refused", "mlx_refused")
    assert started["status"] == "failed"
    assert _banking77_error_code(client, "b77_mlx_refused") == "synth_mlx_rl_endpoint_refused"
    scored = client.post(
        "/reward", json={"rollout_id": "b77_mlx_refused", "mode": "terminal"}
    ).json()
    assert scored["reward"] is None


def test_banking77_still_refuses_unknown_providers_and_hosted_url_swaps() -> None:
    from synth_containers.platform.runtimes.banking77 import _sample_chat_completion

    with pytest.raises(RuntimeError, match="banking77_provider_unsupported"):
        _sample_chat_completion({}, {"provider": "anthropic", "model": "m"})
    with pytest.raises(RuntimeError, match="banking77_chat_endpoint_refused"):
        _sample_chat_completion(
            {},
            {
                "provider": "openai",
                "model": "m",
                "api_key_env": "PATH",
                "base_url": "http://127.0.0.1:8765/v1",
            },
        )


# --- call site 2: react -----------------------------------------------------


def test_react_admits_the_local_provider_over_loopback() -> None:
    policy = OpenRouterReAct(config_id="mlx_local", config=_react_config())
    assert policy.provider == PROVIDER_ID
    assert policy.chat_endpoint == "http://127.0.0.1:8765/v1/chat/completions"


def test_react_admits_an_allowlisted_origin_only_when_named(monkeypatch) -> None:
    config = _react_config(base_url="http://mlx-host.internal:9000/v1")
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    with pytest.raises(PolicyConfigError, match="synth_mlx_rl_endpoint_refused"):
        OpenRouterReAct(config_id="mlx_allow", config=config)
    monkeypatch.setenv(ALLOWLIST_ENV, "http://mlx-host.internal:9000")
    policy = OpenRouterReAct(config_id="mlx_allow", config=config)
    assert policy.chat_endpoint == "http://mlx-host.internal:9000/v1/chat/completions"


def test_react_refuses_a_public_origin_and_a_userinfo_url(monkeypatch) -> None:
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    with pytest.raises(PolicyConfigError, match="synth_mlx_rl_endpoint_refused"):
        OpenRouterReAct(config_id="mlx_bad", config=_react_config(base_url="http://exfil.example/v1"))
    with pytest.raises(PolicyConfigError, match="synth_mlx_rl_endpoint_refused"):
        OpenRouterReAct(
            config_id="mlx_bad",
            config=_react_config(base_url="http://user:pass@127.0.0.1:8765/v1"),
        )


def test_react_never_defaults_the_api_family() -> None:
    with pytest.raises(PolicyConfigError, match="api_family"):
        config = _react_config()
        del config["api_family"]
        OpenRouterReAct(config_id="mlx_nofamily", config=config)


def test_react_speaks_the_responses_family(monkeypatch) -> None:
    """The planner renders a Responses body and reads a function call back.

    Three things differ from chat and each one silently breaks a rollout if
    missed: the tool is flat rather than nested under `function`, the cap is
    `max_output_tokens`, and usage arrives as input/output_tokens.
    """
    policy = OpenRouterReAct(config_id="mlx_resp", config=_react_config(api_family=RESPONSES))
    assert policy.chat_endpoint == "http://127.0.0.1:8765/v1/responses"

    sent: dict[str, object] = {}
    arguments = '{"actions":["noop"]}'
    # Built with json.dumps rather than hand-escaped: the tool arguments are a
    # JSON string nested inside a JSON event, and getting that wrong in a
    # fixture produces a test that fails for a reason the code never had.
    events = "\n\n".join(
        [
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "thinking"}),
            "data: "
            + json.dumps(
                {"type": "response.function_call_arguments.delta", "delta": arguments}
            ),
            "data: "
            + json.dumps(
                {
                    "type": "response.completed",
                    "response": {"usage": {"input_tokens": 11, "output_tokens": 4}},
                }
            ),
            "data: [DONE]",
        ]
    )
    _stub_urlopen(monkeypatch, sent, events, content_type="text/event-stream")

    body = policy._complete("k", ["noop"], None)

    assert sent["url"] == "http://127.0.0.1:8765/v1/responses"
    payload = sent["json"]
    assert "input" in payload and "messages" not in payload
    assert payload["max_output_tokens"] == 1024 and "max_tokens" not in payload
    tool = payload["tools"][0]
    # Flat, not {"type":"function","function":{...}} as chat requires.
    assert tool["type"] == "function" and tool["name"] == "choose_actions"
    assert "function" not in tool

    message = body["choices"][0]["message"]
    assert message["content"] == "thinking"
    assert message["tool_calls"][0]["function"]["arguments"] == arguments
    # Compaction triggers on prompt_tokens; an unmapped usage block reads as a
    # context that never grows, so the transcript would never be compacted.
    assert body["usage"]["prompt_tokens"] == 11
    assert body["usage"]["completion_tokens"] == 4
    assert body["usage"]["total_tokens"] == 15


def test_react_responses_reads_a_terminal_body_with_no_argument_deltas(monkeypatch) -> None:
    """A server may stream nothing and emit only the final response. The plan
    must still survive, or the turn silently falls back to text parsing."""
    policy = OpenRouterReAct(config_id="mlx_resp2", config=_react_config(api_family=RESPONSES))
    sent: dict[str, object] = {}
    body_json = json.dumps(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "choose_actions",
                    "arguments": '{"actions":["noop","noop"]}',
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    )
    _stub_urlopen(monkeypatch, sent, body_json, content_type="application/json")
    result = policy._complete("k", ["noop"], None)
    calls = result["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["arguments"] == '{"actions":["noop","noop"]}'
    assert result["usage"]["prompt_tokens"] == 7


def test_react_openrouter_and_tinker_paths_are_unchanged() -> None:
    openrouter = OpenRouterReAct(
        config_id="luna_med",
        config=_react_config(provider="openrouter", base_url="https://openrouter.ai/api/v1"),
    )
    assert openrouter.chat_endpoint == "https://openrouter.ai/api/v1/chat/completions"
    with pytest.raises(PolicyConfigError, match="provider"):
        OpenRouterReAct(config_id="nope", config=_react_config(provider="openai"))


# --- call site 3: healthbench ----------------------------------------------


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _stub_client(monkeypatch, captured: dict, payload: dict) -> None:
    class Stub:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def post(self, target, *, headers, json, timeout):  # noqa: A002
            del timeout
            captured["base"] = self.base_url
            captured["target"] = target
            captured["headers"] = headers
            captured["json"] = json
            return _StubResponse(payload)

    monkeypatch.setattr(healthbench, "_client", Stub)


def test_healthbench_admits_the_local_provider_on_both_families(monkeypatch) -> None:
    captured: dict = {}
    _stub_client(monkeypatch, captured, {"choices": [{"message": {"content": "hello"}}]})
    result = healthbench._chat(
        {"provider": PROVIDER_ID, "model": "m", "base_url": "http://127.0.0.1:8765/v1"},
        [{"role": "user", "content": "hi"}],
    )
    assert result["text"] == "hello"
    assert captured["target"] == "http://127.0.0.1:8765/v1/chat/completions"
    # No key env is set, and none is invented for a loopback proxy.
    assert captured["headers"] == {}

    captured.clear()
    monkeypatch.setenv(ALLOWLIST_ENV, "http://host.docker.internal:8765")
    _stub_client(monkeypatch, captured, {"output_text": "hello responses"})
    result = healthbench._chat(
        {
            "provider": PROVIDER_ID,
            "model": "m",
            "api_family": RESPONSES,
            "base_url": "http://host.docker.internal:8765/v1",
        },
        [{"role": "user", "content": "hi"}],
    )
    assert result["text"] == "hello responses"
    assert captured["target"] == "http://host.docker.internal:8765/v1/responses"
    assert "input" in captured["json"] and "messages" not in captured["json"]


def test_healthbench_refuses_a_public_local_provider_origin(monkeypatch) -> None:
    monkeypatch.delenv(ALLOWLIST_ENV, raising=False)
    captured: dict = {}
    _stub_client(monkeypatch, captured, {})
    with pytest.raises(RuntimeError, match="synth_mlx_rl_endpoint_refused"):
        healthbench._chat(
            {"provider": PROVIDER_ID, "model": "m", "base_url": "http://exfil.example/v1"},
            [{"role": "user", "content": "hi"}],
        )
    assert captured == {}


def test_healthbench_hosted_providers_keep_their_permissiveness(monkeypatch) -> None:
    captured: dict = {}
    _stub_client(monkeypatch, captured, {"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setenv("SOME_KEY", "opaque")
    healthbench._chat(
        {
            "provider": "some_vendor",
            "model": "m",
            "base_url": "https://anything.example/v1",
            "api_key_env": "SOME_KEY",
        },
        [{"role": "user", "content": "hi"}],
    )
    assert captured["base"] == "https://anything.example/v1"
    assert captured["target"] == "/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer opaque"}

    with pytest.raises(RuntimeError, match="groq_api_key_missing"):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        healthbench._chat({"model": "m"}, [{"role": "user", "content": "hi"}])


def test_httpx_is_still_the_healthbench_transport() -> None:
    # Guards the stub above: if `_client` stops returning an httpx client the
    # stub would be testing nothing.
    assert isinstance(healthbench._client("http://127.0.0.1:1/v1"), httpx.Client)


def test_openrouter_only_fields_do_not_reach_other_providers(monkeypatch) -> None:
    """`reasoning` is an OpenRouter extension, not part of the OpenAI chat
    schema. Sending it to another provider asks for a field that provider never
    defined; a strict server rejects the whole call with a 422 that reads as a
    policy failure. Found against the real local service, which forbids extras.
    """
    sent: dict[str, object] = {}
    _stub_urlopen(
        monkeypatch,
        sent,
        json.dumps({"choices": [{"message": {"content": '{"actions":["noop"]}'}}]}),
        content_type="application/json",
    )
    local = OpenRouterReAct(config_id="mlx_r", config=_react_config())
    local._complete("k", ["noop"], None)
    assert "reasoning" not in sent["json"]

    sent.clear()
    _stub_urlopen(
        monkeypatch,
        sent,
        json.dumps({"choices": [{"message": {"content": '{"actions":["noop"]}'}}]}),
        content_type="application/json",
    )
    hosted = OpenRouterReAct(
        config_id="luna",
        config=_react_config(provider="openrouter", base_url="https://openrouter.ai/api/v1"),
    )
    hosted._complete("k", ["noop"], None)
    assert sent["json"]["reasoning"] == {"effort": "medium"}
