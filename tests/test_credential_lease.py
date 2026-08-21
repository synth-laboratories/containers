from synth_containers.http_models import RolloutPolicySpecModel
from synth_containers.proxying import (
    WORKSHOP_API_KEY_SENTINEL,
    CredentialMode,
    resolve_proxied_inference_url,
    sdk_base_url,
)


def test_credential_mode_workshop_proxy_aliases_proxy() -> None:
    assert CredentialMode.parse("workshop_proxy") is CredentialMode.WORKSHOP_PROXY
    assert CredentialMode.parse("proxy") is CredentialMode.WORKSHOP_PROXY
    assert CredentialMode.parse("proxy").is_proxied()
    assert not CredentialMode.parse("byok").is_proxied()


def test_workshop_proxy_requires_inference_url() -> None:
    try:
        RolloutPolicySpecModel(
            provider="openai",
            model="gpt-4.1-nano",
            credential_mode="workshop_proxy",
        )
    except Exception as exc:
        assert "inference_url" in str(exc)
    else:
        raise AssertionError("missing inference_url must fail closed")


def test_workshop_proxy_uses_inference_url_as_sdk_base() -> None:
    url = "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai/chat/completions"
    policy = RolloutPolicySpecModel(
        provider="openai",
        model="gpt-4.1-nano",
        credential_mode="workshop_proxy",
        inference_url=url,
    )
    assert resolve_proxied_inference_url(policy) == sdk_base_url(url)
    assert WORKSHOP_API_KEY_SENTINEL == "workshop-proxy"


def test_workshop_proxy_rejects_openai_origin() -> None:
    try:
        resolve_proxied_inference_url(
            {
                "credential_mode": "workshop_proxy",
                "inference_url": "https://api.openai.com/v1",
            }
        )
    except ValueError as exc:
        assert "api.openai.com" in str(exc)
    else:
        raise AssertionError("hosted OpenAI origin must fail closed")


def test_workload_proxy_base_honors_env_and_rejects_openai(monkeypatch) -> None:
    from synth_containers.proxying import workload_proxy_base

    monkeypatch.setenv("WORKSHOP_CREDENTIAL_MODE", "workshop_proxy")
    monkeypatch.setenv(
        "WORKSHOP_OPENAI_BASE_URL",
        "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai",
    )
    assert (
        workload_proxy_base({"provider": "openai"})
        == "http://host.docker.internal:9/cap/wcap_x/v1/providers/openai"
    )
    monkeypatch.setenv("WORKSHOP_OPENAI_BASE_URL", "https://api.openai.com/v1")
    try:
        workload_proxy_base({})
    except ValueError as exc:
        assert "api.openai.com" in str(exc)
    else:
        raise AssertionError("hosted OpenAI origin must fail closed")
