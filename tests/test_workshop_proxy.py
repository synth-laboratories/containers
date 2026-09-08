from synth_containers.policies.workshop_proxy import (
    is_workshop_capability_proxy,
    public_proxy_bearer,
)


def test_only_scoped_loopback_capability_routes_receive_public_bearer() -> None:
    route = "http://host.docker.internal:17654/cap/wcap_run_123/v1/providers/openrouter"
    assert is_workshop_capability_proxy(route)
    assert public_proxy_bearer(route) == "workshop-proxy"

    assert public_proxy_bearer("https://openrouter.ai/api/v1") is None
    assert public_proxy_bearer("http://attacker.example/cap/wcap_run_123/v1/providers/openrouter") is None
    assert public_proxy_bearer("http://host.docker.internal:17654/not-a-capability") is None
