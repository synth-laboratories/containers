from synth_containers.policies.codex_agentic import CodexAgenticPolicy, _agent_prompt


def test_codex_agentic_pins_reasoning_effort_in_trace_and_cli() -> None:
    policy = CodexAgenticPolicy(
        config_id="agentic_codex_luna_med",
        config={
            "model": "openai/gpt-5.6-luna",
            "effort": "medium",
            "base_url": "http://proxy.test/v1/providers/openrouter",
            "provider_id": "workshop_proxy",
        },
    )

    assert policy.metadata()["reasoning_effort"] == "medium"
    assert policy._exec_overrides() == [
        "-c",
        'model_reasoning_effort="medium"',
        "-c",
        "model_provider=workshop_proxy",
        "-c",
        'model_providers.workshop_proxy.name="workshop_proxy"',
        "-c",
        'model_providers.workshop_proxy.base_url="http://proxy.test/v1/providers/openrouter"',
        "-c",
        'model_providers.workshop_proxy.env_key="OPENAI_API_KEY"',
        "-c",
        'model_providers.workshop_proxy.wire_api="responses"',
    ]


def test_codex_agentic_refuses_unknown_reasoning_effort() -> None:
    try:
        CodexAgenticPolicy(config_id="bad", config={"effort": "max"})
    except RuntimeError as exc:
        assert str(exc) == "codex_reasoning_effort_invalid:max"
    else:
        raise AssertionError("unknown effort must be refused")


def test_codex_agentic_treats_done_as_terminal_for_workspace_tasks() -> None:
    prompt = _agent_prompt(
        env_name="DeepSWE",
        objective="Solve the task.",
        valid=["done"],
        plan_max=5,
    )

    assert "`done` is only the terminal signal" in prompt
    assert "implement the requested change" in prompt
    assert "run the relevant tests" in prompt


def test_codex_agentic_keeps_action_environments_concise() -> None:
    prompt = _agent_prompt(
        env_name="Craftax",
        objective="Make progress.",
        valid=["left", "right"],
        plan_max=5,
    )

    assert "`done` is only the terminal signal" not in prompt
