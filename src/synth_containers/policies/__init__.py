"""Policy harnesses. The **policy** half of the env/policy split.

An image is the ENVIRONMENT: it bakes a world and serves the container contract.
A *policy* is bound per rollout by ``policy_ref = {harness, config}``; the same
image is run by every rung of this ladder without a rebuild.

    single_call             one LLM call decides the whole action sequence
    react                   chat-completions ReAct loop (OpenRouter / Tinker)
    responses_react         OpenAI Responses-API ReAct, state via response id
    mini_swe                chat-completions agent: one bash command per turn
    codex_agentic           Codex CLI agent with a workspace and tools
    isolated_policy_process out-of-process candidate code (GEPA / DEO lanes)
    nanohorizon             PUT policy.py + sampler config; experiments ReAct
    scripted_react          authored plan; engine acceptance, never a graded eval
    valid_action_uniform    seeded legal walk; transport baseline

``build_planner`` resolves most harness names. ``isolated_policy_process`` and
``nanohorizon`` are constructed by the gold runtime (they need PUT ``policy.py``).
Images still do not branch on harness themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .codex_agentic import CodexAgenticPolicy
from .mini_swe import HARNESS as MINI_SWE_HARNESS
from .mini_swe import MiniSweAgent
from .nanohorizon import HARNESS as NANOHORIZON_HARNESS
from .react import OpenRouterReAct, ScriptedReAct, TinkerReAct, UniformEnginePolicy
from .responses import ResponsesReAct
from .single_call import SingleCallPolicy

__all__ = [
    "CodexAgenticPolicy",
    "HARNESSES",
    "MINI_SWE_HARNESS",
    "MiniSweAgent",
    "NANOHORIZON_HARNESS",
    "OpenRouterReAct",
    "Planner",
    "ResponsesReAct",
    "ScriptedReAct",
    "SingleCallPolicy",
    "TinkerReAct",
    "UniformEnginePolicy",
    "UnknownHarness",
    "build_planner",
]


class UnknownHarness(RuntimeError):
    """The rollout named a harness this build does not implement. Never defaulted."""


@runtime_checkable
class Planner(Protocol):
    def plan(self, observation: dict[str, Any], on_delta: Any = None) -> list[str]: ...

    def usage(self) -> dict[str, Any]: ...

    def metadata(self) -> dict[str, Any]: ...


def _react(*, config_id: str, config: dict[str, Any]) -> Any:
    """`react` is chat-completions; Tinker is a sampler variant of the same harness."""

    target = config.get("inference_target")
    if isinstance(target, dict) and str(target.get("provider_endpoint_id") or "").startswith(
        "tinker://"
    ):
        return TinkerReAct(config_id=config_id, config=config)
    return OpenRouterReAct(config_id=config_id, config=config)


HARNESSES: dict[str, Callable[..., Any]] = {
    "single_call": lambda *, config_id, config: SingleCallPolicy(
        config_id=config_id, config=config
    ),
    "react": _react,
    "responses_react": lambda *, config_id, config: ResponsesReAct(
        config_id=config_id, config=config
    ),
    "codex_agentic": lambda *, config_id, config: CodexAgenticPolicy(
        config_id=config_id, config=config
    ),
    MINI_SWE_HARNESS: lambda *, config_id, config: MiniSweAgent(
        config_id=config_id, config=config
    ),
    "scripted_react": lambda *, config_id, config: ScriptedReAct(config_id=config_id),
    "valid_action_uniform": lambda *, config_id, config: UniformEnginePolicy(
        seed=int(config.get("seed") or 0)
    ),
}


def build_planner(harness: str, *, config_id: str, config: dict[str, Any] | None = None) -> Any:
    """Resolve a harness name to a planner. Unknown names refuse; they never default.

    ``isolated_policy_process`` and ``nanohorizon`` are deliberately absent:
    both need the platform's current PUT ``policy.py``. Isolated also refuses a
    config; nanohorizon requires both the revision *and* a sampler config, so
    GoldRuntime constructs it directly.
    """

    name = (harness or "").strip()
    if not name:
        raise UnknownHarness("policy_ref.harness is required; start must not fill a default")
    factory = HARNESSES.get(name)
    if factory is None:
        raise UnknownHarness(f"unknown_policy_harness:{name}")
    return factory(config_id=config_id, config=dict(config or {}))
