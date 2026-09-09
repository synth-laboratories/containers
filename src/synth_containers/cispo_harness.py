"""Handing a shipped harness a per-attempt sampler instead of a global one.

The harnesses in ``synth_containers.policies`` were written for a fixed
provider: they default ``base_url`` to that provider's public endpoint and read
their credential from ``os.environ[api_key_env]``, which is a process-wide
name. A CISPO binding gives the container something else entirely -- a
:class:`~synth_containers.cispo_policy.SamplerOriginV1` whose path carries the
attempt's own ``proxy_request_id``, plus a credential scoped to that one
session. Both are per attempt. Neither belongs in a process that outlives the
attempt or is shared with another one.

Two hooks already exist and this module uses exactly those two, changing
nothing in either harness:

* every one of them reads ``base_url`` and ``api_key_env`` from its config
  dict, so the bound origin goes in as configuration;
* ``platform.policy_process.IsolatedPolicyProcess`` builds its child
  environment explicitly rather than inheriting one, so the credential goes in
  as one entry of a child environment that starts empty.

**The isolation is what makes the environment variable safe, and it is
per attempt.** :class:`BoundHarnessProcess` spawns one child per launch, hands
it a whitelisted environment carrying exactly one credential, and kills it at
``close``. A shared harness process could not hold two attempts' credentials
under one variable name without racing, so there is no shared-process path
here: a launch is a process, and a process is an attempt.

Three rules close the leak paths the config-plus-env split opens:

1. **The credential never enters the config.** :meth:`BoundHarnessLaunch.config`
   returns the origin, the model and the *name* of the variable, and
   :func:`assert_config_carries_no_credential` refuses a config that carries
   the value anyway. A config is written to disk, echoed into a trace, and
   logged; a credential in one is a credential in all three.
2. **The variable is never a global provider name.** A per-attempt credential
   published as ``OPENAI_API_KEY`` is picked up by anything that inherits the
   environment, which is the thing process isolation was for.
   :data:`GLOBAL_CREDENTIAL_ENV_NAMES` is refused.
3. **The child environment is built, not inherited.** ``PATH`` and the import
   root are named explicitly; nothing else from this process crosses over, so
   an ambient provider key cannot silently satisfy a harness whose bound
   sampler is down.

No task or environment name appears in this module.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cispo_policy import SamplerBinding, SamplerOriginV1

HARNESS_LAUNCH_SCHEMA_VERSION = "cispo.harness_launch.v1"

#: The default variable a bound harness reads its carried credential from. It
#: is deliberately not any provider's conventional name: see rule 2 above.
BOUND_CREDENTIAL_ENV = "SYNTH_CISPO_SESSION_CREDENTIAL"

#: Process-wide provider variable names. A session credential published under
#: one of these is inherited by every child of this process.
GLOBAL_CREDENTIAL_ENV_NAMES: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "TINKER_API_KEY",
        "SYNTH_API_KEY",
    }
)

#: Which wire a shipped harness actually speaks. The two shapes are two
#: datasets, so a ``responses`` binding handed to a chat-completions harness is
#: a refusal rather than a base-url swap.
HARNESS_WIRE_APIS: Mapping[str, frozenset[str]] = {
    "single_call": frozenset({"chat_completions", "responses"}),
    "react": frozenset({"chat_completions"}),
    "responses_react": frozenset({"responses"}),
    "mini_swe": frozenset({"chat_completions"}),
}

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class BoundHarnessError(RuntimeError):
    """A harness launch was refused. Never degrade one of these to a rollout."""


class CredentialLeak(BoundHarnessError):
    """A carried credential reached somewhere it is never allowed to be."""


# --------------------------------------------------------------------------- #
# The one place the carried value is materialized
# --------------------------------------------------------------------------- #


def carried_credential(origin: SamplerOriginV1) -> str:
    """The session credential, read out for the child environment and nowhere else.

    :class:`~synth_containers.cispo_policy.SamplerCredential` closes every other
    path to the value on purpose -- repr, str, format, equality, ``asdict``,
    pickle -- and leaves exactly one: the ``Authorization`` header the outbound
    request needs. This function is the container's second and last use of that
    one exit, and its result goes straight into a per-attempt process
    environment without passing through a payload, a log line, or a file.
    """

    header = str(origin.credential.authorization().get("Authorization") or "")
    scheme, _, value = header.partition(" ")
    if scheme != "Bearer" or not value.strip():
        raise BoundHarnessError(
            "the bound sampler origin carries no bearer credential; a harness that "
            "falls back to an ambient key is sampling something nobody bound"
        )
    return value


def assert_config_carries_no_credential(
    config: Mapping[str, Any], credential: str
) -> None:
    """Refuse a config that carries the value anywhere inside it.

    Checked against the serialized config rather than its top-level keys: a
    credential nested in an ``inference_target`` block is written to the same
    disk and echoed into the same trace as one at the top.
    """

    if not credential:
        return
    blob = json.dumps(config, sort_keys=True, default=str)
    if credential in blob:
        raise CredentialLeak(
            "the harness config carries the session credential; a config is written "
            "to disk, echoed into a trace, and logged, so a credential in one is a "
            "credential in all three"
        )


# --------------------------------------------------------------------------- #
# The launch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BoundHarnessLaunch:
    """One shipped harness, configured from one bound policy, for one attempt.

    ``config`` and ``environment`` split the binding in two along the line the
    harnesses already draw: the origin is configuration, the credential is
    environment, and only the second of the two ever holds a secret.
    """

    harness: str
    binding: SamplerBinding
    config_id: str = ""
    api_key_env: str = BOUND_CREDENTIAL_ENV
    extra_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.harness not in HARNESS_WIRE_APIS:
            raise BoundHarnessError(
                f"{self.harness!r} is not a harness this module binds; it binds "
                f"{sorted(HARNESS_WIRE_APIS)}"
            )
        if self.api_key_env in GLOBAL_CREDENTIAL_ENV_NAMES:
            raise BoundHarnessError(
                f"{self.api_key_env!r} is a process-wide provider variable; a "
                "session-scoped credential published under one is inherited by every "
                "child of this process"
            )
        if not _ENV_NAME.fullmatch(self.api_key_env):
            raise BoundHarnessError(
                f"{self.api_key_env!r} is not a usable environment variable name"
            )
        if self.binding.origin is None:
            raise BoundHarnessError(
                f"binding {self.binding.binding_id!r} drives a pinned identity with no "
                "sampler origin; there is no session for a harness to sample through"
            )
        if not self.binding.dispatchable:
            raise BoundHarnessError(
                f"binding {self.binding.binding_id!r} is bound but its sampler was not "
                "reachable at bind time; a revision is dispatchable only once its "
                "sampler is ready"
            )
        wires = HARNESS_WIRE_APIS[self.harness]
        if self.binding.wire_api not in wires:
            raise BoundHarnessError(
                f"harness {self.harness!r} speaks {sorted(wires)}, the binding is "
                f"{self.binding.wire_api!r}; the two wire shapes are two datasets and "
                "are not interchangeable"
            )
        if self.binding.sampling_transport != "message_in_capture_out":
            raise BoundHarnessError(
                "the shipped harnesses render their own messages; a "
                f"{self.binding.sampling_transport!r} binding would make the harness a "
                "second renderer"
            )

    # -- the two halves --------------------------------------------------- #

    @property
    def origin(self) -> SamplerOriginV1:
        origin = self.binding.origin
        assert origin is not None  # enforced in __post_init__
        return origin

    def config(self) -> dict[str, Any]:
        """The harness config: the bound origin, and the credential's *name*."""

        origin = self.origin
        config: dict[str, Any] = {
            **dict(self.extra_config),
            "base_url": origin.base_url,
            "api": origin.wire_api,
            "api_key_env": self.api_key_env,
            "model": self.binding.model_id,
        }
        if self.binding.sampling.max_tokens is not None:
            config["max_tokens"] = int(self.binding.sampling.max_tokens)
        assert_config_carries_no_credential(config, carried_credential(origin))
        return config

    def environment(self) -> dict[str, str]:
        """The child's whole environment, built rather than inherited.

        ``PATH`` and the import root are named because the child is a Python
        process that has to start and has to import this package. Everything
        else this process holds -- including any ambient provider key -- stays
        on this side of the boundary.
        """

        return {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": _import_root(),
            "PYTHONDONTWRITEBYTECODE": "1",
            self.api_key_env: carried_credential(self.origin),
        }

    def launch_payload(self) -> dict[str, Any]:
        """What the child is told to build. Never carries the credential."""

        return {
            "schema_version": HARNESS_LAUNCH_SCHEMA_VERSION,
            "harness": self.harness,
            "config_id": self.config_id or self.binding.binding_id,
            "config": self.config(),
        }

    # -- the process ------------------------------------------------------ #

    def spawn(self) -> "BoundHarnessProcess":
        """Start the one child process this attempt's harness runs in."""

        return BoundHarnessProcess(self)


def _import_root() -> str:
    """The directory this package is importable from, named explicitly."""

    return str(Path(__file__).resolve().parent.parent)


# --------------------------------------------------------------------------- #
# The child
# --------------------------------------------------------------------------- #

_SERVE = r'''
from __future__ import annotations
import json, os, sys

def main() -> int:
    launch = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
    from synth_containers.policies import build_planner

    config = dict(launch["config"])
    planner = build_planner(
        launch["harness"], config_id=str(launch["config_id"]), config=config
    )
    sys.stdout.write(json.dumps({"op": "ready", "ok": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        request = json.loads(line)
        op = request.get("op")
        response = {"id": request.get("id"), "ok": True}
        try:
            if op == "close":
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                return 0
            if op == "describe":
                key_env = str(getattr(planner, "api_key_env", ""))
                response["describe"] = {
                    "harness": launch["harness"],
                    "base_url": str(getattr(planner, "base_url", "")),
                    "api_key_env": key_env,
                    # The presence of the carried value, never the value.
                    "credential_present": bool(os.environ.get(key_env, "").strip()),
                    "environment_names": sorted(os.environ),
                    "model": str(getattr(planner, "model", "")),
                    "metadata": planner.metadata(),
                }
            elif op == "plan":
                response["actions"] = list(
                    planner.plan(dict(request.get("observation") or {}))
                )
            else:
                response = {"id": request.get("id"), "ok": False, "error": f"unknown_op:{op}"}
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            response = {
                "id": request.get("id"),
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}",
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


class BoundHarnessProcess:
    """One shipped harness in one child process, for one attempt.

    Satisfies :class:`~synth_containers.cispo_rollout.BackgroundProcess`, so an
    attempt can adopt it with ``register_background_process`` and the lifecycle
    stops it at the horizon -- which is also when the carried credential stops
    existing anywhere, because it existed only in this child's environment.
    """

    def __init__(self, launch: BoundHarnessLaunch) -> None:
        self._launch = launch
        self._sandbox = tempfile.TemporaryDirectory(prefix="synth-cispo-harness-")
        root = Path(self._sandbox.name)
        server = root / "serve.py"
        server.write_text(_SERVE, encoding="utf-8")
        plan = root / "launch.json"
        payload = launch.launch_payload()
        plan.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, str(server), str(plan)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(root),
            env=launch.environment(),
            close_fds=True,
        )
        if self._proc.stdin is None or self._proc.stdout is None:  # pragma: no cover
            raise BoundHarnessError("bound_harness_pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._lock = threading.Lock()
        self._request_id = 0
        ready = self._stdout.readline()
        if not ready:
            detail = ""
            if self._proc.stderr is not None:
                detail = self._proc.stderr.read()[-400:]
            raise BoundHarnessError(
                f"bound_harness_startup_failed:rc={self._proc.poll()}:{detail}"
            )
        first = json.loads(ready)
        if first.get("op") != "ready" or first.get("ok") is not True:
            raise BoundHarnessError("bound_harness_not_ready")
        origin = launch.origin
        self.isolation_receipt = {
            "contract": HARNESS_LAUNCH_SCHEMA_VERSION,
            "platform": sys.platform,
            "sandbox": "process",
            "scope": "per_attempt",
            "credential_scope": "per_attempt_process_environment",
            "credential_variable": launch.api_key_env,
            "sampler_origin": origin.base_url,
            "proxy_request_id": origin.proxy_request_id,
            "binding_id": launch.binding.binding_id,
            "policy_revision": int(launch.binding.policy_revision),
            "environment_inherited": False,
            "pid": self._proc.pid,
        }

    # -- introspection ---------------------------------------------------- #

    @property
    def launch(self) -> BoundHarnessLaunch:
        return self._launch

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def describe(self) -> dict[str, Any]:
        """What the child actually resolved: origin, variable name, presence.

        The credential's *presence* comes back; its value has no path out of
        the child at all.
        """

        return dict(self._exchange({"op": "describe"}).get("describe") or {})

    def plan(self, observation: Mapping[str, Any]) -> list[str]:
        """One planning turn, through the bound session."""

        response = self._exchange({"op": "plan", "observation": dict(observation)})
        return [str(item) for item in response.get("actions") or ()]

    # -- protocol --------------------------------------------------------- #

    def _exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                raise BoundHarnessError("bound_harness_dead")
            self._request_id += 1
            request = {**request, "id": self._request_id}
            self._stdin.write(json.dumps(request) + "\n")
            self._stdin.flush()
            line = self._stdout.readline()
        if not line:
            raise BoundHarnessError("bound_harness_closed_the_stream")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise BoundHarnessError(str(response.get("error") or "bound_harness_failed"))
        return response

    def close(self) -> None:
        """Stop the child. The carried credential stops existing with it."""

        with self._lock:
            if self._proc.poll() is None:
                try:
                    self._request_id += 1
                    self._stdin.write(
                        json.dumps({"id": self._request_id, "op": "close"}) + "\n"
                    )
                    self._stdin.flush()
                    self._proc.wait(timeout=5)
                except Exception:  # noqa: BLE001 - a child that will not stop is killed
                    self._proc.kill()
            self._sandbox.cleanup()

    def __enter__(self) -> "BoundHarnessProcess":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def launch_bound_harness(
    harness: str,
    binding: SamplerBinding,
    *,
    config_id: str = "",
    api_key_env: str = BOUND_CREDENTIAL_ENV,
    extra_config: Mapping[str, Any] | None = None,
) -> BoundHarnessProcess:
    """Build and start one per-attempt bound harness in a single call."""

    return BoundHarnessLaunch(
        harness=harness,
        binding=binding,
        config_id=config_id,
        api_key_env=api_key_env,
        extra_config=dict(extra_config or {}),
    ).spawn()


__all__: Sequence[str] = (
    "BOUND_CREDENTIAL_ENV",
    "GLOBAL_CREDENTIAL_ENV_NAMES",
    "HARNESS_LAUNCH_SCHEMA_VERSION",
    "HARNESS_WIRE_APIS",
    "BoundHarnessError",
    "BoundHarnessLaunch",
    "BoundHarnessProcess",
    "CredentialLeak",
    "assert_config_carries_no_credential",
    "carried_credential",
    "launch_bound_harness",
)
