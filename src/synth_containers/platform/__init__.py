"""Containers-compat façade: pins, durable logs, /reward, in-process target stubs."""

from .affordances import AffordanceMap, bind_recipe
from .app import create_compat_app
from .extensions import DockEvalExtension, create_dock_eval_app
from .reducer import project_envelopes
from .reward import RewardStreamer, reward_api_catalog
from .runtimes.harbor import project_harbor_atif
from .state import CompatPlatform
from .targets import (
    PAID_TARGETS,
    PR_TARGETS,
    TARGETS,
    RewardCalculatorFamily,
    RewardKind,
    TargetSpec,
    advertised_reward_authority,
    advertised_reward_calculator,
)

__all__ = [
    "AffordanceMap",
    "CompatPlatform",
    "PAID_TARGETS",
    "PR_TARGETS",
    "TARGETS",
    "RewardCalculatorFamily",
    "RewardKind",
    "RewardStreamer",
    "TargetSpec",
    "advertised_reward_authority",
    "advertised_reward_calculator",
    "bind_recipe",
    "create_compat_app",
    "create_dock_eval_app",
    "DockEvalExtension",
    "project_envelopes",
    "project_harbor_atif",
    "reward_api_catalog",
]
