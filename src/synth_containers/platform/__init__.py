"""Containers-compat façade: pins, durable logs, /reward, in-process target stubs."""

from .affordances import AffordanceMap, bind_recipe
from .app import create_compat_app
from .extensions import DockEvalExtension, create_dock_eval_app
from .reducer import project_envelopes
from .runtimes.harbor import project_harbor_atif
from .state import CompatPlatform
from .targets import PAID_TARGETS, PR_TARGETS, TARGETS, TargetSpec

__all__ = [
    "AffordanceMap",
    "CompatPlatform",
    "PAID_TARGETS",
    "PR_TARGETS",
    "TARGETS",
    "TargetSpec",
    "bind_recipe",
    "create_compat_app",
    "create_dock_eval_app",
    "DockEvalExtension",
    "project_envelopes",
    "project_harbor_atif",
]
