"""`NestedTrialRuntime._trial_for`: the rollout gets the trial it named.

A nested platform runs one of several trial images. Resolving the wrong one is
not a visible failure — the run completes and reports a reward under the id the
caller asked for, so a wrong number looks like a real one. These pin the
selector and the refusal.
"""

from __future__ import annotations

import pytest

from synth_containers.nested import NestedError
from synth_containers.nested_runtime import NestedTrialRuntime, TrialImage


def _trial(trial_id: str) -> TrialImage:
    return TrialImage(id=trial_id, image=f"{trial_id}:local", allow_unpinned=True)


IMAGES = {"gb-cpo-craftax": _trial("gb-cpo-craftax"), "gb-cpo-rogue": _trial("gb-cpo-rogue")}


class _Pin:
    def __init__(self, **fields: object) -> None:
        self.recipe = None
        self.metadata = None
        self.task_instance_id = ""
        for key, value in fields.items():
            setattr(self, key, value)


def _runtime(*, default_trial: str = "gb-cpo-craftax") -> NestedTrialRuntime:
    return NestedTrialRuntime(
        environment_ref="env:test",
        trial_images=dict(IMAGES),
        default_trial=default_trial,
    )


def test_task_instance_id_selects_the_trial() -> None:
    trial = _runtime()._trial_for(_Pin(task_instance_id="gb-cpo-rogue"))
    assert trial.id == "gb-cpo-rogue"


def test_recipe_overrides_task_instance_id() -> None:
    pin = _Pin(recipe={"trial_image": "gb-cpo-craftax"}, task_instance_id="gb-cpo-rogue")
    assert _runtime()._trial_for(pin).id == "gb-cpo-craftax"


def test_an_unknown_named_trial_refuses_instead_of_defaulting() -> None:
    # The failure this guards: the rollout asked for one environment, silently
    # got the default, and its reward was filed under the name it asked for.
    with pytest.raises(NestedError, match="nested_trial_image_unknown:gb-cpo-nope"):
        _runtime()._trial_for(_Pin(task_instance_id="gb-cpo-nope"))


def test_the_default_applies_only_when_nothing_was_named() -> None:
    assert _runtime()._trial_for(_Pin()).id == "gb-cpo-craftax"


def test_no_name_and_no_default_refuses() -> None:
    with pytest.raises(NestedError, match="nested_trial_image_unnamed"):
        _runtime(default_trial="")._trial_for(_Pin())


def test_a_broken_default_names_itself_as_the_default() -> None:
    with pytest.raises(NestedError, match="nested_default_trial_unknown"):
        _runtime(default_trial="gb-cpo-missing")._trial_for(_Pin())
