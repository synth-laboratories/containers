"""TS-E is a Workshop consumer gate. This kit does not claim it."""

from __future__ import annotations

import pytest

TS_E = (
    "TS-E01",
    "TS-E02",
    "TS-E03",
    "TS-E04",
    "TS-E05",
    "TS-E06",
    "TS-E07",
    "TS-E08",
)


@pytest.mark.parametrize("test_id", TS_E)
def test_workshop_consumer_gate(test_id: str) -> None:
    pytest.skip(
        f"{test_id} is Workshop/Desktop (A1/A5). "
        "Containers owns the producer profile in Gates A–D."
    )
