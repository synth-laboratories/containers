"""Delegated routes cannot escape a local origin's host or parent directory."""

import pytest

from synth_containers.tunnels.paths import confined_origin_url


@pytest.mark.parametrize(
    "path",
    [
        "../sibling",
        "artifacts/../../sibling",
        "%2e%2e/sibling",
        "%252e%252e/sibling",
        "%2e%2e%2fsibling",
        "..%5csibling",
        "http://unapproved.invalid/work",
        "https://unapproved.invalid/work",
        "artifacts/file?other=1",
        "artifacts/file#other",
        "%00file",
        "file\\sibling",
    ],
)
def test_refuses_origin_escape(path):
    with pytest.raises(ValueError):
        confined_origin_url("http://127.0.0.1:1234/v1/runs/parent", path, "")


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/",
        "optimizer-events/stream",
        "artifacts/checkpoint%20one",
        "nested//file",
        "file%25.txt",
    ],
)
def test_preserves_trusted_parent_and_query(path):
    assert (
        confined_origin_url("http://127.0.0.1:1234/v1/runs/parent/", path, "cursor=next")
        == "http://127.0.0.1:1234/v1/runs/parent/" + path.lstrip("/") + "?cursor=next"
    )
