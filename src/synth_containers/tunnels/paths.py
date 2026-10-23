"""Origin path confinement for HTTP and WebSocket forwarding.

A locally approved origin's directory prefix is part of its authority. Remote
request paths must never replace its host or climb out of that directory.
"""

import re
from urllib.parse import unquote, urlsplit


def confined_origin_url(base_url: str, path: str, query: str) -> str:
    relative = path.lstrip("/")
    parsed = urlsplit(relative)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or "?" in relative
        or "#" in relative
    ):
        raise ValueError("invalid_origin_path")
    decoded = relative
    # Bound nested decoding: intermediaries may decode more than once. Refuse
    # deeper encodings rather than treating an unchecked suffix as safe.
    for _ in range(8):
        if "\\" in decoded or any(
            ord(character) < 32 or ord(character) == 127 for character in decoded
        ):
            raise ValueError("invalid_origin_path")
        if any(segment in {".", ".."} for segment in decoded.split("/")):
            raise ValueError("invalid_origin_path")
        next_value = unquote(decoded, errors="strict")
        if next_value == decoded:
            break
        decoded = next_value
    else:
        raise ValueError("origin_path_encoding_depth")
    # Concatenation preserves the trusted authority and directory. urljoin can
    # replace either one before the origin credential is attached.
    upstream = base_url.rstrip("/") + "/" + relative
    return upstream + "?" + query if query else upstream


# A worker-supplied forward path is only ever a path *under* the leased target.
# Anything that carries a URI scheme (``http:``, ``file:``, ``ws:`` …) escapes it.
_FORWARD_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


def confined_forward_path(path: str) -> str:
    """Reduce a worker-supplied forward path to a plain path under the target.

    The rust gateway forwards ``/s/{route}/{*path}`` verbatim, so ``path`` is
    attacker-controlled. ``urljoin`` treats a value that is itself an absolute
    URL (``/http://evil/x``) or an authority (``//evil/x``) as replacing the
    base, which is an SSRF primitive. Reject any scheme, authority, or upward
    traversal and return only a relative path confined beneath the lease.
    """

    raw = str(path or "/").replace("\\", "/")
    stripped = raw.lstrip("/")
    if not stripped:
        return "/"
    # ``scheme://host`` / ``ws://…`` / ``file:///…`` — an absolute URL; a bare
    # ``//authority``; or any ``scheme:`` prefix. None is a relative path.
    if (
        "://" in stripped
        or stripped.startswith("//")
        or _FORWARD_SCHEME_RE.match(stripped)
    ):
        return "/"
    trailing = stripped.endswith("/")
    segments: list[str] = []
    for segment in stripped.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            # Refuse to let ``..`` climb above the lease's own path prefix.
            return "/"
        segments.append(segment)
    result = "/" + "/".join(segments)
    if trailing and not result.endswith("/"):
        result += "/"
    return result
