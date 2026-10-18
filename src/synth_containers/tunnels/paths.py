"""Origin path confinement for HTTP and WebSocket forwarding.

A locally approved origin's directory prefix is part of its authority. Remote
request paths must never replace its host or climb out of that directory.
"""

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
