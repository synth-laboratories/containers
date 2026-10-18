from __future__ import annotations
import json
import uuid
from pathlib import Path
import pytest
from synth_containers.tunnels.protocol import DataFrame, DataKind


def test_shared_golden_vectors():
    for vector in json.loads(
        (Path(__file__).parent / "fixtures/tunnel-v2-vectors.json").read_text()
    ):
        wire = bytes.fromhex(vector["wire_hex"])
        decoded = DataFrame.decode(wire)
        assert decoded.stream == uuid.UUID(vector["stream"])
        assert decoded.generation == vector["generation"]
        assert decoded.payload == bytes.fromhex(vector["payload_hex"])
        assert decoded.encode() == wire


def test_invalid_identity_length_and_flags_fail():
    frame = DataFrame(DataKind.RESPONSE_BODY, uuid.UUID(int=42), 3, True, b"hello")
    wire = frame.encode()
    for invalid in [wire[:-1], b"bad", wire[:5] + b"\x02" + wire[6:]]:
        with pytest.raises(ValueError):
            DataFrame.decode(invalid)
    with pytest.raises(ValueError):
        DataFrame(DataKind.REQUEST_BODY, uuid.UUID(int=0), 1, True, b"").encode()
