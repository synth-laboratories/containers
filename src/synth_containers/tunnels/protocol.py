"""Canonical Synth Tunnel v2 binary data envelope (matches Rust vectors)."""

from __future__ import annotations
from dataclasses import dataclass
import struct
import uuid

from .wire_generated import SUBPROTOCOL, CHUNK_BYTES, MAGIC, HEADER_FORMAT, DataKind

__all__ = ["DataFrame", "DataKind", "SUBPROTOCOL", "CHUNK_BYTES"]

HEADER = struct.Struct(HEADER_FORMAT)


@dataclass(frozen=True, slots=True)
class DataFrame:
    kind: DataKind
    stream: uuid.UUID
    generation: int
    final_chunk: bool
    payload: bytes

    def encode(self) -> bytes:
        if self.stream.int == 0 or not 0 < self.generation < 2**64:
            raise ValueError("invalid stream or attachment identity")
        if len(self.payload) > CHUNK_BYTES:
            raise ValueError("binary frame exceeds chunk limit")
        return (
            HEADER.pack(
                MAGIC,
                self.kind,
                int(self.final_chunk),
                0,
                self.stream.bytes,
                self.generation,
                len(self.payload),
            )
            + self.payload
        )

    @classmethod
    def decode(cls, raw: bytes) -> DataFrame:
        if len(raw) < HEADER.size:
            raise ValueError("truncated binary frame")
        magic, kind, final, reserved, stream, generation, length = HEADER.unpack_from(raw)
        if (
            magic != MAGIC
            or final not in (0, 1)
            or reserved
            or length > CHUNK_BYTES
            or len(raw) != HEADER.size + length
        ):
            raise ValueError("invalid binary frame header or length")
        identity = uuid.UUID(bytes=stream)
        if identity.int == 0 or generation == 0:
            raise ValueError("invalid stream or attachment identity")
        return cls(DataKind(kind), identity, generation, bool(final), raw[HEADER.size :])
