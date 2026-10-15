"""Read an ONNX model's metadata without loading it.

sherpa-onnx checks a model's ``model_type`` metadata natively and ends the
process when it does not match what the config asked for, out of reach of
any ``except``.  Reading the same metadata first lets a provider refuse the
file with an exception instead.  No ``onnx`` dependency: the metadata is a
top-level field of the ``ModelProto`` protobuf, read here while the graph
(the bulk of the file) is skipped.
"""

from __future__ import annotations

import os
from typing import BinaryIO

_METADATA_PROPS = 14  # ModelProto.metadata_props: repeated StringStringEntryProto


def read_onnx_metadata(path: str) -> dict[str, str]:
    """The ``metadata_props`` of the ONNX model at *path*, as a dict.

    Raises:
        OSError: The file cannot be read.
        ValueError: The file is not an ONNX model (not a protobuf it can walk).
    """
    metadata: dict[str, str] = {}
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        while (key := _next_key(f)) is not None:
            field, wire = key >> 3, key & 7
            if wire == 0:
                _read_varint(f)
            elif wire == 1:
                f.seek(8, 1)
            elif wire == 5:
                f.seek(4, 1)
            elif wire == 2:
                length = _read_varint(f)
                # A corrupt length past the file's end would make the read
                # allocate it (MemoryError) or the seek overflow.
                if length > size - f.tell():
                    raise ValueError(f"{path} ends inside a field")
                if field == _METADATA_PROPS:
                    name, value = _entry(f.read(length))
                    metadata[name] = value
                else:
                    f.seek(length, 1)
            else:
                raise ValueError(f"{path} is not an ONNX model")
    return metadata


def _next_key(f: BinaryIO) -> int | None:
    """The next field's key, or None at the end of the file."""
    if not f.read(1):
        return None
    f.seek(-1, 1)
    return _read_varint(f)


def _read_varint(f: BinaryIO) -> int:
    result = shift = 0
    while True:
        byte = f.read(1)
        if not byte:
            raise ValueError("truncated protobuf")
        result |= (byte[0] & 0x7F) << shift
        if not byte[0] & 0x80:
            return result
        shift += 7
        if shift > 63:
            raise ValueError("malformed protobuf varint")


def _entry(data: bytes) -> tuple[str, str]:
    """A StringStringEntryProto: key (field 1) and value (field 2)."""
    fields: dict[int, str] = {}
    i = 0
    while i < len(data):
        key, i = _varint_at(data, i)
        field, wire = key >> 3, key & 7
        if wire != 2:
            raise ValueError("malformed ONNX metadata entry")
        length, i = _varint_at(data, i)
        if i + length > len(data):
            raise ValueError("truncated ONNX metadata entry")
        fields[field] = data[i : i + length].decode("utf-8", "replace")
        i += length
    return fields.get(1, ""), fields.get(2, "")


def _varint_at(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while i < len(data):
        byte = data[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7
    raise ValueError("truncated ONNX metadata entry")
