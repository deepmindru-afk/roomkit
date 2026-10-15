"""An ONNX model's metadata, read without onnx or onnxruntime."""

from __future__ import annotations

from pathlib import Path

import pytest

from roomkit.voice._onnx_metadata import read_onnx_metadata


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def onnx_bytes(**metadata: str) -> bytes:
    """A ModelProto: an ir_version, a graph to skip, then metadata_props."""
    model = _varint(1 << 3) + _varint(8)  # ir_version = 8 (a varint field)
    model += _field(7, b"\x0a" * 3000)  # graph: opaque bytes the reader skips
    for key, value in metadata.items():
        model += _field(14, _field(1, key.encode()) + _field(2, value.encode()))
    return model


def test_metadata_is_read_past_the_graph(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(onnx_bytes(model_type="dpdfnet", sample_rate="16000"))

    assert read_onnx_metadata(str(path)) == {"model_type": "dpdfnet", "sample_rate": "16000"}


def test_a_model_without_metadata_has_none(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(onnx_bytes())

    assert read_onnx_metadata(str(path)) == {}


def test_a_truncated_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(onnx_bytes(model_type="gtcrn")[:-3])  # cut inside the entry

    with pytest.raises(ValueError):
        read_onnx_metadata(str(path))


@pytest.mark.parametrize("length", [2**40, 2**69])
def test_a_length_past_the_end_is_refused(tmp_path: Path, length: int) -> None:
    # A corrupt length made the read allocate it (MemoryError) or the seek
    # overflow (OverflowError): neither is the ValueError the provider catches.
    path = tmp_path / "model.onnx"
    for number in (7, 14):  # a field skipped, and the metadata read
        path.write_bytes(_varint(number << 3 | 2) + _varint(length) + b"\x00" * 16)

        with pytest.raises(ValueError, match="ends inside"):
            read_onnx_metadata(str(path))
