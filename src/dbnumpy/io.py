"""Small, host-memory-bounded parsers for database-native input paths."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class _BinaryHeaderStream(Protocol):
    def readline(self, size: int = -1, /) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class MatrixMarketHeader:
    """Metadata from a coordinate-format Matrix Market header.

    Parsing stops at the size line. Matrix values are deliberately not exposed:
    execution backends can stream the body without first materializing it in Python.
    """

    rows: int
    columns: int
    entries: int
    field: str
    symmetry: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.columns)


def read_matrix_market_header(path: str | Path) -> MatrixMarketHeader:
    """Read only the header of a coordinate-format Matrix Market file.

    Plain text and gzip-compressed inputs are supported. Complex-valued matrices
    are rejected because dbnumpy's current IR has a float64 value contract.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Matrix Market input is not a file: {source}")

    opener = gzip.open if source.suffix.lower() == ".gz" else open
    try:
        with opener(source, "rb") as stream:
            return _read_header(stream)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read Matrix Market header from {source}") from exc


def _decode_header_line(raw: bytes, *, line_number: int) -> str:
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Matrix Market header line {line_number} is not ASCII text"
        ) from exc


def _read_header(stream: _BinaryHeaderStream) -> MatrixMarketHeader:
    raw_banner = stream.readline()
    if not raw_banner:
        raise ValueError("Matrix Market file is empty")
    banner = _decode_header_line(raw_banner, line_number=1).split()
    if len(banner) != 5 or banner[0].lower() != "%%matrixmarket":
        raise ValueError("invalid Matrix Market banner")

    _, object_kind, layout, field, symmetry = (token.lower() for token in banner)
    if object_kind != "matrix":
        raise ValueError(f"Matrix Market object must be 'matrix', got {object_kind!r}")
    if layout != "coordinate":
        raise ValueError(
            "only coordinate-format Matrix Market inputs are supported; "
            f"got {layout!r}"
        )
    if field not in {"real", "integer", "pattern"}:
        if field == "complex":
            raise ValueError("complex Matrix Market values are not supported")
        raise ValueError(f"unsupported Matrix Market field {field!r}")
    if symmetry not in {"general", "symmetric", "skew-symmetric", "hermitian"}:
        raise ValueError(f"unsupported Matrix Market symmetry {symmetry!r}")

    line_number = 1
    for raw_line in stream:
        line_number += 1
        line = _decode_header_line(raw_line, line_number=line_number)
        if not line or line.startswith("%"):
            continue
        dimensions = line.split()
        if len(dimensions) != 3:
            raise ValueError(
                "Matrix Market coordinate size line must contain rows, columns, "
                "and entries"
            )
        try:
            if not all(re.fullmatch(r"[+-]?[0-9]+", value) for value in dimensions):
                raise ValueError("invalid integer token")
            rows, columns, entries = (int(value) for value in dimensions)
        except ValueError as exc:
            raise ValueError(
                "Matrix Market dimensions and entry count must be integers"
            ) from exc
        if rows < 0 or columns < 0 or entries < 0:
            raise ValueError(
                "Matrix Market dimensions and entry count must be nonnegative"
            )
        int64_max = 2**63 - 1
        if rows > int64_max or columns > int64_max or entries > int64_max:
            raise OverflowError("Matrix Market header values must fit signed int64")
        if symmetry != "general" and rows != columns:
            raise ValueError(f"{symmetry} Matrix Market inputs must be square")
        return MatrixMarketHeader(rows, columns, entries, field, symmetry)

    raise ValueError("Matrix Market coordinate size line is missing")
