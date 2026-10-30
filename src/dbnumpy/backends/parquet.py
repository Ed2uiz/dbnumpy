"""Parquet input and temporary-directory checks shared by file-based backends."""

from __future__ import annotations

import os
from os import PathLike, fspath
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Sequence

type ParquetPath = str | PathLike[str]


def resolve_temp_directory(directory: str | Path | None) -> Path | None:
    if directory is None:
        return None
    resolved = Path(directory).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"temp_directory must name an existing directory; got {resolved}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ValueError(f"temp_directory is not writable: {resolved}")
    return resolved


def validate_parquet_paths(
    paths: ParquetPath | Sequence[ParquetPath],
) -> tuple[str, ...]:
    candidates: tuple[ParquetPath, ...]
    candidates = (paths,) if isinstance(paths, (str, PathLike)) else tuple(paths)
    if not candidates:
        raise ValueError("at least one Parquet file path is required")

    validated = []
    for candidate in candidates:
        try:
            raw_path = fspath(candidate)
        except TypeError as exc:
            raise TypeError(
                "Parquet paths must be strings or path-like objects"
            ) from exc
        if not isinstance(raw_path, str):
            raise TypeError("Parquet paths must resolve to strings, not bytes")
        if not raw_path or "\x00" in raw_path:
            raise ValueError("Parquet file paths must be nonempty and valid")
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Parquet file does not exist: {path}") from exc
        if not resolved.is_file():
            raise ValueError(f"Parquet path must identify a regular file: {path}")
        normalized = str(resolved)
        if normalized in validated:
            raise ValueError(f"Parquet file paths must be unique: {path}")
        validated.append(normalized)
    return tuple(validated)


def validate_coordinate_schema(schema: pa.Schema, *, path: str) -> None:
    try:
        i_type = schema.field("i").type
        j_type = schema.field("j").type
        x_type = schema.field("x").type
    except KeyError as exc:
        raise ValueError(
            f"Parquet file {path!r} must expose numeric i, j, x columns"
        ) from exc
    if not (
        pa.types.is_integer(i_type)
        and pa.types.is_integer(j_type)
        and (pa.types.is_integer(x_type) or pa.types.is_floating(x_type))
    ):
        raise ValueError(
            f"Parquet file {path!r} must expose integer i/j and numeric x columns"
        )
