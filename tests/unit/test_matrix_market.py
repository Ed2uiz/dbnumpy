from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from dbnumpy import MatrixMarketHeader, read_matrix_market_header


def _write(path: Path, text: str) -> Path:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="ascii") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="ascii")
    return path


@pytest.mark.parametrize("suffix", [".mtx", ".mtx.gz"])
def test_reads_plain_and_gzip_headers(tmp_path: Path, suffix: str) -> None:
    path = _write(
        tmp_path / f"matrix{suffix}",
        "%%MatrixMarket matrix coordinate real general\n"
        "% generated fixture\n\n"
        "3 4 2\n"
        "1 1 7\n",
    )

    header = read_matrix_market_header(path)

    assert header == MatrixMarketHeader(3, 4, 2, "real", "general")
    assert header.shape == (3, 4)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("not matrix market\n", "banner"),
        ("%%MatrixMarket vector coordinate real general\n1 1 0\n", "object"),
        ("%%MatrixMarket matrix array real general\n1 1\n", "coordinate"),
        ("%%MatrixMarket matrix coordinate complex general\n1 1 0\n", "complex"),
        ("%%MatrixMarket matrix coordinate real unknown\n1 1 0\n", "symmetry"),
        ("%%MatrixMarket matrix coordinate real general\n1 two 0\n", "integers"),
        ("%%MatrixMarket matrix coordinate real general\n1 -2 0\n", "nonnegative"),
        ("%%MatrixMarket matrix coordinate real symmetric\n1 2 0\n", "square"),
        ("%%MatrixMarket matrix coordinate real general\n% no size\n", "missing"),
    ],
)
def test_rejects_invalid_headers(tmp_path: Path, text: str, message: str) -> None:
    path = _write(tmp_path / "invalid.mtx", text)
    with pytest.raises((ValueError, OverflowError), match=message):
        read_matrix_market_header(path)


def test_requires_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_matrix_market_header(tmp_path / "missing.mtx")
