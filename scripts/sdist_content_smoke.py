"""Verify that a source distribution contains and runs its local documentation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REQUIRED_PATHS = (
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "mkdocs.yml",
    "pyproject.toml",
    "src/dbnumpy/__init__.py",
    "src/dbnumpy/py.typed",
    "tests/conftest.py",
    "benchmarks/reduction_overhead.py",
    "scripts/artifact_smoke.py",
    "docs/guides/operations.md",
    "examples/vignettes/arithmetic.py",
    "examples/vignettes/numerical_surface.py",
    "examples/vignettes/operations.py",
    "examples/vignettes/overview.py",
)


def _distribution_root(extraction_dir: Path) -> Path:
    roots = [path for path in extraction_dir.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(
            f"expected exactly one source-distribution root, found {len(roots)}"
        )
    return roots[0]


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)  # noqa: S603


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="dbnumpy-sdist-") as temporary:
        extraction_dir = Path(temporary)
        with tarfile.open(archive, mode="r:gz") as source_distribution:
            source_distribution.extractall(extraction_dir, filter="data")

        root = _distribution_root(extraction_dir)
        missing = [path for path in REQUIRED_PATHS if not (root / path).is_file()]
        if missing:
            raise RuntimeError(f"source distribution is missing: {', '.join(missing)}")

        env = os.environ.copy()
        source_path = str(root / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else source_path + os.pathsep + existing_pythonpath
        )

        for vignette in sorted((root / "examples" / "vignettes").glob("*.py")):
            _run([sys.executable, str(vignette)], cwd=root, env=env)

        _run(
            [
                sys.executable,
                "-m",
                "mkdocs",
                "build",
                "--strict",
                "--site-dir",
                str(extraction_dir / "site"),
            ],
            cwd=root,
            env=env,
        )

    print(f"source-distribution content smoke passed: {archive.name}")


if __name__ == "__main__":
    main()
