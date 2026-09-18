from __future__ import annotations

from importlib import resources
from importlib.metadata import metadata, requires, version
from pathlib import Path

import dbnumpy


def test_imported_and_distribution_versions_match() -> None:
    assert dbnumpy.__version__ == version("dbnumpy")


def test_typing_marker_is_packaged() -> None:
    assert resources.files("dbnumpy").joinpath("py.typed").is_file()


def test_distribution_declares_mit_license() -> None:
    assert metadata("dbnumpy")["License-Expression"] == "MIT"


def test_sql_compiler_runtime_dependency_is_direct() -> None:
    requirements = requires("dbnumpy") or []
    assert any(
        requirement.startswith("packaging") and "extra ==" not in requirement
        for requirement in requirements
    )


def test_artifact_commands_use_current_distribution_version() -> None:
    root = Path(__file__).parents[2]
    release = dbnumpy.__version__
    paths = (root / ".github" / "workflows" / "ci.yml", root / "CONTRIBUTING.md")
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert f"dbnumpy-{release}.tar.gz" in text
        assert f"dbnumpy-{release}-py3-none-any.whl" in text
