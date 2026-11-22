"""The package imports and reports a version."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import evalgate


def test_version_is_exported() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project.open("rb") as source:
        name = tomllib.load(source)["project"]["name"]
    assert evalgate.__version__ == version(name)


def test_public_surface_is_only_the_version() -> None:
    # A growing __all__ without tests behind it is how an unchecked API appears.
    assert evalgate.__all__ == ["__version__"]
