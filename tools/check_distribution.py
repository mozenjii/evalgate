"""Verify package contents and release metadata before any PyPI publishing job.

The release workflow owns execution and permissions. This check reads archives without
extracting them, and applies the same archive allowlist to candidate and tagged builds.
"""

from __future__ import annotations

import argparse
import re
import stat
import tarfile
from collections.abc import Set
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile


class DistributionError(ValueError):
    """The built archives do not meet the release contract."""


@dataclass(frozen=True)
class _Distribution:
    metadata: Message
    has_license_file: bool


_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
_PROJECT = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_LICENSES = frozenset({"MIT", "Apache-2.0"})
_SOURCE_FILES = frozenset({"PKG-INFO", "README.md", "CHANGELOG.md", "pyproject.toml", "LICENSE"})
_SOURCE_EXTRAS = frozenset({".gitignore"})
_WHEEL_FILES = frozenset({"METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE"})


def _parts(name: str) -> tuple[str, ...]:
    """Require a canonical package-relative path."""
    parts = tuple(name.split("/"))
    if any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts):
        raise DistributionError(f"invalid archive member path {name!r}")
    return parts


def _python_module(parts: tuple[str, ...]) -> bool:
    """Recognize a Python source file within the import package."""
    return (
        len(parts) >= 2
        and parts[0] == "evalgate"
        and parts[-1].endswith(".py")
        and all(part.isidentifier() for part in (*parts[:-1], parts[-1][:-3]))
    )


def _require_files(names: set[str], required: Set[str]) -> None:
    """Refuse missing minimum package contents before inspecting metadata."""
    if missing := required - names:
        raise DistributionError(f"archive is missing {', '.join(sorted(missing))}")


def _read_wheel(path: Path) -> _Distribution:
    """Inspect one pure Python wheel and verify its CLI entry point."""
    with ZipFile(path) as wheel:
        names: set[str] = set()
        metadata_dirs: set[str] = set()
        for item in wheel.infolist():
            if item.is_dir():
                continue
            parts = _parts(item.filename)
            mode = stat.S_IFMT(item.external_attr >> 16)
            if mode not in (0, stat.S_IFREG) or item.filename in names:
                raise DistributionError(f"invalid or duplicate wheel member {item.filename!r}")
            names.add(item.filename)
            if _python_module(parts):
                continue
            if parts[0].endswith(".dist-info") and "/".join(parts[1:]) in _WHEEL_FILES:
                metadata_dirs.add(parts[0])
                continue
            raise DistributionError(f"unexpected file in wheel: {item.filename!r}")
        if len(metadata_dirs) != 1:
            raise DistributionError("wheel needs exactly one metadata directory")
        directory = next(iter(metadata_dirs))
        _require_files(
            names,
            {"evalgate/__init__.py"}
            | {f"{directory}/{name}" for name in _WHEEL_FILES - {"licenses/LICENSE"}},
        )
        entry_points = wheel.read(f"{directory}/entry_points.txt").decode("utf-8").splitlines()
        if entry_points != ["[console_scripts]", "evalgate = evalgate.cli:main"]:
            raise DistributionError("wheel is missing the expected evalgate CLI entry point")
        metadata = BytesParser().parsebytes(wheel.read(f"{directory}/METADATA"))
    return _Distribution(metadata, f"{directory}/licenses/LICENSE" in names)


def _read_source(path: Path) -> _Distribution:
    """Inspect the source archive under its single package root."""
    root = path.name.removesuffix(".tar.gz")
    with tarfile.open(path, "r:gz") as source:
        names: set[str] = set()
        metadata_file: tarfile.TarInfo | None = None
        for item in source:
            parts = _parts(item.name)
            if parts[0] != root:
                raise DistributionError("unexpected source archive root")
            if item.isdir():
                continue
            name = "/".join(parts[1:])
            if not item.isfile() or name in names:
                raise DistributionError(f"invalid or duplicate source member {item.name!r}")
            names.add(name)
            if name == "PKG-INFO":
                metadata_file = item
            if name not in _SOURCE_FILES | _SOURCE_EXTRAS and not (
                parts[1:2] == ("src",) and _python_module(parts[2:])
            ):
                raise DistributionError(f"unexpected file in source archive: {name!r}")
        _require_files(names, (_SOURCE_FILES - {"LICENSE"}) | {"src/evalgate/__init__.py"})
        file = source.extractfile(metadata_file) if metadata_file is not None else None
        if file is None:
            raise DistributionError("cannot read source package metadata")
        with file:
            metadata = BytesParser().parse(file)
    return _Distribution(metadata, "LICENSE" in names)


def _validate_release(
    wheel: _Distribution, source: _Distribution, tag: str, project: str | None
) -> None:
    """Require the approved project, chosen license and exact version tag."""
    version = wheel.metadata["Version"]
    if _VERSION.fullmatch(version) is None or version == "0.0.0" or tag != f"v{version}":
        raise DistributionError("release tag must exactly match a nonzero vX.Y.Z package version")
    if (
        not project
        or _PROJECT.fullmatch(project) is None
        or project == "evalgate"
        or project != wheel.metadata["Name"]
    ):
        raise DistributionError("PYPI_PROJECT_NAME must match the approved, distinct distribution")
    if any(
        archive.metadata["License-Expression"] not in _LICENSES
        or archive.metadata.get_all("License-File") != ["LICENSE"]
        or not archive.has_license_file
        for archive in (wheel, source)
    ):
        raise DistributionError("release requires an approved MIT or Apache-2.0 license and file")


def verify_distributions(
    directory: Path, *, tag: str | None = None, project: str | None = None
) -> None:
    """Verify one wheel and sdist, then apply the publishing checks when given a tag."""
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise DistributionError("expected exactly one wheel and one source archive")
    try:
        wheel, source = _read_wheel(wheels[0]), _read_source(sources[0])
    except (OSError, BadZipFile, tarfile.TarError, UnicodeError) as error:
        raise DistributionError(f"cannot inspect distribution archives: {error}") from error
    for field in ("Name", "Version", "License-Expression"):
        values = (wheel.metadata.get_all(field), source.metadata.get_all(field))
        if values[0] != values[1] or (values[0] is None and field != "License-Expression"):
            raise DistributionError(f"archive metadata does not agree on {field}")
        if values[0] is not None and (len(values[0]) != 1 or not values[0][0]):
            raise DistributionError(f"archive metadata needs one value for {field}")
    if tag is not None:
        _validate_release(wheel, source, tag, project)


def main() -> int:
    """Return a nonzero status when an archive or release prerequisite fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--project")
    args = parser.parse_args()
    try:
        verify_distributions(args.directory, tag=args.tag, project=args.project)
    except DistributionError as error:
        parser.exit(1, f"error: {error}\n")
    print("wheel and source archive verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
