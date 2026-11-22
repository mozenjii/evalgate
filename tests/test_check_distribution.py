"""Release archive and tag cases exercised without network or publishing permission."""

import io
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from check_distribution import DistributionError, verify_distributions


def distributions(folder: Path, *, sdist_extra: str | None = None, licensed: bool = True) -> None:
    """Build minimal archives that match the metadata shape Hatch produces."""
    metadata = "Metadata-Version: 2.4\nName: some-evalgate\nVersion: 0.1.0\n"
    if licensed:
        metadata += "License-Expression: MIT\nLicense-File: LICENSE\n"
    for path, entries in (
        (
            folder / "some_evalgate-0.1.0-py3-none-any.whl",
            {
                "evalgate/__init__.py": "",
                "some_evalgate-0.1.0.dist-info/METADATA": metadata,
                "some_evalgate-0.1.0.dist-info/WHEEL": "",
                "some_evalgate-0.1.0.dist-info/RECORD": "",
                "some_evalgate-0.1.0.dist-info/entry_points.txt": "[console_scripts]\n"
                "evalgate = evalgate.cli:main\n",
                **(
                    {"some_evalgate-0.1.0.dist-info/licenses/LICENSE": "license"}
                    if licensed
                    else {}
                ),
            },
        ),
        (
            folder / "some_evalgate-0.1.0.tar.gz",
            {
                "src/evalgate/__init__.py": "",
                "PKG-INFO": metadata,
                "pyproject.toml": "",
                "README.md": "",
                "CHANGELOG.md": "",
                **({"LICENSE": "license"} if licensed else {}),
                **({sdist_extra: "not a package file"} if sdist_extra is not None else {}),
            },
        ),
    ):
        if path.suffix == ".whl":
            with ZipFile(path, "w") as wheel:
                for name, body in entries.items():
                    wheel.writestr(name, body)
        else:
            with tarfile.open(path, "w:gz") as source:
                for name, body in entries.items():
                    value = body.encode()
                    member = tarfile.TarInfo(name=f"some_evalgate-0.1.0/{name}")
                    member.size = len(value)
                    source.addfile(member, io.BytesIO(value))


def test_release_requires_exact_version_project_and_license(tmp_path: Path) -> None:
    distributions(tmp_path)
    verify_distributions(tmp_path, tag="v0.1.0", project="some-evalgate")
    with pytest.raises(DistributionError, match="tag"):
        verify_distributions(tmp_path, tag="v0", project="some-evalgate")
    with pytest.raises(DistributionError, match="tag"):
        verify_distributions(tmp_path, tag="v0.1.1", project="some-evalgate")
    with pytest.raises(DistributionError, match="PYPI_PROJECT_NAME"):
        verify_distributions(tmp_path, tag="v0.1.0", project="different-project")


def test_unlicensed_archive_is_buildable_but_not_releaseable(tmp_path: Path) -> None:
    distributions(tmp_path, licensed=False)
    verify_distributions(tmp_path)
    with pytest.raises(DistributionError, match="license"):
        verify_distributions(tmp_path, tag="v0.1.0", project="some-evalgate")


def test_archive_refuses_files_outside_package_allowlist(tmp_path: Path) -> None:
    distributions(tmp_path, sdist_extra="local-notes.md")
    with pytest.raises(DistributionError, match="unexpected file"):
        verify_distributions(tmp_path)
