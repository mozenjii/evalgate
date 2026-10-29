"""The package imports and reports a version."""

import evalgate


def test_version_is_exported() -> None:
    assert evalgate.__version__ == "0.0.0"


def test_public_surface_is_only_the_version() -> None:
    # A growing __all__ without tests behind it is how an unchecked API appears.
    assert evalgate.__all__ == ["__version__"]
