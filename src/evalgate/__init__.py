"""evalgate: a CI check that fails a pull request when an LLM feature gets worse.

The package is deliberately empty at this stage. B0 delivers the workspace and the
verification pipeline; the first product modules (dataset loading, the runner, scorers
and the paired statistics) arrive in B1, each behind its own tests.

Keeping the public surface to a version string until then means the import graph never
grows a module that nothing checks.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
"""The released version.

Still ``0.0.0`` on purpose: nothing is released, and a version number that moves before
there is anything to install is a claim the repository cannot support.
"""
