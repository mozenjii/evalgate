"""evalgate: a CI check that fails a pull request when an LLM feature gets worse.

Modules, from the bottom up: `dataset` loads and validates versioned datasets, `scorers`
turns one output into one number, `stats` holds the paired statistics, `runs` stores run
records, `runner` executes a dataset against a target, `compare` turns two runs into
per-slice verdicts, and `cli` is the `evalgate` command.

The package root exports only the version; import from the module you need. Keeping the
root that small means nothing reaches the public surface without a test behind it.
"""

__all__ = ["__version__"]

__version__ = "0.0.0"
"""The released version.

Still ``0.0.0`` on purpose: nothing is released, and a version number that moves before
there is anything to install is a claim the repository cannot support.
"""
