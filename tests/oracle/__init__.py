"""Independent oracles: tests authored from a contract, never from the code they judge.

Owning artifact: `app/technical context/02_STATISTICS_CONTRACT.md`.

Deliberate boundary: everything under this package may import scipy and statsmodels,
which are test-only dependencies. Nothing under `src/` may.
"""
