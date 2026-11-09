# Measured error rates

Written by `tools/error_rates.py`; do not edit by hand. CI regenerates this file and
fails if it differs, so these are the numbers the current code produces.

Every case puts the true difference **exactly at the tolerance**, the hardest point for
both errors. A *false pass* calls such a slice non-inferior; a *false fail* calls it
regressed. At 95% confidence each should happen at most **2.5%** of the time. The
method is in [statistics.md](statistics.md).

## Binary scorers, one slice

Exact: every possible outcome table is weighed by its probability, so there is no
simulation error, except in the Newcombe column, which needs the full table and is
simulated (20,000 tables per row, 80% baseline accuracy).
*Discordance* is the share of items the two versions disagree on; when it equals the
tolerance, the change breaks items and fixes none.

| n | tolerance | discordance | before: false pass | Newcombe: false pass | **Tango: false pass** | **Tango: false fail** |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 0.02 | 0.02 | 0.0% | 2.4% | 0.0% | 0.0% |
| 10 | 0.02 | 0.04 | 0.0% | 1.7% | 0.0% | 0.0% |
| 10 | 0.02 | 0.20 | 0.0% | 1.7% | 0.4% | 0.1% |
| 10 | 0.02 | 0.40 | 0.2% | 2.2% | 1.9% | 1.1% |
| 10 | 0.05 | 0.05 | 0.0% | 2.0% | 0.0% | 0.0% |
| 10 | 0.05 | 0.10 | 0.0% | 1.6% | 0.0% | 0.0% |
| 10 | 0.05 | 0.20 | 0.0% | 3.2% | 0.2% | 0.3% |
| 10 | 0.05 | 0.40 | 0.4% | 3.4% | 1.5% | 1.6% |
| 10 | 0.10 | 0.10 | 0.0% | 5.4% | 0.0% | 0.0% |
| 10 | 0.10 | 0.20 | 6.7% | 4.5% | 0.4% | 0.1% |
| 10 | 0.10 | 0.40 | 2.2% | 3.1% | 2.2% | 0.9% |
| 20 | 0.02 | 0.02 | 0.0% | 0.7% | 0.0% | 0.0% |
| 20 | 0.02 | 0.04 | 0.0% | 0.4% | 0.0% | 0.0% |
| 20 | 0.02 | 0.20 | 0.4% | 3.3% | 1.6% | 1.4% |
| 20 | 0.02 | 0.40 | 1.0% | 2.6% | 2.7% | 2.7% |
| 20 | 0.05 | 0.05 | 0.0% | 4.1% | 0.0% | 0.0% |
| 20 | 0.05 | 0.10 | 6.8% | 3.1% | 0.3% | 0.2% |
| 20 | 0.05 | 0.20 | 2.8% | 3.0% | 2.1% | 1.0% |
| 20 | 0.05 | 0.40 | 1.0% | 2.9% | 3.1% | 2.3% |
| 20 | 0.10 | 0.10 | 27.0% | 8.2% | 0.0% | 0.2% |
| 20 | 0.10 | 0.20 | 12.2% | 4.4% | 1.6% | 1.1% |
| 20 | 0.10 | 0.40 | 1.6% | 2.8% | 2.5% | 2.6% |
| 40 | 0.02 | 0.02 | 0.0% | 0.2% | 0.0% | 0.1% |
| 40 | 0.02 | 0.04 | 0.0% | 0.3% | 0.2% | 0.5% |
| 40 | 0.02 | 0.20 | 0.9% | 2.9% | 2.1% | 2.3% |
| 40 | 0.02 | 0.40 | 1.5% | 2.6% | 2.5% | 2.4% |
| 40 | 0.05 | 0.05 | 27.1% | 6.3% | 0.0% | 0.3% |
| 40 | 0.05 | 0.10 | 13.3% | 4.1% | 1.6% | 1.2% |
| 40 | 0.05 | 0.20 | 1.9% | 2.6% | 2.6% | 2.6% |
| 40 | 0.05 | 0.40 | 1.4% | 2.5% | 2.6% | 2.5% |
| 40 | 0.10 | 0.10 | 42.3% | 5.8% | 1.5% | 1.5% |
| 40 | 0.10 | 0.20 | 7.9% | 3.2% | 1.8% | 2.6% |
| 40 | 0.10 | 0.40 | 1.7% | 2.9% | 2.3% | 2.5% |
| 100 | 0.02 | 0.02 | 27.1% | 4.0% | 0.0% | 1.5% |
| 100 | 0.02 | 0.04 | 13.8% | 3.5% | 1.6% | 3.2% |
| 100 | 0.02 | 0.20 | 1.4% | 2.7% | 2.3% | 2.5% |
| 100 | 0.02 | 0.40 | 1.8% | 2.5% | 2.5% | 2.4% |
| 100 | 0.05 | 0.05 | 43.6% | 3.8% | 0.6% | 2.8% |
| 100 | 0.05 | 0.10 | 7.5% | 3.0% | 1.8% | 3.0% |
| 100 | 0.05 | 0.20 | 1.9% | 3.0% | 2.2% | 2.4% |
| 100 | 0.05 | 0.40 | 1.8% | 2.7% | 2.4% | 2.4% |
| 100 | 0.10 | 0.10 | 45.1% | 4.9% | 2.4% | 4.0% |
| 100 | 0.10 | 0.20 | 4.1% | 3.0% | 2.2% | 2.6% |
| 100 | 0.10 | 0.40 | 2.0% | 2.6% | 2.6% | 2.5% |
| 200 | 0.02 | 0.02 | 43.1% | 4.7% | 1.8% | 4.9% |
| 200 | 0.02 | 0.04 | 9.6% | 3.1% | 1.9% | 4.1% |
| 200 | 0.02 | 0.20 | 1.8% | 2.5% | 2.5% | 2.4% |
| 200 | 0.02 | 0.40 | 1.9% | 2.6% | 2.4% | 2.5% |
| 200 | 0.05 | 0.05 | 45.5% | 4.6% | 0.9% | 2.4% |
| 200 | 0.05 | 0.10 | 4.6% | 3.0% | 2.2% | 2.7% |
| 200 | 0.05 | 0.20 | 2.1% | 2.7% | 2.4% | 2.5% |
| 200 | 0.05 | 0.40 | 2.0% | 2.6% | 2.5% | 2.5% |
| 200 | 0.10 | 0.10 | 46.6% | 3.2% | 1.7% | 2.7% |
| 200 | 0.10 | 0.20 | 3.8% | 2.6% | 2.4% | 2.7% |
| 200 | 0.10 | 0.40 | 2.2% | 2.7% | 2.5% | 2.5% |
| 500 | 0.02 | 0.02 | 45.7% | 4.3% | 1.0% | 2.6% |
| 500 | 0.02 | 0.04 | 4.9% | 3.2% | 2.3% | 2.7% |
| 500 | 0.02 | 0.20 | 2.0% | 2.5% | 2.5% | 2.5% |
| 500 | 0.02 | 0.40 | 2.1% | 2.7% | 2.5% | 2.5% |
| 500 | 0.05 | 0.05 | 47.1% | 3.3% | 2.0% | 3.0% |
| 500 | 0.05 | 0.10 | 4.0% | 2.8% | 2.4% | 2.6% |
| 500 | 0.05 | 0.20 | 2.3% | 2.6% | 2.5% | 2.5% |
| 500 | 0.05 | 0.40 | 2.2% | 2.4% | 2.5% | 2.5% |
| 500 | 0.10 | 0.10 | 47.8% | 3.5% | 1.9% | 2.5% |
| 500 | 0.10 | 0.20 | 3.7% | 2.9% | 2.4% | 2.6% |
| 500 | 0.10 | 0.40 | 2.4% | 2.6% | 2.5% | 2.5% |

Worst false pass: **47.8%** for the conditional interval evalgate used before, 8.2% for Newcombe's, **3.1%** for Tango's. Tango's worst false fail is 4.9%.

## Continuous scorers, one slice

Simulated through `paired_continuous` at a tolerance of 0.05, 1,000 slices per row and 2,000 resamples per interval (the shipped default is 10,000, which only steadies the bounds). A share *moved* of
items change; the rest score identically, which is what most evaluation data looks
like. *Spread* is the standard deviation of a moved item's change. *Power* is how often
a slice with no change on average is certified safe.

| n | moved | spread | false pass | false fail | power |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.2 | 0.1 | 0.0% | 0.0% | 0.0% |
| 10 | 0.2 | 0.3 | 0.0% | 0.0% | 0.0% |
| 10 | 1.0 | 0.1 | 2.8% | 3.3% | 38.5% |
| 10 | 1.0 | 0.3 | 2.7% | 3.4% | 9.9% |
| 20 | 0.2 | 0.1 | 0.0% | 1.3% | 2.5% |
| 20 | 0.2 | 0.3 | 0.0% | 0.8% | 1.2% |
| 20 | 1.0 | 0.1 | 2.6% | 2.9% | 62.4% |
| 20 | 1.0 | 0.3 | 1.7% | 2.8% | 12.9% |
| 40 | 0.2 | 0.1 | 0.0% | 2.3% | 57.1% |
| 40 | 0.2 | 0.3 | 1.6% | 2.2% | 32.3% |
| 40 | 1.0 | 0.1 | 2.6% | 2.2% | 88.6% |
| 40 | 1.0 | 0.3 | 3.0% | 2.7% | 21.0% |
| 100 | 0.2 | 0.1 | 2.2% | 2.5% | 99.9% |
| 100 | 0.2 | 0.3 | 2.6% | 3.1% | 93.9% |
| 100 | 1.0 | 0.1 | 3.1% | 2.5% | 99.9% |
| 100 | 1.0 | 0.3 | 2.6% | 2.4% | 41.3% |

### Why a slice needs 8 moved items before it is bootstrapped

A plain BCa bootstrap (scipy's), on the same kind of data at the margin, grouped by how
many items actually moved. Below the threshold evalgate uses the count bound instead,
which holds at any size.

| items moved | bootstrap false pass | slices |
|--:|--:|--:|
| 0 | 100.0% | 299 |
| 1 | 20.0% | 774 |
| 2 | 9.9% | 1,171 |
| 3 | 4.9% | 1,302 |
| 4 | 3.9% | 1,350 |
| 5 | 3.2% | 1,237 |
| 6 | 3.6% | 1,175 |
| 7 | 2.8% | 1,072 |
| 8 | 2.6% | 1,004 |
| 9 | 1.5% | 863 |
| 10 | 1.7% | 687 |
| 11 | 2.3% | 655 |
| 12 or more | 1.9% | 6,411 |

## Many slices

Simulated through `paired_binary` and `family_verdicts`: 2,000 checks per
row, 100 items per slice, tolerance 0.05. *False pass*: one slice is truly worse by the
tolerance (20% discordance) and the rest are unchanged (2% discordance, so they can
pass), yet the check passes. *False fail*: every slice is exactly at the tolerance (20%
discordance), yet the check fails.

| gate slices | false pass | false fail, no correction | **false fail, Holm** |
|--:|--:|--:|--:|
| 1 | 1.8% | 2.3% | 2.3% |
| 4 | 0.4% | 8.4% | 1.7% |
| 10 | 0.0% | 21.1% | 2.7% |

The false-pass rate needs no correction: a check passes only when every slice does.
Without a correction the false-fail rate grows with the number of slices; Holm holds it
at or below one slice's.
