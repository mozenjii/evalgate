# Statistics contract

This document states **what must be true** about evalgate's statistics, not how to compute it.
The independent oracle in `tests/oracle/` is written from this file and from scipy and
statsmodels, never from `src/evalgate/stats.py`. If the implementation and the oracle disagree,
the disagreement is resolved by finding the cause in one of them. The oracle is never edited to
agree.

## 1. The question the gate asks

Not "is the candidate different from the baseline" but **"is the candidate at most `delta` worse
than the baseline"**. This is a non-inferiority question, and it is the whole reason the project
exists.

A significance test is the wrong tool. On a slice of 40 items almost nothing reaches significance,
so a gate built on p-values passes nearly every change and its greenness means nothing. The gate
therefore asks about the **lower bound of an interval on the difference**, and `delta` is a
per-slice number chosen by a human and recorded in config.

Sign convention, fixed everywhere: **`difference = candidate - baseline`**. Negative is worse.

## 2. Definitions

For `n` items evaluated under both versions, paired by item id:

- Binary scorer: `baseline[i]`, `candidate[i]` in {0, 1}.
- Continuous scorer: `baseline[i]`, `candidate[i]` real.

Pairing is not optional. Comparing two independent means throws away the item-level correlation
that makes small slices readable at all, and inflates the variance for no reason.

## 3. Binary outcomes

### 3.1 Counts

From the paired outcomes, form the discordant counts:

- `favour_candidate` = number of items where baseline = 0 and candidate = 1
- `favour_baseline`  = number of items where baseline = 1 and candidate = 0
- `discordant` = `favour_candidate + favour_baseline`

Concordant items (both right, both wrong) carry no information about the *difference* and must not
enter the variance.

### 3.2 Point estimate

```
difference = (favour_candidate - favour_baseline) / n
```

This is identical to `mean(candidate) - mean(baseline)`, and that identity must hold exactly.

### 3.3 Interval: Tango's score interval

For a hypothesised difference `D` in `[-1, 1]`, with `b = favour_candidate` and
`c = favour_baseline`, Tango's score statistic is

```
W(D)  = -b - c + (2n - b + c) * D
q(D)  = ( sqrt( W(D)^2 + 8 * n * c * D * (1 - D) ) - W(D) ) / (4n)
Z(D)  = (b - c - n * D) / sqrt( n * (2 * q(D) + D * (1 - D)) )
```

`q(D)` is the maximum-likelihood probability that an item favours the baseline, *restricted* to
the true difference being `D`. When the denominator is zero, `Z(D)` is `0` if the numerator is
`0`, and otherwise `+inf` or `-inf` with the numerator's sign.

With `z = Phi^-1(1 - alpha/2)` and `alpha = 1 - confidence`, the interval is the set of
differences the score test does not reject:

```
interval = { D in [-1, 1] : -z <= Z(D) <= z }
low  = the smallest such D
high = the largest such D
```

**Why Tango, and why not the exact conditional interval used before.** The earlier interval took
a Clopper-Pearson interval on the share of discordant items favouring the candidate and scaled it
by the *observed* discordance rate, treating that rate as fixed. That leaves out a source of
variance exactly where the gate decides. The worst case is a change that breaks items and fixes
none, so every discordant item favours the baseline: the lower bound then sits on the point
estimate, and the interval cannot extend below what was observed. Measured by exact enumeration
at a true difference of exactly `-delta`, that interval reported `non_inferior` up to 48% of the
time against a nominal 2.5%, and the error grew with `n` instead of shrinking. Tango's interval
puts the variance under the hypothesis being tested; on the same grid its worst case is about 3%.
Newcombe's hybrid score interval was also measured, and reached 8%. The figures are in
`docs/error-rates.md`, produced by `tools/error_rates.py`.

A second reason: because the interval is the inversion of a test, the same statistic gives the
p-values of section 3.5, and they agree with the interval exactly. The correction across slices
in section 5 needs exactly that.

### 3.4 Edge cases, which are not optional

- **`discordant = 0`.** Every item scored the same under both versions. The point estimate is
  exactly `0`, and Tango's interval is `[-z^2 / (n + z^2), +z^2 / (n + z^2)]`, which is not a
  point. No special case is needed, and none is allowed: zero observed disagreements on 20 items
  is not proof of equality, and `[0, 0]` would make every such slice `non_inferior` by arithmetic.
- **Every item favours the baseline** (`c = n`). `low = -1`: nothing in the data excludes the
  worst case. Symmetrically, `b = n` gives `high = 1`.
- **`n = 0`.** There is no estimate. Raise, rather than return a number that means nothing.

### 3.5 Margin p-values

The gate asks one question of each slice, at the slice's tolerance `delta`. It is answered by two
one-sided p-values from the same statistic, evaluated at `D = -delta`:

```
p_non_inferior = 1 - Phi( Z(-delta) )     # H0: difference <= -delta
p_regressed    =     Phi( Z(-delta) )     # H0: difference >= -delta
```

with `Phi(+inf) = 1` and `Phi(-inf) = 0`. They agree with the interval exactly:
`low > -delta` if and only if `p_non_inferior < alpha/2`, and `high < -delta` if and only if
`p_regressed < alpha/2`.

McNemar's test, which the first version reported, is gone. It tests "no difference at all",
which is not the gate's question, and nothing read it.

## 4. Continuous outcomes

### 4.1 Point estimate

```
differences[i] = candidate[i] - baseline[i]
difference = mean(differences)
```

### 4.2 Interval: bootstrap, BCa

Used when at least 8 items moved and the `n` differences, unmoved items included, are not all
the same; sections 4.4 and 4.5 cover the other cases.

Resample the **paired differences** with replacement, `resamples` times, recomputing the mean each
time. Resampling the two arrays independently would destroy the pairing.

Use the bias-corrected and accelerated (BCa) percentile interval, not the plain percentile
interval, because scorer distributions are bounded and skewed and the plain interval is visibly
off-centre on them.

- Bias correction: `z0 = Phi^-1( p )`, where `p` counts resample means below the observed mean
  **with ties counted as one half**: `p = ( #{less than} + #{equal} / 2 ) / resamples`.
  On continuous data exact ties essentially never occur and this agrees with the simpler
  "strictly less than" reading. They diverge systematically once the differences take few
  distinct values -- a scorer rounded to two decimals, or set F1 over small sets -- and the
  strict form then biases the correction downwards on every such slice. No tolerance in a
  bootstrap comparison separates that from a bug, so the convention is pinned here.
- Acceleration, from the jackknife over the differences, with `theta_i` the mean leaving item `i`
  out and `theta_bar` their mean:

```
a = sum( (theta_bar - theta_i)^3 ) / ( 6 * ( sum( (theta_bar - theta_i)^2 ) )^(3/2) )
```

- With `z_alpha = Phi^-1(alpha/2)` and `z_1malpha = Phi^-1(1 - alpha/2)`, the adjusted percentiles are

```
lower_pct = Phi( z0 + (z0 + z_alpha)   / (1 - a * (z0 + z_alpha)) )
upper_pct = Phi( z0 + (z0 + z_1malpha) / (1 - a * (z0 + z_1malpha)) )
```

and the interval is those percentiles of the resample means.

The acceleration `a` is a deterministic function of the data. It does not depend on the resampling
and must agree with an independent computation exactly, not approximately.

### 4.3 Scores are bounded

Continuous scores lie in `[0, 1]`, so every difference lies in `[-1, 1]`. A score outside that
range is an error. The bound is what makes section 4.4 possible: it caps how far an item the data
never saw move could move.

### 4.4 Few items moved: the count bound

An item **moved** if its difference is not zero. Let `up` be the number of items with a positive
difference, `down` the number with a negative one, and `moved = up + down`.

The bootstrap cannot describe a change it has barely seen. If two of forty items moved, most
resamples contain one or none of them, and the interval comes out far narrower than the evidence
supports. Measured at a true difference of exactly `-delta`, BCa reported `non_inferior`:

- every time when no item moved, because the interval collapsed to `[0, 0]`;
- about 20% of the time with one item moved, 10% with two, 3-5% with three to seven;
- at or below the nominal 2.5% from eight moved items up.

So **when fewer than 8 items moved**, the interval is not bootstrapped. It is the count bound:

```
low  = -U(down, n)
high = +U(up, n)
```

where `U(k, n)` is the Clopper-Pearson upper bound for `k` successes in `n` trials at the stated
confidence. The bound holds at every `n` without approximation: a difference is never below `-1`,
so the mean difference is never below minus the share of items that moved down.

Clopper-Pearson bounds are defined by the binomial tails, with `alpha = 1 - confidence`:

- the lower bound solves `P(X >= k | n, lo) = alpha / 2`, and is `0` when `k = 0`;
- the upper bound solves `P(X <= k | n, hi) = alpha / 2`, and is `1` when `k = n`.

The count bound's margin p-values, consistent with it as in section 3.5:

```
p_non_inferior = P( Binomial(n, delta) <= down )
p_regressed    = 1
```

`p_regressed` is `1` because `high >= 0 > -delta` always: a slice where almost nothing moved can
never be shown to have regressed. Eight is a measured threshold, not a derived one, and
`tools/error_rates.py` re-measures it.

### 4.5 Every item moved by the same amount

If every one of the `n` differences is the same value `v` (so all `n` items moved, and `n >= 8`), there is nothing for a
resample to vary. The interval must not collapse to `[v, v]`, for the same reason as section 3.4:
items the data never saw could differ from `v` by up to the score range. With
`u = U(0, n) = 1 - (alpha/2)^(1/n)` bounding the share of such items:

```
low  = v - u * (1 + v)
high = v + u * (1 - v)
```

Its margin p-values, consistent with it:

```
p_non_inferior = 1                                 if v <= -delta
                 (1 - (v + delta) / (1 + v))^n     otherwise
p_regressed    = 1                                 if v >= -delta
                 (1 - (-delta - v) / (1 - v))^n    otherwise
```

"The same value" is judged against a tolerance, `max(diff) - min(diff) <= 1e-12 * max(|mean|, 1)`,
not by exact equality: a constant shift computed in floating point leaves differences that
disagree in the last bit.

### 4.6 Margin p-values for the bootstrap

From the same resample means `m_1 .. m_B` that built the BCa interval, with the same `z0` and `a`:

```
F = ( #{m_i < -delta} + #{m_i = -delta} / 2 ) / B, clamped to [1/(2B), 1 - 1/(2B)]
w = Phi^-1(F) - z0
p_non_inferior = Phi( w / (1 + a * w) - z0 )   when 1 + a * w > 0
               = 0 if w < 0, else 1            otherwise
p_regressed    = 1 - p_non_inferior
```

This is the BCa percentile mapping run backwards: `p_non_inferior` is the one-sided level at which
the BCa lower bound would sit exactly on `-delta`. When `1 + a * w <= 0`, `-delta` lies beyond
every percentile the mapping can reach, and the p-value is at its limit. The p-values use the
empirical distribution while the interval interpolates between order statistics, so the two can
disagree by a hair at the margin. Section 6.2 therefore never lets a p-value alone produce a
verdict.

### 4.7 Other edge cases

- **`n = 0`.** Raise.
- **`n = 1`.** No interval is meaningful. Raise rather than report a degenerate one.
- Results must be reproducible: the same seed gives the same interval. **With no seed given,
  one is derived from the data**, by hashing the differences. The default must not be a fresh
  random stream: a gate whose verdict can change between two builds of the same commit is the
  failure a CI check exists to prevent, and leaving the common path unseeded would make that
  the normal case rather than an edge one.
- When the jackknife spread is `0`, `a` is `0/0`; define `a = 0`, which keeps the function total.

## 5. Many slices: two errors, two controls

A check over many slices can be wrong in two ways, and they need different controls.

**False pass**: a slice that is truly worse than its tolerance is reported `non_inferior`. The
check reports "ok" only when *every* gate slice is `non_inferior`. That is an intersection-union
test: if any one slice is truly worse than its tolerance, an "ok" needs that particular slice's
test to err, which happens with probability at most `alpha/2`. So the false-pass rate of "ok" is
controlled at `alpha/2` with **no correction**. Correcting here would only cost power.

That guarantee is about the check's answer, "ok". A single row that says `non_inferior` in a
check that is not ok, because another slice regressed or is not guarded, carries its own
per-slice guarantee of `alpha/2`, and no more.

**False fail**: a slice that is truly within its tolerance is reported `regressed`. The check
fails if *any* gate slice regressed, so this error grows with the number of slices. It is
controlled across the gate slices with **Holm**, applied to the `p_regressed` values of every gate
slice that produced an interval. Holm controls the family-wise error rate.

**Watch slices** never fail the build; a false flag costs a reader one line. Their `p_regressed`
values are adjusted with **Benjamini-Hochberg**, which controls the false discovery rate among
the slices flagged. The two tiers are adjusted separately and never share a family.

Slices with no interval (no paired items, or one) are not tested and are in neither family.

Both corrections return adjusted p-values in the **input order**, both are monotone after
adjustment, and both clamp to `1.0`. With `m` p-values sorted ascending as
`p_(1) <= ... <= p_(m)`:

- Holm: `adj_(i) = max over j <= i of min(1, (m - j + 1) * p_(j))`
- Benjamini-Hochberg: `adj_(i) = min over j >= i of min(1, (m / j) * p_(j))`

Edge cases: an empty input returns empty. A single p-value is returned unchanged except for the
clamp. Ties must be handled without reordering the output.

The first version applied Holm to McNemar p-values. Those tested the wrong hypothesis and the
verdict never read them, so the control it described did not exist.

## 6. The verdict

### 6.1 One slice

Given an interval on `candidate - baseline` and a tolerance `delta > 0`:

| Condition | Verdict | Meaning |
|---|---|---|
| `low > -delta` | `non_inferior` | The data rule out a regression worse than `delta`. |
| `high < -delta` | `regressed` | The data rule out the candidate being within `delta`. |
| otherwise | `inconclusive` | The interval straddles `-delta`. Nothing is ruled out. |

`inconclusive` is a first-class state and **must never collapse into `non_inferior`**. A thin
slice that cannot detect the regression it was asked about is a gap in coverage, and reporting it
as a pass is the precise failure this project exists to prevent. All three states must be
reachable and tested.

`delta` must be strictly positive; a tolerance of zero turns the question back into a
significance test. It must also be at most `1`: no difference can exceed the score range, so a
larger tolerance asks nothing. Reject both.

### 6.2 A family of slices

Within one family (section 5), a slice's final verdict is:

- `non_inferior` exactly when its own verdict is `non_inferior`. There is no adjustment.
- `regressed` when its own verdict is `regressed` **and** its adjusted `p_regressed` is below
  `alpha/2`.
- `inconclusive` otherwise. That includes a slice whose own interval says `regressed` but which,
  once the number of slices tested is accounted for, is within what chance produces. The report
  must say so for that slice rather than leave the reader to wonder.

Every result in one family must share a confidence level; mixing them is an error.

## 7. Interface

```python
class Verdict(StrEnum):
    NON_INFERIOR = "non_inferior"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"

class Correction(StrEnum):
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"

@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    confidence: float

@dataclass(frozen=True)
class PairedResult:
    n: int
    interval: Interval
    method: str     # "tango_score" | "bootstrap_bca" | "count_bound" | "constant_shift"
    tolerance: float
    p_non_inferior: float
    p_regressed: float

def paired_binary(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    tolerance: float,
    confidence: float = 0.95,
) -> PairedResult: ...

def paired_continuous(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    tolerance: float,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = None,
) -> PairedResult: ...

def tango_statistic(
    favour_candidate: int, favour_baseline: int, n: int, difference: float
) -> float: ...
def tango_interval(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float = 0.95
) -> tuple[float, float]: ...
def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]: ...
def jackknife_acceleration(values: Sequence[float]) -> float: ...
def holm(p_values: Sequence[float]) -> list[float]: ...
def benjamini_hochberg(p_values: Sequence[float]) -> list[float]: ...
def judge(interval: Interval, tolerance: float) -> Verdict: ...
def family_verdicts(results: Sequence[PairedResult], correction: Correction) -> list[Verdict]: ...
```

`family_verdicts` returns one verdict per result, in the input order, by section 6.2.

## 7a. Errors

Every "raise" in this document means **`ValueError`**, with a message naming the offending
value.

Mismatched `baseline` and `candidate` lengths is an error for both paired functions. So is a
`tolerance` outside `(0, 1]`, and a continuous score outside `[0, 1]`. NaN is outside both.

`clopper_pearson` requires `trials > 0`, `0 <= successes <= trials` and `0 < confidence < 1`.
`tango_statistic` and `tango_interval` require `n > 0`, non-negative counts with
`favour_candidate + favour_baseline <= n`, and `tango_statistic` requires `difference` in
`[-1, 1]`. `family_verdicts` rejects results built at different confidence levels.

## 8. Invariants

These hold for every input and are the cheapest thing for the oracle to attack:

1. `low <= point <= high` always.
2. `paired_binary` point estimate equals `mean(candidate) - mean(baseline)` to floating point.
3. Swapping baseline and candidate negates the point estimate and mirrors the interval. This is
   exact for `paired_binary`, which is deterministic. For `paired_continuous` it holds only up
   to resampling noise when the bootstrap runs, because negating the differences can move a
   single order statistic across the bias correction's comparison.
4. A wider confidence level gives a wider interval, never a narrower one. Exact for
   `paired_binary`; for `paired_continuous` it is a statement at a fixed seed and resample
   count, since two different resample streams can cross by a hair at adjacent levels.
5. `judge` is total: every interval and every positive tolerance yields one of the three verdicts.
6. Adjusted p-values are >= the raw p-values, and <= 1.
7. `paired_continuous` with a fixed seed is deterministic.
8. For `paired_binary`, the count bound and the constant shift: `low > -delta` exactly when
   `p_non_inferior < alpha/2`, and `high < -delta` exactly when `p_regressed < alpha/2`.
9. Over two or more items, no interval is a single point.
10. `family_verdicts` never produces `non_inferior` or `regressed` for a result that `judge` did
    not already call that.

## 9. Dependency rule

`src/evalgate/stats.py` must not import scipy or statsmodels, at any point, including inside a
function. Those two packages are declared test-only in `pyproject.toml`. If the implementation
imported them, "the statistics module agrees with scipy" would be a statement about nothing.
stdlib `math` and `statistics`, plus numpy for array arithmetic, are what it is allowed.

## 10. Measured error rates

The guarantees above are nominal. How close the methods come to them is measured, not assumed,
by `tools/error_rates.py`, which writes `docs/error-rates.md`. It is seeded and runs
from a clean checkout. CI regenerates the file and fails if the committed copy differs, so the
published numbers are always the ones the current code produces.
