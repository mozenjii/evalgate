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

### 3.3 Interval: exact conditional

Condition on the number of discordant pairs. Under this conditioning,
`favour_candidate ~ Binomial(discordant, pi)`, where `pi` is the probability that a discordant item
favours the candidate. `pi = 1/2` is the null of no difference.

Take a **Clopper-Pearson** (exact binomial) interval `[pi_lo, pi_hi]` at the stated confidence for
`favour_candidate` successes in `discordant` trials, and map it back:

```
low  = (2 * pi_lo - 1) * discordant / n
high = (2 * pi_hi - 1) * discordant / n
```

Clopper-Pearson bounds are defined by the binomial tails, with `alpha = 1 - confidence`:

- `pi_lo` solves `P(X >= k | discordant, pi_lo) = alpha / 2`, and is `0` when `k = 0`
- `pi_hi` solves `P(X <= k | discordant, pi_hi) = alpha / 2`, and is `1` when `k = discordant`

**Why exact rather than Wald.** The Wald interval for paired proportions has coverage well below
its nominal level when the discordant count is small, which is exactly the regime a per-slice gate
lives in. An interval that is too narrow on a thin slice produces confident false verdicts, which
is the failure this project is supposed to prevent.

**The assumption this makes, stated plainly:** conditioning treats `discordant / n` as fixed rather
than estimated. The interval is therefore exact *conditional on the observed discordance*, and
slightly optimistic as an unconditional statement. This is a deliberate trade for correct
small-sample behaviour, and it must be named in the report rather than hidden.

### 3.4 Edge cases, which are not optional

- **`discordant = 0`.** Every item scored the same under both versions. The point estimate is
  exactly `0`. The conditional interval is undefined, so instead bound the discordance rate itself:
  take the Clopper-Pearson upper bound `u` for `0` successes in `n` trials and report `[-u, +u]`.
  That is the **two-sided** bound of section 3.3, `u = 1 - (alpha/2)**(1/n)`, which is about
  `3.7/n` at 95% confidence. It is deliberately not the one-sided rule of three, `3/n`: every
  other interval here is two-sided at `alpha/2`, and mixing the two would make this one slice
  narrower than the rest for no stated reason. It must never be reported as `[0, 0]`: zero
  observed disagreements on 20 items is not proof of equality.

  This prohibition is load-bearing rather than tasteful. `judge` sees only an interval, never
  `n` and never the method, so a zero-discordance slice and a well-powered one are
  indistinguishable to it. Reporting `[0, 0]` would therefore make every such slice
  `non_inferior` by arithmetic, which is exactly the collapse section 6 forbids.
- **`n = 0`.** There is no estimate. Raise, rather than return a number that means nothing.
- **All items favour one side.** `pi_lo = 0` or `pi_hi = 1` as above; the interval is still finite
  because it is scaled by `discordant / n`.

### 3.5 p-value

Exact McNemar, which is the two-sided binomial test of `favour_candidate` out of `discordant`
against `pi = 1/2`. Reported for information. **The gate verdict never depends on it**, because a
p-value answers a question the gate is not asking.

## 4. Continuous outcomes

### 4.1 Point estimate

```
differences[i] = candidate[i] - baseline[i]
difference = mean(differences)
```

### 4.2 Interval: bootstrap, BCa

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

### 4.3 Edge cases

- **All differences identical** (including all zero): the jackknife spread is `0`, `a` is undefined
  by the formula above. Define `a = 0` in that case, and the interval collapses to the point
  estimate, which is the honest answer when every item moved by the same amount.
- **`n = 0`.** Raise.
- **`n = 1`.** No interval is meaningful. Raise rather than report a degenerate one.
- Results must be reproducible: the same seed gives the same interval. **With no seed given,
  one is derived from the data**, by hashing the differences. The default must not be a fresh
  random stream: a gate whose verdict can change between two builds of the same commit is the
  failure a CI check exists to prevent, and leaving the common path unseeded would make that
  the normal case rather than an edge one.
- When every difference is identical, two degeneracies arrive together, not one. The jackknife
  spread is `0`, so `a` is `0/0` as above; but every resample mean also equals the observed
  mean, so the bias-correction proportion is `1/2` under the tie convention and the adjusted
  percentiles are taken over a distribution with no spread. The collapse to the point estimate
  covers both, and is detected by comparing the spread of the differences against a tolerance
  rather than by exact equality, because a constant shift computed in floating point leaves
  differences that disagree in the last bit.

## 5. Many slices

Two tiers, because they answer different questions and must not share a correction:

- **Gate slices** are named in config and can fail the build. Their p-values are corrected with
  **Holm-Bonferroni**, which controls the family-wise error rate.
- **Watch slices** are reported and never fail the build. Theirs are corrected with
  **Benjamini-Hochberg**, which controls the false discovery rate.

Both return adjusted p-values in the **input order**, both are monotone after adjustment, and both
clamp to `1.0`. With `m` p-values sorted ascending as `p_(1) <= ... <= p_(m)`:

- Holm: `adj_(i) = max over j <= i of min(1, (m - j + 1) * p_(j))`
- Benjamini-Hochberg: `adj_(i) = min over j >= i of min(1, (m / j) * p_(j))`

Edge cases: an empty input returns empty. A single p-value is returned unchanged except for the
clamp. Ties must be handled without reordering the output.

## 6. The verdict

Given an interval on `candidate - baseline` and a tolerance `delta > 0`:

| Condition | Verdict | Meaning |
|---|---|---|
| `low > -delta` | `non_inferior` | The data rule out a regression worse than `delta`. |
| `high < -delta` | `regressed` | The data rule out the candidate being within `delta`. |
| otherwise | `inconclusive` | The interval straddles `-delta`. Nothing is ruled out. |

`inconclusive` is a first-class state and **must never collapse into `non_inferior`**. A thin slice
that cannot detect the regression it was asked about is a gap in coverage, and reporting it as a
pass is the precise failure this project exists to prevent. All three states must be reachable and
tested.

`delta` must be strictly positive; a tolerance of zero turns the question back into a significance
test. Reject it.

## 7. Interface

```python
class Verdict(StrEnum):
    NON_INFERIOR = "non_inferior"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"

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
    method: str                 # "exact_conditional" | "bootstrap_bca"
    p_value: float | None       # exact McNemar for binary; None for continuous

def paired_binary(
    baseline: Sequence[bool], candidate: Sequence[bool], confidence: float = 0.95
) -> PairedResult: ...

def paired_continuous(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = None,
) -> PairedResult: ...

def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]: ...
def mcnemar_exact_p(favour_candidate: int, favour_baseline: int) -> float: ...
def jackknife_acceleration(values: Sequence[float]) -> float: ...
def holm(p_values: Sequence[float]) -> list[float]: ...
def benjamini_hochberg(p_values: Sequence[float]) -> list[float]: ...
def judge(interval: Interval, tolerance: float) -> Verdict: ...
```

## 7a. Errors

Every "raise" in this document means **`ValueError`**, with a message naming the offending
value. This was unstated in the first draft, which left the implementation and the oracle each
assuming it separately; they happened to agree, which is not the same as having decided.

Mismatched `baseline` and `candidate` lengths is an error for both paired functions. Section 2
implies it -- pairing is not optional -- but implication is not specification.

`clopper_pearson` requires `trials > 0`, `0 <= successes <= trials` and `0 < confidence < 1`.
The `trials = 0` case is reachable from the zero-discordance path by an implementation that
forgets to branch, so it must raise rather than return something shaped like an answer.

## 8. Invariants

These hold for every input and are the cheapest thing for the oracle to attack:

1. `low <= point <= high` always.
2. `paired_binary` point estimate equals `mean(candidate) - mean(baseline)` to floating point.
3. Swapping baseline and candidate negates the point estimate and mirrors the interval. This is
   exact for `paired_binary`, which is deterministic. For `paired_continuous` it holds only up
   to resampling noise, because negating the differences can move a single order statistic
   across the bias correction's comparison.
4. A wider confidence level gives a wider interval, never a narrower one. Exact for
   `paired_binary`; for `paired_continuous` it is a statement at a fixed seed and resample
   count, since two different resample streams can cross by a hair at adjacent levels.
5. `judge` is total: every interval and every positive tolerance yields one of the three verdicts.
6. Adjusted p-values are >= the raw p-values, and <= 1.
7. `paired_continuous` with a fixed seed is deterministic.

## 9. Dependency rule

`src/evalgate/stats.py` must not import scipy or statsmodels, at any point, including inside a
function. Those two packages are declared test-only in `pyproject.toml`. If the implementation
imported them, "the statistics module agrees with scipy" would be a statement about nothing.
stdlib `math` and `statistics`, plus numpy for array arithmetic, are what it is allowed.
