# evalgate

A CI check for LLM features. It fails a pull request when quality drops on any slice of your data, and tells you whether the drop is real or just noise.

> **Status:** early, but it runs. Datasets, scorers, the paired statistics, run records and the
> `run` / `compare` commands work end to end on the example below. There is no LLM judge yet and
> nothing is released. The results table stays empty until a script in this repo produces the
> numbers.

## The problem

When you change a prompt, swap a model or tweak a retriever, a handful of spot checks usually look fine. Meanwhile some other part of your traffic gets worse and nothing in CI notices, because model output isn't deterministic and "the tests pass" doesn't mean much for free text. Teams end up shipping on gut feeling and finding the regressions from user complaints.

## Try it

The repository ships a 120-item synthetic triage set and a keyword classifier in two versions. The
second has two warranty cues removed, so it is worse on the warranty slice and identical everywhere
else. No model is involved, so this runs instantly and identically on any machine.

```bash
uv sync --all-groups
export PYTHONPATH=examples/triage
uv run evalgate run --dataset examples/triage/manifest.json --target target:classify_v1 --label v1
uv run evalgate run --dataset examples/triage/manifest.json --target target:classify_v2 --label v2
uv run evalgate runs
uv run evalgate compare --baseline <v1-id> --candidate <v2-id> --dataset examples/triage/manifest.json
```

```
slice                       n                  difference  verdict
intent=billing             40   +0.000 (-0.088 to +0.088)  inconclusive
intent=refunds             40   +0.000 (-0.088 to +0.088)  inconclusive
intent=warranty            40   -0.600 (-0.600 to -0.429)  regressed
```

It finds the regression in the slice that has it. Look at the other two lines though: nothing
changed in billing or refunds, and it still will not call them a pass. Forty items with zero
disagreements cannot rule out a five-point drop, so it says so. Most tools would print three green
ticks there, and two of them would mean nothing.

That is the whole idea. Exit code 1 means something regressed, 2 means the data could not answer,
and 0 means it could and nothing did.

## What it does

- Versioned datasets where every item records where it came from — hand-written, mined from logs,
  synthetic, or promoted from a production failure — and which slice it belongs to. Both are
  required; an item that declares neither is refused rather than filed under "unknown".
- Deterministic scorers first: exact match, regex, JSON schema, set F1, edit similarity, or your own
  function. Each declares whether it is binary or continuous, because that picks the statistical
  test and guessing it is silently wrong.
- Paired comparison per slice. Binary scorers get an exact conditional interval, continuous ones a
  BCa bootstrap. Never two independent means.
- Non-inferiority, not significance: "is this at most `delta` worse", answered from the lower bound
  of an interval. A significance test on a forty-item slice passes almost everything.

## Not yet

An LLM judge and its calibration against human labels, log mining, promoting a failure to a
permanent case, the pull-request comment, and the seeded-regression benchmark that measures the
gate's own detection and false-alarm rates.

## How I will know it works

A tool that judges other tools needs its own evaluation, and I would rather show that than assert it.

The statistics are checked by an oracle written from a written contract and from scipy and
statsmodels, with no sight of the implementation it judges: 820 cases covering the exact binomial
intervals, McNemar, Holm and Benjamini-Hochberg, the jackknife acceleration and the BCa bootstrap.
`src/evalgate/stats.py` imports neither scipy nor statsmodels, and one of the oracle's tests parses
the module to enforce that — otherwise "it agrees with scipy" would be a statement about nothing.
The gates have each been watched failing on purpose, which `tools/verify_gates.py` re-checks.

Still to come: inject regressions of known size into a demo feature and measure how often the gate
catches them, and how often it fires on reruns of unchanged code.

| Measurement | How | Result |
|---|---|---|
| Judge agreement with human labels, per rubric | Labelled set, second rater on a subset | not measured |
| Detection rate on injected regressions, by size | Known regressions in the demo feature | not measured |
| False alarms on reruns of unchanged code | Repeated reruns | not measured |
| Cost and time per pull request check | Metered | not measured |

## Design notes

**Non-inferiority, not significance.** The question a gate should ask is "is the new version at most δ worse?", not "is there a statistically significant difference?". On small slices the second question lets almost everything through. The gate will compare the lower bound of a paired confidence interval against a tolerance set per slice.

**Say when a slice can't be tested.** A slice of 25 items can't detect a five point drop however you test it. evalgate will report the smallest regression each slice could have caught, so thin slices show up as a coverage gap instead of a green tick.

**Don't trust the judge until it's calibrated.** Judges drift towards longer answers, first positions and their own model family. I'll measure agreement per rubric and run position-swap and length probes. Where a cheap judge scores a large set and humans label a small one, prediction-powered inference can correct the judge's bias and still give honest intervals.

**Record and replay model calls.** Replayed runs make CI cheap and reproducible. Live reruns are what measure the noise floor.

## Roadmap

- [ ] Dataset format and run records
- [ ] Deterministic scorers and the paired comparison
- [ ] LLM judge with human calibration
- [ ] GitHub Action and pull request comment
- [ ] Failure-to-case promotion and log mining
- [ ] Injected-regression benchmark and write-up

## Development

Requires [uv](https://docs.astral.sh/uv/). The interpreter and every tool version are
pinned, so a fresh checkout runs the same toolchain I do.

```bash
uv sync --all-groups
uv run python tools/check.py
```

That one command runs every gate: formatting, lint, types, documentation links, and
tests against a coverage floor. It is exactly what CI runs. Pass a gate name to run one
of them while you work, for example `uv run python tools/check.py types`.

A gate that has never failed is not known to work, so there is a second command that
makes each one fail on purpose and requires it to notice:

```bash
uv run python tools/verify_gates.py
```

Tested on Python 3.11 and 3.12.

## Related projects

Part of a set of six projects about building LLM systems you can trust:
[evalgate](https://github.com/mozenjii/evalgate),
[policylens](https://github.com/mozenjii/policylens),
[docproof](https://github.com/mozenjii/docproof),
[tollgate](https://github.com/mozenjii/tollgate),
[redline](https://github.com/mozenjii/redline),
[casefile](https://github.com/mozenjii/casefile).

Built by Mohib Ahmad.
