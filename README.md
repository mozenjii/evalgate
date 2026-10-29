# evalgate

A CI check for LLM features. It fails a pull request when quality drops on any slice of your data, and tells you whether the drop is real or just noise.

> **Status:** early. I'm at the design stage and there is no released code yet. The results table below stays empty until a script in this repo produces the numbers.

## The problem

When you change a prompt, swap a model or tweak a retriever, a handful of spot checks usually look fine. Meanwhile some other part of your traffic gets worse and nothing in CI notices, because model output isn't deterministic and "the tests pass" doesn't mean much for free text. Teams end up shipping on gut feeling and finding the regressions from user complaints.

## What it will do

- Keep versioned evaluation datasets where every item records where it came from (hand-labelled, mined from logs, or synthetic) and which slice it belongs to.
- Score runs with deterministic checks first and an LLM judge second, and only trust a judge once its agreement with human labels has been measured.
- Compare an old run with a new one per slice using paired statistics, so "worse" means worse by more than a margin you chose, not just "the number moved".
- Run in CI and comment on the pull request with the slices that regressed and a few example items.
- Turn a failing production case into a permanent regression case.

## How I plan to know it works

A tool that judges other tools needs its own evaluation. I'll build a small demo feature (support ticket triage), inject regressions of known size into it, and measure how often the gate catches them and how often it fires when I rerun unchanged code.

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
