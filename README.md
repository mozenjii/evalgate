# evalgate

A CI check for LLM features. It fails a pull request when quality drops on any slice of your data,
and tells you whether the drop is real or just noise.

## The problem

When you change a prompt, swap a model or tweak a retriever, a handful of spot checks usually look
fine. Meanwhile some other part of your traffic gets worse and nothing in CI notices, because model
output isn't deterministic and "the tests pass" doesn't mean much for free text. Teams end up
shipping on gut feeling and finding the regressions from user complaints.

## Try it

The repository ships a 120-item synthetic triage set and a keyword classifier in two versions. The
second has two warranty cues removed, so it's worse on the warranty slice and identical everywhere
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
changed in billing or refunds, and it still won't call them a pass. Forty items with zero
disagreements can't rule out a five-point drop, so it says so. Most tools would print three green
ticks there, and two of them would mean nothing.

That's the whole idea. Exit code 1 means something regressed, 2 means the data couldn't answer, and
0 means it could and nothing did. Inconclusive gets its own exit code because sharing one with
success is exactly how an underpowered slice becomes a green tick.

## What's built

- **Versioned datasets.** JSONL plus a manifest, with a content hash over the items so an edit that
  nobody meant to make is caught rather than silently changing what a run means. Every item records
  where it came from — hand-written, mined from logs, synthetic, or promoted from a production
  failure — and which slice it's in. Both are required; an item that declares neither is refused
  rather than filed under "unknown".
- **Deterministic scorers.** Exact match, regex, JSON schema, set F1, edit similarity, or your own
  function. Each declares whether it's binary or continuous, because that picks the statistical test
  and inferring it from the values is silently wrong on a slice where everything passed.
- **Paired statistics.** Binary scorers get an exact conditional interval — Clopper-Pearson over the
  discordant pairs, since only disagreements carry information about a difference. Continuous ones
  get a BCa bootstrap over the paired differences. Never two independent means.
- **Non-inferiority, not significance.** The gate asks "is this at most δ worse?" and reads the lower
  bound of the interval. A significance test on a forty-item slice passes almost everything, which is
  how a green check comes to mean nothing. δ is per slice and is a person's judgement, recorded next
  to the verdict.
- **Run records.** SQLite, with output, latency, tokens and cost per item. Pairing two runs names the
  items only one of them scored instead of quietly intersecting, because the item count is what you
  use to judge whether a verdict means anything.
- **A CLI and CI workflow** that turn all of that into an exit code.

## Not built yet

The LLM judge and its calibration against human labels, record-and-replay for model calls, log
mining, promoting a production failure to a permanent case, the pull-request comment, and the
seeded-regression benchmark that measures the gate's own detection and false-alarm rates.

## How I know the statistics are right

A tool that judges other tools needs its own evaluation, and I'd rather show that than assert it.

I wrote the mathematics down first, in [docs/statistics.md](docs/statistics.md). Then the
statistics are checked against scipy and statsmodels by an oracle written from that document alone,
with no sight of the implementation it judges: 820 cases covering the exact binomial intervals, McNemar, Holm and Benjamini-Hochberg, the
jackknife acceleration and the BCa bootstrap. `src/evalgate/stats.py` imports neither scipy nor
statsmodels — one of the oracle's own tests parses the module to enforce it — because otherwise
"it agrees with scipy" would be a statement about nothing. That's why the exact binomial tails and
the Clopper-Pearson inversion are written out here instead of imported.

A passing oracle proves nothing on its own, so I checked it can fail: flipping the sign of the
binary point estimate breaks it, and restoring the sign fixes it.

It caught three real defects in my first implementation. The bootstrap drew a fresh random stream
when no seed was given, so two runs of the same commit could reach different verdicts — on the
default path, which is the exact failure a CI gate exists to prevent. The bias correction counted
resample means strictly below the observed mean, which is fine on continuous data and systematically
wrong once scores take few distinct values. And my own contract contradicted itself about the
zero-discordance bound.

The same discipline applies to the build gates: format, lint, types, documentation links and tests
behind one command, each one watched failing on purpose, which `tools/verify_gates.py` re-checks so
it can't rot.

## Results

Nothing goes in this table until a committed script in this repo produces it.

| Measurement | How | Result |
|---|---|---|
| Judge agreement with human labels, per rubric | Labelled set, second rater on a subset | not measured |
| Detection rate on injected regressions, by size | Known regressions in the demo feature | not measured |
| False alarms on reruns of unchanged code | Repeated reruns | not measured |
| Cost and time per pull request check | Metered | not measured |

## Design notes

**Say when a slice can't be tested.** A slice of 25 items can't detect a five-point drop however you
test it. Thin slices show up as a coverage gap rather than a green tick, and that's load-bearing
rather than tasteful: the verdict function sees an interval and never the item count, so if a slice
with zero disagreements reported a zero-width interval it would pass by arithmetic instead of by
evidence.

**Don't trust the judge until it's calibrated.** Judges drift towards longer answers, first positions
and their own model family. I'll measure agreement per rubric and run position-swap and length
probes. Where a cheap judge scores a large set and humans label a small one, prediction-powered
inference can correct the judge's bias and still give honest intervals.

**Record and replay model calls.** Replayed runs make CI cheap and reproducible. Live reruns are what
measure the noise floor.

## Roadmap

- [x] Dataset format, slices and provenance
- [x] Run records
- [x] Deterministic scorers
- [x] Paired comparison, non-inferiority verdicts and the CLI
- [ ] Redaction, a no-egress default for tests, and a budget cap on model calls
- [ ] LLM judge with human calibration
- [ ] GitHub Action and pull request comment
- [ ] Failure-to-case promotion and log mining
- [ ] Injected-regression benchmark and write-up

## Development

Requires [uv](https://docs.astral.sh/uv/). The interpreter and every tool version are pinned, so a
fresh checkout runs the same toolchain I do.

```bash
uv sync --all-groups
uv run python tools/check.py
```

That one command runs every gate: formatting, lint, types, documentation links, and tests against a
coverage floor. It's exactly what CI runs, so a gate can't exist in one place and be missing from the
other. Pass a gate name to run one while you work, for example `uv run python tools/check.py types`.

A gate that has never failed isn't known to work, so there's a second command that makes each one
fail on purpose and requires it to notice:

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
