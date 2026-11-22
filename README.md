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
config=examples/triage/evalgate.toml
uv run evalgate run --config $config --label v1 --out v1.jsonl
uv run evalgate run --config $config --label v2 --out v2.jsonl --target target:classify_v2
uv run evalgate compare v1.jsonl v2.jsonl --config $config
```

```
evalgate: regression found
1 regressed / 2 not guarded / 0 ok
Comparing `v2` against `v1` on `triage` v1 (120 items).

Gate slices:
  slice            scorer        n  difference (95% CI)     rules out drops over  tolerance  verdict
  intent=warranty  exact_match  40  -60.0 (-73.7 to -44.6)                     -        5.0  regressed
  intent=billing   exact_match  40  +0.0 (-8.8 to +8.8)                      8.8        5.0  not guarded
  intent=refunds   exact_match  40  +0.0 (-8.8 to +8.8)                      8.8        5.0  not guarded
```

It finds the regression in the slice that has it. Look at the other two lines though: nothing
changed in billing or refunds, and it still won't call them a pass. Forty items with no
disagreements can rule out a drop of 8.8 points, not the 5 the tolerance asks for, and the report
says exactly that. Most tools would print three green ticks there, and two of them would mean
nothing. The fix is yours to choose: add items, or widen the tolerance on purpose.

That's the whole idea. A slice that can't answer is never reported as a pass. Whether it fails the
build is a policy you set: by default it warns, and `on_inconclusive = "fail"` makes it block.

| Exit code | Meaning |
|---|---|
| 0 | Nothing regressed. Slices that are not guarded are reported, and only warn by default. |
| 1 | A gate slice regressed by more than its tolerance. |
| 2 | Nothing regressed, but a gate slice is not guarded and the policy is `fail`. |
| 3 | The check could not run: a bad config, a missing file, a target that won't import. |

## Use it on pull requests

Add an `evalgate.toml` and a dataset to your repository (the [example](examples/triage/evalgate.toml)
shows every part), then:

```yaml
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  evalgate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt   # your project's own dependencies
      - uses: mozenjii/evalgate@main
```

The action runs your target on the pull request's base commit and on its head, each in its own
checkout, compares them, writes the report to the job summary and keeps one comment on the pull
request up to date. It installs into your workflow's Python on purpose: your target runs inside
evalgate, so it needs your dependencies beside it. A pull request from a fork gets a read-only
token, so it gets the job summary but no comment.

The [local Ollama walkthrough](docs/ollama.md) shows how to estimate calls, record a run at
zero provider cost, and replay it offline in the CLI or Action.

## Bring your own outputs

evalgate doesn't have to call your model. If your pipeline already produces outputs, write them as
a results file, one JSON object per line, and compare two of them:

```json
{"id": "billing-000", "output": "billing"}
{"id": "billing-001", "output": "refunds", "latency_ms": 412.5}
```

Scoring happens at comparison time, against the dataset, so the file only needs ids and outputs.
Unknown fields are refused rather than ignored, because a mistyped `ouput` would otherwise read as
an item that produced nothing.

## What's built

- **Versioned datasets.** JSONL plus a manifest, with a content hash over the items so an edit that
  nobody meant to make is caught rather than silently changing what a run means. Every item records
  where it came from — hand-written, mined from logs, synthetic, or promoted from a production
  failure — and which slice it's in. Both are required; an item that declares neither is refused
  rather than filed under "unknown".
- **Deterministic scorers.** Exact match, regex, JSON schema, set F1, edit similarity, or your own
  function. Each declares whether it's binary or continuous, because that picks the statistical test
  and inferring it from the values is silently wrong on a slice where everything passed.
- **Paired statistics, chosen by measuring them.** Binary scorers get Tango's score interval.
  Continuous ones get a BCa bootstrap over the paired differences, except where fewer than eight
  items moved: there the bootstrap can't see the change, so an exact count bound takes over. Never
  two independent means. Each choice beat the alternatives on measured error at the tolerance;
  the numbers are below.
- **Two errors, two controls.** A check passes only if every gate slice passes, so a false pass
  needs no correction. It fails if any slice regresses, so false failures grow with the number of
  slices; those are held down with a Holm correction, and the report names any slice it held
  back. Watch slices use Benjamini-Hochberg.
- **Non-inferiority, not significance.** The gate asks "is this at most δ worse?" and reads the lower
  bound of the interval. A significance test on a forty-item slice passes almost everything, which is
  how a green check comes to mean nothing. δ is per slice and is a person's judgement, recorded next
  to the verdict.
- **Results files.** One JSON object per item, with output, latency, tokens and cost. Each records a
  hash of the input it answered: in CI the baseline runs from the base commit, so if a pull request
  edits an item's question, the two answers are left out and listed instead of being compared as if
  they answered the same thing. A crash counts as a failure, not a missing item.
- **Every slice gets an answer.** Gate slices can fail the build; watch slices are reported only.
  A declared slice with no items, or one paired item, is reported as a gap rather than dropped, so
  the report is never quietly shorter than the config.
- **A config file** holds everything a person decides. Tolerances have no default: how much worse a
  slice may get is a judgement about that slice, and a default would be a number nobody chose.
- **A CLI and a GitHub Action** that turn all of that into a pull-request comment and an exit code.
- **Local model capture and replay.** The Ollama runner estimates live calls before it runs,
  enforces a call cap and can redact outputs before persistence. The Action replays saved outputs
  and reports missing recordings.

## Not built yet

The LLM judge and its calibration against human labels, telling you before you build a dataset
how many items a slice needs,
log mining, promoting a production failure to a permanent case, and the seeded-regression benchmark
that measures the gate's own detection and false-alarm rates.

## How I know the statistics are right

A tool that judges other tools needs its own evaluation, and I'd rather show that than assert it.

I wrote the mathematics down first, in [docs/statistics.md](docs/statistics.md). Then the
statistics are checked against scipy and statsmodels by an oracle written from that document alone,
with no sight of the implementation it judges: 1,229 cases covering Tango's interval against a
numerical maximum-likelihood reference, the p-values at each tolerance, the count bound,
Clopper-Pearson, Holm and Benjamini-Hochberg, the jackknife acceleration and the BCa bootstrap. `src/evalgate/stats.py` imports neither scipy nor
statsmodels — one of the oracle's own tests parses the module to enforce it — because otherwise
"it agrees with scipy" would be a statement about nothing. That's why the exact binomial tails and
the Clopper-Pearson inversion are written out here instead of imported.

A passing oracle proves nothing on its own, so I check it can fail: shrinking Tango's variance by
10% breaks it, and so does moving the bootstrap threshold from eight items to seven.

It caught three real defects in my first implementation. The bootstrap drew a fresh random stream
when no seed was given, so two runs of the same commit could reach different verdicts — on the
default path, which is the exact failure a CI gate exists to prevent. The bias correction counted
resample means strictly below the observed mean, which is fine on continuous data and systematically
wrong once scores take few distinct values. And my own contract contradicted itself about the
zero-discordance bound.

Agreeing with scipy only shows the code computes what I asked for. Whether I asked for the right
thing is a separate question, so I measured how often each method is wrong at exactly the
tolerance, where both errors are hardest to avoid. That found two worse problems. The binary
interval I had shipped treated the disagreement rate as known; when a change breaks items and
fixes none, it called a slice that was truly 10 points worse "safe" up to 48% of the time, and it
got worse with more data. And on continuous scores, a slice where nothing happened to move came
out as a zero-width interval, so it passed every time. Both are replaced, and the measurements
that caught them are now a committed script.

The same discipline applies to the build gates: format, lint, types, documentation links and tests
behind one command, each one watched failing on purpose, which `tools/verify_gates.py` re-checks so
it can't rot.

## Results

Nothing goes in this table until a committed script in this repo produces it. The full tables are
in [docs/error-rates.md](docs/error-rates.md), written by `tools/error_rates.py`; CI regenerates
them and fails if they've drifted from the code.

| Measurement | How | Result |
|---|---|---|
| False pass at the tolerance, binary slice | Exact, 66 cells, n 10 to 500 | worst 3.1% (nominal 2.5%); the interval it replaced reached 47.8% |
| False fail at the tolerance, binary slice | Exact, same cells | worst 4.9%, at n = 200 when every disagreement goes one way |
| False pass at the tolerance, continuous slice | Simulated, 16 settings, n 10 to 100 | worst 3.1% |
| False fail across 10 gate slices | Simulated | 2.7% with Holm, 21.1% without |
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
- [x] Deterministic scorers
- [x] Paired comparison, non-inferiority verdicts and the CLI
- [x] Config file, results files, and a GitHub Action that comments on the pull request
- [x] The statistics settled by evidence: interval error at the tolerance, a correction across
      slices, published error rates
- [ ] A real model: record-and-replay, redaction, a budget cap, no network in tests
- [ ] A release: PyPI package and a tagged action
- [ ] LLM judge with human calibration
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
