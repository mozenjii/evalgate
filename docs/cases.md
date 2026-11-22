# Mine and promote evaluation cases

Start with synthetic traces or an export you have already minimized and scrubbed. The tools
run locally and use the classification you declare on each record. Scrubbing, label review,
and permission to store the resulting cases belong to the person preparing the export.

## Prepare a review file

Save a trace like this as `traces.jsonl`, with one JSON object per line. This is a synthetic
example for the [triage dataset](../examples/triage/manifest.json).

```json
{"classification":"synthetic","input":"Does my device guarantee cover a cracked screen?","slices":{"intent":"warranty","length":"short"},"origin_run":"triage-demo-42","suspected_stage":"classification","failure":true,"judge_votes":[true,false]}
```

```bash
uv run evalgate mine traces.jsonl --out review.jsonl
```

- Set `classification` to `synthetic` or `redacted`. Supply the minimized `input`, its proposed
  `slices`, and a scrubbed `origin_run` identifier. Raw model outputs and arbitrary log metadata
  are outside the accepted schema.
- `failure` records whether an observed run failed. `suspected_stage` records where it may have
  gone wrong. Include both when you have a failure you plan to turn into a regression case.
- An optional array of binary `judge_votes` moves candidates with disagreement to the front.
  The tool keeps the first representative of each canonical input and slice combination after
  that prioritization. JSON key order and formatting do not affect exact duplicates.
- `--near-duplicate-threshold` optionally clusters similar text within the same classification
  and slice assignment. It compares up to 64 minimum hashes of three-word shingles, keeping
  the first selected representative. Choose a threshold greater than zero and at most one.
  Structured inputs and short text still use exact matching.

The output is a new JSONL file for review. Inspect each row you intend to keep. Confirm the
input reproduces the issue, minimize it further if possible, correct the slices, and supply a
non-null `expected` result. For this example the expected label is `"warranty"`. Set
`reviewed` to `true` only after checking the case. Keep only
reviewed failures in the file you pass to promotion.

## Write the next dataset version

Promoted cases are marked for strict regression protection. If a case passed in the baseline
and fails a configured deterministic binary gate in the candidate, the check fails and names
that case individually. Watch rules and model-judged comparisons continue to use slice inference.
An omitted protected case is reported as unguarded. Keep promoted cases in later dataset versions
unless you record a reason for removing one during review.

```bash
uv run evalgate promote review.jsonl \
  --manifest examples/triage/manifest.json --version 2 --out-dir triage-v2
```

Promotion refuses an existing destination, a reused source version, unknown slices, and
duplicates within the batch or source dataset. It keeps the existing cases in order and
appends the reviewed failures under stable IDs derived from input and slices. Redacted
failure cases carry `promoted_failure` provenance. Synthetic cases keep their `synthetic`
provenance. Both retain the scrubbed origin run and suspected stage in `source`.
The destination filesystem must support hard links for publishing complete files without
overwriting an existing file.

Point your eval configuration at the new manifest after reviewing the dataset diff. The
normal gate and its configured tolerances then apply to the added cases.
