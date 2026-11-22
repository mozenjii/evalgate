# Calibrate a binary judge

Calibration compares frozen judge decisions with independently reviewed human labels. Use one
row per item and rubric. The report is offline and reads decisions and optional costs. It
needs no raw prompts or responses.

```bash
evalgate calibrate decisions.jsonl
```

## Decision file

These two rows are synthetic examples of the input format. They carry no evidence about model
quality or human agreement.

```jsonl
{"id":"example-1","rubric":"grounded","judge":true,"human":true,"repeat":true}
{"id":"example-2","rubric":"grounded","judge":false,"human":null}
```

Each `id` and `rubric` pair is unique. `judge` and the optional reviewed `human` use JSON
booleans. `true` means the response passes that rubric. Omit `human` or use `null` when there
is no review. The report includes both confusion-matrix classes, Cohen's kappa, and the signed
judge-minus-human acceptance bias on the labeled subset. Kappa is unavailable when chance
agreement is 100%, including when both raters chose the same single class throughout.

Keep the `judgement_sha256` fingerprint when preparing rows from EvalGate judge output. It
identifies the grading protocol, rubric and judge model used for the decisions. Every row
in a rubric must carry the same 64-character lowercase hexadecimal fingerprint. A manually
assembled rubric can omit the fingerprint on every row.

Optional fields carry additional decisions for the same cases.

| Field | Evidence |
| --- | --- |
| `repeat` | The same fixed judge rerun on the same input at temperature zero. Use a fresh response rather than reading the original recording. |
| `position_swap` | Decision on the same logical target after exchanging its presentation position. Normalize both decisions to the original target before recording them. |
| `length_variant` | Decision on a version with altered verbosity and equivalent target content. Check that the case still has the same expected label. |
| `family_variant` | Decision on a matched answer from a different model family. Control for answer quality and length before interpreting the change as a family preference. |
| `second_human` | An independent human decision used to compute human-to-human kappa on shared cases. |
| `ensemble_2`, `ensemble_3` | Decisions from frozen two- and three-judge ensembles. Record results only after defining the aggregation and tie policy. |
| `judge_cost_usd`, `ensemble_2_cost_usd`, `ensemble_3_cost_usd` | The total provider costs for each decision. Omit unknown costs and use zero only when known. |

Probe results show the matched denominator, number of flipped decisions, and variant-minus-original
acceptance change. Unrecorded probes are shown as unmeasured. The ensemble rows use only cases
with human, single-judge, and both ensemble decisions available. Gain per added US dollar is
relative to the single judge on this matched set and requires complete costs and a positive cost
increase.

## Estimate the finite dataset acceptance rate

Freeze a dataset and a judge's decisions across it. Draw a uniform random sample without
replacement separately for each rubric. Obtain human labels for every sampled item without
changing the judge or selecting items according to its decisions. Include all judge decisions in
the file, including cases without human labels. Declare this design with:

```bash
evalgate calibrate decisions.jsonl --random-sample --confidence 0.95
```

The prediction-powered estimate adds the sample mean of `human - judge` to the full dataset's
mean judge decision. This corrects the judge's aggregate acceptance rate, and individual
decisions remain unchanged. The interval targets the fixed dataset's mean human decision under
the declared review design. A fully reviewed dataset has an exact rate. When partial reviews
were curated, filtered, or taken from an active-learning queue, use the command without the flag
to get descriptive agreement for that subset. [Prediction-Powered Inference](https://arxiv.org/abs/2301.09633)

EvalGate applies the empirical Bernstein–Serfling bound to both signs of the sampled residuals,
allocating one tenth of the allowed error to the theorem's probability parameter. It intersects
the result with the deterministic bounds implied by observed labels. The human-only comparison
uses the same procedure with human decisions in place of residuals. The intervals are per rubric,
with no simultaneous guarantee across rubrics or future datasets. The corrected interval can be
wider when the judge's mistakes are noisy or the reviewed sample is small.
[Bardenet and Maillard, Theorem 4.3](https://arxiv.org/abs/1309.4029)

Run `uv run pytest tests/test_calibration.py -k seeded` to repeat a synthetic simulation where a
fixed judge overaccepts by five percentage points. The test checks that correction reduces bias
and that the corrected intervals are narrower on average in that scenario.
