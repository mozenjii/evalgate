# Judge a target's stored answers

The judge applies one binary rubric to each target answer and writes a separate results file.
It only saves boolean verdicts. The original answer, reference, and private item data do not enter
the judge cassette. Live execution sends them to a cloud-disabled local Ollama daemon, so use
public, synthetic, or explicitly scrubbed inputs. Model-generated verdicts should be checked
against independently supplied human labels before they govern a production release.

## Try the synthetic reply example

The [example configuration](../examples/judge/evalgate.toml) uses a fictional support policy.
The candidate deliberately breaks the returns reply and leaves delivery replies unchanged.
Run the following from the checkout in PowerShell, with the model installed and
[Ollama running with cloud disabled](ollama.md).

```powershell
$config = "examples/judge/evalgate.toml"
$store = "runs/reply-judge"
$argsForJudge = @("--config", $config, "--rubric", "examples/judge/policy-accuracy.txt",
  "--ollama-model", "qwen2.5:0.5b", "--cassette-dir", "$store/cassettes")

uv run evalgate run --config $config --label baseline --out "$store/base.jsonl"
uv run evalgate run --config $config --label candidate --out "$store/head.jsonl" --target reply_target:candidate
uv run evalgate judge "$store/base.jsonl" @argsForJudge --out "$store/base-graded.jsonl" --dry-run
uv run evalgate judge "$store/base.jsonl" @argsForJudge --out "$store/base-graded.jsonl" --allow-local-model --max-calls 16
uv run evalgate judge "$store/head.jsonl" @argsForJudge --out "$store/head-graded.jsonl" --allow-local-model --max-calls 16
uv run evalgate compare "$store/base-graded.jsonl" "$store/head-graded.jsonl" --config $config
```

Rerun the two grading commands without `--allow-local-model` to replay recordings with Ollama
stopped. An absent recording leaves an unpaired item. Malformed verdicts and failed judge calls
abort grading, while a recorded failure in the source target remains a target failure.
Comparison requires `judge_pass` and two results files made with the same rubric, model tag,
and current dataset. Changing a reference answer requires regrading both versions.
Use a new cassette directory after changing the model associated with an existing tag.

For an independent repeat on a running local daemon, add `--replicate 1`. This uses a separate
recording namespace and leaves the rubric and model unchanged. Export the original predictions as
JSONL, then add independent labels and optional probe decisions as described in the
[calibration guide](calibration.md).

```powershell
uv run evalgate judge-labels "$store/base-graded.jsonl" > "$store/review.jsonl"
uv run evalgate calibrate "$store/review.jsonl"
```

The fictional reference answers help the example demonstrate the pipeline. They provide no
evidence of a human review study.

In one local smoke run, `qwen2.5:0.5b` accepted the baseline and the broken
candidate, and the gate left both slices unguarded. The replay reproduced all 16 decisions on
each side with the daemon stopped. The offline acceptance test supplies known verdicts and
checks that the same pipeline correctly flags the planted returns regression.
