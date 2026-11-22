# Run and replay a local Ollama model

The [Iris example](../examples/ollama/evalgate.toml) runs nine classification prompts. EvalGate
checks each JSON response against the labeled species, then reports the paired scores by species.
Run the commands from the repository root. They use the loopback Ollama service with the
`qwen2.5:0.5b` model already installed. For a daemon restricted to local inference, start Ollama
with `OLLAMA_NO_CLOUD=1`. EvalGate requires an Ollama version that reports cloud status
(verified with 0.40.17) and checks the installed model has no remote upstream before sending
a prompt. The commands are shown for PowerShell. Use `--ollama-port` for a separate
cloud-disabled daemon on another loopback port.
[Ollama local-only configuration](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)

```powershell
$demo = @("--config", "examples/ollama/evalgate.toml", "--ollama-model", "qwen2.5:0.5b", "--cassette-dir", "runs/iris-cassettes")
uv run evalgate run @demo --label live --out runs/iris-live.jsonl --max-calls 9 --max-cost-usd 0 --dry-run
uv run evalgate run @demo --label live --out runs/iris-live.jsonl --max-calls 9 --max-cost-usd 0 --allow-local-model
uv run evalgate run @demo --label replay --out runs/iris-replay.jsonl
uv run evalgate compare runs/iris-live.jsonl runs/iris-replay.jsonl --config examples/ollama/evalgate.toml
```

The dry run shows how many requests would need to reach Ollama. The live command can record up
to nine requests at zero provider cost. The replay command reads the recorded responses and
works with Ollama stopped. The results headers carry independent labels and timestamps.
Successful generations are recorded. A failed live generation is scored as an error and its
uncached replay is reported as not run. The call cap counts distinct uncached generation
requests in one invocation. The USD estimate covers provider charges for the local adapter.

Three cases per species exercise the reporting path. At the example's five-point tolerance,
an identical replay still reports the slices as **not guarded**. The sample supports a check
that replay works and is too small to evaluate model quality.

The measurements and original labels come from R. A. Fisher (1936), *Iris*, UCI Machine
Learning Repository. The example selects rows 1, 11, 21, 51, 61, 71, 101, 111 and 121 of
`iris.data`, wraps the measurements in handwritten prompts and normalizes the species names
into lowercase labels without the `Iris-` prefix. Each case records its source row.
[Dataset and citation](https://doi.org/10.24432/C56C76) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Regenerate the checked-in data with `uv run python examples/ollama/generate.py`.

## Redaction and recording directories

For data that needs scrubbing, pass `--redact package.module:function` with a trusted callable
that takes and returns a JSON value. It runs on newly generated outputs before they enter either
the cassette or the results file. The cassette stores only that output under a hash of the
provider, model, generation options, prompt, and redactor reference. Prompts still go to local
Ollama. Use a new recording directory after changing the model behind a tag or changing the
redactor's implementation. Keep the directory on a filesystem supporting hard links for atomic
publication and share it only when the recordings are suitable for everyone who can restore it.

## Replay on GitHub Actions

Restore recordings with `actions/cache/restore` before calling the composite Action. Pass
`ollama-model` and `cassette-dir`, plus `redactor` if recordings were keyed with a redactor.
The Action uses the same directory for both checkouts. Forks can replay accessible cache
entries. The report lists items omitted on either side, and slices with too few pairs follow
the configured inconclusive policy.

The repository's replay job restores `evalgate-iris-qwen2.5-0.5b-v1`. On a cache miss it
uses the [nine checked-in recordings](../examples/ollama/recordings) from the public Iris
example, captured with `qwen2.5:0.5b` on Ollama 0.40.17. It verifies all nine items on both
sides and saves the cache on trusted runs.
