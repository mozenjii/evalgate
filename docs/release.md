# Release EvalGate

EvalGate's Python import and command are both `evalgate`. Install the distribution
`mozenjii-evalgate` from PyPI or download a release artifact from GitHub.
[PyPI listing](https://pypi.org/project/mozenjii-evalgate/) · [GitHub Releases](https://github.com/mozenjii/evalgate/releases)

## Verify a candidate

The release workflow can be run manually from GitHub Actions to build a wheel and a source
distribution, check their contents, and run the triage example from an isolated install.
It uploads the checked files as workflow artifacts for inspection. For a local build, use a
new directory so older candidates cannot be picked up from the output path.

```bash
out=$(mktemp -d)
uv build --no-sources --out-dir "$out"
uv run --no-project python tools/check_distribution.py "$out"
```

The smoke test runs the two versions of the triage target and compares their results. The
installed CLI must report the seeded warranty regression and exit with code `1`.
The [changelog](../CHANGELOG.md) records the changes included in the candidate.

## Set up publishing

The first release uses the [MIT license](../LICENSE) and the distribution name
`mozenjii-evalgate`. Set the GitHub repository Actions variable `PYPI_PROJECT_NAME` to
`mozenjii-evalgate`. For a future release, update `__version__` in
`src/evalgate/__init__.py` and refresh `uv.lock`. A release tag must match
the package version exactly, for example `v0.1.0` for `0.1.0`. To check these release
prerequisites locally, pass `--tag` and `--project` to the archive checker.

Create the GitHub Actions environment named `pypi` and restrict it to approved release tags.
Configure required reviewers where available. Configure a pending Trusted Publisher on PyPI
for the approved project name, owner `mozenjii`, repository `evalgate`, workflow filename
`release.yml`, and environment `pypi`. GitHub grants the publishing job a short-lived OIDC
credential. The build job has no publishing credentials. PyPI registers a new project on
its first successful upload. [GitHub deployment environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments) · [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) · [Trusted Publishing on GitHub Actions](https://docs.pypi.org/trusted-publishers/using-a-publisher/)

## Publish and verify

After the owner authorizes the release, push the exact version tag on the reviewed commit.
That tag starts the same package checks as a manual run. Once they pass, the `pypi` environment
controls the publishing job, which uploads the two verified artifacts.

After publishing, verify the exact version from PyPI in a fresh environment. Run this from a
checkout of the release tag in Bash with the approved distribution name and the published version
in the `project` and `version` variables. The commands below run outside the source checkout.

```bash
project='mozenjii-evalgate'
version='0.1.0'
config="$PWD/examples/triage/evalgate.toml"
work=$(mktemp -d)
cd "$work"
run_evalgate() {
  uv run --no-project --isolated --default-index https://pypi.org/simple \
    --with "$project==$version" evalgate "$@"
}
run_evalgate run --config "$config" --label v1 --out v1.jsonl
run_evalgate run --config "$config" --label v2 --out v2.jsonl --target target:classify_v2
status=0
run_evalgate compare v1.jsonl v2.jsonl --config "$config" || status=$?
test "$status" -eq 1
```

The comparison must also name the warranty regression from the
[triage example](../README.md#try-it). [Isolated execution with uv](https://docs.astral.sh/uv/concepts/projects/run/)

After the index check passes, the owner can authorize publishing the Action alias `v0` at the
same release commit. Confirm that a clean workflow resolves `uses: mozenjii/evalgate@v0`, runs the
example, and reports its verdict. The alias identifies the Action's current major release. Use
the exact tag or commit SHA when a workflow needs a fixed revision.
