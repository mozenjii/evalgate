# Changelog

## Unreleased

### 0.1.0 candidate

- Versioned JSONL datasets and results files, deterministic scorers, and a CLI for running
  and comparing targets across slices.
- Paired statistical intervals with explicit tolerance and coverage gaps, correction for
  multiple slices, and published false-pass and false-fail rates.
- A GitHub Action that compares base and head results and reports the gate verdict.
- Local Ollama capture with a dry run, a call budget, optional output redaction, and offline
  replay. The public Iris example exercises capture and replay end to end.
- Isolated distribution builds with release verification and GitHub Actions Trusted Publishing
  support.
