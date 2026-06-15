# Reproducibility

This document describes how to reproduce GAIA-style experiments with ToolSelf.

## Configuration

1. Install dependencies from `requirements.txt`.
2. Configure an OpenAI-compatible API endpoint for the main agent through `MAIN_LLM_*`.
3. Configure an OpenAI-compatible judge endpoint through `JUDGE_*`.
4. Configure web access through `SEARX_HOST` and optionally `JINA_KEY`.
5. Set `DATA_ROOT` to a local directory containing the benchmark files.

## Datasets

The repository includes loaders and config templates only. Users must obtain the datasets separately and place them according to the layout in `README.md`, or edit the config templates.

## Runner

Use `scripts/run_eval.sh`, which calls the per-sample isolated runner:

```bash
scripts/run_eval.sh --config run_GAIA/configs/gaia.example.json
```

The isolated runner writes one result file per task and a `summary.json` file after each completed task. Set `--sample-timeout-seconds 0` only when no sample-level timeout is desired.
