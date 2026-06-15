# Datasets

This repository does not redistribute benchmark data.

## GAIA and GAIA(WS)

Expected JSON fields:

- `task_id`
- `question`
- `final_answer`
- optional `level`

## FRAMES

FRAMES should be supplied as a JSON file using the same GAIA-style schema. If your local FRAMES file uses different field names, convert it before running ToolSelf or update the loader in `run_GAIA/run_submit_eval.py`.

## XBench DeepSearch-2510

DeepSearch-2510 is loaded from the XBench CSV format. The loader decodes each row using the `canary` field and reads:

- `prompt`
- `answer`
- optional `reference_steps`

The decoded rows are mapped to the GAIA-style schema before evaluation.
