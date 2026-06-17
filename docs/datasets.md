# Dataset Preparation

This repository does not redistribute benchmark data. Download the data from official sources, keep it outside git, and point the example configs to your local files through `DATA_ROOT`.

## Official Sources

| Benchmark | Source | Local file expected by the example config |
|---|---|---|
| GAIA | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | `${DATA_ROOT}/GAIA.json` |
| GAIA(WS) | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | `${DATA_ROOT}/GAIA(WS).json` |
| FRAMES | [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) | `${DATA_ROOT}/FRAMES/frames_subset_200.json` |
| XBench DeepSearch-2510 | [xbench/DeepSearch-2510](https://huggingface.co/datasets/xbench/DeepSearch-2510), [xbench.org](https://xbench.org) | `${DATA_ROOT}/DeepSearch-2510.csv` |

GAIA is access-restricted on Hugging Face. Request access, then authenticate with `huggingface-cli login` or set `HF_TOKEN` before loading it.

## Expected Layout

```bash
export DATA_ROOT="/path/to/datasets"
mkdir -p "${DATA_ROOT}/FRAMES"
```

```text
${DATA_ROOT}/GAIA.json
${DATA_ROOT}/GAIA(WS).json
${DATA_ROOT}/FRAMES/frames_subset_200.json
${DATA_ROOT}/DeepSearch-2510.csv
```

If you use different paths, update `dataset_path` in the corresponding file under `run_GAIA/configs/`.

## GAIA-style JSON

GAIA, GAIA(WS), and FRAMES are loaded as JSON lists with this schema:

| Field | Description |
|---|---|
| `task_id` | Unique task identifier |
| `question` | User task/question |
| `final_answer` | Reference answer |
| `level` | Optional difficulty or dataset level |

Example:

```json
[
  {
    "task_id": "gaia_validation_0001",
    "question": "Question text...",
    "final_answer": "Reference answer...",
    "level": "1"
  }
]
```

This schema is enough for text-only samples. Some GAIA tasks depend on local files or media; for those, keep the downloaded files locally and include their accessible local paths in the `question` text, or extend the runner to copy attachment fields into each task workspace.

## Conversion Example

Install the optional Hugging Face loader:

```bash
pip install datasets
```

Use the script below as a starting point. Dataset column names can differ by split or version, so the helper accepts common alternatives and can be adjusted for your local copy.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

DATA_ROOT = Path(os.environ["DATA_ROOT"])
DATA_ROOT.mkdir(parents=True, exist_ok=True)
(DATA_ROOT / "FRAMES").mkdir(parents=True, exist_ok=True)

def pick(row, *keys, default=""):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default

def to_gaia_style(dataset, output_path, prefix, limit=None):
    rows = []
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break

        rows.append({
            "task_id": str(pick(row, "task_id", "id", default=f"{prefix}_{index}")),
            "question": str(pick(row, "question", "Question", "prompt", "Prompt")),
            "final_answer": str(pick(row, "final_answer", "Final answer", "answer", "Answer")),
            "level": str(pick(row, "level", "Level", default="")),
        })

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows)} samples to {output_path}")

# GAIA: choose the split/config you want to evaluate.
gaia = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
to_gaia_style(gaia, DATA_ROOT / "GAIA.json", "gaia")

# GAIA(WS) uses the same target schema. Save a separate copy if you maintain a
# different split or filtered version for the web-search setting.
to_gaia_style(gaia, DATA_ROOT / "GAIA(WS).json", "gaia_ws")

# FRAMES: the example config evaluates a 200-sample local subset.
frames = load_dataset("google/frames-benchmark", split="test")
to_gaia_style(frames, DATA_ROOT / "FRAMES" / "frames_subset_200.json", "frames", limit=200)
PY
```

## DeepSearch-2510

DeepSearch-2510 is loaded directly from the official XBench CSV format. The loader decodes each row using the `canary` field and reads:

- `prompt`
- `answer`
- optional `reference_steps`

Download the encrypted CSV:

```bash
huggingface-cli download xbench/DeepSearch-2510 DeepSearch-2510.csv \
  --repo-type dataset \
  --local-dir "${DATA_ROOT}"
```

Or copy a manually downloaded file:

```bash
cp /path/to/DeepSearch-2510.csv "${DATA_ROOT}/DeepSearch-2510.csv"
```

The decoded rows are mapped to the GAIA-style schema before evaluation. Do not upload decrypted DeepSearch plaintext online.
