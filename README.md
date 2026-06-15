<div align="center">

# [ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Intrinsic Adaptation](https://arxiv.org/html/2602.07883v1)

**Runtime self-reconfiguration for long-horizon, tool-use agents**

</div>

<div align="center">
  <a href="https://arxiv.org/html/2602.07883v1"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv"></a>
  <a href="https://github.com/lian-tian-mo-zun/ToolSelf"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="API" src="https://img.shields.io/badge/API-OpenAI--Compatible-green">
  <img alt="Benchmarks" src="https://img.shields.io/badge/Eval-GAIA%20%7C%20FRAMES%20%7C%20XBench-purple">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"></a>
</div>

<div align="center">
  <a href="https://arxiv.org/html/2602.07883v1">Paper</a> •
  <a href="#-introduction">Introduction</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-reproducibility">Reproducibility</a> •
  <a href="#-citation">Citation</a>
</div>

<br>

<div align="center">
  <img src="assets/toolself_overview.png" alt="ToolSelf overview" width="100%">
</div>

## 🚀 News

- **[2026-06-15]** Initial open-source release of ToolSelf code, evaluation runners, and reproducibility configs.
- **[2026-02-08]** Paper available on arXiv: [ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Intrinsic Adaptation](https://arxiv.org/html/2602.07883v1).

## 💡 Introduction

ToolSelf is a tool-use agent framework that unifies **task execution** and **configuration generation** in one iterative loop. Instead of fixing the agent configuration before execution, ToolSelf lets the agent update its sub-goals, strategy, toolbox, context management, and inter-stage knowledge at runtime.

This repository includes:

- the ToolSelf benchmark runner,
- a ReAct-style execution agent,
- web, search, file, code-interpreter, reconfiguration, and termination tools,
- GAIA-style evaluation scripts,
- loaders and config templates for GAIA, GAIA(WS), FRAMES, and XBench DeepSearch-2510,
- reproducibility notes and result summarization utilities.

> Datasets, API keys, private model endpoints, and run outputs are intentionally not included.

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/lian-tian-mo-zun/ToolSelf.git
cd ToolSelf

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env`, then load it:

```bash
source .env
```

### Smoke Test

Run a small evaluation job:

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/gaia.example.json \
  --max-samples 2 \
  --max-parallel-workers 1 \
  --sample-timeout-seconds 600
```

Summarize a completed run:

```bash
python scripts/summarize_results.py outputs/gaia
```

## ⚙️ Configuration

ToolSelf reads model, web, and dataset settings from environment variables. The repository provides `.env.example` as a template.

| Variable | Required | Purpose |
|---|---:|---|
| `MAIN_LLM_API_KEY` | Yes | API key for the main ToolSelf agent |
| `MAIN_LLM_API_BASE_URL` | Yes | OpenAI-compatible API base URL |
| `MAIN_LLM_MODEL` | Yes | Main model name |
| `JUDGE_API_KEY` | Yes | API key for the LLM-as-judge |
| `JUDGE_BASE_URL` | Yes | Judge API base URL |
| `JUDGE_MODEL` | Yes | Judge model name |
| `SEARX_HOST` | Yes | Searx search endpoint |
| `DATA_ROOT` | Yes | Root directory for local benchmark files |
| `JINA_KEY` | Optional | Jina Reader API key |
| `JINA_READER_URL` | Optional | Jina Reader endpoint |
| `VISIT_LLM_*` | Optional | Separate webpage-summary model config |
| `FILE_ANALYZER_*` | Optional | Separate file-analysis model config |
| `MAX_LLM_CALL_PER_RUN` | Optional | Per-run LLM call budget |

## 📁 Project Structure

```text
ToolSelf/
├── config.py                         # Environment-driven model/tool config
├── toolself_gaia.py                  # ToolSelf benchmark runner
├── execution_agent/                  # ReAct-style execution agent
├── tools/                            # Tool implementations
├── run_GAIA/
│   ├── evaluator.py                  # GAIA-style evaluator
│   ├── run_eval.py                   # Direct evaluation entry point
│   ├── run_eval_isolated.py          # Per-sample isolated runner
│   └── configs/                      # Dataset config templates
├── scripts/
│   ├── run_eval.sh                   # Convenience runner
│   └── summarize_results.py          # Result summary utility
├── docs/
│   ├── datasets.md
│   └── reproducibility.md
├── requirements.txt
└── .env.example
```

## 🗂️ Data Preparation

The repository provides loaders and config templates, but not benchmark data.

Default expected layout:

```text
${DATA_ROOT}/GAIA.json
${DATA_ROOT}/GAIA(WS).json
${DATA_ROOT}/FRAMES/frames_subset_200.json
${DATA_ROOT}/DeepSearch-2510.csv
```

GAIA-style JSON entries should contain:

| Field | Description |
|---|---|
| `task_id` | Unique task identifier |
| `question` | User task/question |
| `final_answer` | Reference answer |
| `level` | Optional difficulty or dataset level |

DeepSearch-2510 is loaded from the XBench CSV format. The loader decodes `prompt`, `answer`, and optional `reference_steps` with the row-level `canary` field and maps each row to the GAIA-style schema.

More details: [`docs/datasets.md`](docs/datasets.md).

## 🧪 Reproducibility

Use the isolated runner for full benchmark runs. It launches each sample in a child process, so a hung tool call or network request does not block the entire job.

### GAIA

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/gaia.example.json \
  --max-parallel-workers 4 \
  --sample-timeout-seconds 1800
```

### GAIA(WS)

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/gaia_ws.example.json \
  --max-parallel-workers 4 \
  --sample-timeout-seconds 1800
```

### FRAMES

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/frames.example.json \
  --max-parallel-workers 4 \
  --sample-timeout-seconds 1800
```

### XBench DeepSearch-2510

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/deepsearch.example.json \
  --max-parallel-workers 4 \
  --sample-timeout-seconds 1800
```

More details: [`docs/reproducibility.md`](docs/reproducibility.md).

## 📦 Outputs

Each run writes an output directory with:

```text
results/task_<task_id>_result.json
results/summary.json
logs/
workspaces/
isolated_runs/
```

Do not commit `.env`, datasets, output directories, workspaces, logs, or isolated run artifacts.

## 🔒 Release Hygiene

Before publishing or pushing updates:

```bash
rg -n "api_key|secret|token|password|PRIVATE_ENDPOINT|ABSOLUTE_LOCAL_PATH" .
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

## 📌 Citation

```bibtex
@article{toolself2026,
  title={ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Intrinsic Adaptation},
  author={Zhou, Jingqi and Wang, Sheng and Deng, DeZhao and Lu, Junwen and Su, Junwei and Li, Qintong and Gao, Jiahui and Wu, Hao and Jiang, Jiyue and Kong, Lingpeng and Wu, Chuan},
  year={2026},
  eprint={2602.07883},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

Please cite our paper if you find ToolSelf useful for your research.

## 📄 License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
