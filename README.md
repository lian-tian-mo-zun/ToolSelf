<div align="center">

# [ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Emergent Adaptation](https://arxiv.org/abs/2602.07883)

**A tool-use agent framework that lets agents reconfigure themselves while solving the task.**

</div>

<div align="center">
  <a href="https://arxiv.org/abs/2602.07883"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv"></a>
  <a href="https://github.com/lian-tian-mo-zun/ToolSelf"><img src="https://img.shields.io/badge/Code-GitHub-black" alt="GitHub"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/Code%20License-MIT-blue" alt="Code License"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="API" src="https://img.shields.io/badge/API-OpenAI--Compatible-green">
  <img alt="Benchmarks" src="https://img.shields.io/badge/Eval-GAIA%20%7C%20FRAMES%20%7C%20XBench-purple">
</div>

<p align="center">
  <a href="#news">News</a> ·
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#datasets">Datasets</a> ·
  <a href="#citation">Citation</a>
</p>

ToolSelf is a tool-use agent framework that unifies task execution and configuration generation in one iterative loop. Instead of fixing the agent configuration before execution, ToolSelf lets the agent update its sub-goals, strategy, toolbox, context-management mode, and inter-stage knowledge at runtime.

This repository includes the ToolSelf benchmark runner, a ReAct-style execution agent, web/search/file/code tools, reconfiguration and termination tools, GAIA-style evaluation scripts, benchmark config templates, and result summarization utilities.

<a id="news"></a>

## News

- **2026-06-15**: ToolSelf code, evaluation runners, config templates, and documentation are released.
- **2026-05-31**: ToolSelf v3 is available on arXiv: [arXiv:2602.07883](https://arxiv.org/abs/2602.07883).

<a id="overview"></a>

## Overview

Most agent systems choose a configuration before execution: a task decomposition, a toolset, a prompting strategy, and a context policy. ToolSelf makes that configuration a first-class, tool-updatable object. During solving, the agent can call a reconfiguration tool, summarize the current stage, and generate the next configuration before continuing.

<div align="center">
<img src="./assets/toolself_overview.png" width="90%">
</div>

<p align="center"><em>Overview of ToolSelf. Configuration becomes a dynamic variable that can be updated through tool calls during execution.</em></p>

## Highlights

- **Runtime self-reconfiguration**: update sub-goals, strategy, tools, task knowledge, and context mode while solving.
- **One policy, one action space**: task execution and adaptation happen inside the same ReAct-style loop.
- **Tool-native adaptation**: reconfiguration is represented as a standard tool call, making it traceable and evaluable.
- **Long-horizon focus**: designed for deep research, general assistance, and software engineering workflows.
- **Reproducible evaluation**: includes isolated benchmark runners, config templates, and result summaries.

<a id="method"></a>

## Method

ToolSelf equips the execution agent with ordinary environment tools plus two special tools:

| Tool | Purpose |
|---|---|
| Reconfiguration tool | Updates the current sub-goal, execution strategy, toolbox, knowledge, and context-management mode |
| Termination tool | Returns the final answer when the task is complete |

At stage `i`, the agent operates under a configuration `C_i = (q_i, sigma_i, T_i, K_i, m_i)`, where `q_i` is the sub-goal, `sigma_i` is the execution strategy, `T_i` is the toolbox, `K_i` is inter-stage knowledge, and `m_i` is the context-management mode. When progress or feedback indicates that the current configuration is no longer suitable, the agent invokes the reconfiguration tool and continues under `C_{i+1}`.

The paper also introduces **Configuration-Aware Two-stage Training (CAT)**: rejection sampling fine-tuning for cold-start trajectories, followed by trajectory-level KTO reinforcement learning to improve runtime adaptation. See the [paper](https://arxiv.org/abs/2602.07883) for full training and evaluation details.

## Repository Contents

| Path | Description |
|---|---|
| `toolself_gaia.py` | Main ToolSelf benchmark runner |
| `execution_agent/` | ReAct-style execution agent |
| `tools/` | Search, browse, file analysis, code interpreter, bash, editor, reconfiguration, and termination tools |
| `run_GAIA/` | GAIA-style runners, evaluator, and dataset configs |
| `scripts/` | Evaluation launcher and result summarization |
| `docs/` | Dataset and reproducibility notes |

<a id="quick-start"></a>

## Quick Start

### Installation

```bash
git clone https://github.com/lian-tian-mo-zun/ToolSelf.git
cd ToolSelf

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your local endpoints:

```bash
cp .env.example .env
source .env
```

Minimum required settings:

```bash
export MAIN_LLM_API_KEY="your-api-key"
export MAIN_LLM_API_BASE_URL="https://your-endpoint/v1"
export MAIN_LLM_MODEL="your-model-name"

export JUDGE_API_KEY="your-judge-api-key"
export JUDGE_BASE_URL="https://your-judge-endpoint/v1"
export JUDGE_MODEL="your-judge-model"

export SEARX_HOST="http://localhost:8888"
export DATA_ROOT="/path/to/datasets"
```

Additional optional variables are listed in [`.env.example`](.env.example) and [`config.py`](config.py).

### Run

After preparing the benchmark files under `DATA_ROOT`, run a two-sample check:

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

<a id="datasets"></a>

## Datasets

This repository provides loaders and config templates, but does **not** redistribute benchmark data. Download the data from official sources and keep it outside git.

| Benchmark | Source | Expected local path |
|---|---|---|
| GAIA | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | `${DATA_ROOT}/GAIA.json` |
| GAIA(WS) | [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | `${DATA_ROOT}/GAIA(WS).json` |
| FRAMES | [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) | `${DATA_ROOT}/FRAMES/frames_subset_200.json` |
| XBench DeepSearch-2510 | [xbench/DeepSearch-2510](https://huggingface.co/datasets/xbench/DeepSearch-2510), [xbench.org](https://xbench.org) | `${DATA_ROOT}/DeepSearch-2510.csv` |

GAIA, GAIA(WS), and FRAMES should be converted into a GAIA-style JSON list:

```json
[
  {
    "task_id": "sample_0001",
    "question": "Question text...",
    "final_answer": "Reference answer...",
    "level": "optional"
  }
]
```

DeepSearch-2510 should remain in the official encrypted CSV format. ToolSelf decodes it locally during evaluation; do not upload decrypted plaintext data online.

For download commands and conversion examples, see [`docs/datasets.md`](docs/datasets.md).

<a id="reproducibility"></a>

## Reproducibility

Use the isolated runner for full benchmark jobs. It evaluates each sample in a child process, so one timeout does not block the entire run.

| Benchmark | Config |
|---|---|
| GAIA | `run_GAIA/configs/gaia.example.json` |
| GAIA(WS) | `run_GAIA/configs/gaia_ws.example.json` |
| FRAMES | `run_GAIA/configs/frames.example.json` |
| XBench DeepSearch-2510 | `run_GAIA/configs/deepsearch.example.json` |

```bash
scripts/run_eval.sh \
  --config run_GAIA/configs/gaia.example.json \
  --max-parallel-workers 4 \
  --sample-timeout-seconds 1800
```

Each run writes:

```text
results/task_<task_id>_result.json
results/summary.json
logs/
workspaces/
isolated_runs/
```

More details are available in [`docs/reproducibility.md`](docs/reproducibility.md).

<a id="citation"></a>

## Citation

```bibtex
@article{zhou2026toolself,
  title={ToolSelf: Unifying Task Execution and Self-Reconfiguration via Tool-Driven Emergent Adaptation},
  author={Zhou, Jingqi and Wang, Sheng and Deng, Dezhao and Lu, Junwen and Su, Junwei and Li, Qintong and Gao, Jiahui and Wu, Hao and Jiang, Jiyue and Kong, Lingpeng and Jin, Dunhong and Wu, Chuan},
  journal={arXiv preprint arXiv:2602.07883},
  year={2026}
}
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
