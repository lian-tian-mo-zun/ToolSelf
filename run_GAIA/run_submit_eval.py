"""
Run ToolSelf on GAIA-style datasets and evaluate predictions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
TOOLSELF_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(TOOLSELF_DIR))


print("\n" + "=" * 80)
print("Initializing ToolSelf configuration...")
print("=" * 80)
try:
    from config import build_llm_cfg

    build_llm_cfg()

    jina_key = os.getenv("JINA_KEY") or os.getenv("JINA_API_KEYS")
    print(f"JINA keys configured: {len([k for k in (jina_key or '').split(',') if k.strip()])}")
    print(f"Main model: {os.getenv('MAIN_LLM_MODEL', '')}")
    print(f"Main API base: {os.getenv('MAIN_LLM_API_BASE_URL', '')}")
    print(f"Searx host: {os.getenv('SEARX_HOST', '')}")
    print("=" * 80 + "\n")
except Exception as exc:
    print(f"Warning: failed to initialize config: {exc}")
    traceback.print_exc()


from evaluator import GAIAEvaluator
from toolself_gaia import ToolSelfGAIA


def xor_decrypt(data: bytes, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    key_length = len(key_bytes)
    return bytes(data[i] ^ key_bytes[i % key_length] for i in range(len(data)))


def load_deepsearch_csv(dataset_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            key = row["canary"]
            question = xor_decrypt(base64.b64decode(row["prompt"]), key).decode("utf-8")
            answer = xor_decrypt(base64.b64decode(row["answer"]), key).decode("utf-8")
            sample: Dict[str, Any] = {
                "index": index,
                "task_id": f"deepsearch_{row.get('id', index)}",
                "question": question,
                "final_answer": answer,
                "level": "DeepSearch",
                "source": "DeepSearch-2510",
                "xbench_id": row.get("id", ""),
            }
            if row.get("reference_steps"):
                sample["reference_steps"] = xor_decrypt(
                    base64.b64decode(row["reference_steps"]), key
                ).decode("utf-8")
            samples.append(sample)
    return samples


def extract_final_prediction(raw_prediction: Any) -> str:
    """Extract the concise final answer from the terminate tool report when present."""
    text = str(raw_prediction or "").strip()
    if not text:
        return ""

    result_match = None
    patterns = [
        r"## Final Result\s*\n\s*\*\*Result:\*\*\s*(.+?)(?:\n\s*##|\n\s*\*\*Status:\*\*|\Z)",
        r"\*\*Result:\*\*\s*(.+?)(?:\n\s*##|\n\s*\*\*Status:\*\*|\Z)",
    ]
    for pattern in patterns:
        result_match = __import__("re").search(pattern, text, flags=__import__("re").DOTALL)
        if result_match:
            break
    if result_match:
        return result_match.group(1).strip()

    answer_match = __import__("re").search(r"<answer>(.*?)</answer>", text, flags=__import__("re").DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    return text


class ToolSelfSubmitEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(self.config["output_dir"]).resolve()
        self.results_dir = self.output_dir / "results"
        self.workspaces_dir = self.output_dir / "workspaces"
        self.logs_dir = self.output_dir / "logs"
        self.checkpoint_file = self.results_dir / "checkpoint.json"
        self.summary_file = self.results_dir / "summary.json"

        for path in [self.results_dir, self.workspaces_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)

        judge_api_key = (
            self.config.get("judge_api_key")
            or os.getenv("JUDGE_API_KEY")
            or os.getenv("MAIN_LLM_API_KEY")
        )
        judge_base_url = (
            self.config.get("judge_base_url")
            or os.getenv("JUDGE_BASE_URL")
            or os.getenv("MAIN_LLM_API_BASE_URL")
        )
        judge_model = (
            self.config.get("judge_model")
            or os.getenv("JUDGE_MODEL")
            or "Qwen/Qwen2.5-72B-Instruct"
        )

        if not judge_api_key or judge_api_key == "YOUR_JUDGE_API_KEY":
            raise ValueError("Judge API key is not configured. Set JUDGE_API_KEY or judge_api_key.")

        self.evaluator = GAIAEvaluator(
            api_key=judge_api_key,
            base_url=judge_base_url,
            judge_model=judge_model,
            verbose=self.config.get("verbose", True),
        )

        self.results: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
        self.checkpoint_lock = threading.Lock()
        self.start_time: Optional[float] = None

    def _load_dataset(self) -> List[Dict[str, Any]]:
        dataset_path = Path(self.config["dataset_path"]).expanduser().resolve()
        print(f"Loading dataset from: {dataset_path}")
        if dataset_path.suffix.lower() == ".csv":
            data = load_deepsearch_csv(dataset_path)
        else:
            with dataset_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        max_samples = self.config.get("max_samples")
        if max_samples is not None:
            data = data[: int(max_samples)]

        print(f"Loaded {len(data)} samples")
        return data

    def _completed_task_ids(self) -> set[str]:
        completed: set[str] = set()
        for result_file in self.results_dir.glob("task_*_result.json"):
            task_id = result_file.stem.replace("task_", "").replace("_result", "")
            completed.add(task_id)
        print(f"Found {len(completed)} completed result files")
        return completed

    def _save_checkpoint(self, completed_count: int, total_count: int) -> None:
        with self.checkpoint_lock:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "completed": completed_count,
                "total": total_count,
                "output_dir": str(self.output_dir),
            }
            with self.checkpoint_file.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)

    def _save_summary(self, dataset_size: int) -> Dict[str, Any]:
        result_files = sorted(self.results_dir.glob("task_*_result.json"))
        all_results: List[Dict[str, Any]] = []
        for result_file in result_files:
            try:
                all_results.append(json.loads(result_file.read_text(encoding="utf-8")))
            except Exception as exc:
                all_results.append({"result_file": str(result_file), "error": f"Could not read: {exc}"})

        successful = [item for item in all_results if item.get("error") is None]
        evaluated = [item for item in successful if item.get("evaluation") is not None]
        correct = [
            item
            for item in evaluated
            if item.get("evaluation", {}).get("final_verdict") is True
        ]
        by_level: Dict[str, Dict[str, Any]] = {}
        for item in evaluated:
            level = str(item.get("level", "unknown"))
            stats = by_level.setdefault(level, {"total": 0, "correct": 0, "accuracy": 0.0})
            stats["total"] += 1
            if item.get("evaluation", {}).get("final_verdict") is True:
                stats["correct"] += 1
        for stats in by_level.values():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": str(Path(self.config["dataset_path"]).expanduser().resolve()),
            "output_dir": str(self.output_dir),
            "dataset_size": dataset_size,
            "result_files": len(result_files),
            "successful": len(successful),
            "evaluated": len(evaluated),
            "correct": len(correct),
            "accuracy": len(correct) / len(evaluated) if evaluated else 0.0,
            "accuracy_over_dataset": len(correct) / dataset_size if dataset_size else 0.0,
            "by_level": by_level,
            "judge_statistics": self.evaluator.get_statistics(),
            "config": {
                key: value
                for key, value in self.config.items()
                if "key" not in key.lower()
            },
        }
        self.summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _run_single_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        task_id = sample.get("task_id", f"task_{index}")
        question = sample["question"]
        gold_answer = sample.get("final_answer", "")
        level = sample.get("level", "unknown")

        print(f"\n{'=' * 80}")
        print(f"Task ID: {task_id} | Index: {index} | Level: {level}")
        print(f"Question: {question[:160]}{'...' if len(question) > 160 else ''}")
        print(f"Gold Answer: {gold_answer}")
        print(f"{'=' * 80}")

        result: Dict[str, Any] = {
            "index": index,
            "task_id": task_id,
            "level": level,
            "question": question,
            "gold_answer": gold_answer,
            "prediction": None,
            "finished": False,
            "total_stages": 0,
            "execution_time": 0.0,
            "error": None,
            "workspace_dir": None,
            "evaluation": None,
            "retry_info": None,
        }

        max_retries = int(self.config.get("max_retries", 1))
        retry_count = 0
        retry_errors: List[Dict[str, Any]] = []
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                workspace_dir = self.workspaces_dir / f"task_{task_id}"
                if attempt > 0 and workspace_dir.exists():
                    import shutil

                    print(f"[Retry {attempt}/{max_retries}] Cleaning old workspace")
                    shutil.rmtree(workspace_dir)
                workspace_dir.mkdir(parents=True, exist_ok=True)

                agent = ToolSelfGAIA(
                    question=question,
                    workspace_dir=str(workspace_dir),
                    function_list=self.config.get("function_list"),
                    verbose=self.config.get("verbose", True),
                    api_key=self.config.get("agent_api_key"),
                    base_url=self.config.get("agent_base_url"),
                    model=self.config.get("agent_model"),
                )

                agent_result = agent.run(max_iterations=int(self.config.get("max_iterations", 30)))
                execution_time = time.time() - start_time
                final_result = agent_result.get("final_result", {})
                raw_prediction = final_result.get("prediction", "")
                prediction = extract_final_prediction(raw_prediction)
                execution_status = final_result.get("execution_status", "")
                error_message = final_result.get("error_message", "")

                if execution_status == "error" and error_message and attempt < max_retries:
                    retry_errors.append(
                        {
                            "attempt": attempt + 1,
                            "error_message": error_message,
                            "stages": agent_result.get("total_stages", 0),
                        }
                    )
                    retry_count += 1
                    print(f"[Agent Internal Error]: {error_message}")
                    continue

                result.update(
                    {
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "finished": agent_result.get("finished", False),
                        "total_stages": agent_result.get("total_stages", 0),
                        "execution_time": execution_time,
                        "workspace_dir": str(workspace_dir),
                        "retry_info": {
                            "retry_count": retry_count,
                            "retry_errors": retry_errors,
                        }
                        if retry_errors
                        else None,
                    }
                )

                print(f"[Agent Finished]: {result['finished']}")
                print(f"[Stages]: {result['total_stages']}/{self.config.get('max_iterations', 30)}")
                print(f"[Execution Time]: {execution_time:.2f}s")
                print(f"[Prediction]: {str(prediction)[:220]}{'...' if len(str(prediction)) > 220 else ''}")

                print("[Starting Evaluation]")
                evaluation = self.evaluator.evaluate_single(
                    pred=str(prediction),
                    gold=str(gold_answer),
                    question=str(question),
                )
                result["evaluation"] = evaluation
                verdict = "CORRECT" if evaluation.get("final_verdict") else "INCORRECT"
                print(f"[Final Verdict]: {verdict} ({evaluation.get('match_type')})")
                break

            except Exception as exc:
                execution_time = time.time() - start_time
                error_msg = f"{type(exc).__name__}: {exc}"
                if attempt >= max_retries:
                    result["error"] = error_msg
                    result["execution_time"] = execution_time
                    result["retry_info"] = {
                        "retry_count": retry_count,
                        "retry_errors": retry_errors,
                    } if retry_errors else None
                    print(f"[ERROR]: {error_msg}")
                    if self.config.get("verbose", True):
                        traceback.print_exc()
                    break

                retry_errors.append(
                    {
                        "attempt": attempt + 1,
                        "error_message": error_msg,
                        "stages": 0,
                    }
                )
                retry_count += 1
                print(f"[EXCEPTION]: {error_msg}")

        result_file = self.results_dir / f"task_{task_id}_result.json"
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def run(self) -> Dict[str, Any]:
        self.start_time = time.time()
        dataset = self._load_dataset()
        completed = self._completed_task_ids() if self.config.get("resume_from_checkpoint", True) else set()
        remaining = [
            (index, sample)
            for index, sample in enumerate(dataset)
            if sample.get("task_id", f"task_{index}") not in completed
        ]

        print(f"\n{'=' * 80}")
        print("Starting ToolSelf GAIA-style evaluation")
        print(f"Total samples: {len(dataset)}")
        print(f"Completed: {len(dataset) - len(remaining)}")
        print(f"Remaining: {len(remaining)}")
        print(f"Max parallel workers: {self.config.get('max_parallel_workers', 1)}")
        print(f"{'=' * 80}\n")

        if remaining:
            with ThreadPoolExecutor(max_workers=int(self.config.get("max_parallel_workers", 1))) as executor:
                future_to_task = {
                    executor.submit(self._run_single_sample, sample, index): (
                        sample.get("task_id", f"task_{index}"),
                        index,
                    )
                    for index, sample in remaining
                }
                completed_count = len(dataset) - len(remaining)
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Tasks"):
                    task_id, index = future_to_task[future]
                    try:
                        result = future.result()
                        with self.results_lock:
                            self.results.append(result)
                    except Exception as exc:
                        print(f"[Error] Task {task_id} (index {index}) failed outside runner: {exc}")
                    completed_count += 1
                    self._save_checkpoint(completed_count, len(dataset))
                    summary = self._save_summary(len(dataset))
                    print(
                        f"[Progress] {completed_count}/{len(dataset)} | "
                        f"evaluated={summary['evaluated']} correct={summary['correct']} "
                        f"accuracy={summary['accuracy']:.2%}"
                    )

        summary = self._save_summary(len(dataset))
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"Output: {self.output_dir}")
        print(f"Evaluated: {summary['evaluated']}/{summary['dataset_size']}")
        print(f"Correct: {summary['correct']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Accuracy over dataset: {summary['accuracy_over_dataset']:.2%}")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"{'=' * 80}")
        return summary


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.exists() else CURRENT_DIR / path
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    for key in ["dataset_path", "output_dir"]:
        if key in config and isinstance(config[key], str):
            config[key] = os.path.expandvars(config[key])
            if "$" in config[key]:
                raise ValueError(f"Unexpanded environment variable in {key}: {config[key]}")

    if not Path(config["dataset_path"]).is_absolute():
        config["dataset_path"] = str((CURRENT_DIR / config["dataset_path"]).resolve())
    if not Path(config["output_dir"]).is_absolute():
        config["output_dir"] = str((CURRENT_DIR / config["output_dir"]).resolve())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ToolSelf on GAIA-style datasets")
    parser.add_argument("--config", default="configs/gaia.example.json", help="Path to JSON config")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--max-parallel-workers", type=int, help="Override worker count")
    parser.add_argument("--max-samples", type=int, help="Run only the first N samples")
    parser.add_argument("--max-iterations", type=int, help="Override max ToolSelf stages per sample")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose agent logs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset:
        config["dataset_path"] = args.dataset
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_parallel_workers is not None:
        config["max_parallel_workers"] = args.max_parallel_workers
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.quiet:
        config["verbose"] = False

    ToolSelfSubmitEvaluator(config).run()


if __name__ == "__main__":
    main()
