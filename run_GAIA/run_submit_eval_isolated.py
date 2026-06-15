"""
Run ToolSelf GAIA-style evaluation with per-sample process isolation.

The original runner uses a ThreadPoolExecutor. A hung network/tool call can keep
one worker alive indefinitely. This wrapper runs each sample through the normal
run_submit_eval.py entry point in a child process and kills the whole process
group on timeout, while preserving the same result-file layout.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
RUNNER = CURRENT_DIR / "run_submit_eval.py"


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


def load_dataset(path: str, max_samples: int | None = None) -> List[Dict[str, Any]]:
    dataset_path = Path(path).expanduser().resolve()
    if dataset_path.suffix.lower() == ".csv":
        data = load_deepsearch_csv(dataset_path)
    else:
        with dataset_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    return data[:max_samples] if max_samples is not None else data


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
            sample: Dict[str, Any] = {
                "index": index,
                "task_id": f"deepsearch_{row.get('id', index)}",
                "question": xor_decrypt(base64.b64decode(row["prompt"]), key).decode("utf-8"),
                "final_answer": xor_decrypt(base64.b64decode(row["answer"]), key).decode("utf-8"),
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


def completed_task_ids(results_dir: Path) -> set[str]:
    completed = set()
    for result_file in results_dir.glob("task_*_result.json"):
        task_id = result_file.stem.replace("task_", "").replace("_result", "")
        completed.add(task_id)
    return completed


def extract_result_from_child(child_output_dir: Path, task_id: str) -> Dict[str, Any] | None:
    result_file = child_output_dir / "results" / f"task_{task_id}_result.json"
    if not result_file.exists():
        return None
    return json.loads(result_file.read_text(encoding="utf-8"))


def copy_child_artifacts(child_output_dir: Path, output_dir: Path, task_id: str) -> None:
    child_result = child_output_dir / "results" / f"task_{task_id}_result.json"
    if child_result.exists():
        target_result = output_dir / "results" / child_result.name
        target_result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child_result, target_result)

    child_workspace = child_output_dir / "workspaces" / f"task_{task_id}"
    if child_workspace.exists():
        target_workspace = output_dir / "workspaces" / f"task_{task_id}"
        if target_workspace.exists():
            shutil.rmtree(target_workspace)
        target_workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(child_workspace, target_workspace)

    child_summary = child_output_dir / "results" / "summary.json"
    if child_summary.exists():
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child_summary, logs_dir / f"child_summary_{task_id}.json")


def timeout_result(sample: Dict[str, Any], index: int, timeout_seconds: int, elapsed: float) -> Dict[str, Any]:
    task_id = sample.get("task_id", f"task_{index}")
    return {
        "index": index,
        "task_id": task_id,
        "level": sample.get("level", "unknown"),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("final_answer", ""),
        "prediction": "",
        "raw_prediction": "",
        "finished": False,
        "total_stages": 0,
        "execution_time": elapsed,
        "error": f"timeout after {timeout_seconds}s",
        "workspace_dir": None,
        "evaluation": {
            "prediction": "",
            "gold": sample.get("final_answer", ""),
            "question": sample.get("question", ""),
            "final_verdict": False,
            "match_type": "timeout",
            "confidence": 0.0,
        },
        "retry_info": None,
    }


def failure_result(sample: Dict[str, Any], index: int, error: str, elapsed: float) -> Dict[str, Any]:
    task_id = sample.get("task_id", f"task_{index}")
    return {
        "index": index,
        "task_id": task_id,
        "level": sample.get("level", "unknown"),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("final_answer", ""),
        "prediction": "",
        "raw_prediction": "",
        "finished": False,
        "total_stages": 0,
        "execution_time": elapsed,
        "error": error,
        "workspace_dir": None,
        "evaluation": {
            "prediction": "",
            "gold": sample.get("final_answer", ""),
            "question": sample.get("question", ""),
            "final_verdict": False,
            "match_type": "runner_error",
            "confidence": 0.0,
        },
        "retry_info": None,
    }


def write_result(output_dir: Path, result: Dict[str, Any]) -> None:
    task_id = result["task_id"]
    result_file = output_dir / "results" / f"task_{task_id}_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def run_one(
    sample: Dict[str, Any],
    index: int,
    config_path: str,
    config: Dict[str, Any],
    output_dir: Path,
    timeout_seconds: int,
    quiet: bool,
) -> Dict[str, Any]:
    task_id = sample.get("task_id", f"task_{index}")
    scratch_root = output_dir / "isolated_runs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    single_dataset = scratch_root / f"{task_id}.json"
    child_output_dir = scratch_root / f"output_{task_id}"
    single_dataset.write_text(json.dumps([sample], indent=2, ensure_ascii=False), encoding="utf-8")
    if child_output_dir.exists():
        shutil.rmtree(child_output_dir)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path),
        "--dataset",
        str(single_dataset),
        "--output-dir",
        str(child_output_dir),
        "--max-parallel-workers",
        "1",
        "--max-iterations",
        str(config.get("max_iterations", 30)),
    ]
    if quiet:
        cmd.append("--quiet")

    env = os.environ.copy()
    start = time.time()
    log_file = output_dir / "logs" / f"task_{task_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(CURRENT_DIR.parent),
            env=env,
            start_new_session=True,
            text=True,
        )
        try:
            if timeout_seconds and timeout_seconds > 0:
                proc.wait(timeout=timeout_seconds)
            else:
                proc.wait()
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            elapsed = time.time() - start
            result = timeout_result(sample, index, timeout_seconds, elapsed)
            write_result(output_dir, result)
            return result

    elapsed = time.time() - start
    if proc.returncode != 0:
        result = failure_result(sample, index, f"child return code {proc.returncode}", elapsed)
        write_result(output_dir, result)
        return result

    child_result = extract_result_from_child(child_output_dir, task_id)
    if child_result is None:
        result = failure_result(sample, index, "child produced no result file", elapsed)
        write_result(output_dir, result)
        return result

    child_result["execution_time"] = child_result.get("execution_time") or elapsed
    copy_child_artifacts(child_output_dir, output_dir, task_id)
    return child_result


def save_summary(output_dir: Path, dataset_path: str, dataset_size: int, config: Dict[str, Any]) -> Dict[str, Any]:
    result_files = sorted((output_dir / "results").glob("task_*_result.json"))
    all_results = []
    for result_file in result_files:
        try:
            all_results.append(json.loads(result_file.read_text(encoding="utf-8")))
        except Exception as exc:
            all_results.append({"result_file": str(result_file), "error": f"Could not read: {exc}"})

    evaluated = [item for item in all_results if item.get("evaluation") is not None]
    correct = [item for item in evaluated if item.get("evaluation", {}).get("final_verdict") is True]
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
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_size": dataset_size,
        "result_files": len(result_files),
        "successful": len([item for item in all_results if item.get("error") is None]),
        "evaluated": len(evaluated),
        "correct": len(correct),
        "accuracy": len(correct) / len(evaluated) if evaluated else 0.0,
        "accuracy_over_dataset": len(correct) / dataset_size if dataset_size else 0.0,
        "by_level": by_level,
        "config": {key: value for key, value in config.items() if "key" not in key.lower()},
    }
    summary_file = output_dir / "results" / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ToolSelf with per-sample process timeout")
    parser.add_argument("--config", default="configs/gaia.example.json")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-parallel-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--sample-timeout-seconds", type=int, default=1800)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config_path = str((Path.cwd() / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
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

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    for subdir in ["results", "workspaces", "logs", "isolated_runs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config["dataset_path"], config.get("max_samples"))
    done = completed_task_ids(output_dir / "results") if config.get("resume_from_checkpoint", True) else set()
    indexed = [(idx, sample) for idx, sample in enumerate(dataset)]
    remaining = [(idx, sample) for idx, sample in indexed if sample.get("task_id", f"task_{idx}") not in done]

    print(f"Dataset: {config['dataset_path']}")
    print(f"Output: {output_dir}")
    print(f"Total: {len(dataset)} | completed: {len(done)} | remaining: {len(remaining)}")
    print(f"Workers: {config.get('max_parallel_workers', 1)} | sample timeout: {args.sample_timeout_seconds}s")

    with ThreadPoolExecutor(max_workers=int(config.get("max_parallel_workers", 1))) as executor:
        futures = {
            executor.submit(
                run_one,
                sample,
                idx,
                config_path,
                config,
                output_dir,
                args.sample_timeout_seconds,
                args.quiet or not config.get("verbose", True),
            ): (idx, sample)
            for idx, sample in remaining
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tasks"):
            idx, sample = futures[future]
            task_id = sample.get("task_id", f"task_{idx}")
            try:
                result = future.result()
            except Exception as exc:
                result = failure_result(sample, idx, f"isolated runner exception: {exc}", 0.0)
                write_result(output_dir, result)
            summary = save_summary(output_dir, config["dataset_path"], len(dataset), config)
            verdict = result.get("evaluation", {}).get("final_verdict")
            print(
                f"[{task_id}] verdict={verdict} evaluated={summary['evaluated']} "
                f"correct={summary['correct']} acc={summary['accuracy']:.2%}"
            )

    summary = save_summary(output_dir, config["dataset_path"], len(dataset), config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
