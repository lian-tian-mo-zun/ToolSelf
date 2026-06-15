#!/usr/bin/env python
"""Print compact summaries for one or more ToolSelf evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    summary_path = path
    if path.is_dir():
        summary_path = path / "results" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Output dirs or summary.json files")
    args = parser.parse_args()

    print("| output | evaluated | correct | accuracy |")
    print("|---|---:|---:|---:|")
    for raw_path in args.paths:
        path = Path(raw_path).expanduser()
        summary = load_summary(path)
        name = path.name if path.is_dir() else path.parent.parent.name
        evaluated = summary.get("evaluated", 0)
        correct = summary.get("correct", 0)
        accuracy = summary.get("accuracy", 0.0)
        print(f"| {name} | {evaluated} | {correct} | {accuracy:.4f} |")


if __name__ == "__main__":
    main()
