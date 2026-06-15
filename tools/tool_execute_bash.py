from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, Union

from qwen_agent.tools.base import BaseTool, register_tool


MAX_OUTPUT_CHARS = int(os.getenv("BASH_MAX_OUTPUT", "40000"))
DEFAULT_TIMEOUT_SEC = int(os.getenv("BASH_TIMEOUT", "120"))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: max(0, limit - 1000)]
    tail = text[-1000:]
    return f"{head}\n<response clipped>\n...\n{tail}"


@register_tool("execute_bash", allow_overwrite=True)
class ExecuteBash(BaseTool):
    name = "execute_bash"
    description = (
        "Execute a bash command in a non-interactive subprocess. "
        "Use absolute paths when possible. Output is truncated if too long."
    )
    arguments: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute (single command; use '&&' to chain).",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT_SEC}).",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (absolute path recommended).",
            },
        },
        "required": ["command"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not isinstance(params, dict):
            return "[execute_bash] Invalid request: expected JSON object with 'command' field."
        command: str = str(params.get("command", "")).strip()
        if not command:
            return "[execute_bash] 'command' must be a non-empty string."

        timeout_sec: int = int(params.get("timeout", DEFAULT_TIMEOUT_SEC) or DEFAULT_TIMEOUT_SEC)
        cwd_param: str | None = params.get("cwd")
        working_directory = str(cwd_param) if cwd_param else None

        try:
            completed = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                cwd=working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            stdout_text = _truncate(stdout_text, MAX_OUTPUT_CHARS)
            stderr_text = _truncate(stderr_text, MAX_OUTPUT_CHARS)
            return (
                f"[execute_bash] exit_code={completed.returncode}\n"
                f"[stdout]\n{stdout_text}\n\n[stderr]\n{stderr_text}"
            ).strip()
        except subprocess.TimeoutExpired as exc:
            partial_out = (exc.stdout or "") + ("\n" + (exc.stderr or ""))
            partial_out = _truncate(partial_out, MAX_OUTPUT_CHARS)
            return (
                f"[execute_bash] timeout after {timeout_sec}s\n"
                f"[partial_output]\n{partial_out}"
            ).strip()
        except Exception as e:
            return f"[execute_bash] error: {type(e).__name__}: {e}"


