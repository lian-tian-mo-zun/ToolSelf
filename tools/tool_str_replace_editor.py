from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


MAX_VIEW_BYTES = int(os.getenv("STR_EDITOR_MAX_VIEW_BYTES", "200000"))


def _read_text(path: str, view_range: Optional[list[int]] = None) -> str:
    if not os.path.exists(path):
        return f"[str_replace_editor] File or directory not found: {path}"
    if os.path.isdir(path):
        try:
            entries = []
            for root, dirs, files in os.walk(path):
                depth = root.replace(path, "").count(os.sep)
                if depth > 2:
                    dirs[:] = []
                    continue
                for d in dirs:
                    if not d.startswith('.'):
                        entries.append(os.path.join(root, d))
                for f in files:
                    if not f.startswith('.'):
                        entries.append(os.path.join(root, f))
            return "\n".join(sorted(entries))
        except Exception as e:
            return f"[str_replace_editor] error listing directory: {e}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if view_range and len(view_range) == 2 and all(isinstance(x, int) for x in view_range):
            start, end = view_range
            if start == -1 and end == -1:
                pass
            else:
                lines = content.splitlines()
                if end == -1:
                    end = len(lines)
                start = max(1, start)
                end = min(len(lines), end)
                content = "\n".join(lines[start - 1 : end])
        if len(content.encode("utf-8")) > MAX_VIEW_BYTES:
            raw = content.encode("utf-8")[:MAX_VIEW_BYTES]
            try:
                content = raw.decode("utf-8")
            except Exception:
                content = raw.decode("utf-8", errors="ignore")
            content = content + "\n<response clipped>"
        return content
    except Exception as e:
        return f"[str_replace_editor] read error: {e}"


def _create_file(path: str, text: str) -> str:
    if os.path.exists(path):
        return f"[str_replace_editor] create failed: path already exists: {path}"
    parent = os.path.dirname(path) or "/"
    if not os.path.exists(parent):
        return f"[str_replace_editor] create failed: parent not found: {parent}"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return f"[str_replace_editor] created: {path}"
    except Exception as e:
        return f"[str_replace_editor] create error: {e}"


def _str_replace(path: str, old_str: str, new_str: Optional[str]) -> str:
    if not os.path.isfile(path):
        return f"[str_replace_editor] replace failed: not a file: {path}"
    if old_str is None or old_str == "":
        return "[str_replace_editor] replace failed: 'old_str' must be non-empty."
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if old_str not in content:
            return "[str_replace_editor] replace failed: 'old_str' not found."
        new_content = content.replace(old_str, new_str or "")
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return "[str_replace_editor] replace success."
    except Exception as e:
        return f"[str_replace_editor] replace error: {e}"


def _insert(path: str, new_str: str, insert_line: int) -> str:
    if not os.path.isfile(path):
        return f"[str_replace_editor] insert failed: not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        insert_at = max(1, int(insert_line))
        if insert_at > len(lines):
            lines.append(new_str)
            if not new_str.endswith("\n"):
                lines.append("\n")
        else:
            lines.insert(insert_at, new_str + ("\n" if not new_str.endswith("\n") else ""))
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return "[str_replace_editor] insert success."
    except Exception as e:
        return f"[str_replace_editor] insert error: {e}"


@register_tool("str_replace_editor", allow_overwrite=True)
class StrReplaceEditor(BaseTool):
    name = "str_replace_editor"
    description = (
        "View, create, and edit text files. Commands: `view`, `create`, `str_replace`, `insert`. "
        "Paths should be absolute. Large outputs may be clipped."
    )
    arguments: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert"],
                "description": "Operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "Absolute file or directory path.",
            },
            "file_text": {
                "type": "string",
                "description": "Required for 'create': new file content.",
            },
            "old_str": {
                "type": "string",
                "description": "Required for 'str_replace': the exact old text.",
            },
            "new_str": {
                "type": "string",
                "description": "New text for 'str_replace' (optional) or required for 'insert'.",
            },
            "insert_line": {
                "type": "integer",
                "description": "Required for 'insert': insert AFTER this 1-indexed line.",
            },
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional for 'view' on files: [start, end], 1-indexed, -1 for end.",
            },
        },
        "required": ["command", "path"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not isinstance(params, dict):
            return (
                "[str_replace_editor] Invalid request: expected JSON object with "
                "'command' and 'path'."
            )

        command = str(params.get("command", "")).strip()
        path = str(params.get("path", "")).strip()
        if not command or not path:
            return "[str_replace_editor] 'command' and 'path' are required."
        if not path.startswith("/"):
            return "[str_replace_editor] Use absolute paths starting with '/'."

        if command == "view":
            view_range = params.get("view_range")
            return _read_text(path, view_range)
        elif command == "create":
            text = params.get("file_text", "")
            return _create_file(path, text)
        elif command == "str_replace":
            old = params.get("old_str")
            new = params.get("new_str")
            return _str_replace(path, old, new)
        elif command == "insert":
            new = str(params.get("new_str", ""))
            line = int(params.get("insert_line", 1))
            if new == "":
                return "[str_replace_editor] 'new_str' required for insert."
            return _insert(path, new, line)

        return f"[str_replace_editor] Unknown command: {command}"


