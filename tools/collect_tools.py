from __future__ import annotations

import inspect
import json
import pkgutil
from importlib import import_module
import os
from typing import Dict, List


def _collect_tool_schemas() -> Dict[str, dict]:
    try:
        from qwen_agent.tools.base import BaseTool
    except Exception:
        BaseTool = object

    schemas: Dict[str, dict] = {}
    package_name = __package__ or "tools"
    package_dir = os.path.dirname(__file__)

    for _finder, module_name, _ispkg in pkgutil.iter_modules([package_dir]):
        if module_name in {"collect_tools", "__init__"} or module_name.startswith("_"):
            continue

        try:
            module = import_module(f"{package_name}.{module_name}")
        except Exception:
            continue

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "__module__", "") != module.__name__:
                continue
            try:
                if not issubclass(obj, BaseTool):
                    continue
            except Exception:
                continue

            tool_name = getattr(obj, "name", None)
            description = getattr(obj, "description", None)
            arguments = getattr(obj, "arguments", None)
            if tool_name and description and arguments:
                schemas[str(tool_name)] = {
                    "name": tool_name,
                    "description": description,
                    "arguments": arguments,
                }

    return schemas


def build_tool_prompt(function_list: List[str]) -> str:
    all_schemas = _collect_tool_schemas()
    selected = []
    for name in function_list:
        schema = all_schemas.get(name)
        if schema is not None:
            selected.append(schema)

    serialized = ",\n".join(
        json.dumps(schema, ensure_ascii=False, indent=2) for schema in selected
    )
    return f"<tools>\n{serialized}\n</tools>"



def build_tool_catalog_text(include_parameters: bool = True, function_list: List[str] | None = None) -> str:
    all_schemas = _collect_tool_schemas()
    lines: List[str] = []
    lines.append("Tool Catalog (for reference only; you cannot call these directly):")
    if function_list:
        ordered_names = [n for n in function_list if n in all_schemas]
    else:
        ordered_names = sorted(all_schemas.keys())
    for name in ordered_names:
        schema = all_schemas[name]
        desc = schema.get("description", "")
        lines.append(f"- {name}: {desc}")
        if include_parameters:
            try:
                params_str = json.dumps(schema.get("arguments", {}), ensure_ascii=False, indent=2)
            except Exception:
                params_str = "{}"
            lines.append("  arguments:\n" + "\n".join("  " + ln for ln in params_str.splitlines()))
    return "\n".join(lines)
