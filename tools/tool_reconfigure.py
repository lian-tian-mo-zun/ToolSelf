from __future__ import annotations

import os
import sys
from typing import Any, Dict, Union

from qwen_agent.tools.base import BaseTool, register_tool

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def _format_context_management_modes() -> str:
    return """[
  {
    "name": "basic",
    "description": "General mode for ordinary reasoning, web search, and tool-use stages. When the visible context becomes too long, historical interaction content is cleared while recent iterations are retained."
  },
  {
    "name": "truncation",
    "description": "Mode for stages likely to involve very long tool calls or tool responses, such as software engineering, PDF reading, document analysis, or file editing. Long non-current tool messages and older observations may be truncated or replaced with placeholders."
  }
]"""


def build_configuration_generation_spec(
    main_task: str,
    available_tools: str,
    execution_history: str,
    update_requirement: str,
) -> str:
    return f"""## Role and Core Objective
You are a configuration generator for a ReAct-based Execution Agent. Your core
objective is to generate the next-stage runtime configuration for the Execution Agent
executing a complex main task. You must:
1. Mentally plan the overall task execution to understand the big picture
2. Identify the immediate next step based on current progress
3. Generate appropriate configuration (sub_goal, strategy, toolbox, knowledge,
   context_management_mode)

## Input Information
You will receive the following parts:
1. **Main Task `<main_task>`**: The final goal the Execution Agent needs to accomplish.
   <main_task>
   {main_task}
   </main_task>
2. **Available Tools `<available_tools>`**: Complete list of tools available
   to the Execution Agent during the entire task lifecycle.
   <available_tools>
   {available_tools}
   </available_tools>

3. **Context-Management Modes `<context_management_modes>`**: Complete list of
   context-management modes available for the next execution stage. Select
   exactly one mode from this list as `context_management_mode`.
   <context_management_modes>
   {_format_context_management_modes()}
   </context_management_modes>

4. **Execution History `<execution_history>`**: Summarized record of all
   previous steps. Can be empty ("NONE").
   <execution_history>
   {execution_history}
   </execution_history>

5. **Update Requirement `<update_requirement>`**: Update suggestions from
   the current Execution Agent. Can be empty ("NONE").
   <update_requirement>
   {update_requirement}
   </update_requirement>

## Configuration Generation Process
### Step 1: Determine Next Sub-Goal
Carefully consider the proposed sub-goal in `<update_requirement>` and adopt
it whenever possible. Only re-plan a new sub-goal if the proposal is entirely
irrelevant to the task.
### Step 2: Generate Other Configuration Components
Based on the determined sub-goal and other requirements specified in
`<update_requirement>`, generate the following elements:
**2.1 Execution Strategy**
For **execution_strategy**, design a heuristic algorithm specifically for
the next sub_goal:
1. **Define Persona**: Assign an expert persona (e.g., "Senior Software
   Engineer", "Data Analyst").
2. **Create a Step-by-Step Plan**: Simulate the thinking process to
   accomplish the next sub_goal.
3. **Identify Decision Points**: Mark key decision points in the plan.
4. **Link to Tools**: Specify which concrete tool should be used in each step.
**2.2 Toolbox Selection**
Select tools for the **toolbox**:
- **CRITICAL**: The toolbox must include at least two tools specifically
  needed for the NEXT sub_goal ONLY.
- **VERY IMPORTANT**: Focus EXCLUSIVELY on tools required to complete the
  next sub_goal.
- **IMPORTANT**: The toolbox must be a **strict subset** of
  `<available_tools>`.
**2.3 Inter-Agent Knowledge**
Determine the `inter_agent_knowledge` field based on `<execution_history>`:
1. **Initial Step**: If `<execution_history>` is `NONE` or empty,
   `inter_agent_knowledge` must be an empty string `""`.
2. **Use ALL**: If history is concise and relevant, use the string `"ALL"`.
3. **Summarize**: If history is long or contains exploratory steps, extract
   a concise summary.
**2.4 Context-Management Mode**
Select the `context_management_mode` field from `<context_management_modes>`
based on the next sub_goal, expected tool outputs, and context requirements:
1. Use `"basic"` for ordinary reasoning, web search, and tool-use stages where
   tool outputs are expected to be moderate.
2. Use `"truncation"` when the next stage is likely to involve very long tool
   calls or tool responses, such as software engineering, PDF/document analysis,
   repository editing, or other context-heavy interactions.
## Output Format
Your final output **must** be a strict JSON object:
{{
  "next_sub_goal": "Detailed description of the next step to execute",
  "execution_strategy": "As a [expert persona], I will follow these steps to accomplish the NEXT sub_goal: 1. [first step and tool] 2. [second step and tool] 3. [key decision points]...",
  "toolbox": ["ToolName1", "ToolName2", "ToolName3"],
  "inter_agent_knowledge": "One of three forms: summary, 'ALL', or ''",
  "context_management_mode": "One of: basic, truncation"
}}"""


@register_tool("reconfigure", allow_overwrite=True)
class Reconfigure(BaseTool):
    name = "reconfigure"
    description = (
        "Return a configuration-generation specification for the next execution stage. "
        "Use when the current sub-task is completed but the main task is not, when current sub-task execution fails, "
        "when the current toolbox is insufficient, when knowledge lacks necessary information or contains irrelevant information, "
        "when the execution strategy no longer fits the task requirements, or when the context-management mode should change for subsequent execution."
    )
    arguments: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "execution_summary": {
                "type": "string",
                "description": "Step-by-step detailed summary of current Execution Agent's task execution process"
            },
            "update_reason": {
                "type": "string",
                "description": "Why this update is needed"
            },
            "new_sub_goal": {
                "type": "string",
                "description": "The new sub_goal that the Execution Agent needs to execute. If the current task is complete, set to 'Task completed, use terminate tool next'"
            },
            "additional_details": {
                "type": "object",
                "properties": {
                    "toolbox_requirements": {
                        "type": "string",
                        "description": "Requirements for the new toolbox based on current limitations"
                    },
                    "knowledge_requirements": {
                        "type": "string",
                        "description": "Requirements for the new knowledge based on missing information"
                    },
                    "execution_strategy_requirements": {
                        "type": "string",
                        "description": "Requirements for the new execution strategy based on current limitations"
                    },
                    "context_management_requirements": {
                        "type": "string",
                        "description": "Requirements for the next context-management mode based on expected tool outputs, history length, or context-heavy interactions"
                    }
                },
                "description": "Required adjustments for generating the next configuration"
            }
        },
        "required": ["execution_summary", "update_reason", "new_sub_goal"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not isinstance(params, dict):
            return "[reconfigure] Invalid request: expected JSON object with required fields."

        execution_summary = params.get("execution_summary", "").strip()
        update_reason = params.get("update_reason", "").strip()
        new_sub_goal = params.get("new_sub_goal", "").strip()
        additional_details = params.get("additional_details", {})

        if not execution_summary:
            return "[reconfigure] Missing required field: execution_summary"

        if not update_reason:
            return "[reconfigure] Missing required field: update_reason"

        if not new_sub_goal:
            return "[reconfigure] Missing required field: new_sub_goal"

        if not isinstance(additional_details, dict):
            additional_details = {}

        adjustment_parts = []
        for key, label in [
            ("toolbox_requirements", "Toolbox Requirements"),
            ("knowledge_requirements", "Knowledge Requirements"),
            ("execution_strategy_requirements", "Execution Strategy Requirements"),
            ("context_management_requirements", "Context-Management Requirements"),
        ]:
            value = str(additional_details.get(key, "")).strip()
            if value:
                adjustment_parts.append(f"{label}: {value}")

        update_requirement_parts = [
            f"Proposed Sub_goal (q_{{i+1}}^prop): {new_sub_goal}",
            f"Update Reason (rho_i): {update_reason}",
        ]
        update_requirement_parts.extend(adjustment_parts)
        update_requirement = "\n".join(update_requirement_parts)

        agent_runtime_context = kwargs.get("agent_runtime_context") or {}
        main_task = agent_runtime_context.get("main_task") or new_sub_goal
        function_list = agent_runtime_context.get("function_list") or []
        previous_history = agent_runtime_context.get("execution_history") or "NONE"
        if previous_history == "NONE":
            execution_history = execution_summary
        else:
            execution_history = previous_history + "\n\nCurrent Stage Summary:\n" + execution_summary

        from tools.collect_tools import build_tool_catalog_text

        execution_tools = [name for name in function_list if name not in {"reconfigure", "terminate"}]
        available_tools = build_tool_catalog_text(include_parameters=True, function_list=execution_tools)
        spec = build_configuration_generation_spec(
            main_task=main_task,
            available_tools=available_tools,
            execution_history=execution_history,
            update_requirement=update_requirement,
        )

        output_lines = ["[reconfigure] Configuration-generation specification returned"]
        output_lines.append("")
        output_lines.append("## Execution Summary (H_i)")
        output_lines.append(execution_summary)
        output_lines.append("")
        output_lines.append("## Reconfiguration Request (r_i)")
        output_lines.append(update_requirement)
        output_lines.append("")
        output_lines.append("## Configuration-Generation Specification")
        output_lines.append(spec)
        output_lines.append("")
        output_lines.append("Return only the strict JSON object required by the specification above. Do not call any more tools.")

        return "\n".join(output_lines)
