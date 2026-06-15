from __future__ import annotations

from typing import Any, Dict, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("terminate", allow_overwrite=True)
class Terminate(BaseTool):
    name = "terminate"
    description = (
        "Signal the completion of the main task and end the entire task execution. "
        "Use only when the current Execution Agent has completely finished the main task."
    )
    arguments: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "task_completion_status": {
                "type": "string",
                "enum": ["complete", "partial", "incomplete"],
                "description": "Main task Q completion status: complete/partial/incomplete"
            },
            "final_result": {
                "type": "string",
                "description": "Final result of the main task execution"
            },
            "execution_summary": {
                "type": "object",
                "properties": {
                    "detailed_execution": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Detailed step-by-step execution process including: steps taken, key decisions made, challenges encountered and how resolved, blocking factors, progress updates, and critical insights."
                    },
                    "tools_used": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of tools used during the task"
                    },
                    "key_achievements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Major achievements and milestones reached"
                    }
                },
                "required": ["detailed_execution", "tools_used"],
                "description": "Comprehensive summary of the task execution process"
            },
            "results_details": {
                "type": "object",
                "properties": {
                    "deliverables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete deliverables produced"
                    },
                    "evidence": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Evidence supporting the results"
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["High", "Medium", "Low"],
                        "description": "Confidence level in the results"
                    }
                },
                "description": "Detailed information about the results achieved"
            },
            "issues_encountered": {
                "type": "object",
                "properties": {
                    "blocking_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Issues that prevented full completion"
                    },
                    "workarounds_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workarounds employed to overcome issues"
                    },
                    "unresolved_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Issues that remain unresolved"
                    }
                },
                "description": "Issues encountered during execution (optional)"
            }
        },
        "required": ["task_completion_status", "final_result", "execution_summary"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not isinstance(params, dict):
            return (
                "[terminate] Invalid request: expected JSON object with required fields."
            )

        task_completion_status = params.get("task_completion_status", "").strip()
        final_result = params.get("final_result", "").strip()
        execution_summary = params.get("execution_summary", {})
        results_details = params.get("results_details", {})
        issues_encountered = params.get("issues_encountered", {})

        if task_completion_status not in ["complete", "partial", "incomplete"]:
            return "[terminate] Invalid task_completion_status. Must be one of: complete, partial, incomplete"
        
        if not final_result:
            return "[terminate] Missing required field: final_result"
            
        if not isinstance(execution_summary, dict) or not execution_summary.get("detailed_execution"):
            return "[terminate] Missing required field: execution_summary with detailed_execution"

        output_lines = [f"[terminate] Main task Q execution completed (T_term invoked)"]
        output_lines.append(f"**Completion Status:** {task_completion_status}")
        
        output_lines.append("\n## Final Result")
        output_lines.append(f"**Result:** {final_result}")
        
        output_lines.append("\n## Execution Summary")
        if execution_summary.get("detailed_execution"):
            output_lines.append("**Detailed Execution:**")
            for item in execution_summary["detailed_execution"]:
                output_lines.append(f"- {item}")
        
        if execution_summary.get("tools_used"):
            output_lines.append("**Tools Used:** " + ", ".join(execution_summary["tools_used"]))
            
        if execution_summary.get("key_achievements"):
            output_lines.append("**Key Achievements:**")
            for achievement in execution_summary["key_achievements"]:
                output_lines.append(f"- {achievement}")

        if results_details:
            output_lines.append("\n## Results Details")
            if results_details.get("deliverables"):
                output_lines.append("**Deliverables:**")
                for deliverable in results_details["deliverables"]:
                    output_lines.append(f"- {deliverable}")
            if results_details.get("evidence"):
                output_lines.append("**Evidence:** " + ", ".join(results_details["evidence"]))
            if results_details.get("confidence_level"):
                output_lines.append(f"**Confidence Level:** {results_details['confidence_level']}")

        if issues_encountered:
            output_lines.append("\n## Issues Encountered")
            if issues_encountered.get("blocking_issues"):
                output_lines.append("**Blocking Issues:**")
                for issue in issues_encountered["blocking_issues"]:
                    output_lines.append(f"- {issue}")
            if issues_encountered.get("workarounds_used"):
                output_lines.append("**Workarounds Used:**")
                for workaround in issues_encountered["workarounds_used"]:
                    output_lines.append(f"- {workaround}")
            if issues_encountered.get("unresolved_issues"):
                output_lines.append("**Unresolved Issues:**")
                for unresolved in issues_encountered["unresolved_issues"]:
                    output_lines.append(f"- {unresolved}")

        output_lines.append("\n**Status:** Execution-reconfiguration loop terminated")

        return "\n".join(output_lines)
