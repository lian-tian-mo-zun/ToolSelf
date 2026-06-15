import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from execution_agent.run_single_ReAct_agent import run_single_ReAct_agent
from tools.collect_tools import build_tool_catalog_text
from tools.tool_reconfigure import build_configuration_generation_spec


class ToolSelfGAIA:
    def __init__(
        self,
        question: str,
        workspace_dir: str = None,
        function_list: List[str] = None,
        verbose: bool = True,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.question = question
        self.verbose = verbose
        self.iteration_count = 0
        self.finished = False
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.function_list = function_list or [
            "visit", "code_interpreter", "file_analyzer", "execute_bash",
            "str_replace_editor", "search", "terminate", "reconfigure"
        ]
        if api_key:
            os.environ["MAIN_LLM_API_KEY"] = api_key
        if base_url:
            os.environ["MAIN_LLM_API_BASE_URL"] = base_url
        if model:
            os.environ["MAIN_LLM_MODEL"] = model

        if workspace_dir is None:
            question_short = self._generate_question_short(question)
            self.workspace_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "workspace",
                question_short
            )
        else:
            self.workspace_dir = workspace_dir

        os.makedirs(self.workspace_dir, exist_ok=True)
        self.context_file = os.path.join(self.workspace_dir, "context.md")
        self.iterations_dir = os.path.join(self.workspace_dir, "iterations")
        os.makedirs(self.iterations_dir, exist_ok=True)
        self._initialize_context_file()
        self.execution_history = "NONE"
        self.current_config: Optional[Dict[str, Any]] = None

    def _generate_question_short(self, question: str) -> str:
        clean_question = re.sub(r'[^\w\s]', '', question)
        short_name = clean_question[:50].strip().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{short_name}_{timestamp}"

    def _initialize_context_file(self):
        with open(self.context_file, "w", encoding="utf-8") as f:
            f.write("# ToolSelf Task Execution Context\n\n")
            f.write(f"**Main Task (Q):** {self.question}\n\n")
            f.write(f"**Started:** {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")

    def _update_context_file(self, iteration: int, sub_goal: str, summary: str):
        new_entry = f"## Stage {iteration}\n"
        new_entry += f"**Sub-goal (q_{iteration}):** {sub_goal}\n\n"
        new_entry += f"**Execution Summary (H_{iteration}):** {summary}\n\n"
        new_entry += "---\n\n"
        with open(self.context_file, "a", encoding="utf-8") as f:
            f.write(new_entry)

    def _read_context_content(self) -> str:
        try:
            with open(self.context_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    @staticmethod
    def _extract_json_object(text: str):
        if not isinstance(text, str):
            return None
        cleaned = text.strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        if "```" in cleaned:
            parts = cleaned.split("```")
            for i in range(len(parts) - 1):
                block = parts[i + 1]
                candidate = block.lstrip()[4:].strip() if block.lstrip().startswith("json") else block.strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        starts = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
        ends = [idx for idx, ch in enumerate(cleaned) if ch == "}"]
        for s in starts:
            for e in reversed(ends):
                if e < s:
                    continue
                try:
                    return json.loads(cleaned[s:e + 1])
                except Exception:
                    continue
        return None

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            config = {}
        normalized = {
            "next_sub_goal": str(config.get("next_sub_goal", "")).strip(),
            "execution_strategy": str(config.get("execution_strategy", "")).strip(),
            "toolbox": config.get("toolbox", []),
            "inter_agent_knowledge": config.get("inter_agent_knowledge", ""),
            "context_management_mode": str(config.get("context_management_mode", "basic")).strip(),
        }
        if not isinstance(normalized["toolbox"], list):
            normalized["toolbox"] = [str(normalized["toolbox"])]
        normalized["toolbox"] = [
            str(name).strip()
            for name in normalized["toolbox"]
            if str(name).strip() and str(name).strip() not in {"reconfigure", "terminate"}
        ]
        if "visit" in normalized["toolbox"] and "search" not in normalized["toolbox"]:
            normalized["toolbox"].append("search")
        if "search" in normalized["toolbox"] and "visit" not in normalized["toolbox"]:
            normalized["toolbox"].append("visit")
        if normalized["context_management_mode"] not in {"basic", "truncation"}:
            normalized["context_management_mode"] = "basic"
        return normalized

    def _build_initial_config_prompt(self) -> str:
        execution_tools = [name for name in self.function_list if name not in {"reconfigure", "terminate"}]
        available_tools = build_tool_catalog_text(include_parameters=True, function_list=execution_tools)
        return build_configuration_generation_spec(
            main_task=self.question,
            available_tools=available_tools,
            execution_history="NONE",
            update_requirement="NONE",
        )

    def _build_system_prompt(self, config: Dict[str, Any]) -> str:
        main_task = self.question
        next_sub_goal = config.get("next_sub_goal", "")
        if not next_sub_goal:
            next_sub_goal = "Task completed. TERMINATE in this round and output results."
        execution_strategy = config.get("execution_strategy", "")
        toolbox_str = "\n".join(config.get("toolbox", []))
        inter_agent_knowledge = config.get("inter_agent_knowledge", "")
        if inter_agent_knowledge == "ALL":
            inter_agent_knowledge = self._read_context_content()
        context_management_mode = config.get("context_management_mode", "basic")

        return f"""## Role and Core Objective
You are a professional Execution Agent. Your core objective is to
efficiently complete the specified task using the given main task, sub-goal,
and available toolset.

## Input Information
You will receive the following information, please analyze carefully:

1. **Main Task `<main_task>`**: This is the ultimate goal to be completed.
   <main_task>
   {main_task}
   </main_task>

2. **Current Sub_goal `<sub_goal>`**: This is the specific task you need to
   focus on completing now.
   <sub_goal>
   {next_sub_goal}
   </sub_goal>

3. **Execution Strategy `<strategy>`**: The execution methodology for the
   current sub_goal.
   <strategy>
   {execution_strategy}
   </strategy>

4. **Toolbox `<toolbox>`**: This is the list of tools available for the
   current sub_goal.
   <toolbox>
   {toolbox_str}
   </toolbox>

5. **Knowledge Information `<knowledge>`**: This is the background information
   and constraints related to the task.
   <knowledge>
   {inter_agent_knowledge}
   </knowledge>

6. **Context-Management Mode `<context_management_mode>`**: This is the
   current runtime mode that controls how the agent maintains, compresses,
   truncates, and organizes the visible context.
   <context_management_mode>
   {context_management_mode}
   </context_management_mode>

## Tool Usage Guidelines
- **Precision**: Select the most appropriate tools based on the current
  sub-goal
- **Efficiency**: Prioritize tools that can directly solve the problem
- **Completeness**: Ensure the task execution process is complete and
  logically clear
- **Context Awareness**: Respect the current context-management mode when
  reading history and processing long tool outputs.

## Agent Management Tools
**Reconfiguration Tool** - Use when the Execution Agent needs configuration updates:
- Current sub-task completed but main task not finished → update sub_goal
- Current sub-task execution failed → try new sub-task approach
- Current toolbox insufficient → update toolbox
- Knowledge lacks necessary information → update knowledge
- Execution strategy no longer fits → update execution_strategy
- Current context-management mode no longer fits → update context_management_mode

**Termination Tool** - Use only when:
- Main task is completely finished (not just sub-goal)
- All objectives have been achieved and no further work is needed

## Execution Requirements
1. **CRITICAL - Focus ONLY on Sub_goal**: Your ONLY objective is to complete
   the current sub_goal. DO NOT attempt to complete the entire main task.
2. **Main Task Usage**: Use the main task information ONLY to understand the
   broader context and make informed decisions about the current sub_goal.
3. **Tool Selection**: Choose appropriate tools from the toolbox
   to complete the sub_goal.
4. **Result Reporting**: Use the termination tool only when the MAIN TASK is
   completely finished. Use the reconfiguration tool when needing to update Execution Agent
   configuration.
5. **Runtime Configuration**: Treat `<context_management_mode>` as an explicit
   part of the current runtime configuration, not as hidden implementation
   detail.
6. **Error Handling**: When encountering problems, use the reconfiguration tool to
   modify agent configuration rather than simply reporting errors."""

    def _parse_execution_summary(self, execution_result: Dict[str, Any]) -> str:
        prediction = str(execution_result.get("prediction", "")).strip()
        summary_patterns = [
            r"(?:Process Summary|Summary|Execution Summary):\s*(.+?)(?=\n\n|\n#|\n\*\*|$)",
            r"(?:Result|Outcome):\s*(.+?)(?=\n\n|\n#|\n\*\*|$)",
        ]
        for pattern in summary_patterns:
            match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        if prediction:
            clean_pred = re.sub(r"\[.*?\]", "", prediction).strip()
            return clean_pred[:500] + ("..." if len(clean_pred) > 500 else "")
        request = execution_result.get("reconfiguration_request") or {}
        if isinstance(request, dict):
            return str(request.get("execution_summary", "")).strip() or "N/A"
        return "N/A"

    def _save_iteration_result(
        self,
        config: Dict[str, Any],
        execution_result: Dict[str, Any],
        execution_summary: str,
    ):
        next_sub_goal = config.get("next_sub_goal", "")
        iteration_data = {
            "stage": self.iteration_count,
            "timestamp": datetime.now().isoformat(),
            "configuration_C_i": {
                "q_i_next_sub_goal": next_sub_goal,
                "sigma_i_execution_strategy": config.get("execution_strategy", ""),
                "T_i_toolbox": config.get("toolbox", []),
                "K_i_inter_agent_knowledge": config.get("inter_agent_knowledge", ""),
                "m_i_context_management_mode": config.get("context_management_mode", "basic"),
            },
            "execution_result": {
                "termination": execution_result.get("termination", ""),
                "H_i_execution_summary": execution_summary,
                "r_i_reconfiguration_request": execution_result.get("reconfiguration_request", ""),
            },
            "finished": self.finished
        }

        paths = [
            (f"iteration_{self.iteration_count:03d}.json", iteration_data),
            (f"agent_setting_{self.iteration_count:03d}.json", config),
            (f"result_{self.iteration_count:03d}.json", execution_result),
        ]
        for filename, payload in paths:
            with open(os.path.join(self.iterations_dir, filename), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    def _check_terminate_condition(self, termination: str, prediction: str) -> bool:
        return termination == "terminate" or "[terminate]" in prediction.lower()

    def _run_configuration_generation(self, prompt: str, save_path: str) -> Dict[str, Any]:
        result = run_single_ReAct_agent(
            question=prompt,
            SEARCH_AGENT_SYSTEM_PROMPT_MULTI="You are a configuration generator. Output only the strict JSON object required by the user-provided configuration-generation specification.",
            function_lis=[],
            save_path=save_path,
            verbose=self.verbose,
            main_task=self.question,
            global_function_list=self.function_list,
            execution_history=self.execution_history,
            context_management_mode="basic",
            expect_configuration_json=True,
        )
        config = result.get("next_config") or self._extract_json_object(result.get("prediction", ""))
        return self._normalize_config(config or {})

    def run(self, max_iterations: int = 10) -> Dict[str, Any]:
        if self.verbose:
            print("=" * 80)
            print("ToolSelf GAIA - Execution-Reconfiguration Loop Started")
            print("=" * 80)
            print(f"Main Task (Q): {self.question}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"Global Tool Pool (T_all): {', '.join(self.function_list)}")
            print("=" * 80)

        final_result = {}

        try:
            while not self.finished and self.iteration_count < max_iterations:
                self.iteration_count += 1

                if self.current_config is None:
                    prompt = self._build_initial_config_prompt()
                    config_save_path = os.path.join(self.iterations_dir, f"config_generation_{self.iteration_count:03d}")
                    self.current_config = self._run_configuration_generation(prompt, config_save_path)

                config = self._normalize_config(self.current_config)
                self.current_config = None

                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"Stage {self.iteration_count}/{max_iterations}")
                    print(f"{'='*80}")
                    print(f"Configuration C_{self.iteration_count} generated")
                    print(f"Sub-goal (q_i): {config.get('next_sub_goal', 'N/A')}")
                    print(f"Toolbox (T_i): {', '.join(config.get('toolbox', []))}")
                    print(f"Context mode (m_i): {config.get('context_management_mode', 'basic')}")

                system_prompt = self._build_system_prompt(config)
                current_sub_goal = config.get("next_sub_goal", "") or "Task completed. TERMINATE in this round and output results."
                tools_list = [
                    name for name in config.get("toolbox", [])
                    if name not in {"reconfigure", "terminate"}
                ]
                tools_list.append("reconfigure")
                tools_list.append("terminate")

                process_save_path = os.path.join(self.iterations_dir, f"process_{self.iteration_count:03d}")
                execution_result = run_single_ReAct_agent(
                    question=current_sub_goal,
                    SEARCH_AGENT_SYSTEM_PROMPT_MULTI=system_prompt,
                    function_lis=tools_list,
                    save_path=process_save_path,
                    verbose=self.verbose,
                    main_task=self.question,
                    global_function_list=self.function_list,
                    execution_history=self.execution_history,
                    context_management_mode=config.get("context_management_mode", "basic"),
                )

                termination = execution_result.get("termination", "")
                prediction = execution_result.get("prediction", "")
                execution_summary = self._parse_execution_summary(execution_result)

                if self._check_terminate_condition(termination, prediction):
                    self.finished = True
                    self._update_context_file(self.iteration_count, current_sub_goal, execution_summary)
                    self._save_iteration_result(config, execution_result, execution_summary)
                    final_result = execution_result
                    break

                next_config = execution_result.get("next_config")
                if next_config is None and termination == "reconfigure":
                    next_config = self._extract_json_object(prediction)

                self._update_context_file(self.iteration_count, current_sub_goal, execution_summary)
                iteration_summary = f"Stage {self.iteration_count}:\nSub-goal: {current_sub_goal}\nSummary: {execution_summary}\n"
                if self.execution_history == "NONE":
                    self.execution_history = iteration_summary
                else:
                    self.execution_history += f"\n{iteration_summary}"

                self._save_iteration_result(config, execution_result, execution_summary)

                if next_config is not None:
                    self.current_config = self._normalize_config(next_config)
                else:
                    prompt = self._build_initial_config_prompt()
                    config_save_path = os.path.join(self.iterations_dir, f"config_generation_{self.iteration_count + 1:03d}")
                    self.current_config = self._run_configuration_generation(prompt, config_save_path)

                final_result = execution_result

        except Exception as e:
            if self.verbose:
                print(f"\nError during execution: {str(e)}")
            final_result = {
                "execution_status": "error",
                "error_message": str(e),
                "stage_count": self.iteration_count,
                "finished": False
            }

        summary_result = {
            "question": self.question,
            "total_stages": self.iteration_count,
            "finished": self.finished,
            "final_result": final_result,
            "workspace_dir": self.workspace_dir,
            "context_file": self.context_file,
            "completion_time": datetime.now().isoformat()
        }

        summary_file = os.path.join(self.workspace_dir, "execution_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"\n{'='*80}")
            print("Execution Summary:")
            print(f"Workspace: {self.workspace_dir}")
            print(f"Summary File: {summary_file}")
            print(f"Context File: {self.context_file}")
            print(f"{'='*80}\n")

        return summary_result


def main():
    test_question = """
    "url": "https://en.wikipedia.org/wiki/Moon", "goal": "Extract the minimum perigee distance of the Moon from Earth from the content of the Wikipedia page."
    """
    agent = ToolSelfGAIA(
        question=test_question,
        verbose=True
    )
    result = agent.run(max_iterations=20)
    print("\n" + "=" * 80)
    print("Execution Complete")
    print(f"Result: {result.get('finished', False)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
