from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from execution_agent.react_agent import MultiTurnReactAgent
from execution_agent.prompt import (
    USER_PROMPT_PREFIX,
    USER_PROMPT_SUFFIX,
)
from tools.collect_tools import build_tool_prompt
from config import build_llm_cfg


def run_single_ReAct_agent(
    question: str,
    SEARCH_AGENT_SYSTEM_PROMPT_MULTI: str,
    function_lis: list[str],
    save_path: str | None,
    verbose: bool = True,
    remove_think_tags: bool = False,
    review_function: callable | None = None,
    max_review_attempts: int = 10,
    main_task: str | None = None,
    global_function_list: list[str] | None = None,
    execution_history: str = "NONE",
    context_management_mode: str = "basic",
    expect_configuration_json: bool = False,
    history_processor_enabled: bool = False,
    history_last_n_observations: int = 10,
    history_max_observation_length: int = 10000,
) -> dict:

    llm_cfg = build_llm_cfg()
    os.environ["REMOVE_THINK_TAGS"] = str(remove_think_tags)

    if "terminate" in function_lis:
        from tools import tool_terminate

    if "reconfigure" in function_lis:
        from tools import tool_reconfigure

    if "update" in function_lis:
        from tools import tool_update

    if "search" in function_lis:
        from tools import tool_search

    if "searx_search" in function_lis:
        from tools import tool_searx_search

    if "visit" in function_lis:
        from tools import tool_visit

    if "file_analyzer" in function_lis:
        from tools import tool_file_analyzer

    from tools import tool_code_interpreter
    from tools import tool_execute_bash
    from tools import tool_str_replace_editor

    system_message = SEARCH_AGENT_SYSTEM_PROMPT_MULTI + "\nCurrent date: " + datetime.now().strftime("%Y-%m-%d")
    function_list = function_lis

    history_processor = None
    if context_management_mode == "truncation" or history_processor_enabled:
        from execution_agent.history_processors import TruncationHistoryProcessor
        history_processor = TruncationHistoryProcessor(
            last_n_observations=history_last_n_observations,
            max_observation_length=history_max_observation_length,
            remove_think_tags=False,
        )
        if verbose:
            print(f"[History Processor] Enabled with mode={context_management_mode}, last_n_observations={history_last_n_observations}, "
                  f"max_observation_length={history_max_observation_length}")

    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=function_list,
        system_message=system_message,
        verbose=verbose,
        review_function=review_function,
        max_review_attempts=max_review_attempts,
        agent_runtime_context={
            "main_task": main_task or question,
            "function_list": global_function_list or function_lis,
            "execution_history": execution_history,
            "expect_configuration_json": expect_configuration_json,
        },
        history_processor=history_processor,
    )

    data = {
        "item": {
            "question": "" if expect_configuration_json else question,
            "answer": "",
        },
        "rollout_id": 1,
    }

    if expect_configuration_json:
        USER_PROMPT = question
    else:
        TOOL_PROMPT = build_tool_prompt(function_list)
        USER_PROMPT = USER_PROMPT_PREFIX + "\n" + TOOL_PROMPT + USER_PROMPT_SUFFIX
    result = agent._run(data=data, model=llm_cfg["model"], user_prompt=USER_PROMPT)

    if save_path:
        process_dir = save_path
        os.makedirs(process_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_filename = f"process_{current_time}.json"
        process_filepath = os.path.join(process_dir, process_filename)

        with open(process_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
        if verbose:
            print(f"✅ Full result saved to: {process_filepath}")

        if result.get('history_processor_enabled', False) and result.get('messages_sent_to_llm'):
            processed_filename = f"process_{current_time}_processed.json"
            processed_filepath = os.path.join(process_dir, processed_filename)

            processed_result = {
                "question": result.get("question"),
                "answer": result.get("answer"),
                "rollout_id": result.get("rollout_id"),
                "messages": result.get("messages_sent_to_llm"),
                "prediction": result.get("prediction"),
                "termination": result.get("termination"),
                "history_processor_enabled": True,
                "note": "This file contains messages as sent to LLM (after history processing)"
            }

            with open(processed_filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps(processed_result, ensure_ascii=False, indent=2))

            if verbose:
                print(f"✅ Processed messages saved to: {processed_filepath}")

                original_chars = sum(len(str(m.get('content', ''))) for m in result.get('messages', []))
                processed_chars = sum(len(str(m.get('content', ''))) for m in result.get('messages_sent_to_llm', []))
                if original_chars > 0:
                    saved_chars = original_chars - processed_chars
                    saved_percent = (saved_chars / original_chars) * 100
                    print(f"📊 History Processor Stats:")
                    print(f"   Original: {original_chars:,} chars")
                    print(f"   Processed: {processed_chars:,} chars")
                    print(f"   Saved: {saved_chars:,} chars ({saved_percent:.1f}%)")

    return result
