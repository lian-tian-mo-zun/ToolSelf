import json
import re
import os
from typing import Dict, List, Optional, Union
import time
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.llm.schema import Message
from collections import defaultdict, deque

try:
    from execution_agent.history_processors import (
        AbstractHistoryProcessor,
        DefaultHistoryProcessor,
        TruncationHistoryProcessor,
        create_history_processor,
    )
except ImportError:
    from history_processors import (
        AbstractHistoryProcessor,
        DefaultHistoryProcessor,
        TruncationHistoryProcessor,
        create_history_processor,
    )


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 200))
MAX_TOKEN_LENGTH = int(os.getenv('MAX_LENGTH', 31 * 1024 - 500))
_TOKENIZER_CACHE = {}
_TOKENIZER_MODEL_ALIASES = {
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-14B-FP8": "Qwen/Qwen3-14B",
}


def _load_tokenizer(model_name: str):
    candidates = []
    alias = _TOKENIZER_MODEL_ALIASES.get(model_name)
    if alias:
        candidates.append(alias)
    if model_name:
        candidates.append(model_name)

    last_error = None
    for candidate in candidates:
        if candidate in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[candidate]
        try:
            tok = AutoTokenizer.from_pretrained(candidate)
            _TOKENIZER_CACHE[candidate] = tok
            if model_name and model_name not in _TOKENIZER_CACHE:
                _TOKENIZER_CACHE[model_name] = tok
            return tok
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError("No tokenizer model name provided")


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 verbose: bool = True,
                 review_function: Optional[callable] = None,
                 max_review_attempts: int = 10,
                 history_processor: Optional[AbstractHistoryProcessor] = None,
                 agent_runtime_context: Optional[Dict] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.verbose = verbose
        self.original_function_list = function_list or []
        self.review_function = review_function
        self.max_review_attempts = max_review_attempts
        self.history_processor = history_processor if history_processor is not None else DefaultHistoryProcessor()
        self._last_processed_messages = []
        self.agent_runtime_context = agent_runtime_context or {}

    def call_server(self, msgs, max_tries=10):
        openai_api_key = os.getenv("MAIN_LLM_API_KEY", "")
        openai_api_base = os.getenv("MAIN_LLM_API_BASE_URL", "")
        openai_model = os.getenv("MAIN_LLM_MODEL", self.llm_local_path)

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=60,
        )

        request_max_tokens = int(os.getenv("MAIN_LLM_MAX_TOKENS", "1024"))
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=openai_model,
                    messages=msgs,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=request_max_tokens,
                )
                content = chat_response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == (max_tries - 1):
                    if self.verbose:
                        print(f"DeepInfra API error {e}")
                    return f"DeepInfra API error"
                time.sleep(min(5, 0.5 * (2 ** attempt)))
                continue
        
        return "DeepInfra API empty response"

    def count_tokens(self, messages, model="gpt-4o"):
        try:
            tokenizer = _load_tokenizer(self.llm_local_path)
        except Exception as e:
            tokenizer = tiktoken.encoding_for_model(model)

        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)

        return len(tokenizer.encode(full_prompt))

    def _call_tool(self, tool_name: str, tool_args: dict = None, **kwargs):
        if tool_args is None:
            tool_args = {}

        registered_tool_names = []
        if hasattr(self, 'function_map') and self.function_map:
            registered_tool_names = list(self.function_map.keys())
        elif hasattr(self, 'original_function_list'):
            for item in self.original_function_list:
                if isinstance(item, str):
                    registered_tool_names.append(item)
                elif isinstance(item, dict) and 'name' in item:
                    registered_tool_names.append(item['name'])
                elif hasattr(item, 'name'):
                    registered_tool_names.append(item.name)

        if registered_tool_names and tool_name not in registered_tool_names:
            error_message = f"[Tool Validation Error]\n"
            error_message += f"Tool '{tool_name}' is not registered.\n\n"
            error_message += f"Available tools: {', '.join(registered_tool_names)}\n\n"

            if tool_name == 'pytest':
                error_message += "To run tests, use: execute_bash with command='python -m pytest tests/'\n"
            elif tool_name == 'pip':
                error_message += "To install packages, use: execute_bash with command='pip install package_name'\n"
            elif tool_name == 'python':
                error_message += "To run Python scripts, use: execute_bash with command='python script.py'\n"
            else:
                error_message += f"Hint: Most command-line operations can be executed using 'execute_bash' tool.\n"

            if self.verbose:
                print(f"[Warning] {error_message}")

            return error_message

        try:
            return super()._call_tool(tool_name, tool_args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if 'not registered' in error_str.lower() or 'not found' in error_str.lower():
                return f"[Tool Error] {error_str}\nAvailable tools: {', '.join(registered_tool_names)}"
            else:
                raise

    def _call_review_function(self, messages):
        if self.review_function is None:
            return {'is_solved': True, 'reason': ''}

        try:
            review_result = self.review_function(messages)

            if not isinstance(review_result, dict):
                if self.verbose:
                    print(f"[Warning] Review function returned non-dict: {type(review_result)}")
                return {'is_solved': True, 'reason': 'Invalid review result format'}

            if 'is_solved' not in review_result:
                if self.verbose:
                    print(f"[Warning] Review result missing 'is_solved' key")
                return {'is_solved': True, 'reason': 'Invalid review result format'}

            if 'reason' not in review_result:
                review_result['reason'] = ''

            return review_result

        except Exception as e:
            if self.verbose:
                print(f"[Error] Review function failed: {e}")
            return {'is_solved': True, 'reason': f'Review function error: {str(e)}'}

    def _handle_exit_with_review(self, messages, review_attempts, intended_termination='loop_exit'):
        if self.review_function is None or review_attempts >= self.max_review_attempts:
            if review_attempts >= self.max_review_attempts:
                return (True, review_attempts, 'max_review_attempts_reached')
            return (True, review_attempts, intended_termination)

        review_result = self._call_review_function(messages)

        if review_result['is_solved']:
            if self.verbose:
                print("[Review] Task verified as complete")
            return (True, review_attempts, intended_termination)
        else:
            review_attempts += 1
            if self.verbose:
                print(f"[Review {review_attempts}/{self.max_review_attempts}] Task not complete: {review_result['reason']}")

            if review_attempts >= self.max_review_attempts:
                if self.verbose:
                    print(f"[Review] Maximum review attempts ({self.max_review_attempts}) reached, forcing exit")
                return (True, review_attempts, 'max_review_attempts_reached')

            review_feedback = (
                f"[REVIEW FEEDBACK - Attempt {review_attempts}/{self.max_review_attempts}]\n"
                f"Your work was reviewed, but the task is not yet complete.\n\n"
                f"Reason: {review_result['reason']}\n\n"
                f"Please address the issues mentioned above and continue working on the task. "
                f"When you believe you have completed the task, call the terminate tool again."
            )
            messages.append({"role": "user", "content": review_feedback})
            return (False, review_attempts, None)

    @staticmethod
    def _extract_json_candidate(text: str):
        if not isinstance(text, str):
            return None

        cleaned = text.strip()
        if not cleaned:
            return None

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        fence = "```"
        if fence in cleaned:
            parts = cleaned.split(fence)
            for i in range(len(parts) - 1):
                block = parts[i + 1]
                if block.lstrip().startswith("json"):
                    candidate = block.lstrip()[4:].strip()
                else:
                    candidate = block.strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    continue

        starts = [idx for idx, ch in enumerate(cleaned) if ch == '{']
        ends = [idx for idx, ch in enumerate(cleaned) if ch == '}']
        for s in starts:
            for e in reversed(ends):
                if e < s:
                    continue
                candidate = cleaned[s:e + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        return None

    @staticmethod
    def _is_configuration_json(candidate) -> bool:
        if not isinstance(candidate, dict):
            return False

        required_keys = {
            "next_sub_goal",
            "execution_strategy",
            "toolbox",
            "inter_agent_knowledge",
            "context_management_mode",
        }
        if not required_keys.issubset(candidate.keys()):
            return False
        if not isinstance(candidate.get("toolbox"), list):
            return False
        if candidate.get("context_management_mode") not in {"basic", "truncation"}:
            return False
        return True

    def _run(self, data: str, model: str, user_prompt: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        answer = data['item']['answer']
        self.user_prompt = user_prompt
        self.user_prompt = self.user_prompt + question
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        empty_response_streak = 0
        tool_call_counts = defaultdict(int)
        last_tool_call_signature = None
        repeated_tool_call_count = 0
        tool_response_echo_count = 0
        round = 0
        terminate_tool_called = False
        update_tool_called = False
        awaiting_configuration_json = False
        last_reconfiguration_request = None
        review_attempts = 0
        termination = 'unknown'
        tool_result = None
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            processed_messages = self.history_processor(messages)
            self._last_processed_messages = processed_messages

            content = self.call_server(processed_messages)
            if self.agent_runtime_context.get("expect_configuration_json"):
                maybe_config = self._extract_json_candidate(content)
                if self._is_configuration_json(maybe_config):
                    messages.append({"role": "assistant", "content": json.dumps(maybe_config, ensure_ascii=False, indent=2)})
                    prediction = json.dumps(maybe_config, ensure_ascii=False, indent=2)
                    final_result = {
                        "question": question,
                        "answer": answer,
                        "rollout_id": data['rollout_id'],
                        "messages": messages,
                        "messages_sent_to_llm": self._last_processed_messages,
                        "prediction": prediction,
                        "termination": "configuration_generated",
                        "next_config": maybe_config,
                        "history_processor_enabled": self.history_processor.type != "default"
                    }
                    return final_result

                messages.append({"role": "assistant", "content": content.strip() if content else ""})
                messages.append({
                    "role": "user",
                    "content": "Output only the strict JSON object required by the configuration-generation specification."
                })
                continue
            if awaiting_configuration_json:
                maybe_config = self._extract_json_candidate(content)
                if self._is_configuration_json(maybe_config):
                    messages.append({"role": "assistant", "content": json.dumps(maybe_config, ensure_ascii=False, indent=2)})
                    prediction = json.dumps(maybe_config, ensure_ascii=False, indent=2)
                    final_result = {
                        "question": question,
                        "answer": answer,
                        "rollout_id": data['rollout_id'],
                        "messages": messages,
                        "messages_sent_to_llm": self._last_processed_messages,
                        "prediction": prediction,
                        "termination": "reconfigure",
                        "reconfiguration_request": last_reconfiguration_request,
                        "next_config": maybe_config,
                        "history_processor_enabled": self.history_processor.type != "default"
                    }
                    return final_result

                messages.append({"role": "assistant", "content": content.strip() if content else ""})
                messages.append({
                    "role": "user",
                    "content": "You already received the configuration-generation specification. Do not call tools. Output only the strict JSON object required by that specification."
                })
                continue
            if not content or content.strip() == "" or content.startswith("DeepInfra API"):
                empty_response_streak += 1
                if empty_response_streak >= 3:
                    messages.append({"role": "assistant", "content": "[Error] LLM kept returning empty responses."})
                    should_exit, review_attempts, term = self._handle_exit_with_review(messages, review_attempts, 'empty_response_error')
                    if should_exit:
                        termination = term
                        break
                    else:
                        empty_response_streak = 0
                        continue
                else:
                    continue
            else:
                empty_response_streak = 0
            if content.strip().startswith('<tool_response>') and ('<answer>' not in content and '<tool_call>' not in content):
                tool_response_echo_count += 1
                control_msg = (
                    "Please don't repeat tool returns. Based on the tool return information currently obtained, integrate your thinking and directly provide the final answer, "
                    "using the following format: <think>Your brief thinking</think>\n<answer>Your answer</answer>"
                )
                messages.append({"role": "user", "content": control_msg})
                if tool_response_echo_count >= 3:
                    messages.append({"role": "user", "content": "Please immediately output <answer>, don't call tools or repeat tool returns."})
                continue
            else:
                tool_response_echo_count = 0
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
                if not content.strip():
                    messages.append({
                        "role": "user",
                        "content": (
                            "Tool return received. Please stop repeating tool returns, directly generate final answer based on existing information, "
                            "strictly using the following format: <think>Your brief thinking</think>\n<answer>Your answer</answer>"
                        )
                    })
                    continue
            remove_think = os.getenv("REMOVE_THINK_TAGS", "False").lower() == "true"
            if remove_think:
                content_filtered = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if self.verbose:
                    print(f'Round {round}:\n {content_filtered}')
                messages.append({"role": "assistant", "content": content_filtered})
            else:
                if self.verbose:
                    print(f'Round {round}:\n {content}')
                messages.append({"role": "assistant", "content": content.strip()})
            
            if '<tool_call>' in content and '</tool_call>' in content:
                try:
                    tool_call_blob = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    maybe_tc = json.loads(tool_call_blob)
                    if isinstance(maybe_tc, dict):
                        tool_name = maybe_tc.get('name')
                        available_tool_names = []
                        for tool in self.original_function_list:
                            if hasattr(tool, 'name'):
                                available_tool_names.append(tool.name.lower())
                            elif isinstance(tool, dict) and 'name' in tool:
                                available_tool_names.append(tool['name'].lower())
                            elif isinstance(tool, str):
                                available_tool_names.append(tool.lower())
                            else:
                                available_tool_names.append(str(tool).lower())
                        
                        if tool_name == 'terminate' and 'terminate' in available_tool_names:
                            terminate_tool_called = True
                            tool_args = maybe_tc.get('arguments', {})
                            tool_result = self._call_tool('terminate', tool_args)
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})

                            should_exit, review_attempts, term = self._handle_exit_with_review(messages, review_attempts, 'terminate')
                            if should_exit:
                                termination = term
                                break
                            else:
                                terminate_tool_called = False
                                continue

                        elif tool_name == 'update' and 'update' in available_tool_names:
                            update_tool_called = True
                            tool_args = maybe_tc.get('arguments', {})
                            tool_result = self._call_tool('update', tool_args)
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})

                            termination = 'update'
                            break

                        elif tool_name == 'reconfigure' and 'reconfigure' in available_tool_names:
                            update_tool_called = True
                            tool_args = maybe_tc.get('arguments', {})
                            last_reconfiguration_request = tool_args
                            tool_result = self._call_tool(
                                'reconfigure',
                                tool_args,
                                agent_runtime_context=self.agent_runtime_context,
                            )
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})
                            awaiting_configuration_json = True
                            messages.append({
                                "role": "user",
                                "content": "The configuration-generation specification is now available in the tool response above. Do not call any tools. Output only the final strict JSON object requested there."
                            })
                            continue
                except Exception as e:
                    if self.verbose:
                        print(f"Error in terminate/reconfigure tool detection: {e}")
                    pass
            if '<tool_call>' in content and '</tool_call>' in content:
                try:
                    tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})

                    available_tool_names = []
                    for tool in self.original_function_list:
                        if hasattr(tool, 'name'):
                            available_tool_names.append(tool.name.lower())
                        elif isinstance(tool, dict) and 'name' in tool:
                            available_tool_names.append(tool['name'].lower())
                        elif isinstance(tool, str):
                            available_tool_names.append(tool.lower())
                        else:
                            available_tool_names.append(str(tool).lower())

                    if tool_name.lower() not in available_tool_names:
                        error_msg = f"Tool '{tool_name}' is not available in the current function list. Available tools: {', '.join(available_tool_names)}"
                        messages.append({"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"})
                        if self.verbose:
                            print(f"⚠️  Blocked tool call: {error_msg}")
                        continue

                    if tool_name == 'terminate':
                        terminate_tool_called = True
                    elif tool_name in ('update', 'reconfigure'):
                        update_tool_called = True
                    tool_call_counts[tool_name] += 1
                    signature = json.dumps({"name": tool_name, "arguments": tool_args}, ensure_ascii=False, sort_keys=True)
                    if signature == last_tool_call_signature:
                        repeated_tool_call_count += 1
                    else:
                        repeated_tool_call_count = 0
                        last_tool_call_signature = signature
                    if repeated_tool_call_count >= 200 or tool_call_counts[tool_name] > 500:
                        control_msg = (
                            "Please stop repeated tool calls. Based on the tool return information currently obtained, integrate your thinking and directly provide the final answer, "
                            "using the following format: <think>Your brief thinking</think>\n<answer>Your answer</answer>"
                        )
                        messages.append({"role": "user", "content": control_msg})
                        continue
                    messages_for_tools = [Message(**m) if isinstance(m, dict) else m for m in messages]
                    tool_kwargs = {"messages": messages_for_tools}
                    if tool_name == "reconfigure":
                        tool_kwargs["agent_runtime_context"] = self.agent_runtime_context
                        last_reconfiguration_request = tool_args
                    tool_result = self._call_tool(tool_name, tool_args, **tool_kwargs)

                    tool_result_formatted = "<tool_response>\n" + tool_result + "\n</tool_response>"
                    messages.append({"role": "user", "content": tool_result_formatted})

                    if tool_name == 'terminate':
                        terminate_tool_called = True
                        should_exit, review_attempts, term = self._handle_exit_with_review(messages, review_attempts, 'terminate')
                        if should_exit:
                            termination = term
                            break
                        else:
                            terminate_tool_called = False
                            continue

                    elif tool_name == 'update':
                        update_tool_called = True
                        termination = 'update'
                        break

                    elif tool_name == 'reconfigure':
                        update_tool_called = True
                        awaiting_configuration_json = True
                        messages.append({
                            "role": "user",
                            "content": "The configuration-generation specification is now available in the tool response above. Do not call any tools. Output only the final strict JSON object requested there."
                        })
                        continue

                except Exception as e:
                    if self.verbose:
                        print(f"Error in tool call parsing: {e}")
                        print(f"Tool call content: {tool_call}")
                    continue
            else:
                try:
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        maybe_json = content[left:right+1]
                        tc = json.loads(maybe_json)
                        if isinstance(tc, dict) and 'name' in tc and 'arguments' in tc:
                            tool_name = tc.get('name', '')
                            tool_args = tc.get('arguments', {})

                            available_tool_names = []
                            for tool in self.original_function_list:
                                if hasattr(tool, 'name'):
                                    available_tool_names.append(tool.name.lower())
                                elif isinstance(tool, dict) and 'name' in tool:
                                    available_tool_names.append(tool['name'].lower())
                                elif isinstance(tool, str):
                                    available_tool_names.append(tool.lower())
                                else:
                                    available_tool_names.append(str(tool).lower())

                            if tool_name.lower() not in available_tool_names:
                                error_msg = f"Tool '{tool_name}' is not available in the current function list. Available tools: {', '.join(available_tool_names)}"
                                messages.append({"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"})
                                if self.verbose:
                                    print(f"⚠️  Blocked tool call (fallback): {error_msg}")
                                continue

                            if tool_name == 'terminate':
                                terminate_tool_called = True
                            elif tool_name in ('update', 'reconfigure'):
                                update_tool_called = True
                            tool_kwargs = {}
                            if tool_name == "reconfigure":
                                tool_kwargs["agent_runtime_context"] = self.agent_runtime_context
                                last_reconfiguration_request = tool_args
                            tool_result = self._call_tool(tool_name, tool_args, **tool_kwargs)
                            tool_result = "<tool_response>\n" + tool_result + "\n</tool_response>"
                            messages.append({"role": "user", "content": tool_result})
                            if tool_name == "reconfigure":
                                awaiting_configuration_json = True
                                messages.append({
                                    "role": "user",
                                    "content": "The configuration-generation specification is now available in the tool response above. Do not call any tools. Output only the final strict JSON object requested there."
                                })
                                continue
                except Exception as e:
                    if self.verbose:
                        print(f"Error in JSON parsing fallback: {e}")
                        print(f"Content: {content}")
                    try:
                        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content, re.DOTALL)
                        code_match = re.search(r'"code"\s*:\s*"([\s\S]*?)"\s*\}?\s*([\]\}]|$)', content)
                        if name_match:
                            tool_name = name_match.group(1)

                            available_tool_names = []
                            for tool in self.original_function_list:
                                if hasattr(tool, 'name'):
                                    available_tool_names.append(tool.name.lower())
                                elif isinstance(tool, dict) and 'name' in tool:
                                    available_tool_names.append(tool['name'].lower())
                                elif isinstance(tool, str):
                                    available_tool_names.append(tool.lower())
                                else:
                                    available_tool_names.append(str(tool).lower())

                            if tool_name.lower() not in available_tool_names:
                                error_msg = f"Tool '{tool_name}' is not available in the current function list. Available tools: {', '.join(available_tool_names)}"
                                messages.append({"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"})
                                if self.verbose:
                                    print(f"⚠️  Blocked tool call (regex fallback): {error_msg}")
                                continue

                            if tool_name == 'terminate':
                                terminate_tool_called = True
                            elif tool_name in ('update', 'reconfigure'):
                                update_tool_called = True
                            if code_match and tool_name == 'code_interpreter':
                                raw_code = code_match.group(1)
                                raw_code = raw_code.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                                tool_result = self._call_tool('code_interpreter', {"code": raw_code})
                                tool_result = "<tool_response>\n" + tool_result + "\n</tool_response>"
                                messages.append({"role": "user", "content": tool_result})
                    except Exception as e:
                        if self.verbose:
                            print(f"Error in regex fallback parsing: {e}")
                            print(f"Content: {content}")
                        pass
            if self.verbose:
                print(tool_result)

            if '<answer>' in content and '</answer>' in content:
                should_exit, review_attempts, term = self._handle_exit_with_review(messages, review_attempts, 'answer')
                if should_exit:
                    termination = term
                    break
                else:
                    continue
                
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = MAX_TOKEN_LENGTH
            processed_for_check = self.history_processor(messages)
            token_count = self.count_tokens(processed_for_check)
            if self.verbose:
                print(f"round: {round}, token count: {token_count}")
                print('\n\n\n')

            if token_count > max_tokens:
                if self.verbose:
                    print(f"Token count exceeds limit: {token_count} > {max_tokens}")

                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                processed_messages = self.history_processor(messages)
                self._last_processed_messages = processed_messages
                content = self.call_server(processed_messages)

                remove_think = os.getenv("REMOVE_THINK_TAGS", "False").lower() == "true"
                if remove_think:
                    content_filtered = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    messages.append({"role": "assistant", "content": content_filtered})
                else:
                    messages.append({"role": "assistant", "content": content.strip()})

                should_exit, review_attempts, term = self._handle_exit_with_review(messages, review_attempts, 'token_limit_reached')
                if should_exit:
                    try:
                        if '<answer>' in messages[-1]['content'] and '</answer>' in messages[-1]['content']:
                            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                        else:
                            prediction = messages[-1]['content']
                    except (IndexError, KeyError):
                        prediction = messages[-1].get('content', '') if messages else ''

                    final_result = {
                        "question": question,
                        "answer": answer,
                        "rollout_id": data['rollout_id'],
                        "messages": messages,
                        "messages_sent_to_llm": self._last_processed_messages,
                        "prediction": prediction,
                        "termination": term,
                        "history_processor_enabled": self.history_processor.type != "default"
                    }
                    return final_result
                else:
                    continue
            prune_threshold = int(os.getenv('PRUNE_TOKEN_THRESHOLD', max_tokens * 0.8))
            if token_count > prune_threshold:
                preserved = [messages[0]]
                if len(messages) > 1:
                    preserved.append(messages[1])
                tail = messages[-12:] if len(messages) > 12 else messages
                messages = preserved + tail

        if not terminate_tool_called and not update_tool_called and num_llm_calls_available > 0:
            termination_reminder = (
                "You have completed your work but did not use proper termination tools. "
                "You MUST use either the 'reconfigure' tool (T_reconfig - to modify agent configuration) or 'terminate' tool (T_term - to end main task) to properly terminate.\n\n"
                
                "**EXAMPLES OF PROPER TOOL USAGE:**\n\n"
                
                "**RECONFIGURE TOOL (T_reconfig)** - Use when agent needs configuration updates:\n"
                "<tool_call>{\n"
                "  'name': 'reconfigure',\n"
                "  'arguments': {\n"
                "    'execution_summary': 'Step-by-step summary of current execution process (H_i)',\n"
                "    'update_reason': 'Current sub-task completed, need new sub_goal for next phase (ρ_i)',\n"
                "    'new_sub_goal': 'Specific new sub_goal to execute next (q_{i+1}^prop)'\n"
                "  }\n"
                "}</tool_call>\n\n"
                
                "**TERMINATE TOOL (T_term)** - Use only when main task is completely finished:\n"
                "<tool_call>{\n"
                "  'name': 'terminate',\n"
                "  'arguments': {\n"
                "    'task_completion_status': 'complete',  // or 'partial', 'incomplete'\n"
                "    'final_result': 'Description of final achievement',\n"
                "    'execution_summary': {\n"
                "      'detailed_execution': ['Step 1: what was done', 'Step 2: what was done'],\n"
                "      'tools_used': ['tool1', 'tool2'],\n"
                "      'key_achievements': ['Achievement 1', 'Achievement 2']\n"
                "    }\n"
                "  }\n"
                "}</tool_call>\n\n"
                
                "**IMPORTANT DECISION CRITERIA:**\n"
                "- Use RECONFIGURE tool when: sub_goal done but main task not complete, need new approach, insufficient toolbox/knowledge\n"
                "- Use TERMINATE tool when: main task completely finished, all objectives achieved\n"
                "- NEVER use terminate tool for sub_goal completion - use reconfigure tool instead"
            )
            messages.append({"role": "user", "content": termination_reminder})

            processed_messages = self.history_processor(messages)
            self._last_processed_messages = processed_messages
            content = self.call_server(processed_messages)

            remove_think = os.getenv("REMOVE_THINK_TAGS", "False").lower() == "true"
            if remove_think:
                content_filtered = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if self.verbose:
                    print(f'Final termination reminder response:\n {content_filtered}')
                messages.append({"role": "assistant", "content": content_filtered})
            else:
                if self.verbose:
                    print(f'Final termination reminder response:\n {content}')
                messages.append({"role": "assistant", "content": content.strip()})
            
            if '<tool_call>' in content and '</tool_call>' in content:
                try:
                    tool_call_blob = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    maybe_tc = json.loads(tool_call_blob)
                    if isinstance(maybe_tc, dict):
                        tool_name = maybe_tc.get('name')

                        available_tool_names = []
                        for tool in self.original_function_list:
                            if hasattr(tool, 'name'):
                                available_tool_names.append(tool.name.lower())
                            elif isinstance(tool, dict) and 'name' in tool:
                                available_tool_names.append(tool['name'].lower())
                            elif isinstance(tool, str):
                                available_tool_names.append(tool.lower())
                            else:
                                available_tool_names.append(str(tool).lower())

                        if tool_name == 'terminate' and 'terminate' in available_tool_names:
                            terminate_tool_called = True
                            termination = 'terminate'
                            tool_args = maybe_tc.get('arguments', {})
                            tool_result = self._call_tool('terminate', tool_args)
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})
                        elif tool_name == 'update' and 'update' in available_tool_names:
                            update_tool_called = True
                            termination = 'update'
                            tool_args = maybe_tc.get('arguments', {})
                            tool_result = self._call_tool('update', tool_args)
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})
                        elif tool_name == 'reconfigure' and 'reconfigure' in available_tool_names:
                            update_tool_called = True
                            tool_args = maybe_tc.get('arguments', {})
                            last_reconfiguration_request = tool_args
                            tool_result = self._call_tool(
                                'reconfigure',
                                tool_args,
                                agent_runtime_context=self.agent_runtime_context,
                            )
                            messages.append({"role": "user", "content": "<tool_response>\n" + tool_result + "\n</tool_response>"})
                            awaiting_configuration_json = True
                            messages.append({
                                "role": "user",
                                "content": "The configuration-generation specification is now available in the tool response above. Do not call any tools. Output only the final strict JSON object requested there."
                            })
                            processed_messages = self.history_processor(messages)
                            self._last_processed_messages = processed_messages
                            config_content = self.call_server(processed_messages)
                            maybe_config = self._extract_json_candidate(config_content)
                            if self._is_configuration_json(maybe_config):
                                messages.append({"role": "assistant", "content": json.dumps(maybe_config, ensure_ascii=False, indent=2)})
                                prediction = json.dumps(maybe_config, ensure_ascii=False, indent=2)
                                return {
                                    "question": question,
                                    "answer": answer,
                                    "rollout_id": data['rollout_id'],
                                    "messages": messages,
                                    "messages_sent_to_llm": self._last_processed_messages,
                                    "prediction": prediction,
                                    "termination": "reconfigure",
                                    "reconfiguration_request": last_reconfiguration_request,
                                    "next_config": maybe_config,
                                    "history_processor_enabled": self.history_processor.type != "default"
                                }
                except Exception as e:
                    if self.verbose:
                        print(f"Error in final termination tool detection: {e}")

        prediction = 'No answer found.'
        termination = 'answer not found' if not terminate_tool_called and not update_tool_called else termination
        for m in reversed(messages):
            if m.get('role') == 'user' and isinstance(m.get('content'), str) and m['content'].startswith('<tool_response>'):
                inner = m['content']
                if inner.find('[terminate]') != -1:
                    try:
                        body = inner.split('\n', 1)[1].split('</tool_response>')[0]
                    except Exception:
                        body = inner.replace('<tool_response>', '').replace('</tool_response>', '').strip()
                    prediction = body.strip()
                    termination = 'terminate'
                    break
                elif inner.find('[update]') != -1:
                    try:
                        body = inner.split('\n', 1)[1].split('</tool_response>')[0]
                    except Exception:
                        body = inner.replace('<tool_response>', '').replace('</tool_response>', '').strip()
                    prediction = body.strip()
                    termination = 'update'
                    break
                elif inner.find('[reconfigure]') != -1:
                    try:
                        body = inner.split('\n', 1)[1].split('</tool_response>')[0]
                    except Exception:
                        body = inner.replace('<tool_response>', '').replace('</tool_response>', '').strip()
                    prediction = body.strip()
                    termination = 'reconfigure'
                    break
        if termination not in ['terminate', 'update', 'reconfigure']:
            try:
                if '<answer>' in messages[-1]['content'] and '</answer>' in messages[-1]['content']:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'answer'
            except (IndexError, KeyError):
                prediction = messages[-1].get('content', '') if messages else ''
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        final_result = {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": messages,
            "messages_sent_to_llm": self._last_processed_messages,
            "prediction": prediction,
            "termination": termination,
            "history_processor_enabled": self.history_processor.type != "default"
        }
        return final_result
