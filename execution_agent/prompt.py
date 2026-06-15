USER_PROMPT_PREFIX = """A conversation between User and Assistant. The user asks a question, and the assistant solves it by calling one or more of the following tools.
"""


USER_PROMPT_SUFFIX ="""

The assistant starts with one or more cycles of (thinking about which tool to use -> performing tool call -> waiting for tool response), and ends by calling the `terminate` tool with a final summary and completion status. The thinking processes, tool calls, and tool responses are enclosed within their tags. There could be multiple thinking processes, tool calls, tool call parameters and tool response parameters.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here, "another parameter name here": another parameter value here, ...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<tool_call>
{"name": "another tool name here", "arguments": {...}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
(more thinking processes, tool calls and tool responses here)
<think> thinking process here </think>
<tool_call>
{"name": "terminate", "arguments": {"task_completion_status": "complete", "final_result": "final result here", "execution_summary": {"detailed_execution": ["step-by-step execution process"], "tools_used": ["tool1", "tool2"]}}}
</tool_call>

User: """

