import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os 
from openai import OpenAI
import time
import random

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
IGNORE_JINA = os.getenv("IGNORE_JINA", "false").lower() == "true"
JINA_READER_URL_PREFIX = os.getenv("JINA_READER_URL", "https://r.jina.ai/")

def _get_jina_keys() -> list:
    raw = os.getenv("JINA_API_KEYS") or os.getenv("JINA_KEY") or ""
    if not isinstance(raw, str):
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    arguments = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
      },
      "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
      }
    },
    "required": ["url", "goal"]
  }
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        response.append(future.result())
                    except Exception as e:
                        response.append(f"Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
    
    def call_server(self, msgs, max_tries=10):
        openai_api_key = os.getenv("VISIT_LLM_API_KEY", os.getenv("MAIN_LLM_API_KEY", ""))
        openai_api_base = os.getenv("VISIT_LLM_API_BASE_URL", os.getenv("MAIN_LLM_API_BASE_URL", ""))
        openai_model = os.getenv("VISIT_LLM_MODEL", os.getenv("MAIN_LLM_MODEL", ""))

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=60,
        )
        request_max_tokens = int(os.getenv("VISIT_LLM_MAX_TOKENS", os.getenv("MAIN_LLM_MAX_TOKENS", "1024")))
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=openai_model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.7,
                    max_tokens=request_max_tokens,
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception:
                if attempt == (max_tries - 1):
                    return ""
                time.sleep(min(5, 0.5 * (2 ** attempt)))
                continue

    def jina_readpage(self, url: str) -> str:
        keys = _get_jina_keys()
        headers = {"Authorization": f"Bearer {random.choice(keys)}"} if keys else {}

        jina_base_url = os.getenv("JINA_READER_URL", "https://r.jina.ai/")

        max_retries = 3
        timeout = 10

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{jina_base_url}{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."


    def readpage(self, url: str, goal: str) -> str:
        max_attempts = 10
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            sevice = "jina"

            print(sevice)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = self.call_server(messages)

                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = self.call_server(messages)
                    summary_retries -= 1
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        raw = json.loads(raw)
                        break
                    except:
                        raw = self.call_server(messages)
                        parse_retry_times += 1
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information
