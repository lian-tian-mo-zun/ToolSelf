from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool


def _get_searx_host() -> str:
    return os.getenv("SEARX_HOST", "http://localhost:8888").rstrip("/")


def _detect_language_from_text(text: str) -> str:
    if not text:
        return "en-US"

    counts = {
        "zh": 0,
        "ja": 0,
        "ko": 0,
        "ru": 0,
        "ar": 0,
        "hi": 0,
    }

    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            counts["zh"] += 1
            continue
        if 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
            counts["ja"] += 1
            continue
        if 0xAC00 <= code <= 0xD7AF:
            counts["ko"] += 1
            continue
        if 0x0400 <= code <= 0x04FF:
            counts["ru"] += 1
            continue
        if 0x0600 <= code <= 0x06FF:
            counts["ar"] += 1
            continue
        if 0x0900 <= code <= 0x097F:
            counts["hi"] += 1

    if counts["zh"] > 0 and counts["zh"] >= max(counts["ja"], counts["ko"]):
        return "zh-CN"
    if counts["ja"] > 0 and counts["ja"] >= counts["ko"]:
        return "ja-JP"
    if counts["ko"] > 0:
        return "ko-KR"
    if counts["ru"] > 0:
        return "ru-RU"
    if counts["ar"] > 0:
        return "ar"
    if counts["hi"] > 0:
        return "hi-IN"
    return "en-US"


def _get_searx_language(query: str) -> str:
    detected_lang = _detect_language_from_text(query)
    if detected_lang:
        return detected_lang
    env_lang = os.getenv("SEARX_LANGUAGE")
    return env_lang or "en-US"


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = (
        "Performs batched web searches via a Searx instance. Provide an array 'query'; "
        "the tool retrieves top results for each query in one call."
    )
    arguments = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def _search_once(self, query: str) -> str:
        base_url = _get_searx_host()
        url = f"{base_url}/search"
        params = {
            "q": query,
            "format": "json",
            "language": _get_searx_language(query),
            "safesearch": 1,
        }
        last_exc: Exception | None = None
        for i in range(5):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    last_exc = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    continue
                data = resp.json()
                results = data.get("results") or []

                web_snippets: List[str] = []
                for idx, item in enumerate(results[:10], start=1):
                    title = item.get("title") or "(no title)"
                    link = item.get("url") or ""
                    content = item.get("content") or ""
                    date = item.get("publishedDate")
                    engines = item.get("engines") or []
                    source = ", ".join(engines) if engines else (item.get("source") or "")

                    date_line = f"\nDate published: {date}" if date else ""
                    source_line = f"\nSource: {source}" if source else ""
                    snippet_line = f"\n{content}" if content else ""

                    entry = f"{idx}. [{title}]({link}){date_line}{source_line}{snippet_line}"
                    web_snippets.append(entry)

                content_out = (
                    f"A search for '{query}' found {len(results)} results:\n\n## Web Results\n"
                    + "\n\n".join(web_snippets)
                )
                return content_out
            except Exception as e:
                last_exc = e
                continue

        return (
            f"Search failed for '{query}'. "
            + (f"Last error: {type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown error.")
        )

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"] if isinstance(params, dict) else params
        except Exception:
            return "[search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            return self._search_once(query)

        if not isinstance(query, list):
            return "[search] 'query' must be a string or an array of strings"

        with ThreadPoolExecutor(max_workers=3) as executor:
            response_chunks = list(executor.map(self._search_once, query))
        return "\n=======\n".join(response_chunks)
