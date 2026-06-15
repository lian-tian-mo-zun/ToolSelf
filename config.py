from __future__ import annotations

import os


MAIN_LLM_API_KEY = os.getenv("MAIN_LLM_API_KEY", "")
MAIN_LLM_API_BASE_URL = os.getenv("MAIN_LLM_API_BASE_URL", "")
MAIN_LLM_MODEL = os.getenv("MAIN_LLM_MODEL", "")
MAIN_LLM_MAX_TOKENS = int(os.getenv("MAIN_LLM_MAX_TOKENS", 1024 * 4))

VISIT_LLM_API_KEY = os.getenv("VISIT_LLM_API_KEY", MAIN_LLM_API_KEY)
VISIT_LLM_API_BASE_URL = os.getenv("VISIT_LLM_API_BASE_URL", MAIN_LLM_API_BASE_URL)
VISIT_LLM_MODEL = os.getenv("VISIT_LLM_MODEL", MAIN_LLM_MODEL)
VISIT_LLM_MAX_TOKENS = int(os.getenv("VISIT_LLM_MAX_TOKENS", 1024 * 4))

GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY", "")
JINA_KEY = os.getenv("JINA_KEY", "")

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))
TOP_P = float(os.getenv("TOP_P", 0.95))


def build_llm_cfg() -> dict:

    os.environ["MAIN_LLM_API_KEY"] = MAIN_LLM_API_KEY
    os.environ["MAIN_LLM_API_BASE_URL"] = MAIN_LLM_API_BASE_URL
    os.environ["MAIN_LLM_MODEL"] = MAIN_LLM_MODEL
    os.environ["MAIN_LLM_MAX_TOKENS"] = str(MAIN_LLM_MAX_TOKENS)

    os.environ["VISIT_LLM_API_KEY"] = VISIT_LLM_API_KEY
    os.environ["VISIT_LLM_API_BASE_URL"] = VISIT_LLM_API_BASE_URL
    os.environ["VISIT_LLM_MODEL"] = VISIT_LLM_MODEL
    os.environ["VISIT_LLM_MAX_TOKENS"] = str(VISIT_LLM_MAX_TOKENS)

    os.environ["GOOGLE_SEARCH_KEY"] = GOOGLE_SEARCH_KEY
    os.environ["JINA_KEY"] = JINA_KEY
    
    os.environ.setdefault("SEARX_HOST", "http://localhost:8888")
    os.environ.setdefault("SEARX_LANGUAGE", "en-US")

    return {
        "model": MAIN_LLM_MODEL,
        "generate_cfg": {
            "max_input_tokens": 320000,
            "max_retries": 10,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "model_type": "oai",
        "api_key": MAIN_LLM_API_KEY,
        "base_url": MAIN_LLM_API_BASE_URL,
    }
