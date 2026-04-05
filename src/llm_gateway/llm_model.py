"""
LLM Gateway — singleton wrapper around a HuggingFace chat model.

Loaded once at first use via get_chat_model(); shared across callers
(the model itself is stateless).
"""

import os
import logging
from functools import lru_cache

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Read + coerce env vars ───────────────────────────────────── #

LLM_REPO_ID: str = os.getenv("LLM_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")

# env vars are always strings; cast to the types HuggingFaceEndpoint expects
_raw_max_tokens = os.getenv("LLM_MAX_NEW_TOKENS", "1024")
_raw_rep_penalty = os.getenv("LLM_REPETITION_PENALTY", "1.1")

try:
    LLM_MAX_NEW_TOKENS: int = int(_raw_max_tokens)
except ValueError:
    logger.warning(
        "LLM_MAX_NEW_TOKENS='%s' is not a valid integer; defaulting to 1024.",
        _raw_max_tokens,
    )
    LLM_MAX_NEW_TOKENS = 1024

try:
    LLM_REPETITION_PENALTY: float = float(_raw_rep_penalty)
except ValueError:
    logger.warning(
        "LLM_REPETITION_PENALTY='%s' is not a valid float; defaulting to 1.1.",
        _raw_rep_penalty,
    )
    LLM_REPETITION_PENALTY = 1.1


# ── Singleton factory ────────────────────────────────────────── #

@lru_cache(maxsize=1)
def get_chat_model() -> ChatHuggingFace:
    """
    Return (and cache) the global ChatHuggingFace instance.

    The @lru_cache(maxsize=1) decorator ensures the heavy model is loaded
    exactly once per process, no matter how many times this is called.
    """
    logger.info(
        "Loading LLM: %s  (max_new_tokens=%d, rep_penalty=%.2f)",
        LLM_REPO_ID,
        LLM_MAX_NEW_TOKENS,
        LLM_REPETITION_PENALTY,
    )

    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task="text-generation",
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=LLM_REPETITION_PENALTY,
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)


def get_openai_model(model="gpt-5.4-nano") -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=os.getenv("api_key"))