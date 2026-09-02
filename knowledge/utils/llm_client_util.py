"""LLM client factory for OpenAI-compatible DashScope endpoints."""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from knowledge.utils.logger_util import logger

load_dotenv()

_llm_client = None
_llm_json_client = None


def _create_chat_openai(response_format: bool):
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - depends on optional package
        logger.error("langchain_openai 不可用: {}", exc)
        return None

    kwargs = {
        "model": os.getenv("MODEL") or os.getenv("LLM_DEFAULT_MODEL") or "qwen-flash",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_API_BASE", ""),
        "temperature": float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
    }

    if response_format:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    try:
        return ChatOpenAI(**kwargs)
    except Exception as exc:
        logger.error("创建 LLM 客户端失败: {}", exc)
        return None


def get_llm_client(response_format: bool = False):
    global _llm_client, _llm_json_client

    if response_format:
        if _llm_json_client is None:
            _llm_json_client = _create_chat_openai(True)
        return _llm_json_client

    if _llm_client is None:
        _llm_client = _create_chat_openai(False)
    return _llm_client


def invoke_llm_with_json_fallback(messages):
    """优先 JSON 响应模式；模型不支持时回退到普通模式（提示词仍要求 JSON）。"""
    llm_client = get_llm_client(response_format=True)
    if llm_client is None:
        return None
    try:
        return llm_client.invoke(messages)
    except Exception as exc:
        logger.warning("JSON 模式调用失败，回退普通模式: {}", exc)
        llm_client = get_llm_client(response_format=False)
        if llm_client is None:
            return None
        try:
            return llm_client.invoke(messages)
        except Exception as exc:
            logger.error("普通模式 LLM 调用失败: {}", exc)
            return None
