"""
OpenAI GPT API 封装
================================

设计依据
- 老师 WS8 实际上没有"在代码中调用 OpenAI API"，他是把 ChatGPT 网页对话的输出
  贴回 cell 里运行的。本模块对应作业要求里
  "使用生成/分析 API 和您的 API 密钥... 不是我用 GPT 制作的脚本，创建您自己的脚本！"
  这一条 —— 真正用 SDK 调 API。
- 用 tenacity 自动重试 + 简单并发，控制成本和稳定性。
- 同时支持官方 https://api.openai.com 和国内中转端点（通过 OPENAI_BASE_URL 配置）。
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import env, get_logger, load_config

log = get_logger("gpt")


def make_client() -> OpenAI:
    api_key = env("OPENAI_API_KEY")
    base_url = env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key or api_key.startswith("sk-xxxx"):
        raise RuntimeError(
            "未设置 OPENAI_API_KEY。请把 .env.example 改名为 .env 并填入真实 Key。"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def chat(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 500,
    response_format: dict | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict | list:
    """要求模型返回严格 JSON，自动 parse"""
    raw = chat(
        client,
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 退化：截取 {} / [] 包裹的部分
        for s, e in [("{", "}"), ("[", "]")]:
            if s in raw and e in raw:
                try:
                    return json.loads(raw[raw.index(s) : raw.rindex(e) + 1])
                except Exception:
                    pass
        log.warning(f"JSON 解析失败，原始响应: {raw[:200]}")
        return {}


def get_default_model() -> str:
    return load_config()["openai"]["model"]
