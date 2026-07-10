"""
backend/subtitle_process/llm_client.py
LLM 客户端封装

移植自 VideoCaptioner app/core/llm/client.py，关键改动：
- 移除全局单例 / 线程锁 / 环境变量读取
- 所有配置通过函数参数传入（base_url, api_key, model）
- 异常统一转为 LLMError 层级
- 支持 disable_thinking 参数和自定义 extra_body，便于不同 OpenAI
  兼容接口分别控制思考模式
- 支持 require_json 参数，强制 JSON 输出格式
- 响应自动清理 think 标签和提取 JSON

修改：
  1. _create_client 设置 max_retries=0，禁用 SDK 自动重试
  2. timeout 从硬编码 120s 改为可配置参数（默认 120s）
  3. call_llm 中对 BadRequestError(400) 增加降级重试：
     去掉 extra_body 和 response_format 后重试一次，
     解决代理服务不兼容非标准参数的问题
  4. APIStatusError 中检测伪装成 400 的限流/上游暂时性错误
  5. _create_client / call_llm / check_llm_connection 均接受 timeout 参数透传
"""

import logging
import re
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from optimizer.exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)

logger = logging.getLogger("optimizer.llm_client")

# 默认超时秒数
DEFAULT_TIMEOUT = 120.0


def _normalize_base_url(base_url: str) -> str:
    """规范化 API base URL。"""
    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def _create_client(
    base_url: str,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenAI:
    """创建一次性 OpenAI 客户端实例。

    max_retries=0：禁用 SDK 自动重试。
    重试逻辑由应用层的 agent_loop 控制。

    Args:
        base_url: API base URL
        api_key: API 密钥
        timeout: 请求超时秒数（默认 120s）
    """
    normalized_url = _normalize_base_url(base_url)
    return OpenAI(
        base_url=normalized_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )


def _clean_response_content(response: ChatCompletion) -> None:
    """清理 LLM 响应中的非内容部分。

    1. 移除 <think>...</think> 标签（含未闭合的）
    2. 移除 Markdown 代码围栏 (```json ... ```)
    3. 如果响应包含 JSON 对象但被其他文本包裹，提取 JSON 部分

    直接修改 response.choices[0].message.content。
    """
    if (
        not response
        or not response.choices
        or not response.choices[0].message
        or not response.choices[0].message.content
    ):
        return

    content = response.choices[0].message.content

    # 1. 移除 <think>...</think>（含未闭合）
    content = re.sub(
        r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE,
    )
    content = re.sub(
        r"<think>[\s\S]*$", "", content, flags=re.IGNORECASE,
    )

    # 2. 移除 Markdown 代码围栏
    md_match = re.search(
        r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```", content,
    )
    if md_match:
        content = md_match.group(1)

    # 3. 如果内容包含 JSON 对象但有前后文本，提取第一个 {...}
    content_stripped = content.strip()
    if not content_stripped.startswith("{"):
        first_brace = content_stripped.find("{")
        last_brace = content_stripped.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidate = content_stripped[first_brace : last_brace + 1]
            if '"' in candidate and ":" in candidate:
                content = candidate

    response.choices[0].message.content = content.strip()


def _deep_merge_dict(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """递归合并字典，override 覆盖 base。"""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_extra_kwargs(
    kwargs: dict[str, Any],
    llm_extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合并调用方传入的 extra_body。

    不再猜测不同 provider 的思考模式字段。若需要控制 reasoning/thinking
    等非标准参数，由调用方通过 llm_extra_body 原样传入。
    """
    if not llm_extra_body:
        return kwargs

    extra_body = kwargs.pop("extra_body", {}) or {}
    if not isinstance(extra_body, dict):
        extra_body = {}
    kwargs["extra_body"] = _deep_merge_dict(extra_body, llm_extra_body)
    return kwargs


def _inject_no_think_directive(
    messages: list[dict[str, str]],
    disable_thinking: bool,
) -> list[dict[str, str]]:
    """在 system prompt 末尾注入 /no_think 指令。

    对 Qwen3 API 服务有效（通过 prompt 控制）。
    对其他模型无害（只是额外文本）。
    """
    if not disable_thinking or not messages:
        return messages

    messages = [msg.copy() for msg in messages]

    # 找到 system message 并追加
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = msg["content"].rstrip() + "\n/no_think"
            return messages

    # 没有 system message，插入一个
    messages.insert(0, {"role": "system", "content": "/no_think"})
    return messages


def _do_llm_call(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    **kwargs: Any,
) -> ChatCompletion:
    """执行实际的 LLM API 调用（内部函数）。

    将 OpenAI SDK 异常转为 Voxlign 异常层级。
    """
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            **kwargs,
        )
    except AuthenticationError as e:
        logger.error("LLM 认证失败: %s", e)
        raise LLMAuthError(detail=str(e)) from e
    except RateLimitError as e:
        logger.warning("LLM 速率限制: %s", e)
        raise LLMRateLimitError(detail=str(e)) from e
    except APITimeoutError as e:
        logger.error("LLM 请求超时: %s", e)
        raise LLMConnectionError(detail=f"请求超时: {e}") from e
    except APIConnectionError as e:
        logger.error("LLM 连接失败: %s", e)
        raise LLMConnectionError(detail=str(e)) from e
    except APIStatusError as e:
        error_str = str(e).lower()
        is_upstream_transient = (
            "upstream" in error_str
            or "请求上游服务失败" in str(e)
            or "稍后重试" in str(e)
            or "try again later" in error_str
        )
        if is_upstream_transient:
            logger.warning(
                "LLM 上游暂时性错误 (HTTP %s): %s", e.status_code, e,
            )
            raise LLMRateLimitError(
                detail=f"上游暂时性错误 (HTTP {e.status_code}): {e.message}",
            ) from e

        logger.error(
            "LLM 服务端错误 (HTTP %s, %s): %s",
            e.status_code, type(e).__name__, e,
        )
        raise LLMConnectionError(
            detail=f"HTTP {e.status_code}: {e.message}",
        ) from e

    except OpenAIError as e:
        logger.error("LLM 调用异常 (%s): %s", type(e).__name__, e)
        raise LLMError(f"LLM 调用失败: {e}") from e


def call_llm(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    base_url: str,
    api_key: str,
    disable_thinking: bool = False,
    require_json: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> ChatCompletion:
    """调用 LLM API，返回 ChatCompletion。

    对 BadRequestError(400) 有降级机制：如果请求包含 extra_body 或
    response_format 等非标准参数，400 失败后会去掉这些参数重试一次。
    这解决了部分 API 代理服务不兼容非标准参数的问题。

    Args:
        messages: OpenAI 格式的消息列表
        model: 模型名称
        temperature: 采样温度
        base_url: API base URL
        api_key: API 密钥
        disable_thinking: 是否注入 /no_think 提示词指令
        require_json: 是否强制 JSON 输出格式（optimizer/translator 使用）
        llm_extra_body: 原样透传到 OpenAI 兼容接口 extra_body 的 JSON 对象
        timeout: 请求超时秒数（默认 120s）
        **kwargs: 透传给 chat.completions.create 的额外参数

    Returns:
        ChatCompletion 响应对象

    Raises:
        LLMAuthError / LLMRateLimitError / LLMConnectionError /
        LLMResponseError / LLMError
    """
    # 原样透传 provider-specific extra_body
    kwargs = _build_extra_kwargs(kwargs, llm_extra_body=llm_extra_body)

    # 注入 /no_think 指令到 system prompt
    messages = _inject_no_think_directive(messages, disable_thinking)

    # 强制 JSON 输出格式
    if require_json:
        kwargs["response_format"] = {"type": "json_object"}

    # 记录是否有可降级的非标准参数
    has_extra_body = "extra_body" in kwargs
    has_response_format = "response_format" in kwargs
    can_fallback = has_extra_body or has_response_format

    client = _create_client(base_url, api_key, timeout=timeout)

    try:
        response = _do_llm_call(client, model, messages, temperature, **kwargs)
    except LLMConnectionError as e:
        # 如果是 400 且有非标准参数，尝试去掉后重试
        if can_fallback and "HTTP 400" in (e.message or ""):
            logger.info(
                "400 错误可能由非标准参数引起，去掉 extra_body/response_format 重试",
            )
            fallback_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("extra_body", "response_format")
            }
            response = _do_llm_call(
                client, model, messages, temperature, **fallback_kwargs,
            )
        else:
            raise

    # 清理响应内容
    _clean_response_content(response)

    # 校验响应有效性
    if (
        not response
        or not response.choices
        or not response.choices[0].message
        or not response.choices[0].message.content
    ):
        raise LLMResponseError(
            detail="response.choices is empty or content is None"
        )

    logger.debug(
        "LLM 调用成功: model=%s, usage=%s",
        model,
        response.usage,
    )
    return response


def check_llm_connection(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """验证 LLM 连通性。

    Args:
        base_url: API base URL
        api_key: API 密钥
        model: 模型名称
        timeout: 请求超时秒数（默认 120s）
    """
    start = time.monotonic()
    try:
        call_llm(
            messages=[{"role": "user", "content": "hi"}],
            model=model,
            temperature=0,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=5,
        )
    except LLMError as e:
        logger.info("LLM 连通性检查失败: %s", e.message)
        return {
            "success": False,
            "error": e.message,
        }

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("LLM 连通性检查成功: latency=%dms", elapsed_ms)
    return {
        "success": True,
        "latency_ms": elapsed_ms,
    }
