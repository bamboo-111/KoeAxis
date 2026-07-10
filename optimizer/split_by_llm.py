"""LLM 辅助文本断句 — 移植自 VideoCaptioner app/core/split/split_by_llm.py

使用 LLM 在文本中插入 <br> 标签进行断句。
包含 agent loop：断句 → 验证（内容一致性）→ 反馈 → 重试。

关键改动（相对上游）：
- 导入路径改为 backend.*
- split_by_llm 签名增加 base_url / api_key / timeout，透传给 call_llm
- logger 前缀 optimizer.split_llm

修改：
  1. split_by_llm 不再静默 fallback，失败时抛出 RuntimeError
  2. _split_with_agent_loop 中 call_llm 加入退避重试
  3. prompt 中注入缩减后的字数限制（×PROMPT_LIMIT_RATIO），引导 LLM 产出合理长度
  4. 验证只检查内容一致性，不检查长度 — 长度控制由 splitter 层在匹配回
     时间戳后通过 _split_long_segment 实现，避免纯文本拆分导致匹配失败
  5. 内容一致性验证前对双方做标点归一化，消除断句边界标点误报
  6. split_by_llm / _split_with_agent_loop 接受 timeout 参数透传给 call_llm
"""

import difflib
import logging
import re
import time
from typing import List, Tuple

from optimizer.exceptions import LLMRateLimitError, LLMConnectionError
from optimizer.text_utils import is_mainly_cjk
from optimizer.llm_client import DEFAULT_TIMEOUT, call_llm
from optimizer.prompts import get_prompt

logger = logging.getLogger("optimizer.split_llm")

MAX_STEPS = 3  # Agent loop 最大尝试次数（仅验证内容一致性，通常 1-2 次即过）
DIAGNOSTIC_TEXT_LIMIT = 4000

# 退避参数
RATE_LIMIT_BASE_DELAY = 5.0
CONNECTION_BASE_DELAY = 3.0
MAX_DELAY = 60.0
BACKOFF_FACTOR = 2.0

# Prompt 字数缩减比例：告诉 LLM 的上限 = 实际限制 × 此比例
# 引导 LLM 产出合理长度的分段，即使超标也由下游 splitter 处理
PROMPT_LIMIT_RATIO = 0.8

# 用于内容一致性检查前的标点归一化 —— 这些标点出现在断句边界被吞掉属于正常行为
_BOUNDARY_PUNCTUATION = re.compile(r"[。、，！？,.!?\s]")


def _clip_for_log(text: str, limit: int = DIAGNOSTIC_TEXT_LIMIT) -> str:
    """Limit diagnostic log payloads while preserving exact text content."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}...<omitted {omitted} chars>"


def split_by_llm(
    text: str,
    model: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    base_url: str,
    api_key: str,
    prompt_limit_ratio: float = PROMPT_LIMIT_RATIO,
    disable_thinking: bool = False,
    llm_extra_body: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[str]:
    """使用 LLM 进行文本断句（固定使用句子分段）。

    只验证内容一致性（LLM 是否篡改原文），不验证长度。
    长度控制由调用方（SubtitleSplitter）在匹配回时间戳后处理，
    因为那里有词级时间信息可以精确拆分。

    断句是后续处理的前提，失败时抛出异常而非静默 fallback。

    Args:
        text: 待断句的文本。
        model: LLM 模型名称。
        max_word_count_cjk: 中文最大字符数（传给 prompt 引导 LLM）。
        max_word_count_english: 英文最大单词数（传给 prompt 引导 LLM）。
        base_url: LLM API 基地址。
        api_key: LLM API 密钥。
        prompt_limit_ratio: 注入 prompt 的长度缩减比例。
        disable_thinking: 是否注入 /no_think 指令。
        llm_extra_body: 原样透传到 OpenAI 兼容接口 extra_body 的 JSON。
        timeout: LLM 请求超时秒数（默认 120s）。

    Returns:
        断句后的文本列表。

    Raises:
        RuntimeError: 断句失败时抛出。
    """
    return _split_with_agent_loop(
        text, model, max_word_count_cjk, max_word_count_english,
        base_url, api_key, prompt_limit_ratio, disable_thinking, llm_extra_body, timeout,
    )


# ------------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------------

def _split_with_agent_loop(
    text: str,
    model: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    base_url: str,
    api_key: str,
    prompt_limit_ratio: float = PROMPT_LIMIT_RATIO,
    disable_thinking: bool = False,
    llm_extra_body: dict | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[str]:
    """通过反馈循环进行 LLM 断句，自动验证和修正。

    对 LLM 调用错误（限速、连接、超时）实现退避重试。

    Prompt 中注入的字数限制经过缩减（×PROMPT_LIMIT_RATIO），
    引导 LLM 产出合理长度，但验证时不检查长度。
    长度超标的段由下游 splitter 在匹配回时间戳后拆分。

    Raises:
        RuntimeError: 所有尝试均失败。
    """

    # Prompt 中使用缩减后的限制，引导 LLM 产出合理长度
    prompt_ratio = max(0.1, float(prompt_limit_ratio))
    prompt_cjk_limit = max(1, int(max_word_count_cjk * prompt_ratio))
    prompt_english_limit = max(1, int(max_word_count_english * prompt_ratio))

    system_prompt = get_prompt(
        "split/sentence",
        max_word_count_cjk=prompt_cjk_limit,
        max_word_count_english=prompt_english_limit,
    )

    user_prompt = (
        f"Please use multiple <br> tags to separate the following sentence:\n{text}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    logger.info(
        "断句诊断 | 输入文本 len=%d prompt_cjk_limit=%d prompt_english_limit=%d text=%s",
        len(text),
        prompt_cjk_limit,
        prompt_english_limit,
        _clip_for_log(text),
    )

    last_result: List[str] | None = None
    rate_limit_delay = RATE_LIMIT_BASE_DELAY
    connection_delay = CONNECTION_BASE_DELAY

    for step in range(MAX_STEPS):
        try:
            response = call_llm(
                messages=messages,
                model=model,
                temperature=0.1,
                base_url=base_url,
                api_key=api_key,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            )
        except LLMRateLimitError as e:
            logger.warning(
                "断句 LLM 被限速 (第 %d 次尝试): %s — 等待 %.1f 秒",
                step + 1, e, rate_limit_delay,
            )
            time.sleep(rate_limit_delay)
            rate_limit_delay = min(rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMConnectionError as e:
            logger.warning(
                "断句 LLM 连接/超时错误 (第 %d 次尝试): %s — 等待 %.1f 秒",
                step + 1, e, connection_delay,
            )
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "断句 LLM 调用失败 (第 %d 次尝试): %s", step + 1, e,
            )
            continue

        result_text: str = response.choices[0].message.content or ""
        logger.info(
            "断句诊断 | LLM 原始输出 attempt=%d len=%d text=%s",
            step + 1,
            len(result_text),
            _clip_for_log(result_text),
        )

        # 解析结果
        result_text_cleaned = re.sub(r"\n+", "", result_text)
        split_result = [
            segment.strip()
            for segment in result_text_cleaned.split("<br>")
            if segment.strip()
        ]
        last_result = split_result
        logger.info(
            "断句诊断 | LLM 解析结果 attempt=%d segments=%d merged=%s",
            step + 1,
            len(split_result),
            _clip_for_log(" | ".join(split_result)),
        )

        # 成功收到响应，重置退避
        rate_limit_delay = RATE_LIMIT_BASE_DELAY
        connection_delay = CONNECTION_BASE_DELAY

        # 只验证内容一致性，不验证长度
        is_valid, error_message = _validate_split_result(
            original_text=text,
            split_result=split_result,
        )

        if is_valid:
            logger.info(
                "断句诊断 | 内容一致性通过 attempt=%d segments=%d",
                step + 1,
                len(split_result),
            )
            return split_result

        logger.warning(
            "断句验证失败 (第%d次尝试): %s", step + 1, error_message,
        )
        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {error_message}\n"
                    "Fix the errors above and output the COMPLETE corrected text "
                    "with <br> tags (include ALL segments, not just the fixed ones), "
                    "no explanation."
                ),
            }
        )

    if last_result:
        logger.warning("断句达到最大尝试次数 (%d)，返回最后结果", MAX_STEPS)
        return last_result

    raise RuntimeError(
        f"断句失败：{MAX_STEPS} 次尝试均未获得有效结果"
    )


# ------------------------------------------------------------------
# 验证（仅内容一致性）
# ------------------------------------------------------------------

def _normalize_for_comparison(text: str) -> str:
    """对文本做标点归一化，用于内容一致性比较。

    断句时 LLM 经常在标点边界处"吞掉"句末标点（如 。、，），
    这是正常的断句行为而非内容篡改。移除这些标点后再比较，
    可以消除大量误报。
    """
    return _BOUNDARY_PUNCTUATION.sub("", text)


def _validate_split_result(
    original_text: str,
    split_result: List[str],
) -> Tuple[bool, str]:
    """验证断句结果的内容一致性。

    不检查长度 — 长度控制由 splitter 层在匹配回时间戳后，
    通过 _split_long_segment 基于时间间隔精确拆分。

    Returns:
        (is_valid, error_feedback)
    """
    if not split_result:
        return False, "No segments found. Split the text with <br> tags."

    # --- 内容一致性检查 ---
    original_cleaned = re.sub(r"\s+", " ", original_text)
    text_is_cjk = is_mainly_cjk(original_cleaned)

    merged_char = "" if text_is_cjk else " "
    merged = merged_char.join(split_result)
    merged_cleaned = re.sub(r"\s+", " ", merged)

    # 标点归一化后比较，消除断句边界标点误报
    original_normalized = _normalize_for_comparison(original_cleaned)
    merged_normalized = _normalize_for_comparison(merged_cleaned)

    matcher = difflib.SequenceMatcher(None, original_normalized, merged_normalized)
    similarity_ratio = matcher.ratio()

    if similarity_ratio < 0.96:
        # 用归一化后的文本生成 diff 反馈
        differences: list[str] = []
        context_size = 5 if text_is_cjk else 20

        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "replace":
                before = original_normalized[max(0, a0 - context_size) : a0]
                orig_part = original_normalized[a0:a1]
                after = original_normalized[a1 : a1 + context_size]
                new_part = merged_normalized[b0:b1]
                if orig_part.isspace() or new_part.isspace():
                    continue
                differences.append(
                    f"...{before}[{orig_part}]{after}... → changed to [{new_part}]"
                )
            elif opcode == "delete":
                before = original_normalized[max(0, a0 - context_size) : a0]
                deleted_part = original_normalized[a0:a1]
                after = original_normalized[a1 : a1 + context_size]
                if deleted_part.isspace():
                    continue
                differences.append(
                    f"...{before}[{deleted_part}]{after}... → deleted"
                )
            elif opcode == "insert":
                before = merged_normalized[max(0, b0 - context_size) : b0]
                inserted_part = merged_normalized[b0:b1]
                after = merged_normalized[b1 : b1 + context_size]
                if inserted_part.isspace():
                    continue
                differences.append(
                    f"Wrongly inserted [{inserted_part}] between "
                    f"'...{before}' and '{after}...'"
                )

        if differences:
            error_msg = f"Content modified (similarity: {similarity_ratio:.1%}):\n"
            error_msg += "\n".join(f"- {diff}" for diff in differences)
            error_msg += (
                "\nKeep original text unchanged, only insert <br> between words."
            )
            return False, error_msg

    return True, ""
