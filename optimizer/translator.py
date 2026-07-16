"""字幕翻译模块

新建模块（非移植），仅使用 LLM 进行翻译。
Voxlign 不使用 VideoCaptioner 的 Bing/DeepLx 等翻译服务。

设计模式参考 optimizer.py：
- ThreadPoolExecutor 并发批量处理
- json_repair 解析 LLM 输出
- 异常由 llm_client.py 抛出，translator 不自己构造 LLMError 子类
- 单批翻译失败时保留原文，translate() 允许部分批次失败
- logger 前缀 optimizer.translator

修改：
  1. _agent_loop 中对 LLMRateLimitError 和 LLMConnectionError 均做指数退避
  2. _parallel_translate 中添加提交间隔，避免瞬时并发全部打满
  3. 429/暂时性错误后通过 threading.Event 实现全局冷却
  4. MAX_STEPS 从 3 提升到 5，给退避后重试留出足够机会
  5. __init__ 接受 timeout 参数，透传给 call_llm
"""

import json
import logging
import re
import threading
import time
from concurrent.futures import (  # pylint: disable=no-name-in-module
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any, Callable, Dict, List, Optional

import json_repair

from optimizer.asr_data import ASRData, ASRDataSeg
from optimizer.exceptions import LLMRateLimitError, LLMConnectionError
from optimizer.llm_config import LLMConfig
from optimizer.llm_client import DEFAULT_TIMEOUT, call_llm
from optimizer.prompts import get_prompt

logger = logging.getLogger("optimizer.translator")

MAX_STEPS = 5

# 退避参数
RATE_LIMIT_BASE_DELAY = 5.0      # 首次 429 等待秒数
CONNECTION_BASE_DELAY = 3.0      # 首次连接/上游错误等待秒数
MAX_DELAY = 60.0                 # 最大等待秒数
BACKOFF_FACTOR = 2.0             # 指数退避因子

# 全局冷却参数
GLOBAL_COOLDOWN_SECONDS = 10.0   # 收到 429 后全局冷却秒数
SUBMIT_INTERVAL = 0.5            # 批次提交间隔秒数（避免瞬时并发）

_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_DIGIT_RE = re.compile(r"\d+")
_LATIN_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_KATAKANA_ENTITY_RE = re.compile(r"[\u30a1-\u30ff]{3,}")
_NAME_SUFFIX_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]{2,}(?:\u3055\u3093|\u541b|\u3061\u3083\u3093|\u5148\u751f)")
_QUESTION_SOURCE_RE = re.compile(r"[?？]\s*$|(?:\u304b|\u306e)\s*[?？]?\s*$")
_QUESTION_TARGET_RE = re.compile(r"[?？]|[\u5417\u5462\u5427\u4e48]\s*$")
_NEGATION_SOURCE_RE = re.compile(
    r"\u306a\u3044|\u307e\u305b\u3093|\u306c|\u3058\u3083\u306a\u3044|"
    r"\u3067\u306f\u306a\u3044|\u5acc|\u30c0\u30e1|\u3060\u3081"
)
_NEGATION_TARGET_RE = re.compile(r"\u4e0d|\u6ca1|\u7121|\u65e0|\u522b|\u975e|\u5426|\u672a")
_SHORT_RESPONSE_NORMALIZED = {
    "\u306f\u3044",
    "\u3046\u3093",
    "\u3048\u3048",
    "\u3044\u3044\u3048",
    "\u3044\u3084",
    "\u3046\u3046\u3093",
    "\u3060\u3081",
    "\u30c0\u30e1",
}
_FRAGMENT_NORMALIZED = {
    "\u3042",
    "\u3048",
    "\u3093",
    "\u3042\u306e",
    "\u305d\u306e",
    "\u3067\u3082",
    "\u3058\u3083\u3042",
    "\u3051\u3069",
    "\u3063\u3066",
}


class SubtitleTranslator:
    """字幕翻译器

    使用 LLM 将字幕翻译为目标语言。支持：
    - 并发批量翻译
    - Agent loop 验证和修正
    - 部分失败容忍（保留原文）
    - 429/暂时性错误指数退避和全局冷却
    """

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        model: str,
        base_url: str,
        api_key: str,
        target_language: str,
        custom_prompt: str = "",
        disable_thinking: bool = True,
        llm_extra_body: Optional[dict] = None,
        timeout: float = DEFAULT_TIMEOUT,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """初始化翻译器。

        Args:
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            model: LLM 模型名称
            base_url: LLM API 基地址
            api_key: LLM API 密钥
            target_language: 目标语言（如 "简体中文"、"English"）
            custom_prompt: 自定义翻译提示词（术语表等）
            disable_thinking: 是否注入 /no_think 指令
            llm_extra_body: 原样透传到 OpenAI 兼容接口 extra_body 的 JSON
            timeout: LLM 请求超时秒数（默认 120s）
        """
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.llm_config = LLMConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        self.target_language = target_language
        self.custom_prompt = custom_prompt
        self.disable_thinking = disable_thinking
        self.llm_extra_body = llm_extra_body
        self.timeout = timeout
        self.progress_callback = progress_callback

        self.is_running = True
        self.executor: Optional[ThreadPoolExecutor] = None

        # 全局冷却：收到 429 时所有线程暂停
        self._cooldown_until: float = 0.0
        self._cooldown_lock = threading.Lock()

        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        """初始化线程池。"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)

    def _set_global_cooldown(self, seconds: float) -> None:
        """设置全局冷却截止时间。

        多个线程可能同时遇到 429，取最远的截止时间。
        """
        with self._cooldown_lock:
            deadline = time.monotonic() + seconds
            if deadline > self._cooldown_until:
                self._cooldown_until = deadline
                logger.info(
                    "全局冷却已设置：%.1f 秒后恢复请求", seconds,
                )

    def _wait_for_cooldown(self) -> None:
        """如果处于全局冷却期，等待直到冷却结束。"""
        while self.is_running:
            with self._cooldown_lock:
                remaining = self._cooldown_until - time.monotonic()
            if remaining <= 0:
                return
            # 分段等待，以便及时响应 stop()
            time.sleep(min(remaining, 1.0))

    def translate(self, asr_data: ASRData) -> ASRData:
        """翻译字幕（主入口）。

        将所有 segment 分批并发翻译，翻译结果写入
        各 segment 的 translated_text 字段。

        允许部分批次失败（失败的批次保留原文，translated_text 不修改）。
        全部批次失败时 raise RuntimeError。

        Args:
            asr_data: 待翻译的 ASRData

        Returns:
            修改后的 ASRData（原地修改 segments）

        Raises:
            RuntimeError: 全部批次翻译失败
        """
        if not asr_data.segments:
            return asr_data

        # 分批
        chunks = self._split_chunks(asr_data.segments)

        # 并发翻译
        success_count = self._parallel_translate(chunks, len(asr_data.segments))

        if success_count == 0:
            raise RuntimeError(
                f"翻译失败：全部 {len(chunks)} 个批次均失败"
            )

        logger.info(
            "翻译完成：%d/%d 批次成功", success_count, len(chunks),
        )
        return asr_data

    def _split_chunks(
        self, segments: List[ASRDataSeg],
    ) -> List[List[ASRDataSeg]]:
        """将 segments 列表分割成批次。

        Args:
            segments: 字幕片段列表

        Returns:
            批次列表，每批包含对应的 segments 引用
        """
        return [
            segments[i : i + self.batch_num]
            for i in range(0, len(segments), self.batch_num)
        ]

    def _parallel_translate(
        self, chunks: List[List[ASRDataSeg]],
        total_segments: int,
    ) -> int:
        """并发翻译所有批次。

        使用 as_completed 收集结果，避免按提交顺序阻塞。
        批次之间添加提交间隔，避免瞬时并发打满 API 限制。

        Args:
            chunks: 分批后的 segments 列表

        Returns:
            成功翻译的批次数
        """
        if not self.executor:
            raise ValueError("线程池未初始化")

        future_to_chunk = {}
        for idx, chunk in enumerate(chunks):
            if not self.is_running:
                break
            future = self.executor.submit(self._translate_chunk, chunk)
            future_to_chunk[future] = chunk
            # 批次提交间隔：避免所有请求同时发出
            if idx < len(chunks) - 1:
                time.sleep(SUBMIT_INTERVAL)

        success_count = 0
        processed_segments = 0
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            current = f"{chunk[0].start_time}-{chunk[-1].end_time}"
            try:
                future.result()
                success_count += 1
                processed_segments += len(chunk)
                self._emit_progress(processed_segments, total_segments, current)
            except Exception as e:  # pylint: disable=broad-exception-caught
                processed_segments += len(chunk)
                self._emit_progress(processed_segments, total_segments, f"failed {current}")
                logger.error(
                    "翻译批次失败（segments %d-%d）：%s",
                    chunk[0].start_time,
                    chunk[-1].end_time,
                    e,
                )

        return success_count

    def _emit_progress(self, done: int, total: int, current: str) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(done, total, current)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("翻译进度回调失败: %s", e)

    def _translate_chunk(self, chunk: List[ASRDataSeg]) -> None:
        """翻译单个字幕批次。

        使用 agent loop 进行翻译验证和修正。
        翻译结果直接写入各 segment 的 translated_text。

        Args:
            chunk: 字幕片段列表（引用，原地修改）

        Raises:
            RuntimeError: 翻译失败
        """
        # 构建输入字典 {"1": "text1", "2": "text2", ...}
        subtitle_dict: Dict[str, str] = {
            str(i): seg.text for i, seg in enumerate(chunk, 1)
        }

        # 日志用 segment 实际时间范围（毫秒→秒）更直观
        time_start = chunk[0].start_time / 1000
        time_end = chunk[-1].end_time / 1000
        logger.info(
            "[+]正在翻译字幕：%d 条 (%.1fs - %.1fs)",
            len(chunk), time_start, time_end,
        )

        # 获取翻译 prompt
        system_prompt = get_prompt(
            "translate/standard",
            target_language=self.target_language,
            custom_prompt=self.custom_prompt,
        )

        # Agent loop
        result_dict = self._agent_loop(system_prompt, subtitle_dict)

        if result_dict is None:
            raise RuntimeError(
                "翻译失败：agent loop 未返回有效结果"
            )

        # 将翻译结果写入 segments。兼容旧格式 {"1": "译文"} 和
        # 新格式 {"1": {"translation": "译文", ...疑点字段...}}。
        for i, seg in enumerate(chunk, 1):
            translated, suspect_meta = _normalize_translation_payload(result_dict.get(str(i)))
            if translated is not None:
                seg.translated_text = translated
            suspect_meta = _augment_translation_suspects(seg, translated or "", suspect_meta)
            for key, value in suspect_meta.items():
                setattr(seg, key, value)

    def _agent_loop(
        self,
        system_prompt: str,
        subtitle_dict: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Agent loop 翻译字幕批次。

        LLM → 验证 → 反馈 → 重试（最多 MAX_STEPS 次）
        对 429 和暂时性连接错误均实现指数退避 + 全局冷却。

        Args:
            system_prompt: 系统提示词
            subtitle_dict: 待翻译的字幕字典

        Returns:
            翻译结果字典，全部失败时返回 None
        """
        user_content = (
            "<input_subtitle>"
            f"{json.dumps(subtitle_dict, ensure_ascii=False)}"
            "</input_subtitle>"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_result: Optional[Dict[str, Any]] = None
        rate_limit_delay = RATE_LIMIT_BASE_DELAY
        connection_delay = CONNECTION_BASE_DELAY

        for step in range(MAX_STEPS):
            if not self.is_running:
                break

            # 等待全局冷却期结束
            self._wait_for_cooldown()

            if not self.is_running:
                break

            try:
                response = call_llm(
                    messages=messages,
                    model=self.llm_config.model,
                    temperature=0.3,
                    base_url=self.llm_config.base_url,
                    api_key=self.llm_config.api_key,
                    disable_thinking=self.llm_config.disable_thinking,
                    require_json=True,
                    llm_extra_body=self.llm_config.llm_extra_body,
                    timeout=self.llm_config.timeout,
                )
            except LLMRateLimitError as e:
                # 429：指数退避 + 全局冷却
                logger.warning(
                    "LLM 调用被限速 (第 %d 次尝试): %s — 等待 %.1f 秒后重试",
                    step + 1, e, rate_limit_delay,
                )
                self._set_global_cooldown(rate_limit_delay)
                self._sleep_interruptible(rate_limit_delay)
                rate_limit_delay = min(
                    rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY,
                )
                continue
            except LLMConnectionError as e:
                # 400/5xx 上游暂时性错误：退避等待后重试
                logger.warning(
                    "LLM 连接/上游错误 (第 %d 次尝试): %s — 等待 %.1f 秒后重试",
                    step + 1, e, connection_delay,
                )
                self._sleep_interruptible(connection_delay)
                connection_delay = min(
                    connection_delay * BACKOFF_FACTOR, MAX_DELAY,
                )
                continue
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "LLM 调用失败 (第 %d 次尝试): %s", step + 1, e,
                )
                continue

            result_text = response.choices[0].message.content
            if not result_text:
                logger.warning(
                    "LLM 返回空结果 (第 %d 次尝试)", step + 1,
                )
                continue

            # 解析结果
            parsed = json_repair.loads(result_text.strip())
            if not isinstance(parsed, dict):
                logger.warning(
                    "LLM 返回非 dict (第 %d 次尝试): %s",
                    step + 1,
                    type(parsed).__name__,
                )
                messages.append(
                    {"role": "assistant", "content": result_text},
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Output must be a JSON dictionary, got "
                            f"{type(parsed).__name__}. Use format: "
                            '{"1": "translated text", "2": "..."}'
                        ),
                    }
                )
                continue

            result_dict: Dict[str, Any] = {str(k): v for k, v in parsed.items()}
            last_result = result_dict

            # 成功收到有效响应，重置退避计时器
            rate_limit_delay = RATE_LIMIT_BASE_DELAY
            connection_delay = CONNECTION_BASE_DELAY

            # 验证：检查键匹配
            is_valid, error_message = self._validate_translation_result(
                subtitle_dict, result_dict,
            )

            if is_valid:
                return result_dict

            # 验证失败，添加反馈
            logger.warning(
                "翻译验证失败 (第 %d 次尝试): %s",
                step + 1,
                error_message,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        result_dict, ensure_ascii=False,
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {error_message}\n\n"
                        "Fix the errors above and output ONLY a "
                        "valid JSON dictionary with ALL "
                        f"{len(subtitle_dict)} keys."
                    ),
                }
            )

        # 达到最大步数
        if last_result:
            logger.warning(
                "达到最大尝试次数 (%d)，返回最后结果", MAX_STEPS,
            )
        return last_result

    def _sleep_interruptible(self, seconds: float) -> None:
        """可中断的 sleep：每秒检查 is_running 标志。

        Args:
            seconds: 总等待秒数
        """
        end_time = time.monotonic() + seconds
        while self.is_running:
            remaining = end_time - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    @staticmethod
    def _validate_translation_result(
        original_dict: Dict[str, str],
        translated_dict: Dict[str, Any],
    ) -> tuple[bool, str]:
        """验证翻译结果。

        仅检查键匹配。翻译内容的质量由 LLM 自身保证，
        不做相似度检查（翻译后的文本与原文语言不同）。

        Args:
            original_dict: 原始字幕字典
            translated_dict: 翻译后字幕字典

        Returns:
            (是否有效, 错误反馈)
        """
        expected_keys = set(original_dict.keys())
        actual_keys = set(translated_dict.keys())

        error_parts: list[str] = []
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            if missing:
                error_parts.append(
                    f"Missing keys {sorted(missing)} — "
                    "you must translate these items"
                )
            if extra:
                error_parts.append(
                    f"Extra keys {sorted(extra)} — "
                    "these keys are not in input, remove them"
                )

        for key in expected_keys & actual_keys:
            translated, _ = _normalize_translation_payload(translated_dict.get(key))
            if translated is None or not translated.strip():
                error_parts.append(f"{key} is missing translation")

        return (not error_parts), "; ".join(error_parts)

    def stop(self) -> None:
        """停止翻译器并清理资源。

        可安全多次调用。
        """
        if not self.is_running:
            return

        self.is_running = False

        if self.executor:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            finally:
                self.executor = None


def _normalize_translation_payload(value: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(value, dict):
        translated = value.get("translation")
        if translated is None:
            translated = value.get("translated_text")
        if translated is None:
            translated = value.get("translated_subtitle")
        meta = {
            "asr_suspect": _coerce_bool(value.get("asr_suspect")),
            "needs_audio_review": _coerce_bool(value.get("needs_audio_review")),
            "suspect_types": _coerce_string_list(value.get("suspect_types")),
            "suspect_reason": str(value.get("reason", "")).strip(),
            "suspect_confidence": _coerce_confidence(value.get("confidence")),
        }
        return (str(translated).strip() if translated is not None else None), meta
    if value is None:
        return None, {}
    return str(value).strip(), {}


def _augment_translation_suspects(
    segment: ASRDataSeg,
    translated: str,
    suspect_meta: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(suspect_meta)
    existing_types = _coerce_string_list(meta.get("suspect_types"))
    rule_hits = _detect_translation_suspect_types(segment, translated)
    merged_types = list(dict.fromkeys(existing_types + [hit[0] for hit in rule_hits]))
    if merged_types:
        meta["asr_suspect"] = _coerce_bool(meta.get("asr_suspect")) or bool(rule_hits)
        meta["needs_audio_review"] = _coerce_bool(meta.get("needs_audio_review")) or bool(rule_hits)
        meta["suspect_types"] = merged_types
        reasons = [str(meta.get("suspect_reason", "")).strip()]
        reasons.extend(hit[1] for hit in rule_hits)
        meta["suspect_reason"] = "; ".join(dict.fromkeys(reason for reason in reasons if reason))
        confidence = _coerce_confidence(meta.get("suspect_confidence", 1.0))
        if rule_hits:
            confidence = min(confidence, min(hit[2] for hit in rule_hits))
        meta["suspect_confidence"] = confidence
    else:
        meta["asr_suspect"] = _coerce_bool(meta.get("asr_suspect"))
        meta["needs_audio_review"] = _coerce_bool(meta.get("needs_audio_review"))
        meta["suspect_types"] = []
        meta["suspect_reason"] = str(meta.get("suspect_reason", "")).strip()
        meta["suspect_confidence"] = _coerce_confidence(meta.get("suspect_confidence", 1.0))
    return meta


def _detect_translation_suspect_types(
    segment: ASRDataSeg,
    translated: str,
) -> list[tuple[str, str, float]]:
    source = str(segment.text or "").strip()
    target = str(translated or "").strip()
    source_norm = _normalize_signal_text(source)
    target_norm = _normalize_signal_text(target)
    hits: list[tuple[str, str, float]] = []

    if not source_norm:
        return hits

    if source_norm in _SHORT_RESPONSE_NORMALIZED:
        hits.append(("short_response", "short response needs audio confirmation", 0.72))

    if source_norm in _FRAGMENT_NORMALIZED or source.endswith(("\u2026", "...")):
        hits.append(("fragment", "source looks fragmentary", 0.68))

    if target_norm and (source_norm == target_norm or _HIRAGANA_KATAKANA_RE.search(target)):
        hits.append(("untranslated", "translation appears to contain source-language text", 0.55))

    if _QUESTION_SOURCE_RE.search(source) and target and not _QUESTION_TARGET_RE.search(target):
        hits.append(("question", "source looks interrogative but translation is not marked as a question", 0.70))

    if _NEGATION_SOURCE_RE.search(source) and target and not _NEGATION_TARGET_RE.search(target):
        hits.append(("negation", "source contains negation that is not obvious in translation", 0.68))

    source_digits = _DIGIT_RE.findall(source)
    target_digits = _DIGIT_RE.findall(target)
    if source_digits and source_digits != target_digits:
        hits.append(("quantity", "numeric content differs between source and translation", 0.66))

    if _NAME_SUFFIX_RE.search(source):
        hits.append(("name", "source contains a name-like expression", 0.74))
    elif _KATAKANA_ENTITY_RE.search(source) or _LATIN_ENTITY_RE.search(source):
        hits.append(("entity", "source contains an entity-like expression", 0.76))

    if target and _content_looks_dropped(source, target):
        hits.append(("content_conservation", "translation may have dropped source content", 0.58))

    duration_ms = int(getattr(segment, "end_time", 0) or 0) - int(getattr(segment, "start_time", 0) or 0)
    if 0 < duration_ms <= 350 and len(source_norm) >= 5:
        hits.append(("time", "source is dense for a very short timing span", 0.62))

    if source_norm.startswith(("\u3067\u3082", "\u3060\u304b\u3089", "\u3058\u3083\u3042", "\u305d\u308c\u3067")):
        hits.append(("context_linkage", "source depends on adjacent context", 0.72))

    return list(dict.fromkeys(hits))


def _normalize_signal_text(value: str) -> str:
    return re.sub(r"[\s\u3000、。，,.!?！？\"'`~\-\u2014\u2026]+", "", value)


def _content_looks_dropped(source: str, target: str) -> bool:
    source_units = _signal_unit_count(source)
    target_units = _signal_unit_count(target)
    return source_units >= 8 and target_units <= max(1, source_units // 5)


def _signal_unit_count(value: str) -> int:
    cjk_count = len(_CJK_RE.findall(value))
    kana_count = len(_HIRAGANA_KATAKANA_RE.findall(value))
    latin_count = sum(len(match.group(0)) for match in _LATIN_ENTITY_RE.finditer(value))
    digit_count = sum(len(match.group(0)) for match in _DIGIT_RE.finditer(value))
    return cjk_count + kana_count + latin_count + digit_count


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, confidence))
