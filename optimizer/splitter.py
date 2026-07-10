"""字幕智能分割器 — 移植自 VideoCaptioner app/core/split/split.py

支持两种模式:
  1. LLM 断句 — 并发调用 LLM 进行语义断句，自动验证和修复
  2. 纯规则断句 — 基于时间间隔、常见词、超长段拆分的降级方案

LLM 调用失败或匹配不完整时自动降级为规则模式。

关键改动（相对上游）：
- 导入路径全部改为 backend.*
- __init__ 新增 base_url / api_key 参数
- 移除 atexit.register，提供显式 stop()
- split_subtitle 中字符串输入改为 ASRData.from_srt()
- 使用 asr_data.to_txt() 替代已删除的 _segments_to_txt()
- logger 前缀 optimizer.splitter
- 长度控制统一在 _process_by_llm 层面（匹配回时间戳后），
  split_by_llm 只负责语义断句和内容一致性验证
- 匹配时对双方做标点归一化，提高 LLM 删标点场景下的匹配成功率
- 匹配失败时不偏移 asr_index，避免级联失配；未覆盖区间局部回退到规则分割
"""

import difflib
import logging
import re
from concurrent.futures import (  # pylint: disable=no-name-in-module
    ThreadPoolExecutor,
    as_completed,
)
from typing import List, Union

from optimizer.asr_data import ASRData, ASRDataSeg
from optimizer.split_by_llm import split_by_llm
from optimizer.llm_client import DEFAULT_TIMEOUT
from optimizer.text_utils import (
    count_words,
    is_mainly_cjk,
    is_pure_punctuation,
    is_space_separated_language,
)

logger = logging.getLogger("optimizer.splitter")

DIAGNOSTIC_TEXT_LIMIT = 2000


def _clip_for_log(text: str, limit: int = DIAGNOSTIC_TEXT_LIMIT) -> str:
    """Limit diagnostic log payloads while preserving exact text content."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}...<omitted {omitted} chars>"

# ==================== 配置常量 ====================

# 字数限制
MAX_WORD_COUNT_CJK = 25      # CJK 文本单行最大字数
MAX_WORD_COUNT_ENGLISH = 18   # 英文文本单行最大单词数

# 分段阈值
SEGMENT_WORD_THRESHOLD = 300  # 长文本分段阈值(字数)

# 时间间隔
MAX_GAP = 1500            # 允许的最大时间间隔(毫秒)
MERGE_SHORT_GAP = 200     # 短分段合并时间阈值(毫秒)
MERGE_VERY_SHORT_GAP = 500   # 极短分段合并时间阈值(毫秒)

# 短分段合并阈值
MERGE_MIN_WORDS = 5           # 短分段最小字数阈值
MERGE_VERY_SHORT_WORDS = 3    # 极短分段字数阈值

# 分割相关
SPLIT_SEARCH_RANGE = 30       # 分割点前后搜索范围
TIME_GAP_WINDOW_SIZE = 5      # 时间间隔窗口大小
TIME_GAP_MULTIPLIER = 3       # 大间隔判断倍数
MIN_GROUP_SIZE = 5            # 最小分组大小

# 规则分割
RULE_SPLIT_GAP = 500          # 规则分割时间间隔阈值(毫秒)
RULE_MIN_SEGMENT_SIZE = 4     # 规则分割最小分段大小

# 常见词分割
PREFIX_WORD_RATIO = 0.6       # 前缀词分割比例
SUFFIX_WORD_RATIO = 0.4       # 后缀词分割比例

# 匹配相关
MATCH_SIMILARITY_THRESHOLD = 0.5   # 文本匹配相似度阈值
MATCH_MAX_SHIFT = 30               # 匹配滑动窗口最大偏移
MATCH_LARGE_SHIFT = 100            # 未匹配时的大偏移量
MATCH_NEXT_OVERLAP_PENALTY = 0.35  # 候选窗口侵占下一句时的评分惩罚
MATCH_LENGTH_PENALTY = 0.04        # 窗口归一化长度偏差惩罚
MATCH_START_SHIFT_PENALTY = 0.002  # 起点远离当前索引的惩罚
FILLER_MERGE_MAX_GAP = 220         # 短语气词兜底合并最大间隔(毫秒)
FILLER_MERGE_MAX_CJK = 24          # 短语气词合并后最大 CJK 字数
READABILITY_MERGE_MAX_GAP = 260    # 短片段可读性合并最大间隔(毫秒)
READABILITY_MERGE_MAX_CJK = 32     # 短片段合并后最大 CJK 字数
READABILITY_MIN_DURATION = 1200    # 低于此时长的短片段优先合并(毫秒)
READABILITY_ZERO_DURATION = 100    # 近零时长片段强制尝试合并(毫秒)

# 用于匹配时的标点归一化 — LLM 断句常删标点，归一化后比较提高匹配率
_MATCH_PUNCTUATION = re.compile(r"[。、，！？,.!?;:；：・\s]")
_FILLER_ONLY_TOKENS = (
    "あの",
    "まあ",
    "ね",
    "よ",
    "さ",
)
_DIALOGUE_STANDALONE_RESPONSES = {
    "はい",
    "はいはい",
    "うん",
    "ううん",
    "ええ",
    "いや",
    "いいえ",
    "そう",
    "そうそう",
    "ああ",
    "え",
    "あ",
    "なに",
    "何",
    "なんで",
    "どうして",
}
_SHORT_FILLER_PREFIXES = ("ね", "え", "うん")
_READABILITY_SUFFIXES = (
    "です",
    "でした",
    "ます",
    "ました",
    "でした",
    "けど",
    "けれど",
    "けれども",
    "から",
    "ので",
    "のに",
    "し",
    "て",
    "って",
    "という",
    "とか",
    "など",
    "な",
    "ね",
    "よ",
    "わ",
    "か",
    "に",
    "で",
    "と",
    "が",
    "を",
    "は",
    "も",
    "の",
    "へ",
    "から",
    "ので",
    "こと",
)
_READABILITY_PREFIXES = (
    "ということで",
    "そして",
    "それで",
    "だから",
    "あと",
    "でも",
    "じゃあ",
    "ただ",
    "また",
    "なので",
    "すると",
    "ところで",
    "ちなみに",
)
_READABILITY_STANDALONE_WEAK = (
    "ございます",
    "よろしく",
    "お願いします",
)
_NUMERIC_UNIT_FRAGMENTS = {
    "度",
    "日",
    "月",
    "年",
    "時",
    "分",
    "秒",
    "個",
    "回",
    "人",
    "枚",
    "本",
}
_READABILITY_TRAILING_FRAGMENTS = (
    "んです",
    "んですが",
    "でしたが",
    "ますが",
    "けど",
    "けれど",
    "けれども",
    "から",
    "ので",
    "のに",
    "し",
    "て",
    "って",
    "という",
    "とか",
)
_READABILITY_LEADING_FRAGMENTS = (
    "そして",
    "それで",
    "だから",
    "でも",
    "じゃあ",
    "ただ",
    "また",
    "なので",
    "すると",
    "ところで",
    "ちなみに",
)


# ==================== 模块级预处理函数 ====================

def preprocess_segments(
    segments: List[ASRDataSeg], need_lower: bool = True
) -> List[ASRDataSeg]:
    """预处理 ASR 分段。

    1. 移除纯标点符号的分段
    2. 为需要空格分隔的语言添加尾随空格（英语、俄语等，不包括 CJK）

    Args:
        segments: ASR 数据分段列表。
        need_lower: 是否转小写（仅对拉丁/西里尔字母有效）。

    Returns:
        处理后的分段列表。
    """
    new_segments: List[ASRDataSeg] = []
    for seg in segments:
        if not is_pure_punctuation(seg.text):
            text = seg.text.strip()
            if is_space_separated_language(text):
                if need_lower:
                    text = text.lower()
                seg.text = text + " "
            new_segments.append(seg)
    return new_segments


def _next_sentence_overlap(candidate: str, next_sentence: str) -> int:
    """Return length of next-sentence prefix consumed at candidate tail."""
    if not candidate or len(next_sentence) < 3:
        return 0

    max_size = min(len(candidate), len(next_sentence))
    for size in range(max_size, 1, -1):
        if next_sentence[:size] in candidate:
            return size
    if candidate.endswith(next_sentence[:1]):
        return 1
    return 0


def _is_better_match(
    *,
    score: float,
    ratio: float,
    start: int,
    window_size: int,
    next_overlap: int,
    best_score: float,
    best_ratio: float,
    best_start: int | None,
    best_window_size: int,
    best_next_overlap: int,
) -> bool:
    """Compare candidate matches with deterministic, boundary-aware tie-breaks."""
    if best_start is None:
        return True

    epsilon = 1e-9
    if score > best_score + epsilon:
        return True
    if score < best_score - epsilon:
        return False

    if ratio > best_ratio + epsilon:
        return True
    if ratio < best_ratio - epsilon:
        return False

    if next_overlap != best_next_overlap:
        return next_overlap < best_next_overlap
    if start != best_start:
        return start < best_start
    return window_size < best_window_size


def _normalize_filler_text(text: str) -> str:
    """Normalize a candidate subtitle fragment for filler-word checks."""
    return _MATCH_PUNCTUATION.sub("", text.strip())


def _is_filler_only(text: str) -> bool:
    """Return whether text is only short spoken filler particles."""
    normalized = _normalize_filler_text(text)
    if _is_dialogue_standalone_response(normalized):
        return False
    if not normalized or count_words(normalized) > 6:
        return False

    remaining = normalized
    tokens = sorted(_FILLER_ONLY_TOKENS, key=len, reverse=True)
    while remaining:
        for token in tokens:
            if remaining.startswith(token):
                remaining = remaining[len(token):]
                break
        else:
            return False
    return True


def _is_dialogue_standalone_response(text: str) -> bool:
    """Return whether a short fragment is likely a separate dialogue turn."""
    normalized = _normalize_filler_text(text)
    return normalized in _DIALOGUE_STANDALONE_RESPONSES


def _starts_with_short_filler(text: str) -> bool:
    """Return whether a very short fragment starts with a filler particle."""
    normalized = _normalize_filler_text(text)
    if (
        not normalized
        or _is_filler_only(normalized)
        or count_words(normalized) > 4
    ):
        return False
    return any(normalized.startswith(prefix) for prefix in _SHORT_FILLER_PREFIXES)


def _can_merge_filler(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    """Check whether merging adjacent filler-related fragments is conservative."""
    if (
        _is_dialogue_standalone_response(left.text)
        or _is_dialogue_standalone_response(right.text)
    ):
        return False
    gap = max(0, right.start_time - left.end_time)
    merged_text = f"{left.text}{right.text}"
    return (
        gap <= FILLER_MERGE_MAX_GAP
        and count_words(merged_text) <= FILLER_MERGE_MAX_CJK
    )


def _segment_duration(seg: ASRDataSeg) -> int:
    """Return non-negative segment duration in milliseconds."""
    return max(0, seg.end_time - seg.start_time)


def _is_numeric_fragment(text: str) -> bool:
    """Return whether a fragment is part of a compact numeric expression."""
    normalized = _normalize_filler_text(text)
    return bool(normalized) and (
        normalized.isdigit() or normalized in _NUMERIC_UNIT_FRAGMENTS
    )


def _is_readability_short(seg: ASRDataSeg) -> bool:
    """Return whether a segment is a structural fragment, not merely short."""
    text = _normalize_filler_text(seg.text)
    if _is_dialogue_standalone_response(text):
        return False
    duration = _segment_duration(seg)
    return (
        duration <= READABILITY_ZERO_DURATION
        or text in _READABILITY_STANDALONE_WEAK
        or text in _READABILITY_SUFFIXES
        or text in _READABILITY_PREFIXES
        or text.endswith(_READABILITY_TRAILING_FRAGMENTS)
        or text.startswith(_READABILITY_LEADING_FRAGMENTS)
        or _is_numeric_fragment(text)
    )


def _is_protected_short_utterance(text: str) -> bool:
    """Protect short complete utterances from readability smoothing."""
    normalized = _normalize_filler_text(text)
    if not normalized or count_words(normalized) > 4:
        return False
    if _is_numeric_fragment(normalized) or _is_filler_only(normalized):
        return False
    if normalized in _READABILITY_STANDALONE_WEAK:
        return False
    if normalized in _READABILITY_SUFFIXES or normalized in _READABILITY_PREFIXES:
        return False
    if normalized.endswith(_READABILITY_TRAILING_FRAGMENTS):
        return False
    if normalized.startswith(_READABILITY_LEADING_FRAGMENTS):
        return False
    return True


def _can_merge_readability(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    """Check whether merging short display fragments keeps subtitle size sane."""
    if (
        _is_dialogue_standalone_response(left.text)
        or _is_dialogue_standalone_response(right.text)
        or _is_protected_short_utterance(left.text)
        or _is_protected_short_utterance(right.text)
    ):
        return False
    gap = max(0, right.start_time - left.end_time)
    merged_text = f"{left.text}{right.text}"
    return (
        gap <= READABILITY_MERGE_MAX_GAP
        and count_words(merged_text) <= READABILITY_MERGE_MAX_CJK
    )


def _prefer_merge_next(
    prev_seg: ASRDataSeg | None,
    current: ASRDataSeg,
    next_seg: ASRDataSeg | None,
) -> bool:
    """Choose merge direction for a short fragment."""
    if next_seg is None:
        return False
    if prev_seg is None:
        return True

    current_text = _normalize_filler_text(current.text)
    next_text = _normalize_filler_text(next_seg.text)
    if _is_numeric_fragment(current_text) and _is_numeric_fragment(next_text):
        return True
    if (
        current_text in _READABILITY_PREFIXES
        or current_text.startswith(_READABILITY_LEADING_FRAGMENTS)
    ):
        return True
    if (
        current_text in _READABILITY_SUFFIXES
        or current_text.endswith(_READABILITY_TRAILING_FRAGMENTS)
    ):
        return False

    prev_gap = max(0, current.start_time - prev_seg.end_time)
    next_gap = max(0, next_seg.start_time - current.end_time)
    return next_gap + 100 < prev_gap


def _merge_asr_segments(left: ASRDataSeg, right: ASRDataSeg) -> ASRDataSeg:
    """Merge two adjacent ASR segments without changing text content."""
    text = f"{left.text}{right.text}"
    if left.end_time >= right.start_time:
        left_text = left.text.strip()
        right_text = right.text.strip()
        if right_text and left_text.endswith(right_text):
            text = left.text
        elif left_text and right_text.startswith(left_text):
            text = right.text

    translated_text = ""
    if left.translated_text or right.translated_text:
        translated_text = f"{left.translated_text}{right.translated_text}"
    return ASRDataSeg(
        text,
        min(left.start_time, right.start_time),
        max(left.end_time, right.end_time),
        translated_text=translated_text,
    )


# ==================== SubtitleSplitter ====================

class SubtitleSplitter:
    """字幕智能分割器。

    使用 LLM 进行语义分段，支持并发处理和规则降级。
    长度控制在匹配回时间戳后统一处理，split_by_llm 只负责语义断句。
    """

    def __init__(
        self,
        thread_num: int,
        model: str,
        base_url: str,
        api_key: str,
        max_word_count_cjk: int = MAX_WORD_COUNT_CJK,
        max_word_count_english: int = MAX_WORD_COUNT_ENGLISH,
        prompt_limit_ratio: float = 0.8,
        disable_thinking: bool = False,
        llm_extra_body: dict | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """初始化分割器。

        Args:
            thread_num: 并发线程数。
            model: LLM 模型名称。
            base_url: LLM API 基地址。
            api_key: LLM API 密钥。
            max_word_count_cjk: CJK 最大字数。
            max_word_count_english: 英文最大单词数。
            prompt_limit_ratio: 注入 split prompt 的长度缩减比例。
            disable_thinking: 是否注入 /no_think 指令。
            llm_extra_body: 原样透传到 OpenAI 兼容接口 extra_body 的 JSON。
        """
        self.thread_num = thread_num
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_word_count_cjk = max_word_count_cjk
        self.max_word_count_english = max_word_count_english
        self.prompt_limit_ratio = prompt_limit_ratio
        self.disable_thinking = disable_thinking
        self.llm_extra_body = llm_extra_body
        self.is_running = True
        self.timeout = timeout
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=self.thread_num
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """停止分割器并清理线程池资源。

        调用后 self.executor 被置为 None。
        """
        if not self.is_running:
            return
        self.is_running = False
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:  # pylint: disable=broad-exception-caught
                # stop() 作为清理方法，不应因任何异常而崩溃
                logger.error("关闭线程池时出错: %s", e)
            finally:
                self.executor = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def split_subtitle(self, subtitle_data: Union[str, ASRData]) -> ASRData:
        """分割字幕（主入口）。

        处理流程:
        1. 读取并预处理字幕
        2. 按字数分段
        3. 并发调用 LLM 处理
        4. 合并结果并优化

        Args:
            subtitle_data: SRT 字符串或 ASRData 对象。

        Returns:
            分割后的 ASRData 对象。

        Raises:
            RuntimeError: 分割失败时抛出。
        """
        try:
            # 1. 读取字幕
            if isinstance(subtitle_data, str):
                asr_data = ASRData.from_srt(subtitle_data)
            else:
                asr_data = subtitle_data

            if not asr_data.is_word_timestamp():
                asr_data = asr_data.split_to_word_segments()

            # 2. 预处理
            asr_data.segments = preprocess_segments(
                asr_data.segments, need_lower=False
            )
            txt = asr_data.to_txt().replace("\n", "")

            # 3. 确定分段数并分割
            total_word_count = count_words(txt)
            num_segments = self._determine_num_segments(total_word_count)
            logger.info(
                "根据字数 %d, 确定断句分段数: %d", total_word_count, num_segments,
            )

            asr_data_list = self._split_asr_data(asr_data, num_segments)

            # 4. 并发处理
            processed_segments = self._process_segments(asr_data_list)

            # 5. 合并并排序
            final_segments = self._merge_processed_segments(processed_segments)
            final_segments = self._smooth_short_fillers(final_segments)
            final_segments = self._smooth_readability_segments(final_segments)

            return ASRData(final_segments)

        except RuntimeError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("分割失败: %s", e)
            raise RuntimeError(f"分割失败: {e}") from e

    # ------------------------------------------------------------------
    # 分段数计算
    # ------------------------------------------------------------------

    def _determine_num_segments(
        self, word_count: int, threshold: int = SEGMENT_WORD_THRESHOLD
    ) -> int:
        """根据字数确定分段数。"""
        num_segments = word_count // threshold
        if word_count % threshold > 0:
            num_segments += 1
        return max(1, num_segments)

    # ------------------------------------------------------------------
    # 长文本拆分
    # ------------------------------------------------------------------

    def _split_asr_data(
        self, asr_data: ASRData, num_segments: int
    ) -> List[ASRData]:
        """按时间间隔智能分割长文本。

        策略:
        1. 计算平均分割点
        2. 在分割点附近寻找最大时间间隔
        3. 在间隔处切分以保证语义完整
        """
        total_segs = len(asr_data.segments)
        txt = asr_data.to_txt()
        total_word_count = count_words(txt)
        words_per_segment = total_word_count // max(num_segments, 1)

        if num_segments <= 1 or total_segs <= num_segments:
            return [asr_data]

        # 初始分割点
        split_indices = [i * words_per_segment for i in range(1, num_segments)]

        # 调整分割点：在附近处找最大时间间隔
        adjusted_split_indices: list[int] = []
        for split_point in split_indices:
            start = max(0, split_point - SPLIT_SEARCH_RANGE)
            end = min(total_segs - 1, split_point + SPLIT_SEARCH_RANGE)

            max_gap = -1
            best_index = split_point

            for j in range(start, end):
                if j + 1 >= total_segs:
                    break
                gap = (
                    asr_data.segments[j + 1].start_time
                    - asr_data.segments[j].end_time
                )
                if gap > max_gap:
                    max_gap = gap
                    best_index = j

            adjusted_split_indices.append(best_index)

        # 去重并排序
        adjusted_split_indices = sorted(set(adjusted_split_indices))

        # 执行分割
        segments: List[ASRData] = []
        prev_index = 0
        for index in adjusted_split_indices:
            part = ASRData(asr_data.segments[prev_index : index + 1])
            segments.append(part)
            prev_index = index + 1

        if prev_index < total_segs:
            part = ASRData(asr_data.segments[prev_index:])
            segments.append(part)

        return segments

    # ------------------------------------------------------------------
    # 并发处理
    # ------------------------------------------------------------------

    def _process_segments(
        self, asr_data_list: List[ASRData]
    ) -> List[List[ASRDataSeg]]:
        """并发处理所有分段。"""
        if self.executor is None:
            raise ValueError("线程池未初始化")

        futures = []
        for asr_data in asr_data_list:
            future = self.executor.submit(self._process_single_segment, asr_data)
            futures.append(future)

        processed_segments: List[List[ASRDataSeg]] = []
        for future in as_completed(futures):
            if not self.is_running:
                break
            try:
                result = future.result()
                processed_segments.append(result)
            except Exception as e:  # pylint: disable=broad-exception-caught
                # 单个分段失败不应中断整个流程
                logger.error("处理分段失败: %s", e)

        return processed_segments

    def _process_single_segment(
        self, asr_data_part: ASRData
    ) -> List[ASRDataSeg]:
        """处理单个分段。

        优先使用 LLM 断句；若调用失败或无法完整匹配回时间戳，
        则降级到规则分割。
        """
        if not asr_data_part.segments:
            return []
        try:
            return self._process_by_llm(asr_data_part.segments)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("LLM 分割失败，回退到规则分割: %s", e)
            return self._process_by_rules(asr_data_part.segments)

    # ------------------------------------------------------------------
    # LLM 处理
    # ------------------------------------------------------------------

    def _process_by_llm(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """使用 LLM 进行智能分段。

        流程：
        1. split_by_llm 返回语义断句（只保证内容一致性，不保证长度）
        2. _merge_segments_based_on_sentences 将句子匹配回词级时间戳
        3. _enforce_length_limit 对超长段基于时间间隔拆分，确保长度合规
        """
        txt = "".join([seg.text for seg in segments])
        logger.info("开始调用 API 进行分段，文本长度: %d", count_words(txt))
        logger.info(
            "断句诊断 | 待断句 ASR 文本 segments=%d text=%s",
            len(segments),
            _clip_for_log(txt),
        )

        sentences = split_by_llm(
            text=txt,
            model=self.model,
            max_word_count_cjk=self.max_word_count_cjk,
            max_word_count_english=self.max_word_count_english,
            base_url=self.base_url,
            api_key=self.api_key,
            prompt_limit_ratio=self.prompt_limit_ratio,
            disable_thinking=self.disable_thinking,
            llm_extra_body=self.llm_extra_body,
            timeout=self.timeout,
        )
        logger.info(
            "断句诊断 | LLM 返回句子数=%d sentences=%s",
            len(sentences),
            _clip_for_log(" | ".join(sentences)),
        )

        matched_segments = self._merge_segments_based_on_sentences(
            segments, sentences,
        )

        # 匹配回时间戳后，对超长段基于时间间隔拆分
        return self._enforce_length_limit(matched_segments)

    # ------------------------------------------------------------------
    # 长度合规拆分
    # ------------------------------------------------------------------

    def _enforce_length_limit(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """对超长 ASRDataSeg 进行拆分，确保所有段满足字数限制。

        此方法在匹配回时间戳之后调用，利用词级时间信息
        在最大时间间隔处精确拆分，保证每个子段都有正确的时间范围。

        对于无法继续拆分的段（词数少于 RULE_MIN_SEGMENT_SIZE），
        保留原样。
        """
        result: List[ASRDataSeg] = []
        for seg in segments:
            wc = count_words(seg.text)
            max_wc = (
                self.max_word_count_cjk
                if is_mainly_cjk(seg.text)
                else self.max_word_count_english
            )
            if wc <= max_wc:
                result.append(seg)
            else:
                # 尝试将单个 seg 还原为词级分段再拆分
                sub_segments = self._split_oversized_seg(seg, max_wc)
                result.extend(sub_segments)
        return result

    def _split_oversized_seg(
        self, seg: ASRDataSeg, max_wc: int,
    ) -> List[ASRDataSeg]:
        """将超长 ASRDataSeg 拆分为多个合规段。

        尝试基于 ASRData 的词级分段能力还原子词，
        然后利用 _split_long_segment 按时间间隔拆分。
        如果无法还原词级分段，则按字符位置均分。
        """
        # 尝试还原词级分段
        try:
            temp_asr = ASRData([seg])
            if not temp_asr.is_word_timestamp():
                temp_asr = temp_asr.split_to_word_segments()
            word_segs = temp_asr.segments
            if len(word_segs) >= RULE_MIN_SEGMENT_SIZE:
                return self._split_long_segment(word_segs)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # 无法还原词级分段 — 按字数均分时间
        text = seg.text
        total_wc = count_words(text)
        num_parts = (total_wc + max_wc - 1) // max_wc  # 向上取整

        if num_parts <= 1:
            return [seg]

        duration = seg.end_time - seg.start_time
        part_len = len(text) // num_parts
        parts: List[ASRDataSeg] = []

        for i in range(num_parts):
            if i < num_parts - 1:
                part_text = text[i * part_len : (i + 1) * part_len].strip()
            else:
                part_text = text[i * part_len :].strip()

            if not part_text:
                continue

            # 按比例分配时间
            ratio_start = i / num_parts
            ratio_end = (i + 1) / num_parts
            part_start = seg.start_time + int(duration * ratio_start)
            part_end = seg.start_time + int(duration * ratio_end)

            parts.append(ASRDataSeg(part_text, part_start, part_end))

        return parts if parts else [seg]

    # ------------------------------------------------------------------
    # 规则处理（降级方案）
    # ------------------------------------------------------------------

    def _process_by_rules(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """使用规则进行基础分割（LLM 降级方案）。

        规则:
        1. 按时间间隔分组
        2. 按常见词分割长句
        3. 拆分超长分段
        """
        logger.info("规则分割 — 输入分段数: %d", len(segments))

        # 1. 按时间间隔分组
        segment_groups = self._group_by_time_gaps(
            segments, max_gap=RULE_SPLIT_GAP, check_large_gaps=True
        )
        logger.info("按时间间隔分组: %d", len(segment_groups))

        # 2. 按常见词分割长句
        common_result_groups: List[List[ASRDataSeg]] = []
        for group in segment_groups:
            max_wc = (
                self.max_word_count_cjk
                if is_mainly_cjk("".join(seg.text for seg in group))
                else self.max_word_count_english
            )
            if count_words("".join(seg.text for seg in group)) > max_wc:
                split_groups = self._split_by_common_words(group)
                common_result_groups.extend(split_groups)
            else:
                common_result_groups.append(group)

        # 3. 拆分超长分段
        result_segments: List[ASRDataSeg] = []
        for group in common_result_groups:
            result_segments.extend(self._split_long_segment(group))

        return result_segments

    # ------------------------------------------------------------------
    # 时间间隔分组
    # ------------------------------------------------------------------

    def _group_by_time_gaps(
        self,
        segments: List[ASRDataSeg],
        max_gap: int = MAX_GAP,
        check_large_gaps: bool = False,
    ) -> List[List[ASRDataSeg]]:
        """按时间间隔分组。"""
        if not segments:
            return []

        result: List[List[ASRDataSeg]] = []
        current_group: List[ASRDataSeg] = [segments[0]]
        recent_gaps: list[int] = []

        for i in range(1, len(segments)):
            time_gap = segments[i].start_time - segments[i - 1].end_time

            # 检查异常大间隔
            if check_large_gaps:
                recent_gaps.append(time_gap)
                if len(recent_gaps) > TIME_GAP_WINDOW_SIZE:
                    recent_gaps.pop(0)
                if len(recent_gaps) == TIME_GAP_WINDOW_SIZE:
                    avg_gap = sum(recent_gaps) / len(recent_gaps)
                    if (
                        time_gap > avg_gap * TIME_GAP_MULTIPLIER
                        and len(current_group) > MIN_GROUP_SIZE
                    ):
                        result.append(current_group)
                        current_group = []
                        recent_gaps = []

            # 超过最大间隔则分组
            if time_gap > max_gap:
                result.append(current_group)
                current_group = []
                recent_gaps = []

            current_group.append(segments[i])

        if current_group:
            result.append(current_group)

        return result

    # ------------------------------------------------------------------
    # 常见词分割
    # ------------------------------------------------------------------

    def _split_by_common_words(
        self, segments: List[ASRDataSeg]
    ) -> List[List[ASRDataSeg]]:
        """在常见连接词处分割。"""

        prefix_split_words = {
            # 英文
            "and", "or", "but", "if", "then", "because", "as", "until",
            "while", "what", "when", "where", "nor", "yet", "so", "for",
            "however", "moreover",
            # 中文
            "和", "或", "与", "但", "而", "因", "你", "他",
            "她", "它", "您", "这", "那", "哪",
        }

        suffix_split_words = {
            # 标点
            ".", ",", "!", "?", "。", "，", "！", "？",
            # 中文语气词
            "的", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "啦",
            # 英文代词
            "mine", "yours", "hers", "its", "ours", "theirs", "either",
            "neither",
        }

        result: List[List[ASRDataSeg]] = []
        current_group: List[ASRDataSeg] = []

        for i, seg in enumerate(segments):
            max_wc = (
                self.max_word_count_cjk
                if is_mainly_cjk(seg.text)
                else self.max_word_count_english
            )

            # 前缀词分割
            if (
                any(seg.text.lower().startswith(word) for word in prefix_split_words)
                and len(current_group) >= int(max_wc * PREFIX_WORD_RATIO)
            ):
                result.append(current_group)
                logger.debug("在前缀词 %s 前分割", seg.text)
                current_group = []

            # 后缀词分割
            if (
                i > 0
                and any(
                    segments[i - 1].text.lower().endswith(word)
                    for word in suffix_split_words
                )
                and len(current_group) >= int(max_wc * SUFFIX_WORD_RATIO)
            ):
                result.append(current_group)
                logger.debug("在后缀词 %s 后分割", segments[i - 1].text)
                current_group = []

            current_group.append(seg)

        if current_group:
            result.append(current_group)

        return result

    # ------------------------------------------------------------------
    # 超长段拆分
    # ------------------------------------------------------------------

    def _split_long_segment(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """拆分超长分段 — 寻找最大时间间隔点进行拆分。"""
        result_segs: List[ASRDataSeg] = []
        segments_to_process: List[List[ASRDataSeg]] = [segments]

        while segments_to_process:
            current_segments = segments_to_process.pop(0)

            if not current_segments:
                continue

            merged_text = "".join(seg.text for seg in current_segments)
            max_wc = (
                self.max_word_count_cjk
                if is_mainly_cjk(merged_text)
                else self.max_word_count_english
            )
            n = len(current_segments)

            # 分段足够短或无法继续拆分
            if count_words(merged_text) <= max_wc or n < RULE_MIN_SEGMENT_SIZE:
                merged_seg = ASRDataSeg(
                    merged_text.strip(),
                    current_segments[0].start_time,
                    current_segments[-1].end_time,
                )
                result_segs.append(merged_seg)
                continue

            # 检查时间间隔
            gaps = [
                current_segments[j + 1].start_time - current_segments[j].end_time
                for j in range(n - 1)
            ]
            all_equal = all(abs(gap - gaps[0]) < 1e-6 for gap in gaps)

            if all_equal:
                split_index = n // 2
            else:
                start_idx = max(n // 6, 1)
                end_idx = min((5 * n) // 6, n - 2)
                split_index = max(
                    range(start_idx, end_idx),
                    key=lambda idx: (
                        current_segments[idx + 1].start_time
                        - current_segments[idx].end_time
                    ),
                    default=n // 2,
                )
                if split_index == 0 or split_index == n - 1:
                    split_index = n // 2

            first_segs = current_segments[: split_index + 1]
            second_segs = current_segments[split_index + 1 :]
            segments_to_process.extend([first_segs, second_segs])

        result_segs.sort(key=lambda seg: seg.start_time)
        return result_segs

    # ------------------------------------------------------------------
    # 滑动窗口匹配合并
    # ------------------------------------------------------------------

    def _merge_segments_based_on_sentences(
        self,
        segments: List[ASRDataSeg],
        sentences: List[str],
    ) -> List[ASRDataSeg]:
        """基于 LLM 返回的句子列表合并 ASR 分段。

        使用滑动窗口匹配算法：
        1. 对每个 LLM 句子，查找最佳匹配的 ASR 分段序列
        2. 使用相似度算法进行匹配（匹配前对双方做标点归一化）
        3. 合并匹配的分段为单个 ASRDataSeg（保留完整语义单元）

        长度拆分不在此处进行，由调用方 _process_by_llm 统一处理。

        若部分句子无法匹配，则保留已成功匹配的 LLM 句子，
        未覆盖的 ASR token 区间用规则分割补齐。
        """

        def normalize_for_match(s: str) -> str:
            """移除标点和空白后转小写，用于匹配比较。"""
            return _MATCH_PUNCTUATION.sub("", s).lower()

        asr_texts = [seg.text for seg in segments]
        # 预计算归一化后的 ASR 文本，避免重复计算
        asr_texts_normalized = [normalize_for_match(t) for t in asr_texts]
        asr_len = len(asr_texts)
        asr_index = 0
        threshold = MATCH_SIMILARITY_THRESHOLD
        max_shift = MATCH_MAX_SHIFT
        unmatched_sentences: list[str] = []

        new_segments: List[ASRDataSeg] = []
        emitted_until = 0

        for sentence_index, sentence in enumerate(sentences):
            logger.debug("处理句子: %s", sentence)
            logger.debug(
                "后续句子: %s", "".join(asr_texts[asr_index : asr_index + 10]),
            )

            sentence_normalized = normalize_for_match(sentence)
            next_sentence_normalized = ""
            if sentence_index + 1 < len(sentences):
                next_sentence_normalized = normalize_for_match(
                    sentences[sentence_index + 1]
                )
            wc = max(len(sentence_normalized), 1)
            best_ratio = 0.0
            best_score = float("-inf")
            best_pos: int | None = None
            best_window_size = 0
            best_text = ""
            best_normalized = ""
            best_next_overlap = 0

            # 滑动窗口大小 — 提前绑定 wc 避免 cell-var-from-loop
            max_window_size = min(wc * 2, asr_len - asr_index)
            min_window_size = 1
            _wc = wc  # 绑定到局部变量
            window_sizes = sorted(
                range(min_window_size, max_window_size + 1),
                key=lambda x, _w=_wc: abs(x - _w),
            )

            # 滑动窗口匹配
            for window_size in window_sizes:
                max_start = min(
                    asr_index + max_shift + 1, asr_len - window_size + 1
                )
                for start in range(asr_index, max_start):
                    # 使用归一化后的文本拼接比较
                    substr_normalized = "".join(
                        asr_texts_normalized[start : start + window_size]
                    )
                    ratio = difflib.SequenceMatcher(
                        None, sentence_normalized, substr_normalized
                    ).ratio()
                    next_overlap = _next_sentence_overlap(
                        substr_normalized,
                        next_sentence_normalized,
                    )
                    length_penalty = (
                        abs(len(substr_normalized) - wc)
                        / max(wc, len(substr_normalized), 1)
                    ) * MATCH_LENGTH_PENALTY
                    start_penalty = (
                        max(0, start - asr_index) * MATCH_START_SHIFT_PENALTY
                    )
                    next_penalty = next_overlap * MATCH_NEXT_OVERLAP_PENALTY
                    score = ratio - length_penalty - start_penalty - next_penalty

                    if _is_better_match(
                        score=score,
                        ratio=ratio,
                        start=start,
                        window_size=window_size,
                        next_overlap=next_overlap,
                        best_score=best_score,
                        best_ratio=best_ratio,
                        best_start=best_pos,
                        best_window_size=best_window_size,
                        best_next_overlap=best_next_overlap,
                    ):
                        best_score = score
                        best_ratio = ratio
                        best_pos = start
                        best_window_size = window_size
                        best_text = "".join(asr_texts[start : start + window_size])
                        best_normalized = substr_normalized
                        best_next_overlap = next_overlap

            # 处理匹配结果
            if best_ratio >= threshold and best_pos is not None:
                start_seg_index = best_pos
                end_seg_index = best_pos + best_window_size - 1

                if start_seg_index > emitted_until:
                    gap_segments = segments[emitted_until:start_seg_index]
                    logger.info(
                        "断句诊断 | 局部规则补齐 gap_start=%d gap_end=%d gap_tokens=%d",
                        emitted_until,
                        start_seg_index - 1,
                        len(gap_segments),
                    )
                    new_segments.extend(self._process_by_rules(gap_segments))

                segs_to_merge = segments[start_seg_index : end_seg_index + 1]

                # 按时间切分避免跨度过大
                seg_groups = self._group_by_time_gaps(
                    segs_to_merge, max_gap=MAX_GAP
                )

                for group in seg_groups:
                    # 合并组内所有词级段为一个 ASRDataSeg
                    merged_text = "".join(seg.text for seg in group).strip()
                    if merged_text:
                        merged_seg = ASRDataSeg(
                            merged_text,
                            group[0].start_time,
                            group[-1].end_time,
                        )
                        new_segments.append(merged_seg)

                max_shift = MATCH_MAX_SHIFT
                asr_index = end_seg_index + 1
                emitted_until = end_seg_index + 1
            else:
                context_text = "".join(asr_texts[asr_index : asr_index + 40])
                logger.warning("无法匹配句子: %s", sentence)
                logger.warning(
                    "断句诊断 | 匹配失败 sentence_norm=%s best_ratio=%.3f "
                    "best_score=%.3f best_pos=%s best_window_size=%d "
                    "next_overlap=%d best_text=%s best_norm=%s asr_index=%d "
                    "context=%s",
                    _clip_for_log(sentence_normalized),
                    best_ratio,
                    best_score,
                    best_pos,
                    best_window_size,
                    best_next_overlap,
                    _clip_for_log(best_text),
                    _clip_for_log(best_normalized),
                    asr_index,
                    _clip_for_log(context_text),
                )
                unmatched_sentences.append(sentence)
                # 不移动 asr_index — 跳过此 sentence，
                # 让后续 sentence 从同一位置继续匹配，避免级联失配
                max_shift = MATCH_LARGE_SHIFT

        if emitted_until < asr_len:
            remaining_segments = segments[emitted_until:]
            logger.info(
                "断句诊断 | 局部规则补齐 tail_start=%d tail_tokens=%d",
                emitted_until,
                len(remaining_segments),
            )
            new_segments.extend(self._process_by_rules(remaining_segments))

        if unmatched_sentences:
            logger.warning(
                "LLM 分句部分未匹配，已用规则补齐未覆盖区间: %d",
                len(unmatched_sentences),
            )

        return new_segments

    # ------------------------------------------------------------------
    # 合并处理后的分段
    # ------------------------------------------------------------------

    def _merge_processed_segments(
        self, processed_segments: List[List[ASRDataSeg]]
    ) -> List[ASRDataSeg]:
        """合并所有处理后的分段并排序。"""
        final_segments: List[ASRDataSeg] = []
        for segs in processed_segments:
            final_segments.extend(segs)
        final_segments.sort(key=lambda seg: seg.start_time)
        return final_segments

    def _smooth_short_fillers(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """Conservatively merge standalone short filler fragments."""
        if not segments:
            return []

        smoothed: List[ASRDataSeg] = []
        merge_count = 0

        for seg in segments:
            text = seg.text.strip()
            should_merge_left = (
                bool(smoothed)
                and _is_filler_only(text)
                and _can_merge_filler(smoothed[-1], seg)
            )
            if should_merge_left:
                logger.info(
                    "断句诊断 | 短语气词后处理合并到前句: text=%s",
                    text,
                )
                smoothed[-1] = _merge_asr_segments(smoothed[-1], seg)
                merge_count += 1
            else:
                smoothed.append(seg)

        result: List[ASRDataSeg] = []
        index = 0
        while index < len(smoothed):
            current = smoothed[index]
            if (
                index + 1 < len(smoothed)
                and _is_filler_only(current.text)
                and _can_merge_filler(current, smoothed[index + 1])
            ):
                logger.info(
                    "断句诊断 | 短语气词后处理合并到后句: text=%s",
                    current.text.strip(),
                )
                result.append(_merge_asr_segments(current, smoothed[index + 1]))
                merge_count += 1
                index += 2
                continue

            result.append(current)
            index += 1

        if merge_count:
            logger.info("断句诊断 | 短语气词后处理合并数: %d", merge_count)
        return result

    def _smooth_readability_segments(
        self, segments: List[ASRDataSeg]
    ) -> List[ASRDataSeg]:
        """Merge very short subtitle fragments into nearby readable segments."""
        if not segments:
            return []

        result: List[ASRDataSeg] = []
        merge_count = 0
        index = 0

        while index < len(segments):
            current = segments[index]
            next_seg = segments[index + 1] if index + 1 < len(segments) else None
            prev_seg = result[-1] if result else None

            if not _is_readability_short(current):
                result.append(current)
                index += 1
                continue

            can_merge_prev = (
                prev_seg is not None
                and _can_merge_readability(prev_seg, current)
            )
            can_merge_next = (
                next_seg is not None
                and _can_merge_readability(current, next_seg)
            )

            if not can_merge_prev and not can_merge_next:
                result.append(current)
                index += 1
                continue

            if can_merge_next and _prefer_merge_next(prev_seg, current, next_seg):
                logger.info(
                    "断句诊断 | 短片段可读性合并到后句: text=%s",
                    current.text.strip(),
                )
                result.append(_merge_asr_segments(current, next_seg))
                merge_count += 1
                index += 2
                continue

            if can_merge_prev:
                logger.info(
                    "断句诊断 | 短片段可读性合并到前句: text=%s",
                    current.text.strip(),
                )
                result[-1] = _merge_asr_segments(prev_seg, current)
                merge_count += 1
                index += 1
                continue

            logger.info(
                "断句诊断 | 短片段可读性合并到后句: text=%s",
                current.text.strip(),
            )
            result.append(_merge_asr_segments(current, next_seg))
            merge_count += 1
            index += 2

        if merge_count:
            logger.info("断句诊断 | 短片段可读性合并数: %d", merge_count)
        return result
