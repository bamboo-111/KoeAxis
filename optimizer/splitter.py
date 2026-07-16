"""字幕智能分割器 — 移植自 VideoCaptioner app/core/split/split.py

生产环境仅支持纯规则断句：基于时间间隔、文本边界、短应答保护和可读性规则进行切分。

关键改动（相对上游）：
- 导入路径全部改为 backend.*
- 移除 atexit.register，提供显式 stop()
- split_subtitle 中字符串输入改为 ASRData.from_srt()
- 使用 asr_data.to_txt() 替代已删除的 _segments_to_txt()
- logger 前缀 optimizer.splitter
"""

import logging
from concurrent.futures import (  # pylint: disable=no-name-in-module
    ThreadPoolExecutor,
    as_completed,
)
from typing import List, Union

from optimizer.asr_data import ASRData, ASRDataSeg
from optimizer.splitter_boundaries import (
    STRONG_SENTENCE_END as _STRONG_SENTENCE_END,
    inline_dialogue_parts as _inline_dialogue_parts,
)
from optimizer.splitter_display import (
    MAX_SUBTITLE_DISPLAY_DURATION,
    ORDINARY_SUBTITLE_MIN_DURATION,
    PROTECTED_SHORT_MIN_DURATION,
    cap_long_display_durations as _cap_long_display_durations,
    extend_protected_short_display_durations as _extend_protected_short_display_durations,
    extend_segment_display_duration as _extend_segment_display_duration,
    merge_zero_gap_short_display_fragments as _merge_zero_gap_short_display_fragments,
    minimum_display_duration as _minimum_display_duration,
    redistribute_zero_gap_short_display_durations as _redistribute_zero_gap_short_display_durations,
)
from optimizer.splitter_readability import (
    READABILITY_MERGE_MAX_CJK,
    TAIL_FRAGMENT_MERGE_MAX_GAP,
    can_merge_filler as _can_merge_filler,
    can_merge_readability as _can_merge_readability,
    is_dialogue_standalone_response as _is_dialogue_standalone_response,
    is_filler_only as _is_filler_only,
    is_numeric_fragment as _is_numeric_fragment,
    is_protected_short_display_response as _is_protected_short_display_response,
    is_protected_short_utterance as _is_protected_short_utterance,
    is_readability_short as _is_readability_short,
    is_structural_readability_fragment as _is_structural_readability_fragment,
    is_tail_fragment as _is_tail_fragment,
    normalize_filler_text as _normalize_filler_text,
    prefer_merge_next as _prefer_merge_next,
    segment_duration as _segment_duration,
    starts_with_short_filler as _starts_with_short_filler,
)
from optimizer.splitter_timing import (
    INLINE_SHORT_UTTERANCE_MIN_DURATION,
    extend_inline_short_utterances as _extend_inline_short_utterances,
    parts_to_timed_segments as _parts_to_timed_segments,
    split_text_evenly_with_timing as _split_text_evenly_with_timing,
)
from optimizer.text_utils import (
    count_words,
    is_mainly_cjk,
    is_pure_punctuation,
    is_space_separated_language,
)

logger = logging.getLogger("optimizer.splitter")

# ==================== 配置常量 ====================

# 字数限制
MAX_WORD_COUNT_CJK = 18  # CJK 文本单行最大字数
MAX_WORD_COUNT_ENGLISH = 18  # 英文文本单行最大单词数

# 分段阈值
SEGMENT_WORD_THRESHOLD = 300  # 长文本分段阈值(字数)

# 时间间隔
MAX_GAP = 1500  # 允许的最大时间间隔(毫秒)
MERGE_SHORT_GAP = 200  # 短分段合并时间阈值(毫秒)
MERGE_VERY_SHORT_GAP = 500  # 极短分段合并时间阈值(毫秒)

# 短分段合并阈值
MERGE_MIN_WORDS = 5  # 短分段最小字数阈值
MERGE_VERY_SHORT_WORDS = 3  # 极短分段字数阈值

# 分割相关
SPLIT_SEARCH_RANGE = 30  # 分割点前后搜索范围
TIME_GAP_WINDOW_SIZE = 5  # 时间间隔窗口大小
TIME_GAP_MULTIPLIER = 3  # 大间隔判断倍数
MIN_GROUP_SIZE = 5  # 最小分组大小

# 规则分割
RULE_SPLIT_GAP = 500  # 规则分割时间间隔阈值(毫秒)
RULE_MIN_SEGMENT_SIZE = 4  # 规则分割最小分段大小

# 常见词分割
PREFIX_WORD_RATIO = 0.6  # 前缀词分割比例
SUFFIX_WORD_RATIO = 0.4  # 后缀词分割比例

READABILITY_ZERO_DURATION = 200  # 近零时长片段强制尝试合并(毫秒)
TAIL_FRAGMENT_MERGE_MAX_CJK = 32  # 句尾残片合并后最大 CJK 字数
# ==================== 模块级预处理函数 ====================


def preprocess_segments(segments: List[ASRDataSeg], need_lower: bool = True) -> List[ASRDataSeg]:
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


def _split_inline_dialogue_boundaries(segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
    result: List[ASRDataSeg] = []
    for segment in segments:
        parts = _inline_dialogue_parts(segment.text)
        if len(parts) <= 1:
            result.append(segment)
            continue
        result.extend(_parts_to_timed_segments(segment, parts))
    return result


def _merge_asr_segments(left: ASRDataSeg, right: ASRDataSeg) -> ASRDataSeg:
    """Merge two adjacent ASR segments without changing text content."""
    text = f"{left.text}{right.text}"

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
    """生产规则字幕分割器。"""

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
        timeout: float = 120.0,
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
        del model, base_url, api_key, prompt_limit_ratio, disable_thinking, llm_extra_body, timeout
        self.thread_num = thread_num
        self.max_word_count_cjk = max_word_count_cjk
        self.max_word_count_english = max_word_count_english
        self.is_running = True
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=self.thread_num)

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
            asr_data.segments = preprocess_segments(asr_data.segments, need_lower=False)
            txt = asr_data.to_txt().replace("\n", "")

            # 3. 确定分段数并分割
            total_word_count = count_words(txt)
            num_segments = self._determine_num_segments(total_word_count)
            logger.info(
                "根据字数 %d, 确定断句分段数: %d",
                total_word_count,
                num_segments,
            )

            asr_data_list = self._split_asr_data(asr_data, num_segments)

            # 4. 并发处理
            processed_segments = self._process_segments(asr_data_list)

            # 5. 合并并排序
            final_segments = self._merge_processed_segments(processed_segments)
            final_segments = self._smooth_short_fillers(final_segments)
            final_segments = self._smooth_readability_segments(final_segments)
            final_segments = self._merge_tail_fragments(final_segments)
            final_segments = self._smooth_readability_segments(final_segments)
            final_segments = _extend_protected_short_display_durations(final_segments)

            return ASRData(final_segments)

        except RuntimeError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("分割失败: %s", e)
            raise RuntimeError(f"分割失败: {e}") from e

    # ------------------------------------------------------------------
    # 分段数计算
    # ------------------------------------------------------------------

    def _determine_num_segments(self, word_count: int, threshold: int = SEGMENT_WORD_THRESHOLD) -> int:
        """根据字数确定分段数。"""
        num_segments = word_count // threshold
        if word_count % threshold > 0:
            num_segments += 1
        return max(1, num_segments)

    # ------------------------------------------------------------------
    # 长文本拆分
    # ------------------------------------------------------------------

    def _split_asr_data(self, asr_data: ASRData, num_segments: int) -> List[ASRData]:
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
                gap = asr_data.segments[j + 1].start_time - asr_data.segments[j].end_time
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

    def _process_segments(self, asr_data_list: List[ASRData]) -> List[List[ASRDataSeg]]:
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

    def _process_single_segment(self, asr_data_part: ASRData) -> List[ASRDataSeg]:
        """使用唯一生产规则实现处理单个分段。"""
        if not asr_data_part.segments:
            return []
        return self._process_by_rules(asr_data_part.segments)

    # ------------------------------------------------------------------
    # 长度合规拆分
    # ------------------------------------------------------------------

    def _enforce_length_limit(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """对超长 ASRDataSeg 进行拆分，确保所有段满足字数限制。

        此方法在匹配回时间戳之后调用，利用词级时间信息
        在最大时间间隔处精确拆分，保证每个子段都有正确的时间范围。

        对于无法继续拆分的段（词数少于 RULE_MIN_SEGMENT_SIZE），
        保留原样。
        """
        result: List[ASRDataSeg] = []
        for seg in segments:
            wc = count_words(seg.text)
            max_wc = self.max_word_count_cjk if is_mainly_cjk(seg.text) else self.max_word_count_english
            if wc <= max_wc:
                result.append(seg)
            else:
                # 尝试将单个 seg 还原为词级分段再拆分
                sub_segments = self._split_oversized_seg(seg, max_wc)
                result.extend(sub_segments)
        return result

    def _split_oversized_seg(
        self,
        seg: ASRDataSeg,
        max_wc: int,
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

        return _split_text_evenly_with_timing(seg, num_parts)

    # ------------------------------------------------------------------
    # 规则处理（降级方案）
    # ------------------------------------------------------------------

    def _process_by_rules(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """使用规则进行基础分割（LLM 降级方案）。

        规则:
        1. 按时间间隔分组
        2. 按常见词分割长句
        3. 拆分超长分段
        """
        logger.info("规则分割 — 输入分段数: %d", len(segments))

        # 1. 按时间间隔分组
        segment_groups = self._group_by_time_gaps(segments, max_gap=RULE_SPLIT_GAP, check_large_gaps=True)
        segment_groups = [
            sentence_group for group in segment_groups for sentence_group in self._split_at_sentence_boundaries(group)
        ]
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

        return _split_inline_dialogue_boundaries(result_segments)

    @staticmethod
    def _split_at_sentence_boundaries(
        segments: List[ASRDataSeg],
    ) -> List[List[ASRDataSeg]]:
        """Split short time groups at explicit strong sentence endings."""
        groups: List[List[ASRDataSeg]] = []
        current: List[ASRDataSeg] = []
        for segment in segments:
            current.append(segment)
            if _STRONG_SENTENCE_END.search(segment.text):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

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
                    if time_gap > avg_gap * TIME_GAP_MULTIPLIER and len(current_group) > MIN_GROUP_SIZE:
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

    def _split_by_common_words(self, segments: List[ASRDataSeg]) -> List[List[ASRDataSeg]]:
        """在常见连接词处分割。"""

        prefix_split_words = {
            # 英文
            "and",
            "or",
            "but",
            "if",
            "then",
            "because",
            "as",
            "until",
            "while",
            "what",
            "when",
            "where",
            "nor",
            "yet",
            "so",
            "for",
            "however",
            "moreover",
            # 中文
            "和",
            "或",
            "与",
            "但",
            "而",
            "因",
            "你",
            "他",
            "她",
            "它",
            "您",
            "这",
            "那",
            "哪",
        }

        suffix_split_words = {
            # 标点
            ".",
            ",",
            "!",
            "?",
            "。",
            "，",
            "！",
            "？",
            # 中文语气词
            "的",
            "了",
            "着",
            "过",
            "吗",
            "呢",
            "吧",
            "啊",
            "呀",
            "嘛",
            "啦",
            # 英文代词
            "mine",
            "yours",
            "hers",
            "its",
            "ours",
            "theirs",
            "either",
            "neither",
        }

        result: List[List[ASRDataSeg]] = []
        current_group: List[ASRDataSeg] = []

        for i, seg in enumerate(segments):
            max_wc = self.max_word_count_cjk if is_mainly_cjk(seg.text) else self.max_word_count_english

            # 前缀词分割
            if any(seg.text.lower().startswith(word) for word in prefix_split_words) and len(current_group) >= int(
                max_wc * PREFIX_WORD_RATIO
            ):
                result.append(current_group)
                logger.debug("在前缀词 %s 前分割", seg.text)
                current_group = []

            # 后缀词分割
            if (
                i > 0
                and any(segments[i - 1].text.lower().endswith(word) for word in suffix_split_words)
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

    def _split_long_segment(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """拆分超长分段 — 寻找最大时间间隔点进行拆分。"""
        result_segs: List[ASRDataSeg] = []
        segments_to_process: List[List[ASRDataSeg]] = [segments]

        while segments_to_process:
            current_segments = segments_to_process.pop(0)

            if not current_segments:
                continue

            merged_text = "".join(seg.text for seg in current_segments)
            max_wc = self.max_word_count_cjk if is_mainly_cjk(merged_text) else self.max_word_count_english
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
            gaps = [current_segments[j + 1].start_time - current_segments[j].end_time for j in range(n - 1)]
            all_equal = all(abs(gap - gaps[0]) < 1e-6 for gap in gaps)

            if all_equal:
                split_index = n // 2
            else:
                start_idx = max(n // 6, 1)
                end_idx = min((5 * n) // 6, n - 2)
                split_index = max(
                    range(start_idx, end_idx),
                    key=lambda idx: current_segments[idx + 1].start_time - current_segments[idx].end_time,
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
    # 合并处理后的分段
    # ------------------------------------------------------------------

    def _merge_processed_segments(self, processed_segments: List[List[ASRDataSeg]]) -> List[ASRDataSeg]:
        """合并所有处理后的分段并排序。"""
        final_segments: List[ASRDataSeg] = []
        for segs in processed_segments:
            final_segments.extend(segs)
        final_segments.sort(key=lambda seg: seg.start_time)
        return final_segments

    def _smooth_short_fillers(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """Conservatively merge standalone short filler fragments."""
        if not segments:
            return []

        smoothed: List[ASRDataSeg] = []
        merge_count = 0

        for seg in segments:
            text = seg.text.strip()
            should_merge_left = bool(smoothed) and _is_filler_only(text) and _can_merge_filler(smoothed[-1], seg)
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

    def _smooth_readability_segments(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
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

            can_merge_prev = prev_seg is not None and _can_merge_readability(prev_seg, current)
            can_merge_next = next_seg is not None and _can_merge_readability(current, next_seg)

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

    def _merge_tail_fragments(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """Merge dangling suffix fragments back into the preceding subtitle."""
        if not segments:
            return []

        result: List[ASRDataSeg] = []
        merge_count = 0
        for seg in segments:
            if not result or not _is_tail_fragment(seg.text):
                result.append(seg)
                continue

            previous = result[-1]
            gap = max(0, seg.start_time - previous.end_time)
            merged_text = f"{previous.text}{seg.text}"
            if gap <= TAIL_FRAGMENT_MERGE_MAX_GAP and count_words(merged_text) <= TAIL_FRAGMENT_MERGE_MAX_CJK:
                logger.info(
                    "断句诊断 | 句尾残片合并到前句: text=%s gap=%d",
                    seg.text.strip(),
                    gap,
                )
                result[-1] = _merge_asr_segments(previous, seg)
                merge_count += 1
                continue

            result.append(seg)

        if merge_count:
            logger.info("断句诊断 | 句尾残片合并数: %d", merge_count)
        return result
