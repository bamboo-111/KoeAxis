"""ASRData — 自动语音识别数据结构

移植自 VideoCaptioner app/core/asr/asr_data.py。
已移除：
- SubtitleLayoutEnum 及所有 layout 逻辑
- to_ass / to_lrc / to_vtt 方法
- from_youtube_vtt / from_vtt / from_ass 方法
- handle_long_path 函数
- ASRDataSeg 的 to_lrc_ts / to_ass_ts / _ms_to_lrc_time / _ms_to_ass_ts 方法

保留完整功能：
- ASRDataSeg（text, start_time, end_time, translated_text, to_srt_ts,
  _ms_to_srt_time, transcript property）
- ASRData（segments, __iter__, __len__, has_data, is_word_timestamp,
  _is_word_level_segment, split_to_word_segments, remove_punctuation,
  to_txt, to_srt, to_json, from_srt, from_json, from_subtitle_file,
  merge_segments, merge_with_next_segment, optimize_timing, save）
- _WORD_SPLIT_PATTERN 多语言分词正则

时间单位：毫秒整数（与 VideoCaptioner 一致）。
与 Voxlign 的桥接通过 SRT 字符串完成：
ASRData.to_srt() → 标准 SRT → Voxlign srt_utils.py 解析。
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import List, Optional

from optimizer.text_utils import clean_subtitle_text, is_mainly_cjk

# langdetect 可选依赖：用于 from_srt() 双语检测，不可用时降级为单语模式
_HAS_LANGDETECT = False
try:
    from langdetect import LangDetectException, detect as langdetect_detect

    _HAS_LANGDETECT = True
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("optimizer.asr_data")

# 多语言分词模式（支持词级和字符级语言）
_WORD_SPLIT_PATTERN = (
    r"[a-zA-Z\u00c0-\u00ff\u0100-\u017f']+"  # 拉丁字符（含扩展）
    r"|[\u0400-\u04ff]+"    # 西里尔字母（俄文）
    r"|[\u0370-\u03ff]+"    # 希腊字母
    r"|[\u0600-\u06ff]+"    # 阿拉伯文
    r"|[\u0590-\u05ff]+"    # 希伯来文
    r"|\d+"                 # 数字
    r"|[\u4e00-\u9fff]"     # 中文
    r"|[\u3040-\u309f]"     # 日文平假名
    r"|[\u30a0-\u30ff]"     # 日文片假名
    r"|[\uac00-\ud7af]"     # 韩文
    r"|[\u0e00-\u0e7f][\u0e30-\u0e3a\u0e47-\u0e4e]*"  # 泰文
    r"|[\u0900-\u097f]"     # 天城文（印地语）
    r"|[\u0980-\u09ff]"     # 孟加拉文
    r"|[\u0e80-\u0eff]"     # 老挝文
    r"|[\u1000-\u109f]"     # 缅甸文
)


class ASRDataSeg:
    """单个语音识别片段（词或句子级）。

    Attributes:
        text: 原始文本
        translated_text: 翻译文本（可选）
        start_time: 起始时间（毫秒整数）
        end_time: 结束时间（毫秒整数）
    """

    def __init__(
        self,
        text: str,
        start_time: int,
        end_time: int,
        translated_text: str = "",
    ) -> None:
        self.text = text
        self.translated_text = translated_text
        self.start_time = start_time
        self.end_time = end_time

    def to_srt_ts(self) -> str:
        """转换为 SRT 时间戳格式。

        Returns:
            形如 "HH:MM:SS,mmm --> HH:MM:SS,mmm" 的字符串
        """
        return (
            f"{self._ms_to_srt_time(self.start_time)} --> "
            f"{self._ms_to_srt_time(self.end_time)}"
        )

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        """将毫秒转换为 SRT 时间格式 (HH:MM:SS,mmm)。

        Args:
            ms: 毫秒值

        Returns:
            SRT 时间字符串
        """
        total_seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (
            f"{int(hours):02}:{int(minutes):02}:"
            f"{int(seconds):02},{int(milliseconds):03}"
        )

    @property
    def transcript(self) -> str:
        """返回片段文本。"""
        return self.text

    def __str__(self) -> str:
        return f"ASRDataSeg({self.text!r}, {self.start_time}, {self.end_time})"

    def __repr__(self) -> str:
        return self.__str__()


class ASRData:
    """自动语音识别数据容器。

    持有一组 ASRDataSeg，提供序列化/反序列化、分词、合并等操作。
    初始化时自动过滤空段并按 start_time 排序。

    Attributes:
        segments: 有效片段列表（按 start_time 升序）
    """

    def __init__(self, segments: List[ASRDataSeg]) -> None:
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        self.segments = filtered_segments

    def __iter__(self):  # noqa: ANN204
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def has_data(self) -> bool:
        """检查是否有有效片段。"""
        return len(self.segments) > 0

    # ------------------------------------------------------------------
    # 词级时间戳检测
    # ------------------------------------------------------------------

    def _is_word_level_segment(self, segment: ASRDataSeg) -> bool:
        """判断单个片段是否为词级。

        - CJK 语言：1-2 个字符视为词级
        - 非 CJK 语言（如英文）：单个单词视为词级

        Args:
            segment: 待判断的字幕片段

        Returns:
            True 如果片段符合词级模式
        """
        text = segment.text.strip()

        # CJK 语言：1-2 个字符
        if is_mainly_cjk(text):
            return len(text) <= 2

        # 非 CJK 语言：单个单词
        words = text.split()
        return len(words) == 1

    def is_word_timestamp(self) -> bool:
        """检查时间戳是否为词级（非句子级）。

        词级判定标准：
        - 英文：单个单词
        - CJK/亚洲语言：1-2 个字符
        - 允许 20% 误差容忍（即 80%+ 的片段符合词级模式即判定为词级）

        Returns:
            True 如果 80%+ 的片段符合词级模式
        """
        if not self.segments:
            return False

        word_level_count = sum(
            1 for seg in self.segments if self._is_word_level_segment(seg)
        )

        threshold = 0.8
        word_level_ratio = word_level_count / len(self.segments)

        return word_level_ratio >= threshold

    # ------------------------------------------------------------------
    # 分词与标点移除
    # ------------------------------------------------------------------

    def split_to_word_segments(self) -> "ASRData":
        """将句子级字幕分割为词级字幕，并按音素估算分配时间戳。

        时间戳分配基于音素估算（每 4 个字符约 1 个音素）。

        Returns:
            修改后的 ASRData 实例（self）
        """
        chars_per_phoneme = 4
        new_segments: list[ASRDataSeg] = []

        for seg in self.segments:
            text = seg.text
            duration = seg.end_time - seg.start_time

            # 使用统一的多语言分词模式
            words_list = list(re.finditer(_WORD_SPLIT_PATTERN, text))

            if not words_list:
                continue

            # 计算总音素数
            total_phonemes = sum(
                math.ceil(len(w.group()) / chars_per_phoneme) for w in words_list
            )
            time_per_phoneme = duration / max(total_phonemes, 1)

            # 为每个词分配时间戳
            current_time = seg.start_time
            for word_match in words_list:
                word = word_match.group()
                word_phonemes = math.ceil(len(word) / chars_per_phoneme)
                word_duration = int(time_per_phoneme * word_phonemes)

                word_end_time = min(current_time + word_duration, seg.end_time)
                new_segments.append(
                    ASRDataSeg(
                        text=word,
                        start_time=current_time,
                        end_time=word_end_time,
                    )
                )
                current_time = word_end_time

        self.segments = new_segments
        return self

    def remove_punctuation(self) -> "ASRData":
        """移除片段末尾的中文逗号和句号。

        Returns:
            修改后的 ASRData 实例（self）
        """
        punctuation = r"[，。]"
        for seg in self.segments:
            seg.text = re.sub(f"{punctuation}+$", "", seg.text.strip())
            seg.translated_text = re.sub(
                f"{punctuation}+$", "", seg.translated_text.strip()
            )
        return self

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_txt(self) -> str:
        """Convert to plain text (no timestamps), one segment per line."""
        return "\n".join(seg.text for seg in self.segments)

    def to_srt(self) -> str:
        """转换为标准 SRT 字符串。

        如果片段有 translated_text，输出为双语（原文在上，译文在下）。
        生成的 SRT 与 Voxlign 现有 srt_utils.py 解析器完全兼容。

        Returns:
            标准 SRT 格式字符串
        """
        srt_lines: list[str] = []
        for n, seg in enumerate(self.segments, 1):
            original = clean_subtitle_text(seg.text)
            translated = clean_subtitle_text(seg.translated_text)

            if translated:
                text = f"{original}\n{translated}"
            else:
                text = original

            srt_lines.append(f"{n}\n{seg.to_srt_ts()}\n{text}\n")

        return "\n".join(srt_lines)

    def to_json(self) -> dict:
        """转换为 JSON 格式。

        Returns:
            以序号（字符串键）为 key 的字典
        """
        result_json: dict[str, dict] = {}
        for i, segment in enumerate(self.segments, 1):
            result_json[str(i)] = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "original_subtitle": clean_subtitle_text(segment.text),
                "translated_subtitle": clean_subtitle_text(segment.translated_text),
            }
        return result_json

    def save(self, save_path: str) -> None:
        """保存到文件。

        仅支持 .srt 和 .json 格式。

        Args:
            save_path: 输出文件路径

        Raises:
            ValueError: 不支持的文件扩展名
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        suffix = path.suffix.lower()
        if suffix == ".srt":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_srt())
        elif suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported file extension: {suffix}. Use .srt or .json")

        logger.info("Saved ASRData to %s (%d segments)", save_path, len(self.segments))

    # ------------------------------------------------------------------
    # 反序列化
    # ------------------------------------------------------------------

    @staticmethod
    def from_subtitle_file(file_path: str) -> "ASRData":
        """从字幕文件加载 ASRData。

        仅支持 .srt 和 .json。

        Args:
            file_path: 字幕文件路径

        Returns:
            解析后的 ASRData 实例

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path_obj.read_text(encoding="gbk")

        suffix = file_path_obj.suffix.lower()

        if suffix == ".srt":
            return ASRData.from_srt(content)
        elif suffix == ".json":
            return ASRData.from_json(json.loads(content))
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .srt or .json"
            )

    @staticmethod
    def from_json(json_data: dict) -> "ASRData":
        """从 JSON 数据创建 ASRData。

        Args:
            json_data: 以序号字符串为 key 的字典（与 to_json 输出一致）

        Returns:
            ASRData 实例
        """
        segments: list[ASRDataSeg] = []
        for i in sorted(json_data.keys(), key=int):
            segment_data = json_data[i]
            segment = ASRDataSeg(
                text=segment_data["original_subtitle"],
                translated_text=segment_data.get("translated_subtitle", ""),
                start_time=segment_data["start_time"],
                end_time=segment_data["end_time"],
            )
            segments.append(segment)
        return ASRData(segments)

    @staticmethod
    def from_srt(srt_str: str) -> "ASRData":
        """从 SRT 格式字符串创建 ASRData。

        使用语言检测区分双语字幕（原文+译文）和多行单语字幕。
        如果 langdetect 不可用则默认为非双语模式。

        Args:
            srt_str: SRT 格式字幕字符串

        Returns:
            解析后的 ASRData 实例
        """
        segments: list[ASRDataSeg] = []
        srt_time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s"
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
        )
        blocks = re.split(r"\n\s*\n", srt_str.strip())

        # ------------------------------------------------------------------
        # 双语检测：所有 block 都是 4 行 + 70% 以上检测到不同语言
        # ------------------------------------------------------------------
        is_bilingual = False

        if _HAS_LANGDETECT and blocks:
            def _is_different_lang(block: str) -> bool:
                lines = block.splitlines()
                if len(lines) != 4:
                    return False
                try:
                    assert langdetect_detect is not None
                    return langdetect_detect(lines[2]) != langdetect_detect(lines[3])
                except LangDetectException:
                    return False

            all_four_lines = all(len(b.splitlines()) == 4 for b in blocks)
            if all_four_lines:
                sample_size = min(50, len(blocks))
                if sample_size > 0:
                    diff_count = sum(
                        _is_different_lang(b) for b in blocks[:sample_size]
                    )
                    is_bilingual = diff_count / sample_size >= 0.7

        # ------------------------------------------------------------------
        # 解析所有 block
        # ------------------------------------------------------------------
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3:
                continue

            match = srt_time_pattern.match(lines[1])
            if not match:
                continue

            time_parts = list(map(int, match.groups()))
            start_time = (
                time_parts[0] * 3600000
                + time_parts[1] * 60000
                + time_parts[2] * 1000
                + time_parts[3]
            )
            end_time = (
                time_parts[4] * 3600000
                + time_parts[5] * 60000
                + time_parts[6] * 1000
                + time_parts[7]
            )

            if is_bilingual and len(lines) == 4:
                segments.append(
                    ASRDataSeg(lines[2], start_time, end_time, lines[3])
                )
            else:
                segments.append(
                    ASRDataSeg(" ".join(lines[2:]), start_time, end_time)
                )

        return ASRData(segments)

    # ------------------------------------------------------------------
    # 片段操作
    # ------------------------------------------------------------------

    def merge_segments(
        self,
        start_index: int,
        end_index: int,
        merged_text: Optional[str] = None,
    ) -> None:
        """合并 start_index 到 end_index（含）的片段。

        Args:
            start_index: 起始索引
            end_index: 结束索引（含）
            merged_text: 合并后的文本。若为 None 则拼接所有片段文本。

        Raises:
            IndexError: 索引无效
        """
        if (
            start_index < 0
            or end_index >= len(self.segments)
            or start_index > end_index
        ):
            raise IndexError("Invalid segment index")

        merged_start_time = self.segments[start_index].start_time
        merged_end_time = self.segments[end_index].end_time

        if merged_text is None:
            merged_text = "".join(
                seg.text for seg in self.segments[start_index : end_index + 1]
            )

        merged_seg = ASRDataSeg(merged_text, merged_start_time, merged_end_time)
        self.segments[start_index : end_index + 1] = [merged_seg]

    def merge_with_next_segment(self, index: int) -> None:
        """将指定索引的片段与下一个片段合并。

        Args:
            index: 当前片段索引

        Raises:
            IndexError: 索引越界或没有下一个片段
        """
        if index < 0 or index >= len(self.segments) - 1:
            raise IndexError("Index out of range or no next segment to merge")

        current_seg = self.segments[index]
        next_seg = self.segments[index + 1]
        merged_text = f"{current_seg.text} {next_seg.text}"
        merged_seg = ASRDataSeg(
            merged_text, current_seg.start_time, next_seg.end_time
        )
        self.segments[index] = merged_seg
        del self.segments[index + 1]

    def optimize_timing(self, threshold_ms: int = 1000) -> "ASRData":
        """优化字幕显示时序。

        如果相邻片段间隔小于阈值，将边界调整到两者之间的 3/4 点
        （减少闪烁）。对词级时间戳不做处理。

        Args:
            threshold_ms: 时间间隔阈值（毫秒，默认 1000）

        Returns:
            修改后的 ASRData 实例（self）
        """
        if self.is_word_timestamp() or not self.segments:
            return self

        for i in range(len(self.segments) - 1):
            current_seg = self.segments[i]
            next_seg = self.segments[i + 1]
            time_gap = next_seg.start_time - current_seg.end_time

            if time_gap < threshold_ms:
                mid_time = (
                    (current_seg.end_time + next_seg.start_time) // 2
                    + time_gap // 4
                )
                current_seg.end_time = mid_time
                next_seg.start_time = mid_time

        return self

    def __str__(self) -> str:
        return self.to_srt()
