"""字幕文本对齐器 — 移植自 VideoCaptioner app/core/split/alignment.py

基于 difflib.ndiff 的文本序列对齐。当目标文本缺少某项时，
使用其上一项进行填充，确保输出两个等长列表。

使用示例:
    text1 = ['ab', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    text2 = ['a',  'b', 'c', 'd', 'f', 'g', 'h', 'i']

    aligner = SubtitleAligner()
    aligned_source, aligned_target = aligner.align_texts(text1, text2)
    # aligned_target 中缺失的 'e' 位置由 'd' 填充
"""

import difflib
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("optimizer.alignment")


class SubtitleAligner:
    """字幕文本对齐器，用于对齐两个文本序列，支持基于相似度的匹配。"""

    def __init__(self) -> None:
        self.line_numbers: list[int] = [0, 0]

    def align_texts(
        self, source_text: List[str], target_text: List[str]
    ) -> Tuple[List[str], List[str]]:
        """对齐两个文本序列并返回配对行。

        Args:
            source_text: 源文本行列表。
            target_text: 目标文本行列表。

        Returns:
            (aligned_source, aligned_target) 两个等长列表。
        """
        # 重置行号计数器以支持多次调用
        self.line_numbers = [0, 0]
        diff_iterator = difflib.ndiff(source_text, target_text)
        return self._pair_lines(diff_iterator)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _pair_lines(
        self, diff_iterator: object
    ) -> Tuple[List[str], List[str]]:
        """从 diff 迭代器中配对行。"""
        source_lines: List[str] = []
        target_lines: List[str] = []
        flag = 0

        for source_line, target_line, _ in self._line_iterator(diff_iterator):
            if source_line is not None:
                if source_line[1] == "\n":
                    flag += 1
                    continue
                source_lines.append(source_line[1])
            if target_line is not None:
                if flag > 0:
                    flag -= 1
                    continue
                target_lines.append(target_line[1])

        # 缺失项用上一项填充
        for i in range(1, len(target_lines)):
            if target_lines[i] == "\n":
                target_lines[i] = target_lines[i - 1]

        return source_lines, target_lines

    def _line_iterator(self, diff_iterator: object):  # noqa: C901
        """遍历 diff 行并逐对 yield。

        Yields:
            (source_line, target_line, has_diff)
        """
        lines: list[str] = []
        blank_lines_pending = 0
        blank_lines_to_yield = 0

        source_line: Optional[Tuple[int, str]]
        target_line: Optional[Tuple[int, str]]

        while True:
            while len(lines) < 4:
                lines.append(next(diff_iterator, "X"))  # type: ignore[arg-type]

            diff_type = "".join([line[0] for line in lines])

            if diff_type.startswith("X"):
                blank_lines_to_yield = blank_lines_pending
            elif diff_type.startswith("-?+?"):
                yield (
                    self._format_line(lines, "?", 0),
                    self._format_line(lines, "?", 1),
                    True,
                )
                continue
            elif diff_type.startswith("--++"):
                blank_lines_pending -= 1
                yield self._format_line(lines, "-", 0), None, True
                continue
            elif diff_type.startswith(("--?+", "--+", "- ")):
                source_line = self._format_line(lines, "-", 0)
                target_line = None
                blank_lines_to_yield, blank_lines_pending = blank_lines_pending - 1, 0
            elif diff_type.startswith("-+?"):
                yield (
                    self._format_line(lines, None, 0),
                    self._format_line(lines, "?", 1),
                    True,
                )
                continue
            elif diff_type.startswith("-?+"):
                yield (
                    self._format_line(lines, "?", 0),
                    self._format_line(lines, None, 1),
                    True,
                )
                continue
            elif diff_type.startswith("-"):
                blank_lines_pending -= 1
                yield self._format_line(lines, "-", 0), None, True
                continue
            elif diff_type.startswith("+--"):
                blank_lines_pending += 1
                yield None, self._format_line(lines, "+", 1), True
                continue
            elif diff_type.startswith(("+ ", "+-")):
                source_line = None
                target_line = self._format_line(lines, "+", 1)
                blank_lines_to_yield, blank_lines_pending = blank_lines_pending + 1, 0
            elif diff_type.startswith("+"):
                blank_lines_pending += 1
                yield None, self._format_line(lines, "+", 1), True
                continue
            elif diff_type.startswith(" "):
                yield (
                    self._format_line(lines[:], None, 0),
                    self._format_line(lines, None, 1),
                    False,
                )
                continue
            else:
                # 未知模式，跳过一行避免死循环
                lines.pop(0)
                continue

            while blank_lines_to_yield < 0:
                blank_lines_to_yield += 1
                yield None, ("", "\n"), True
            while blank_lines_to_yield > 0:
                blank_lines_to_yield -= 1
                yield ("", "\n"), None, True

            if diff_type.startswith("X"):
                return
            else:
                yield source_line, target_line, True

    def _format_line(
        self, lines: list[str], format_key: Optional[str], side: int
    ) -> Tuple[int, str]:
        """格式化一行并返回 (行号, 文本)。

        Args:
            lines: 待处理行列表（会 pop）。
            format_key: 格式类型 ('?', '-', '+', 或 None)。
            side: 0=source, 1=target。
        """
        self.line_numbers[side] += 1
        if format_key is None:
            return self.line_numbers[side], lines.pop(0)[2:]
        if format_key == "?":
            text = lines.pop(0)
            lines.pop(0)  # 跳过标记行
            text = text[2:]
        else:
            text = lines.pop(0)[2:]
            if not text:
                text = ""
        return self.line_numbers[side], text
