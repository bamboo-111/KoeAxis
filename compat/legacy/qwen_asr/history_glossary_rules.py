from __future__ import annotations

import re
from typing import Any

from qwen_asr.glossary import GlossaryEntry

JP_TERMS = ("ラジ", "コエログ", "Zepp", "Girls Band Cry", "トゲ")
CN_TERMS = ("广播", "声log", "声日志", "巡演", "乐队", "栏目")
JP_CURATION_HINTS = (
    "トゲ",
    "コエログ",
    "ラジ",
    "Zepp",
    "Girls",
    "Band",
    "川崎",
    "監督",
    "学院",
    "先生",
    "理名",
    "夕莉",
    "朱李",
    "凪都",
)
CN_CURATION_HINTS = (
    "广播",
    "声log",
    "声日志",
    "巡演",
    "川崎",
    "导演",
    "学院",
    "老师",
    "理名",
    "夕莉",
    "朱李",
    "凪都",
    "无刺有刺",
)


def normalize_glossary_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip(" \t\r\n。！？!?，,、;；：:\"'“”‘’")


def looks_like_glossary_candidate(source: str, target: str, item: Any) -> bool:
    if not source or not target:
        return False
    if len(source) > 18 or len(target) > 18:
        return False
    if "\n" in source or "\n" in target:
        return False
    if item.matched_segment_count > 2:
        return False
    if item.length_ratio_score < 0.55:
        return False
    if item.time_overlap_score < 0.7:
        return False
    if item.token_coverage_score < 0.7:
        return False
    if contains_ascii_word(source) and len(source) < 3:
        return False
    if source.lower() == target.lower():
        return False
    if re.fullmatch(r"[A-Za-z0-9 .!?'_-]+", target):
        return False
    if re.fullmatch(r"[A-Za-z0-9 .!?'_-]+", source) and len(source) <= 6:
        return False
    if any(mark in source for mark in ("。", "？", "！", "…")) and len(source) > 12:
        return False
    if any(mark in target for mark in ("。", "？", "！", "…")) and len(target) > 12:
        return False
    if not (has_cjk(source) or has_kana(source) or contains_ascii_word(source)):
        return False
    if not (has_cjk(target) or contains_ascii_word(target)):
        return False
    return True


def guess_glossary_group(source: str, target: str) -> str:
    if any(token in source for token in JP_TERMS) or any(token in target for token in CN_TERMS):
        return "show_terms"
    if any(keyword in source for keyword in ("監督", "学院", "先生", "Zepp", "コエログ", "トゲラジ")):
        return "show_terms"
    if any(keyword in target for keyword in ("导演", "学院", "老师")):
        return "show_terms"
    if 1 < len(source) <= 6 and 1 < len(target) <= 6:
        return "names"
    return "fixed_phrases"


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def has_kana(text: str) -> bool:
    return any(("\u3040" <= char <= "\u30ff") for char in text)


def contains_ascii_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text))


def is_glossary_like_pair(source: str, target: str) -> bool:
    keyword_hit = any(
        keyword in source or keyword in target
        for keyword in (
            "トゲ",
            "コエログ",
            "Zepp",
            "Girls",
            "Band",
            "学院",
            "广播",
            "声log",
            "声日志",
            "巡演",
            "导演",
            "老师",
        )
    )
    if keyword_hit:
        return True

    source_len = len(source)
    target_len = len(target)
    if source_len <= 6 and target_len <= 6 and (has_kana(source) or has_cjk(source)):
        return True
    if source_len <= 10 and target_len <= 10 and (has_kana(source) or contains_ascii_word(source)) and has_cjk(target):
        return True
    return False


def is_curated_priority(source: str, target: str, items: list[Any]) -> bool:
    if any(hint in source for hint in JP_CURATION_HINTS):
        return True
    if any(hint in target for hint in CN_CURATION_HINTS):
        return True
    if len(source) <= 6 and len(target) <= 8 and (has_kana(source) or has_cjk(source)) and has_cjk(target):
        return True
    if any(item.score >= 0.9 for item in items) and len(source) <= 12 and len(target) <= 12:
        return True
    return False


def is_llm_glossary_entry_allowed(entry: GlossaryEntry) -> bool:
    source = entry.source
    target = entry.target
    if len(source) > 20 or len(target) > 20:
        return False
    if looks_like_sentence_text(source) or looks_like_sentence_text(target):
        return False
    if looks_like_contextual_role_phrase(source, target):
        return False
    if entry.group == "names" and (len(source) > 8 or len(target) > 8):
        return False
    if entry.group == "fixed_phrases" and len(target) > 12:
        return False
    if entry.group == "show_terms" and len(target) > 16:
        return False
    return True


def looks_like_sentence_text(text: str) -> bool:
    sentence_markers = (
        "我们",
        "你们",
        "你",
        "我",
        "大家",
        "因为",
        "所以",
        "如果",
        "但是",
        "然后",
        "的话",
        "了",
        "呢",
        "吗",
        "吧",
        "就是",
        "不是",
        "真的",
        "感觉",
        "觉得",
        "想",
        "要",
        "会",
    )
    if len(text) >= 14:
        return True
    if any(marker in text for marker in sentence_markers) and len(text) >= 8:
        return True
    if any(mark in text for mark in ("。", "！", "？", "，", ",")):
        return True
    return False


def looks_like_contextual_role_phrase(source: str, target: str) -> bool:
    role_markers = ("导演", "老师", "同学", "嘉宾", "主持", "监督", "担当", "桑", "さん", "君", "ちゃん")
    if any(marker in target for marker in role_markers) and len(target) >= 4:
        return True
    if any(marker in source for marker in role_markers) and len(source) >= 4:
        return True
    return False


def score_to_level(score: float, min_match_score: float) -> str:
    if score >= min_match_score:
        return "high"
    if score >= max(0.5, min_match_score - 0.18):
        return "medium"
    return "low"
