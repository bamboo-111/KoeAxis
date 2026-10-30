from __future__ import annotations

import re

def _normalize_export_email_variants(text: str) -> str:
    text = re.sub(
        r"(?i)トゲラジ\s*(?:at|@)\s*interfmjptogerajiinterfmjp",
        "トゲラジ togeraji@interfm.jp",
        text,
    )
    text = re.sub(r"(?i)togeraji\s+interfm\s*j\s*p\b", "togeraji@interfm.jp", text)
    text = re.sub(r"(?i)togeraji\s+interfm\.jp\b", "togeraji@interfm.jp", text)
    text = re.sub(r"(?i)togeraji\s*interfmjp\b", "togeraji@interfm.jp", text)
    text = re.sub(r"(?i)togerajiinterfmjp\b", "togeraji@interfm.jp", text)
    text = re.sub(
        r"(?i)トゲラジ\s*(?:at|@)\s*interfm\s*j\s*p",
        "トゲラジ togeraji@interfm.jp",
        text,
    )
    text = re.sub(
        r"(?i)トゲラジ\s*(?:at|@)\s*interfm\.jp",
        "トゲラジ togeraji@interfm.jp",
        text,
    )
    text = re.sub(
        r"(?i)トゲラジ\s*(?:at|@)\s*interfmjp",
        "トゲラジ togeraji@interfm.jp",
        text,
    )
    text = re.sub(r"(?i)interfm\s*j\s*p", "interfm.jp", text)
    text = re.sub(r"(?i)interfmjp", "interfm.jp", text)
    return text

def _normalize_radio_addresses(text: str) -> str:
    text = re.sub(
        r"(?i)(トゲラジ)\s*(?:at|@)?\s*(?:in-?t|inter|int)?fm\.?j?p?(?:d\s*)?(?:[a-z]\s*){0,16}(?:at|@)?\s*(?:in-?t|inter|int)?fm\.?j?p?",
        r"\1 togeraji@interfm.jp",
        text,
    )
    text = re.sub(
        r"(?i)(?:toge|toga|doge|doga|togare|dogea)\s*ラ?ジ?\s*(?:at|@)\s*(?:in-?t|inter|int)?fm\.?j?p?",
        "togeraji@interfm.jp",
        text,
    )
    text = re.sub(
        r"(?i)(?:t\s*o\s*g\s*e\s*r?\s*a?\s*j?\s*i|d\s*o\s*g\s*e\s*a\s*d\s*i|t\s*o\s*g\s*a\s*r\s*e\s*d\s*i)\s*(?:at|@)\s*(?:in-?t|inter|int)?fm\.?j?p?",
        "togeraji@interfm.jp",
        text,
    )
    text = re.sub(
        r"(?i)(?:in-?ta?fm|intfm|interfm)\s*\.?\s*j?p?",
        "interfm.jp",
        text,
    )
    return text

def _normalize_radio_fixed_terms(text: str) -> str:
    replacements = (
        ("ハシタグ", "ハッシュタグ"),
        ("ハッシュタグトゲラシ", "ハッシュタグトゲラジ"),
        ("棘ラジ", "トゲラジ"),
        ("トゲラシ", "トゲラジ"),
        ("トゲイ", "トゲアリ"),
        ("トゲラジトゲアリ", "トゲナシトゲアリ"),
        ("ゲアリー", "トゲアリ"),
        ("トゲナシトゲアリー", "トゲナシトゲアリ"),
        ("トゲナシとゲアリ", "トゲナシトゲアリ"),
        ("トゲナシトギアリ", "トゲナシトゲアリ"),
        ("棘なし棘あり", "トゲナシトゲアリ"),
        ("シュリー", "朱李"),
        ("シュリ", "朱李"),
        ("修理さん", "朱李さん"),
        ("秀利さん", "朱李さん"),
        ("手首さん", "朱李さん"),
        ("ビナ", "仁菜"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text
