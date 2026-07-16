from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any


PROTECTED_SHORT = {
    "はい",
    "うん",
    "ううん",
    "え",
    "あ",
    "いや",
    "いいえ",
    "だめ",
    "ダメ",
    "\u304a",
    "\u304a\u304a",
    "\u306f\u3042",
}
STRUCTURAL_SUFFIXES = {
    "て",
    "で",
    "し",
    "と",
    "が",
    "を",
    "は",
    "も",
    "の",
    "へ",
    "に",
    "か",
    "ね",
    "よ",
    "わ",
    "た",
    "だ",
    "って",
    "けど",
    "から",
    "ので",
    "のに",
    "という",
}
STRUCTURAL_PREFIXES = {
    "そして",
    "それで",
    "だから",
    "でも",
    "じゃあ",
    "ただ",
    "また",
    "なので",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    reports = [diagnose(Path(value)) for value in args.workdir]
    payload = {"workdirs": reports}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return 0


def diagnose(workdir: Path) -> dict[str, Any]:
    segments = load_split_segments(workdir / "split_segments.json")
    lt_500 = []
    protected_lt_120 = []
    ordinary_lt_500 = []
    very_long = []
    nonpositive = []
    overlaps = []
    duplicate_pairs = []
    buckets: dict[str, int] = {}

    for index, item in enumerate(segments):
        duration = item["end_ms"] - item["start_ms"]
        if duration <= 0:
            nonpositive.append(item)
        if duration < 500:
            classification = classify_short_segment(segments, index)
            entry = {**item, "duration_ms": duration, "classification": classification}
            lt_500.append(entry)
            buckets[classification] = buckets.get(classification, 0) + 1
            if is_protected_short(item["text"]):
                if duration < 120:
                    protected_lt_120.append(entry)
            else:
                ordinary_lt_500.append(entry)
        if duration > 8000:
            very_long.append({**item, "duration_ms": duration})
        if index > 0 and item["start_ms"] < segments[index - 1]["end_ms"]:
            overlaps.append({"previous": segments[index - 1], "current": item})
        if index > 0 and normalize(segments[index - 1]["text"]) == normalize(item["text"]) and normalize(item["text"]):
            duplicate_pairs.append({"previous": segments[index - 1], "current": item})

    return {
        "workdir": str(workdir),
        "split_count": len(segments),
        "summary": {
            "lt_500ms_count": len(lt_500),
            "ordinary_lt_500ms_count": len(ordinary_lt_500),
            "protected_lt_120ms_count": len(protected_lt_120),
            "nonpositive_count": len(nonpositive),
            "overlap_count": len(overlaps),
            "very_long_count": len(very_long),
            "adjacent_duplicate_count": len(duplicate_pairs),
            "classification_counts": buckets,
        },
        "ordinary_lt_500ms_examples": ordinary_lt_500[:30],
        "protected_lt_120ms_examples": protected_lt_120[:20],
        "duplicate_examples": duplicate_pairs[:20],
        "very_long_examples": very_long[:20],
    }


def load_split_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result = []
    for key in sorted(payload, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        item = payload[key]
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "key": str(key),
                "text": str(item.get("original_subtitle", item.get("text", ""))),
                "start_ms": int(item.get("start_time", 0)),
                "end_ms": int(item.get("end_time", 0)),
            }
        )
    return result


def classify_short_segment(segments: list[dict[str, Any]], index: int) -> str:
    item = segments[index]
    text = normalize(item["text"])
    if is_protected_short(item["text"]):
        return "protected_short"
    prev_item = segments[index - 1] if index > 0 else None
    next_item = segments[index + 1] if index + 1 < len(segments) else None
    prev_gap = item["start_ms"] - prev_item["end_ms"] if prev_item else None
    next_gap = next_item["start_ms"] - item["end_ms"] if next_item else None
    if is_structural_fragment(text):
        if prev_gap is not None and prev_gap <= 8000:
            return "tail_or_structural_fragment_merge_left"
        if next_gap is not None and next_gap <= 800:
            return "leading_fragment_merge_right"
        return "structural_fragment_no_nearby_merge"
    if next_gap is not None and next_gap <= 80:
        return "can_merge_next_short_gap"
    if prev_gap is not None and prev_gap <= 80:
        return "can_merge_prev_short_gap"
    if len(text) <= 2:
        return "unprotected_tiny_reaction_or_token"
    return "source_timing_too_short_or_boundary_error"


def is_structural_fragment(text: str) -> bool:
    return (
        text in STRUCTURAL_SUFFIXES
        or text in STRUCTURAL_PREFIXES
        or any(text.endswith(value) for value in STRUCTURAL_SUFFIXES)
        or any(text.startswith(value) for value in STRUCTURAL_PREFIXES)
    )


def is_protected_short(text: str) -> bool:
    normalized = normalize(text)
    return normalized in {normalize(value) for value in PROTECTED_SHORT}


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# P2 split 可读性诊断", ""]
    for item in payload["workdirs"]:
        summary = item["summary"]
        lines.extend(
            [
                f"## {item['workdir']}",
                "",
                f"- split 条数：{item['split_count']}",
                f"- `<500ms` 总数：{summary['lt_500ms_count']}",
                f"- 普通 `<500ms`：{summary['ordinary_lt_500ms_count']}",
                f"- 受保护 `<120ms`：{summary['protected_lt_120ms_count']}",
                f"- 非正时长：{summary['nonpositive_count']}",
                f"- 相邻重叠：{summary['overlap_count']}",
                f"- 超长 `>8000ms`：{summary['very_long_count']}",
                f"- 相邻同文重复：{summary['adjacent_duplicate_count']}",
                f"- 分类计数：`{json.dumps(summary['classification_counts'], ensure_ascii=False)}`",
                "",
            ]
        )
        if item["ordinary_lt_500ms_examples"]:
            lines.append("### 普通 `<500ms` 样例")
            lines.append("")
            for example in item["ordinary_lt_500ms_examples"][:20]:
                lines.append(
                    f"- #{example['key']} {example['duration_ms']}ms "
                    f"{example['classification']} `{example['text']}` "
                    f"({example['start_ms']}->{example['end_ms']})"
                )
            lines.append("")
        if item["protected_lt_120ms_examples"]:
            lines.append("### 受保护 `<120ms` 样例")
            lines.append("")
            for example in item["protected_lt_120ms_examples"][:20]:
                lines.append(
                    f"- #{example['key']} {example['duration_ms']}ms `{example['text']}` "
                    f"({example['start_ms']}->{example['end_ms']})"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
