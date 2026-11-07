from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import unicodedata
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    reports = [diagnose_workdir(Path(value)) for value in args.workdir]
    payload = {"workdirs": reports}

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return 0


def diagnose_workdir(workdir: Path) -> dict[str, Any]:
    aligned_path = workdir / "aligned_segments.json"
    transcript_path = workdir / "transcript_segments.json"
    split_path = workdir / "split_segments.json"
    aligned = read_json(aligned_path, [])
    transcript = read_json(transcript_path, [])
    split = read_json(split_path, {})

    source_segments = build_current_split_source(workdir, aligned, transcript)
    source_text = normalize("".join(item["text"] for item in source_segments))
    token_text = normalize("".join(token_texts(aligned)))
    aligned_item_text = normalize("".join(str(item.get("text", "")) for item in aligned if isinstance(item, dict)))
    transcript_text = normalize("".join(str(item.get("text", "")) for item in transcript if isinstance(item, dict)))
    split_segments = split_to_segments(split)
    split_text = normalize("".join(item["text"] for item in split_segments))

    opcodes = list(difflib.SequenceMatcher(None, source_text, split_text, autojunk=False).get_opcodes())
    source_map = char_map(source_segments)
    split_map = char_map(split_segments)
    diff_examples = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        diff_examples.append(
            {
                "opcode": tag,
                "source_range": [i1, i2],
                "split_range": [j1, j2],
                "source_text": source_text[i1:i2],
                "split_text": split_text[j1:j2],
                "source_locations": locations_for_range(source_map, i1, i2),
                "split_locations": locations_for_range(split_map, j1, j2),
            }
        )
        if len(diff_examples) >= 30:
            break

    fallback_reasons: dict[str, int] = {}
    for item in source_segments:
        reason = str(item.get("source_reason", "tokens"))
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1

    protected = {"はい", "うん", "ううん", "え", "いや", "いいえ", "だめ", "ダメ"}
    source_short = count_exact_short(source_segments, protected)
    split_short = count_exact_short(split_segments, protected)
    protected_normalized = {normalize(value) for value in protected}
    duplicate_report = classify_adjacent_duplicates(source_segments, split_segments)

    return {
        "workdir": str(workdir),
        "files": {
            "aligned_sha256": sha256(aligned_path),
            "transcript_sha256": sha256(transcript_path),
            "split_sha256": sha256(split_path),
        },
        "counts": {
            "aligned_items": len(aligned) if isinstance(aligned, list) else None,
            "source_segments": len(source_segments),
            "split_segments": len(split_segments),
            "token_normalized_chars": len(token_text),
            "aligned_item_normalized_chars": len(aligned_item_text),
            "transcript_normalized_chars": len(transcript_text),
            "current_split_source_normalized_chars": len(source_text),
            "split_normalized_chars": len(split_text),
            "matched_chars": sum(i2 - i1 for tag, i1, i2, _j1, _j2 in opcodes if tag == "equal"),
            "delete_count": sum(i2 - i1 for tag, i1, i2, _j1, _j2 in opcodes if tag == "delete"),
            "insert_count": sum(j2 - j1 for tag, _i1, _i2, j1, j2 in opcodes if tag == "insert"),
            "replace_count": sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in opcodes if tag == "replace"),
        },
        "status": "PASS" if source_text == split_text else "FAIL",
        "source_fallback_reason_counts": fallback_reasons,
        "short_response_counts": {
            "source": source_short,
            "split": split_short,
        },
        "protected_occurrences": {
            "source": count_occurrences(source_text, protected_normalized),
            "split": count_occurrences(split_text, protected_normalized),
        },
        "adjacent_duplicates": duplicate_report,
        "diff_examples": diff_examples,
    }


def build_current_split_source(
    workdir: Path,
    aligned: Any,
    transcript: Any,
) -> list[dict[str, Any]]:
    from optimizer.asr_data import ASRData, ASRDataSeg
    from qwen_asr.models import WorkPaths
    from qwen_asr.optimizer_bridge import aligned_manifest_to_asr_data

    try:
        data = aligned_manifest_to_asr_data(WorkPaths.from_workdir(workdir), ASRData, ASRDataSeg)
    except Exception:
        data = None
    if data is not None:
        return [
            {
                "id": str(index + 1),
                "text": str(segment.text),
                "source_reason": "current-code",
                "start_ms": int(segment.start_time),
                "end_ms": int(segment.end_time),
            }
            for index, segment in enumerate(data.segments)
        ]

    transcript_by_id = {
        str(item.get("segment_id")): item
        for item in transcript
        if isinstance(item, dict)
    }
    result: list[dict[str, Any]] = []
    if not isinstance(aligned, list):
        return result
    for item in aligned:
        if not isinstance(item, dict):
            continue
        segment_id = str(item.get("segment_id", ""))
        transcript_text = str(transcript_by_id.get(segment_id, {}).get("text", "")).strip()
        tokens = [
            token
            for token in item.get("tokens", [])
            if isinstance(token, dict) and str(token.get("text", "")).strip()
        ]
        token_norm = normalize("".join(str(token.get("text", "")) for token in tokens))
        transcript_norm = normalize(transcript_text)
        use_transcript = False
        reason = "tokens"
        if item.get("status") != "completed":
            use_transcript = bool(transcript_text)
            reason = "status-not-completed"
        elif transcript_text and token_norm != transcript_norm:
            use_transcript = True
            reason = "content-changed"
        elif not tokens and transcript_text:
            use_transcript = True
            reason = "empty-tokens"
        if use_transcript:
            result.append(
                {
                    "id": segment_id,
                    "text": transcript_text,
                    "source_reason": reason,
                    "start_ms": int(round(float(item.get("global_start_time", 0.0)) * 1000)),
                    "end_ms": int(round(float(item.get("global_end_time", 0.0)) * 1000)),
                }
            )
            continue
        for index, token in enumerate(tokens):
            result.append(
                {
                    "id": f"{segment_id}:token:{index + 1}",
                    "text": str(token.get("text", "")),
                    "source_reason": reason,
                    "start_ms": int(round(float(token.get("start_time", 0.0)) * 1000)),
                    "end_ms": int(round(float(token.get("end_time", 0.0)) * 1000)),
                }
            )
    return result


def split_to_segments(split: Any) -> list[dict[str, Any]]:
    if isinstance(split, dict):
        pairs = sorted(split.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        return [
            {
                "id": str(key),
                "text": str(value.get("original_subtitle", value.get("text", ""))),
                "start_ms": int(value.get("start_time", 0)),
                "end_ms": int(value.get("end_time", 0)),
            }
            for key, value in pairs
            if isinstance(value, dict)
        ]
    if isinstance(split, list):
        return [
            {
                "id": str(index + 1),
                "text": str(value.get("original_subtitle", value.get("text", ""))),
                "start_ms": int(value.get("start_time", 0)),
                "end_ms": int(value.get("end_time", 0)),
            }
            for index, value in enumerate(split)
            if isinstance(value, dict)
        ]
    return []


def token_texts(aligned: Any) -> list[str]:
    if not isinstance(aligned, list):
        return []
    return [
        str(token.get("text", ""))
        for item in aligned
        if isinstance(item, dict)
        for token in item.get("tokens", [])
        if isinstance(token, dict)
    ]


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def char_map(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for segment in segments:
        for character in normalize(str(segment["text"])):
            if character.isalnum():
                mapped.append(segment)
    return mapped


def locations_for_range(
    mapped: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    locations = []
    for item in mapped[start:end]:
        key = str(item["id"])
        if key in seen:
            continue
        seen.add(key)
        locations.append(
            {
                "id": key,
                "text": str(item["text"]),
                "time_ms": [item.get("start_ms"), item.get("end_ms")],
                "source_reason": item.get("source_reason"),
            }
        )
        if len(locations) >= 8:
            break
    return locations


def count_exact_short(
    segments: list[dict[str, Any]],
    protected: set[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in segments:
        text = normalize(str(item["text"]))
        if text in protected:
            counts[text] = counts.get(text, 0) + 1
    return counts


def count_occurrences(text: str, protected: set[str]) -> dict[str, int]:
    return {
        value: text.count(value)
        for value in sorted(protected)
        if value and text.count(value)
    }


def classify_adjacent_duplicates(
    source_segments: list[dict[str, Any]],
    split_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    source_pairs = adjacent_duplicate_pairs(source_segments)
    split_pairs = adjacent_duplicate_pairs(split_segments)
    introduced = []
    inherited = []
    for pair in split_pairs:
        if has_source_duplicate_near(source_segments, source_pairs, pair):
            inherited.append(pair)
        else:
            introduced.append(pair)
    return {
        "split_adjacent_duplicate_count": len(split_pairs),
        "introduced_adjacent_duplicate_count": len(introduced),
        "inherited_adjacent_duplicate_count": len(inherited),
        "introduced_examples": introduced[:20],
        "inherited_examples": inherited[:20],
    }


def adjacent_duplicate_pairs(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for left, right in zip(segments, segments[1:], strict=False):
        left_text = normalize(str(left["text"]))
        right_text = normalize(str(right["text"]))
        if not left_text or left_text != right_text:
            continue
        result.append(
            {
                "text": left_text,
                "left_id": str(left["id"]),
                "right_id": str(right["id"]),
                "left_time_ms": [left.get("start_ms"), left.get("end_ms")],
                "right_time_ms": [right.get("start_ms"), right.get("end_ms")],
                "left_text": str(left["text"]),
                "right_text": str(right["text"]),
            }
        )
    return result


def has_source_duplicate_near(
    source_segments: list[dict[str, Any]],
    source_pairs: list[dict[str, Any]],
    split_pair: dict[str, Any],
) -> bool:
    split_start = int(split_pair["left_time_ms"][0] or 0)
    split_end = int(split_pair["right_time_ms"][1] or split_start)
    for pair in source_pairs:
        if pair["text"] != split_pair["text"]:
            continue
        source_start = int(pair["left_time_ms"][0] or 0)
        source_end = int(pair["right_time_ms"][1] or source_start)
        if source_start <= split_end + 500 and source_end >= split_start - 500:
            return True
    phrase = str(split_pair["text"])
    if not phrase:
        return False
    window_text = "".join(
        normalize(str(item["text"]))
        for item in source_segments
        if int(item.get("end_ms") or 0) >= split_start - 500
        and int(item.get("start_ms") or 0) <= split_end + 500
    )
    return window_text.count(phrase) >= 2


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# P0 split 内容守恒诊断", ""]
    for item in payload["workdirs"]:
        counts = item["counts"]
        lines.extend(
            [
                f"## {item['workdir']}",
                "",
                f"- 状态：{item['status']}",
                f"- 当前 split 输入规范化字符：{counts['current_split_source_normalized_chars']}",
                f"- split 输出规范化字符：{counts['split_normalized_chars']}",
                f"- aligned token 字符：{counts['token_normalized_chars']}",
                f"- transcript 字符：{counts['transcript_normalized_chars']}",
                f"- source segments / split segments：{counts['source_segments']} / {counts['split_segments']}",
                f"- delete / insert / replace：{counts['delete_count']} / {counts['insert_count']} / {counts['replace_count']}",
                f"- source fallback reasons：`{json.dumps(item['source_fallback_reason_counts'], ensure_ascii=False)}`",
                f"- 短应答计数 source：`{json.dumps(item['short_response_counts']['source'], ensure_ascii=False)}`",
                f"- 短应答计数 split：`{json.dumps(item['short_response_counts']['split'], ensure_ascii=False)}`",
                f"- 受保护短应答出现次数 source：`{json.dumps(item['protected_occurrences']['source'], ensure_ascii=False)}`",
                f"- 受保护短应答出现次数 split：`{json.dumps(item['protected_occurrences']['split'], ensure_ascii=False)}`",
                f"- split 相邻重复：{item['adjacent_duplicates']['split_adjacent_duplicate_count']}，"
                f"其中引入重复：{item['adjacent_duplicates']['introduced_adjacent_duplicate_count']}，"
                f"源输入继承重复：{item['adjacent_duplicates']['inherited_adjacent_duplicate_count']}",
                "",
            ]
        )
        if item["diff_examples"]:
            lines.append("### 前 30 个差异样例")
            lines.append("")
            for example in item["diff_examples"][:30]:
                lines.append(
                    f"- {example['opcode']} source[{example['source_range'][0]}:{example['source_range'][1]}] "
                    f"split[{example['split_range'][0]}:{example['split_range'][1]}]："
                    f"`{example['source_text']}` -> `{example['split_text']}`"
                )
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
