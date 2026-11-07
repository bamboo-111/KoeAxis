"""Read-only field comparison against baseline B. Not a correction input."""
import hashlib
import json
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[3] / "compat/legacy/workspaces/radio-54-togeradi-20261018"
RAW = BASELINE / "manifests/transcript_segments.raw.json"
TRANSCRIPT = BASELINE / "manifests/transcript_segments.json"
REPORT = BASELINE / "manifests/corrected_segments.json"
RUN_LOG = BASELINE / "logs/run.log"
GPU_LOG = BASELINE / "logs/pipeline-gpu.log"
SEGMENT_FIELDS = (
    "segment_id", "audio_path", "global_start_time", "global_end_time", "text",
    "language", "raw_model_output", "status", "error")
REPORT_FIELDS = (
    "segment_id", "original_text", "corrected_text", "changed", "reason", "status", "error")


def load(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def _pair(left, right):
    left_map = {row["segment_id"]: row for row in left}
    right_map = {row["segment_id"]: row for row in right}
    return left_map, right_map, [row["segment_id"] for row in left], [row["segment_id"] for row in right]


def field_diffs(left, right, fields):
    left_map, right_map, left_ids, right_ids = _pair(left, right)
    diffs = []
    if left_ids != right_ids:
        diffs.append({"field": "segment_id_order", "left": left_ids, "right": right_ids})
    for segment_id in left_ids:
        if segment_id not in right_map:
            continue
        for field in fields:
            if field == "segment_id":
                continue
            a = left_map[segment_id].get(field)
            b = right_map[segment_id].get(field)
            if a != b:
                item = {"segment_id": segment_id, "field": field}
                if field in ("text", "original_text", "corrected_text", "reason", "status", "error"):
                    item["ours"] = a
                    item["baseline"] = b
                else:
                    item["values_omitted"] = True
                diffs.append(item)
    return diffs


def log_lines():
    notes = []
    if RUN_LOG.is_file():
        for number, line in enumerate(RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "ASR correction report written" in line or "LLM correction:" in line:
                notes.append({"file": "logs/run.log", "line": number, "text": line.strip()})
    if GPU_LOG.is_file():
        for number, line in enumerate(GPU_LOG.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "corrected_segments.json already exists" in line or "LLM correction:" in line:
                notes.append({"file": "logs/pipeline-gpu.log", "line": number, "text": line.strip()})
    return notes


def against_baseline(our_raw, our_transcript, our_report, input_sha256, byte_flags):
    b_raw = load(RAW)
    b_transcript = load(TRANSCRIPT)
    b_report = load(REPORT)
    reasons = {}
    for item in b_report:
        reasons[item.get("reason")] = reasons.get(item.get("reason"), 0) + 1
    transcript_vs_report = 0
    by_id = {row["segment_id"]: row for row in b_transcript}
    for item in b_report:
        row = by_id.get(item["segment_id"])
        if row is None or row.get("text") != item.get("corrected_text"):
            transcript_vs_report += 1
    text_diffs = field_diffs(our_transcript, b_transcript, SEGMENT_FIELDS)
    report_diffs = field_diffs(our_report, b_report, REPORT_FIELDS)
    raw_diffs = field_diffs(our_raw, b_raw, SEGMENT_FIELDS)
    llm_reasons = sum(1 for item in b_report if "llm:" in str(item.get("reason", "")).lower())
    if not text_diffs and not report_diffs:
        relation = "field-identical"
    else:
        relation = "differs"
    return {
        "baseline_root": str(BASELINE).replace("\\", "/"),
        "input_sha256": input_sha256,
        "baseline_raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "baseline_transcript_sha256": hashlib.sha256(TRANSCRIPT.read_bytes()).hexdigest(),
        "baseline_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "input_bytes_equal_baseline_raw": byte_flags["input_bytes_equal_baseline_raw"],
        "transcript_bytes_equal_baseline": byte_flags["transcript_bytes_equal_baseline"],
        "report_bytes_equal_baseline": byte_flags["report_bytes_equal_baseline"],
        "baseline_reason_counts": reasons,
        "baseline_llm_reason_rows": llm_reasons,
        "baseline_transcript_text_ne_report": transcript_vs_report,
        "baseline_changed": sum(1 for item in b_report if item.get("changed")),
        "raw_field_diff_count": len(raw_diffs),
        "transcript_field_diff_count": len(text_diffs),
        "report_field_diff_count": len(report_diffs),
        "raw_field_diffs": raw_diffs,
        "transcript_field_diffs": text_diffs,
        "report_field_diffs": report_diffs,
        "relation": relation,
        "semantic_source": (
            "B corrected_segments reasons are the rules-mode strings from "
            "run_correction_stage (deterministic text cleanup / kept; no llm: prefix). "
            "run.log records the correction report immediately after ASR and before the aligner load. "
            "pipeline-gpu.log later skipped correction because corrected_segments.json already existed. "
            "B transcript text matches that report's corrected_text, so the stored transcript was not "
            "rewritten away from the rules report. Differences, if any, stay differs; rules are not adjusted."
        ),
        "log": log_lines(),
        "chain_passed": False,
        "align_completed": False,
        "migration_equivalence": "not-assessed-beyond-this-comparison",
    }
