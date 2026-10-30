"""Direct normal-import of run_correction_stage. Not the host candidate.

Reads a request JSON and writes a summary JSON. Does not start the legacy CLI.
"""
import dataclasses
import hashlib
import json
import os
import pathlib
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(request):
    root = pathlib.Path(request["legacy_root"])
    profile = json.loads(pathlib.Path(request["profile"]).read_text(encoding="utf-8"))
    sources = profile["legacy_sources"]
    if pathlib.Path(sources["root"]).resolve() != root.resolve():
        raise SystemExit("legacy root mismatch")
    for item in sources["files"]:
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise SystemExit("legacy hash mismatch: " + item["path"])
    sys.path.insert(0, str(root))
    os.environ.pop("PYTHONPYCACHEPREFIX", None)
    from qwen_asr.commands.stages import load_transcripts
    from qwen_asr.corrector import run_correction_stage
    from qwen_asr.models import TranscriptSegment, WorkPaths
    records = []
    for case in request["cases"]:
        source = pathlib.Path(case["raw"])
        before = source.read_bytes()
        backup = pathlib.Path(case["backup"])
        transcript = pathlib.Path(case["transcript"])
        report = pathlib.Path(case["report"])
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(before)
        rows = json.loads(before.decode("utf-8-sig"))
        items = [TranscriptSegment(**row) for row in rows]
        work = WorkPaths.from_workdir(transcript.parent)
        work.raw_transcript_manifest = backup
        work.transcript_manifest = transcript
        work.corrected_manifest = report
        corrected, report_items = run_correction_stage(work, items, "", "", "", mode="rules")
        loaded = load_transcripts(transcript)
        if source.read_bytes() != before:
            raise SystemExit("source raw changed: " + case["id"])
        names = {type(item).__name__ for item in report_items}
        if report_items and names != {"CorrectionReportItem"}:
            raise SystemExit("report kind mismatch: " + case["id"])
        records.append({
            "id": case["id"],
            "report_kind": "CorrectionReportItem",
            "corrected_count": len(corrected),
            "report_count": len(report_items),
            "changed": sum(1 for item in report_items if item.changed),
            "failed": sum(1 for item in report_items if item.status != "completed"),
            "load_transcripts": [item.text for item in loaded] == [item.text for item in corrected],
            "input_sha256": hashlib.sha256(before).hexdigest(),
            "raw_backup_sha256": sha(backup),
            "transcript_sha256": sha(transcript),
            "report_sha256": sha(report),
            "report_fields": sorted(dataclasses.asdict(report_items[0]).keys()) if report_items else [],
        })
    return {"cases": records}


if __name__ == "__main__":
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run(payload)
    pathlib.Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
