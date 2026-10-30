"""stage.correct/run_bound. Thin wrapper around the frozen rules closure.

Does not reimplement cleanup. Does not read baseline B. Does not copy audio.
"""
import hashlib
import json
from pathlib import Path

from host.contracts import ModuleError, effect

LEGACY_ROOT = Path(__file__).resolve().parents[3] / "compat" / "legacy"
REPORT_KIND = "CorrectionReportItem"


def _under(path, root):
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise ModuleError("MissingInput", "raw path cannot be resolved") from exc
    return resolved == root or root in resolved.parents


class Implementation:
    def __init__(self, settings):
        self.settings = settings

    def invoke(self, operation, arguments, api):
        if operation != "run_bound":
            raise ModuleError("UnknownOperation", operation)
        if arguments["mode"] != "rules":
            raise ModuleError(
                "UnsupportedCoverage",
                "only mode=rules is covered; other modes including llm are refused")
        source = Path(arguments["raw_source"])
        legacy = LEGACY_ROOT.resolve()
        if _under(source, legacy):
            raise ModuleError(
                "PermissionDenied",
                "refusing a legacy-repository path; baseline B is not an input")
        if not source.is_file():
            raise ModuleError(
                "MissingInput",
                "explicit raw manifest is missing; refusing without reading baseline B")
        locators = [arguments[name]["locator"] for name in
                    ("raw_backup", "transcript_manifest", "report_manifest")]
        if len(set(locators)) != 3:
            raise ModuleError("UnsupportedCoverage", "raw backup, transcript and report must be distinct")
        backup = api.path(arguments["raw_backup"], "write")
        transcript = api.path(arguments["transcript_manifest"], "write")
        report = api.path(arguments["report_manifest"], "write")
        before = source.read_bytes()
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(before)
        if source.read_bytes() != before:
            raise ModuleError("InputOverwritten", "explicit raw changed while copying the backup")
        try:
            rows = json.loads(before.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ModuleError("InvalidRaw", "raw manifest is not utf-8 json") from exc
        if type(rows) is not list:
            raise ModuleError("InvalidRaw", "raw manifest must be a JSON list")
        closure = api.legacy_modules()
        written = {"raw_backup": backup.is_file(), "transcript_manifest": False, "report_manifest": False}
        try:
            with closure.activate():
                try:
                    models = closure["qwen_asr.models"]
                    items = [models.TranscriptSegment(**row) for row in rows]
                    work = models.WorkPaths.from_workdir(transcript.parent)
                    work.raw_transcript_manifest = backup
                    work.transcript_manifest = transcript
                    work.corrected_manifest = report
                    corrected, report_items = closure["qwen_asr.corrector"].run_correction_stage(
                        work, items, "", "", "", mode="rules")
                    loaded = closure["qwen_asr.commands.stages"].load_transcripts(transcript)
                    load_ok = (
                        [item.segment_id for item in loaded] == [item.segment_id for item in corrected]
                        and [item.text for item in loaded] == [item.text for item in corrected]
                        and [item.status for item in loaded] == [item.status for item in corrected])
                    report_names = {type(item).__name__ for item in report_items}
                    if report_items and report_names != {REPORT_KIND}:
                        raise ModuleError(
                            "UnexpectedReport",
                            "corrected manifest items are not " + REPORT_KIND)
                    value = {
                        "mode": "rules",
                        "report_kind": REPORT_KIND,
                        "corrected_count": len(corrected),
                        "report_count": len(report_items),
                        "changed": sum(1 for item in report_items if item.changed),
                        "failed": sum(1 for item in report_items if item.status != "completed"),
                        "load_transcripts": load_ok,
                        "input_sha256": hashlib.sha256(before).hexdigest(),
                        "input_preserved": source.read_bytes() == before,
                        "raw_backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
                        "transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
                        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                        "audio_copied": False,
                        "audio_references": sorted({row.get("audio_path", "") for row in rows}),
                        "chain_passed": False,
                    }
                except ModuleError:
                    raise
                except Exception as exc:
                    written["transcript_manifest"] = transcript.is_file()
                    written["report_manifest"] = report.is_file()
                    raise ModuleError(
                        "LegacyException",
                        type(exc).__name__ + ": " + str(exc),
                        legacy_identity=type(exc).__module__ + "." + type(exc).__qualname__) from exc
        finally:
            closure.close()
        effects = [
            effect(arguments["raw_backup"], "write", "committed", "exact raw backup; source not overwritten"),
            effect(arguments["transcript_manifest"], "write", "committed",
                   "rules transcript_segments via run_correction_stage"),
            effect(arguments["report_manifest"], "write", "committed",
                   "CorrectionReportItem list; not TranscriptSegment"),
        ]
        return {"report": value}, effects
