"""Root test.real.chain.correct/run. Calls stage.correct/run_bound. Offline is synthetic only."""
import json
import os
import subprocess
import sys
from pathlib import Path

from host.contracts import ModuleError, ref

from modules.temporary.real_chain_correct.fixtures import CASES

V = "0.1.0-real-chain-correct"
OWNER = "test.real.chain.correct"
ROOT = Path(__file__).resolve().parents[3]
EXPLICIT_RAW = ROOT / "runs/experiments/REAL-ASR-001/repair-1/historical-batch/run-1/transcript_segments.legacy-format.json"
PROFILE = ROOT / "module-registry/real_chain_correct/contracts-real-chain-correct.json"
REFERENCE = ROOT / "modules/temporary/real_chain_correct/reference.py"
LEGACY = Path(__file__).resolve().parents[3] / "compat" / "legacy"
STRUCT = {
    "changed": {"changed": 1, "report_count": 1, "corrected_count": 1},
    "unchanged": {"changed": 0, "report_count": 1, "corrected_count": 1},
    "failed": {"changed": 0, "report_count": 0, "corrected_count": 1},
    "empty": {"changed": 0, "report_count": 0, "corrected_count": 1},
    "blank": {"changed": 0, "report_count": 0, "corrected_count": 1},
    "mixed": {"changed": 1, "report_count": 2, "corrected_count": 3},
}


def artifact(locator, module, operation):
    return {
        "artifact_id": locator.replace("/", ".").replace(".json", ""),
        "locator": locator,
        "representation": "application/json; charset=utf-8",
        "permissions": ["read", "write"],
        "content_identity": None,
        "state_owner": OWNER,
        "writer_operation": ref(module, operation, V),
        "lifetime": "run",
    }


class Implementation:
    def __init__(self, settings):
        self.settings = settings

    def invoke(self, operation, arguments, api):
        if operation != "run":
            raise ModuleError("UnknownOperation", operation)
        if arguments["plan_id"] != "REAL-CHAIN-CORRECT-001":
            raise ModuleError("InvalidPlan", arguments["plan_id"])
        mode_value = api.environment("REAL_CHAIN_CORRECT_MODE", "") or ""
        if mode_value in ("", "offline"):
            mode = "offline"
        elif mode_value == "cpu149":
            mode = "cpu149"
        else:
            raise ModuleError("InvalidMode", "REAL_CHAIN_CORRECT_MODE")
        output = (ROOT / api.snapshot()["environment"]["output"]).resolve()
        checks = []

        def check(name, cond, observed=None):
            checks.append({"id": name, "passed": bool(cond)})
            if not cond:
                detail = name if observed is None else name + " " + json.dumps(observed, ensure_ascii=False)
                raise ModuleError("VerificationFailed", detail, partial={"checks": checks})

        def call(module, op, args, artifacts=None):
            return api.call(ref(module, op, V), args, artifacts)

        def write_json(locator, payload):
            resource = artifact(locator, "storage.artifacts", "write_fixture_json")
            result = call("storage.artifacts", "write_fixture_json",
                          {"path": resource, "payload": payload}, [resource])
            check("write-" + locator, result["outcome"] == "returned")
            return result

        def stage(mode_name, raw_source, stem):
            args = {
                "mode": mode_name,
                "raw_source": str(raw_source),
                "raw_backup": artifact(stem + "/transcript_segments.raw.json", "stage.correct", "run_bound"),
                "transcript_manifest": artifact(stem + "/transcript_segments.json", "stage.correct", "run_bound"),
                "report_manifest": artifact(stem + "/corrected_segments.json", "stage.correct", "run_bound"),
            }
            arts = [args["raw_backup"], args["transcript_manifest"], args["report_manifest"]]
            return call("stage.correct", "run_bound", args, arts)

        if mode == "offline":
            body = self._offline(output, check, write_json, stage)
        else:
            body = self._cpu(output, check, write_json, stage)
        started = [event["request"]["target"]["operation_id"]
                   for event in api.snapshot()["trace"] if event.get("event") == "started"]
        check("root-first", started[0] == "run")
        check("called-run-bound", "run_bound" in started)
        body["calls"] = started
        body["chain_passed"] = False
        body["align_completed"] = False
        body["models_run"] = False
        failed = len([item for item in checks if not item["passed"]])
        report = {
            "plan_id": arguments["plan_id"],
            "status": "passed" if failed == 0 else "failed",
            "passed": len(checks) - failed,
            "failed": failed,
            "checks": checks,
            "mode": mode,
            "chain_passed": False,
            "align_completed": False,
            "models_run": False,
            "coverage": "partial",
            "migration_equivalence": body.get("migration_equivalence"),
            "case_count": body.get("case_count", 1 if mode == "cpu149" else len(CASES)),
        }
        write_json("evidence/cases.json", body)
        write_json("evidence/report.json", report)
        return {"report": report}, []

    def _offline(self, output, check, write_json, stage):
        request_cases = []
        for case in CASES:
            locator = "synthetic/" + case["id"] + "/source.json"
            write_json(locator, case["rows"])
            source = output / locator
            request_cases.append({
                "id": case["id"],
                "raw": str(source),
                "backup": str(output / "direct" / case["id"] / "transcript_segments.raw.json"),
                "transcript": str(output / "direct" / case["id"] / "transcript_segments.json"),
                "report": str(output / "direct" / case["id"] / "corrected_segments.json"),
            })
        summary_path = output / "direct" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        request_path = output / "direct" / "request.json"
        request_path.write_text(json.dumps({
            "legacy_root": str(LEGACY),
            "profile": str(PROFILE),
            "cases": request_cases,
        }, ensure_ascii=False), encoding="utf-8", newline="\n")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("REAL_CHAIN_CORRECT_MODE", None)
        for key in list(env):
            if any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                env.pop(key, None)
        completed = subprocess.run(
            [sys.executable, "-B", str(REFERENCE), str(request_path), str(summary_path)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180, check=False)
        check("direct-exit", completed.returncode == 0, {"code": completed.returncode, "err": (completed.stderr or "")[-400:]})
        direct = {row["id"]: row for row in json.loads(summary_path.read_text(encoding="utf-8"))["cases"]}
        rows = []
        for case in CASES:
            result = stage("rules", output / "synthetic" / case["id"] / "source.json", "candidate/" + case["id"])
            check("stage-" + case["id"], result["outcome"] == "returned" and len(result["effects"]) == 3,
                  result.get("raised") or result.get("host_rejected"))
            body = result["returned"]["report"]
            expected = direct[case["id"]]
            shape = STRUCT[case["id"]]
            check("match-" + case["id"],
                  body["transcript_sha256"] == expected["transcript_sha256"]
                  and body["report_sha256"] == expected["report_sha256"]
                  and body["load_transcripts"] is True
                  and body["report_kind"] == "CorrectionReportItem"
                  and body["input_preserved"] is True
                  and body["raw_backup_sha256"] == body["input_sha256"]
                  and body["audio_copied"] is False
                  and body["chain_passed"] is False
                  and {key: body[key] for key in shape} == shape,
                  {key: body[key] for key in ("changed", "report_count", "corrected_count", "load_transcripts")})
            rows.append({"id": case["id"], "direct_bytes_match": True, **shape,
                         "load_transcripts": True, "report_kind": "CorrectionReportItem"})
        refusals = [
            ("invalid-mode", "llm", "not-used", "UnsupportedCoverage"),
            ("missing-raw", "rules", str(output / "synthetic" / "missing-source.json"), "MissingInput"),
            ("legacy-raw", "rules", str(LEGACY / "missing-real-chain-correct.json"), "PermissionDenied"),
        ]
        refusal_rows = []
        for name, mode_name, raw_source, category in refusals:
            result = stage(mode_name, raw_source, "refuse/" + name)
            raised = result.get("raised") or {}
            check("refuse-" + name,
                  result["outcome"] == "raised" and raised.get("category") == category and not result["effects"],
                  {"outcome": result["outcome"], "category": raised.get("category")})
            refusal_rows.append({"id": name, "category": category})
        wavs = [path.relative_to(output).as_posix() for path in output.rglob("*.wav")]
        check("no-audio-copy", wavs == [])
        return {
            "kind": "synthetic-offline",
            "chain_passed": False,
            "align_completed": False,
            "case_count": len(rows),
            "cases": rows,
            "refusals": refusal_rows,
            "migration_equivalence": "synthetic-direct-vs-host-only",
            "roles": [
                {"locator": "synthetic/<id>/source.json", "role": "synthetic-raw", "source": "fixed fixture"},
                {"locator": "candidate/<id>/transcript_segments.raw.json", "role": "raw-backup", "source": "exact copy of that fixture"},
                {"locator": "candidate/<id>/transcript_segments.json", "role": "rules-transcript", "source": "run_correction_stage"},
                {"locator": "candidate/<id>/corrected_segments.json", "role": "correction-report", "source": "CorrectionReportItem list"},
            ],
        }

    def _cpu(self, output, check, write_json, stage):
        from modules.temporary.real_chain_correct.compare import REPORT, TRANSCRIPT, against_baseline
        check("explicit-raw-exists", EXPLICIT_RAW.is_file())
        direct_dir = output / "direct149"
        request_path = direct_dir / "request.json"
        summary_path = direct_dir / "summary.json"
        direct_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps({
            "legacy_root": str(LEGACY),
            "profile": str(PROFILE),
            "cases": [{
                "id": "cpu149",
                "raw": str(EXPLICIT_RAW),
                "backup": str(direct_dir / "transcript_segments.raw.json"),
                "transcript": str(direct_dir / "transcript_segments.json"),
                "report": str(direct_dir / "corrected_segments.json"),
            }],
        }, ensure_ascii=False), encoding="utf-8", newline="\n")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for key in list(env):
            if any(part in key.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                env.pop(key, None)
        completed = subprocess.run(
            [sys.executable, "-B", str(REFERENCE), str(request_path), str(summary_path)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300, check=False)
        check("direct-149-exit", completed.returncode == 0, {"code": completed.returncode, "err": (completed.stderr or "")[-400:]})
        result = stage("rules", EXPLICIT_RAW, "work")
        check("stage-149", result["outcome"] == "returned" and len(result["effects"]) == 3,
              result.get("raised") or result.get("host_rejected"))
        body = result["returned"]["report"]
        direct = json.loads(summary_path.read_text(encoding="utf-8"))["cases"][0]
        check("host-matches-direct",
              body["transcript_sha256"] == direct["transcript_sha256"]
              and body["report_sha256"] == direct["report_sha256"]
              and body["corrected_count"] == 149
              and body["load_transcripts"] is True
              and body["input_preserved"] is True
              and body["audio_copied"] is False
              and body["chain_passed"] is False)
        transcript_path = output / "work" / "transcript_segments.json"
        report_path = output / "work" / "corrected_segments.json"
        raw_path = output / "work" / "transcript_segments.raw.json"
        flags = {
            "input_bytes_equal_baseline_raw": raw_path.read_bytes() == (LEGACY / "workspaces/radio-54-togeradi-20261018/manifests/transcript_segments.raw.json").read_bytes(),
            "transcript_bytes_equal_baseline": transcript_path.read_bytes() == TRANSCRIPT.read_bytes(),
            "report_bytes_equal_baseline": report_path.read_bytes() == REPORT.read_bytes(),
        }
        comparison = against_baseline(
            json.loads(raw_path.read_bytes().decode("utf-8-sig")),
            json.loads(transcript_path.read_bytes().decode("utf-8-sig")),
            json.loads(report_path.read_bytes().decode("utf-8-sig")),
            body["input_sha256"], flags)
        check("comparison-recorded", comparison["relation"] in ("field-identical", "differs") and comparison["chain_passed"] is False)
        write_json("evidence/comparison.json", comparison)
        write_json("evidence/roles.json", {
            "files": [
                {"locator": "work/transcript_segments.raw.json", "role": "raw-backup",
                 "source": "exact bytes of REAL-ASR-001 repair-1 historical-batch run-1 legacy-format; not baseline B"},
                {"locator": "work/transcript_segments.json", "role": "align-input-transcript",
                 "source": "run_correction_stage mode=rules serialize_manifest; text is rules output"},
                {"locator": "work/corrected_segments.json", "role": "correction-report",
                 "source": "CorrectionReportItem list from the same call; not a TranscriptSegment list"},
            ],
            "audio": "audio_path values kept as read-only references; audio files are not copied",
            "align_completed": False,
        })
        return {
            "kind": "cpu149",
            "chain_passed": False,
            "align_completed": False,
            "case_count": 1,
            "input_sha256": body["input_sha256"],
            "corrected_count": body["corrected_count"],
            "report_count": body["report_count"],
            "changed": body["changed"],
            "direct_bytes_match": True,
            "relation": comparison["relation"],
            "transcript_bytes_equal_baseline": flags["transcript_bytes_equal_baseline"],
            "report_bytes_equal_baseline": flags["report_bytes_equal_baseline"],
            "input_bytes_equal_baseline_raw": flags["input_bytes_equal_baseline_raw"],
            "transcript_field_diff_count": comparison["transcript_field_diff_count"],
            "report_field_diff_count": comparison["report_field_diff_count"],
            "migration_equivalence": "not-assessed-beyond-recorded-comparison",
            "align_transcript": "work/transcript_segments.json",
        }
