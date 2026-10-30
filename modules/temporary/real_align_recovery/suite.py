"""Root test.real.align.recovery/run. Acquire the file lock, then stage.align, then release."""
from pathlib import Path

from host.contracts import ModuleError, ref, returned

V = "0.1.0-real-align-recovery"
OWNER = "test.real.align.recovery"
EXEC = "resource.execution"
STAGE = "stage.align"


def artifact(locator):
    return {
        "artifact_id": locator.replace("/", ".").replace(".json", ""),
        "locator": locator,
        "representation": "application/json; charset=utf-8",
        "permissions": ["read", "write"],
        "content_identity": None,
        "state_owner": OWNER,
        "writer_operation": ref("storage.artifacts", "write_fixture_json", V),
        "lifetime": "run",
    }


class Implementation:
    def __init__(self, settings):
        self.settings = settings

    def invoke(self, operation, arguments, api):
        if operation == "run_chain":
            return self.run_chain(arguments, api)
        if operation != "run":
            raise ModuleError("UnknownOperation", operation)
        if arguments["plan_id"] != "REAL-ALIGN-RECOVERY-001":
            raise ModuleError("InvalidPlan", arguments["plan_id"])
        mode_value = api.environment("REAL_ALIGN_RECOVERY_MODE", "") or ""
        if mode_value in ("", "offline"):
            mode = "offline"
        elif mode_value == "real":
            mode = "real"
        else:
            raise ModuleError("InvalidMode", "REAL_ALIGN_RECOVERY_MODE")
        root = Path(__file__).resolve().parents[3]
        output_rel = api.snapshot()["environment"]["output"]
        output_root = str((root / output_rel).resolve())
        lock_path = str((root / output_rel / "gpu.lock").resolve())
        timeout = 5 if mode == "real" else 2
        checks = []

        def check(name, cond):
            checks.append({"id": name, "passed": bool(cond)})

        def child(module, op, args, artifacts=None):
            return api.call(ref(module, op, V), args, artifacts)

        lease = None
        stage_report = None
        release_report = None
        try:
            acquired = returned(child(EXEC, "acquire_gpu_lock", {
                "lock_path": lock_path,
                "output_root": output_root,
                "timeout_seconds": timeout,
                "stage": "align-recovery",
            }))
            lease = acquired["lease"]
            check("lock-acquired", lease["acquired"] is True and "filelock" in lease["implementation"]
                  and lease["rejects_external_gpu_process"] is False and lease["cooperative_only"] is True)
            stage_body = returned(child(STAGE, "recover_segment", {
                "segment_id": "segment_000041",
                "mode": mode,
            }))
            stage_report = stage_body["report"]
        finally:
            if lease and lease.get("acquired") is True:
                release_report = returned(child(EXEC, "release_gpu_lock", {
                    "lock_path": lease["lock_path"],
                    "holder": lease["holder"],
                }))["result"]
        check("lock-released", release_report is not None and release_report["released"] is True
              and release_report["held_after"] is False)
        if mode == "offline":
            check("offline-no-model", stage_report["models_run"] is False and stage_report["mfa_executed"] is False)
            check("offline-transcript", stage_report["segment_text_matches_baseline_transcript"] is True)
            check("offline-no-recovery-branch", stage_report["enter_short_window"] is False
                  and stage_report["enter_mfa"] is False)
        else:
            check("real-legacy-clean", stage_report.get("legacy_clean") is True)
            check("real-finished", stage_report.get("error") is None and stage_report.get("segment_verdict") != "blocked")
        started = [event["request"]["target"]["operation_id"]
                   for event in api.snapshot()["trace"] if event.get("event") == "started"]
        check("root-called-lock-and-align",
              started[0] == "run" and "acquire_gpu_lock" in started and "recover_segment" in started
              and "release_gpu_lock" in started)
        acquire_at = started.index("acquire_gpu_lock")
        align_at = started.index("recover_segment")
        release_at = started.index("release_gpu_lock")
        check("lock-before-align-before-release", acquire_at < align_at < release_at)
        cases = {
            "mode": mode,
            "checks": checks,
            "lock": {
                "implementation": lease["implementation"],
                "cooperative_only": True,
                "rejects_external_gpu_process": False,
                "timeout_seconds": timeout,
                "acquired": True,
                "released": True,
            },
            "stage": {
                "models_run": stage_report.get("models_run"),
                "mfa_executed": stage_report.get("mfa_executed"),
                "segment_verdict": stage_report.get("segment_verdict"),
                "segment_text_matches_baseline_transcript": stage_report.get(
                    "segment_text_matches_baseline_transcript"),
            },
        }
        cases_art = artifact("evidence/cases.json")
        returned(child("storage.artifacts", "write_fixture_json", {"path": cases_art, "payload": cases}, [cases_art]))
        failed = len([item for item in checks if not item["passed"]])
        report = {
            "plan_id": arguments["plan_id"],
            "status": "passed" if failed == 0 else "failed",
            "passed": len(checks) - failed,
            "failed": failed,
            "case_count": len(checks),
            "mode": mode,
            "models_run": stage_report.get("models_run") is True,
            "mfa_executed": stage_report.get("mfa_executed") is True,
            "segment_verdict": stage_report.get("segment_verdict"),
            "legacy_equivalence": False,
            "full_chain": False,
            "lock_scope": "cooperative filelock callers only; not an external GPU process",
            "stage_report": stage_report,
        }
        report_art = artifact("evidence/report.json")
        returned(child("storage.artifacts", "write_fixture_json", {"path": report_art, "payload": report}, [report_art]))
        if failed:
            raise ModuleError("VerificationFailed", "one or more checks failed",
                              partial={"failed": failed, "checks": checks})
        return {"report": report}, []

    def run_chain(self, arguments, api):
        if arguments["plan_id"] != "REAL-CHAIN-ALIGN-001":
            raise ModuleError("InvalidPlan", arguments["plan_id"])
        value = api.environment("REAL_CHAIN_ALIGN_MODE", "") or ""
        if value not in ("", "offline", "real"):
            raise ModuleError("InvalidMode", "REAL_CHAIN_ALIGN_MODE")
        mode = "real" if value == "real" else "offline"
        root = Path(__file__).resolve().parents[3]
        output_root = (root / api.snapshot()["environment"]["output"]).resolve()
        transcript = root / ("runs/experiments/REAL-CHAIN-CORRECT-001/cpu-rules/work/transcript_segments.json" if mode == "real" else "modules/temporary/real_align_recovery/synthetic.json")
        raw = root / "runs/experiments/REAL-CHAIN-CORRECT-001/cpu-rules/work/transcript_segments.raw.json" if mode == "real" else None
        source_audio = str(root / "compat/legacy/workspaces/radio-54-togeradi-20261018/audio/source.wav")
        lock_path = str(output_root / "gpu.lock")
        lease = None
        release = None
        try:
            lease = returned(api.call(ref(EXEC, "acquire_gpu_lock", V), {
                "lock_path": lock_path, "output_root": str(output_root), "timeout_seconds": 5, "stage": "align-chain"}))["lease"]
            report = returned(api.call(ref(STAGE, "run_bound", V), {
                "mode": mode, "transcript_source": str(transcript), "raw_source": str(raw) if raw else None,
                "work_dir": str(output_root / "work"), "source_audio": source_audio}))["report"]
        finally:
            if lease and lease.get("acquired"):
                release = returned(api.call(ref(EXEC, "release_gpu_lock", V), {
                    "lock_path": lease["lock_path"], "holder": lease["holder"]}))["result"]
        checks = [
            {"id": "lock-acquired", "passed": bool(lease and lease.get("acquired"))},
            {"id": "lock-released", "passed": bool(release and release.get("released"))},
            {"id": "nonempty", "passed": report["count"] == (149 if mode == "real" else 3)},
            {"id": "alignment-completed", "passed": report["failed"] == 0},
        ]
        if mode == "offline":
            checks += [{"id": "explicit-neighbors", "passed": report["neighbor_ids"] == ["a", "b", "c"]},
                       {"id": "language-forwarded", "passed": report["language"] == "Japanese"},
                       {"id": "no-model", "passed": report["models_run"] is False}]
        cases = {"mode": mode, "checks": checks}
        failed = sum(not item["passed"] for item in checks)
        result = {"plan_id": arguments["plan_id"], "status": "failed" if failed else "passed",
                  "passed": len(checks) - failed, "failed": failed, "case_count": len(checks),
                  "chain_passed": False, "stage_report": report}
        for filename, payload in (("cases.json", cases), ("report.json", result)):
            item = artifact("evidence/" + filename)
            returned(api.call(ref("storage.artifacts", "write_fixture_json", V), {"path": item, "payload": payload}, [item]))
        if failed:
            raise ModuleError("VerificationFailed", "chain checks failed", partial={"checks": checks})
        return {"report": result}, []
