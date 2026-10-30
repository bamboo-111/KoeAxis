"""stage.align short-window / MFA recovery. Offline does not load a model."""
import importlib.util
from pathlib import Path

from host.contracts import ModuleError, ref, returned

V = "0.1.0-real-align-recovery"
WORK = Path(__file__).resolve().parents[3] / "runs/experiments/REAL-ALIGN-RECOVERY-001/mfa-work"


def _recover():
    path = Path(__file__).resolve().parent / "recover.py"
    spec = importlib.util.spec_from_file_location("real_align_recover", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Implementation:
    def __init__(self, settings):
        self.settings = settings
        self.interface_version = settings.get('interface_version', V)

    def invoke(self, operation, arguments, api):
        if operation == "run_bound":
            return self.bound(arguments, api)
        if operation != "recover_segment":
            raise ModuleError("UnknownOperation", operation)
        if arguments["mode"] == "offline":
            return self.offline(), []
        if arguments["mode"] != "real":
            raise ModuleError("InvalidMode", arguments["mode"])
        return self.real(arguments, api)

    def offline(self):
        recover = _recover()
        row = recover.segment_row(recover.TRANSCRIPT, "segment_000041")
        text = None if row is None else row.get("text")
        return {"report": {
            "mode": "offline",
            "models_run": False,
            "mfa_executed": False,
            "segment_id": "segment_000041",
            "segment_text_matches_baseline_transcript": text == recover.EXPECTED_TEXT,
            "input": "fixed B transcript segment_000041; models and MFA not run",
            "enter_short_window": False,
            "enter_mfa": False,
            "chain": "offline reference only; not a full-chain pass",
        }}

    def real(self, arguments, api):
        recover = _recover()
        work = WORK
        recover.assert_outside(work, "real workdir")
        work.mkdir(parents=True, exist_ok=True)
        before = recover.legacy_snapshot()
        previous_env = recover._offline_env(work)
        previous_cwd = Path.cwd()
        short = None
        mfa = None
        error = None
        try:
            recover.progress("legacy-snapshot-before", git_lines=before["git"]["lines"])
            import os
            os.chdir(work)
            short = recover.run_short_window(work)
            if short.get("enter_mfa"):
                child = api.call(ref("provider.mfa", "align_injected", self.interface_version), {
                    "work_dir": str(work),
                    "segment_id": arguments["segment_id"],
                })
                mfa = returned(child)["report"]
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            recover.progress("real-exception", error_type=error["type"])
        finally:
            import os
            os.chdir(previous_cwd)
            recover._restore_env(previous_env)
        after = recover.legacy_snapshot()
        changed = recover.snapshot_diff(before, after)
        recover.progress("legacy-snapshot-after", changed=len(changed), git_lines=after["git"]["lines"])
        baseline = recover._load_json(recover.BASELINE)
        aligned = (mfa if mfa and mfa.get("tokens") else None) or (short or {}).get("aligned") or {}
        tokens = aligned.get("tokens") or []
        fields = {
            "text": aligned.get("text") or (short or {}).get("transcript_text"),
            "status": aligned.get("status_field") or aligned.get("status"),
            "alignment_backend": aligned.get("alignment_backend"),
            "alignment_state": aligned.get("alignment_state"),
        }
        if mfa and mfa.get("alignment_state") == "completed_exact":
            fields["status"] = "completed"
            fields["alignment_backend"] = mfa.get("alignment_backend") or "mfa-local-recovery"
            fields["alignment_state"] = "completed_exact"
            fields["text"] = mfa.get("text") or fields["text"]
        diffs = recover.compare_result(tokens, fields, baseline) if short else [
            {"field": "run", "actual": "exception", "expected": "completed"}]
        short_window = (short or {}).get("short_window") or {}
        base_short = baseline.get("short_window") or {}
        if short_window.get("content_error") != base_short.get("content_error"):
            diffs.append({"field": "short_window.content_error", "actual": short_window.get("content_error"),
                          "expected": base_short.get("content_error")})
        if short_window.get("merged_asr_text") != base_short.get("merged_asr_text"):
            diffs.append({"field": "short_window.merged_asr_text", "actual": short_window.get("merged_asr_text"),
                          "expected": base_short.get("merged_asr_text")})
        base_recovery = baseline.get("recovery") or {}
        if mfa:
            for key in ("recognized_text", "content_score", "lab_text"):
                if mfa.get(key) != base_recovery.get(key):
                    diffs.append({"field": "recovery." + key, "actual": mfa.get(key),
                                  "expected": base_recovery.get(key)})
        verdict = "blocked" if error or changed or not short else ("identical" if not diffs else "differs")
        report = {
            "mode": "real",
            "models_run": short is not None,
            "mfa_executed": mfa is not None,
            "legacy_clean": not changed,
            "legacy_changes": changed,
            "git_after_lines": after["git"]["lines"],
            "error": error,
            "branch": None if short is None else {
                "enter_short_window": short.get("enter_short_window"),
                "enter_mfa": short.get("enter_mfa"),
                "mfa_skipped_reason": short.get("mfa_skipped_reason"),
                "initial_error": (short.get("initial") or {}).get("error"),
                "one_model_at_a_time": True,
            },
            "input": None if short is None else short.get("input"),
            "raw_differs_from_transcript": None if short is None else short.get("raw_differs_from_transcript"),
            "short_window": short_window or None,
            "mfa": mfa,
            "comparison_fields": fields,
            "tokens": tokens,
            "diffs": diffs,
            "segment_verdict": verdict,
            "chain": "segment_000041 only; not the full ASR-to-subtitle chain",
        }
        return {"report": recover.plain(report)}, []

    def bound(self, arguments, api):
        path = Path(__file__).resolve().parent / "chain.py"
        spec = importlib.util.spec_from_file_location("real_align_chain", path)
        chain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chain)
        try:
            rows = chain.load_rows(arguments["transcript_source"])
            raw = arguments["raw_source"]
            if raw:
                chain.load_rows(raw)
            if arguments["mode"] == "offline":
                def initial(row):
                    return {"segment_id": row["segment_id"], "status": "failed" if row["segment_id"] == "b" else "completed"}
                def short(row, first, neighbors):
                    return {"aligned_row": first, "enter_mfa": True,
                            "neighbor_ids": [item["segment_id"] for item in neighbors], "language": row.get("language")}
                def mfa(row, report):
                    return {"aligned_row": {"segment_id": row["segment_id"], "status": "completed"}}
                aligned, branches = chain.align_rows(rows, initial, short, mfa)
                report = {"mode": "offline", "count": len(aligned), "failed": sum(x["status"] != "completed" for x in aligned),
                          "neighbor_ids": branches[0]["short_window"]["neighbor_ids"],
                          "language": branches[0]["short_window"]["language"], "models_run": False}
            elif arguments["mode"] == "real":
                recover = _recover()
                resources = arguments.get('resources')
                if resources is not None:
                    recover.configure_resources(resources)
                def mfa(work, segment_id, source_audio):
                    mfa_args = {"work_dir": work, "segment_id": segment_id, "source_audio": source_audio}
                    if resources is not None:
                        mfa_args['resources'] = resources
                    return returned(api.call(ref("provider.mfa", "align_bound", self.interface_version), mfa_args))["report"]
                report = chain.real(recover, rows, arguments["work_dir"], arguments["source_audio"], mfa)
                report.update({"mode": "real", "models_run": True, "transcript_sha256": chain.digest(arguments["transcript_source"]),
                               "raw_sha256": chain.digest(raw) if raw else None})
            else:
                raise ValueError("invalid mode")
            return {"report": report}, []
        except Exception as exc:
            raise ModuleError("BoundAlignmentFailed", type(exc).__name__ + ": " + str(exc)) from exc
