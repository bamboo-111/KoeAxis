"""Targeted offline tests for explicit alignment lineage and failure handling."""
import json
import tempfile
import unittest
from pathlib import Path

from modules.temporary.real_align_recovery import chain


class ChainAlignTests(unittest.TestCase):
    def test_missing_and_invalid_input_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            with self.assertRaises(FileNotFoundError):
                chain.load_rows(path)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                chain.load_rows(path)
            path.write_text(json.dumps([{"segment_id": "x"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                chain.load_rows(path)

    def test_current_initial_neighbors_and_mfa(self):
        rows = [{"segment_id": x, "language": "Japanese"} for x in ("a", "b", "c")]
        seen = []
        def initial(row):
            return {"segment_id": row["segment_id"], "status": "failed" if row["segment_id"] == "b" else "completed", "origin": "this-run"}
        def short(row, initial_row, neighbors):
            seen.append((row["language"], [n["origin"] for n in neighbors]))
            return {"aligned_row": initial_row, "enter_mfa": True}
        def mfa(row, report):
            return {"aligned_row": {"segment_id": row["segment_id"], "status": "completed"}}
        result, branches = chain.align_rows(rows, initial, short, mfa)
        self.assertEqual([x["status"] for x in result], ["completed"] * 3)
        self.assertEqual(seen, [("Japanese", ["this-run"] * 3)])
        self.assertEqual(branches[0]["segment_id"], "b")

    def test_recovery_manifest_uses_only_explicit_neighbors(self):
        from modules.temporary.real_align_recovery import recover
        recover._legacy()
        from qwen_asr.models import WorkPaths
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rows = [{"segment_id": name, "origin": "current"} for name in ("a", "b", "c")]
            target = {"segment_id": "b", "origin": "failed-current"}
            recover._write_recovery_inputs(work, {"segment_id": "b"}, target, rows, work / "source.wav")
            paths = WorkPaths.from_workdir(work)
            written = json.loads(paths.aligned_manifest.read_text(encoding="utf-8"))
            self.assertEqual([r["origin"] for r in written], ["current", "failed-current", "current"])
    def test_root_releases_lock_after_stage_exception(self):
        from modules.temporary.real_align_recovery.suite import Implementation
        class FakeAPI:
            def __init__(self):
                self.calls = []
            def environment(self, name, default):
                return default
            def snapshot(self):
                return {"environment": {"output": "runs/experiments/REAL-CHAIN-ALIGN-OFF"}}
            def call(self, target, args, artifacts=None):
                op = target["operation_id"]
                self.calls.append(op)
                if op == "acquire_gpu_lock":
                    return {"outcome": "returned", "returned": {"lease": {"acquired": True, "lock_path": args["lock_path"], "holder": "h"}}}
                if op == "release_gpu_lock":
                    return {"outcome": "returned", "returned": {"result": {"released": True}}}
                raise RuntimeError("synthetic stage failure")
        api = FakeAPI()
        with self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
            Implementation({}).invoke("run_chain", {"plan_id": "REAL-CHAIN-ALIGN-001"}, api)
        self.assertEqual(api.calls, ["acquire_gpu_lock", "run_bound", "release_gpu_lock"])


    def test_wrong_recovery_id_rejected(self):
        rows = [{"segment_id": "a"}]
        with self.assertRaises(ValueError):
            chain.align_rows(rows, lambda row: {"segment_id": "a", "status": "failed"},
                             lambda *args: {"aligned_row": {"segment_id": "wrong"}, "enter_mfa": False},
                             lambda *args: None)


if __name__ == "__main__":
    unittest.main()
