"""Refusal checks for stage.correct/run_bound. No host and no baseline read."""
import unittest
from pathlib import Path

from host.contracts import ModuleError
from modules.temporary.real_chain_correct.fixtures import CASES
from modules.temporary.real_chain_correct.operations import Implementation


class Stop(Exception):
    pass


class FakeApi:
    def path(self, resource, access):
        raise Stop("path")

    def legacy_modules(self):
        raise Stop("legacy")


def args(mode, raw):
    return {
        "mode": mode,
        "raw_source": raw,
        "raw_backup": {"locator": "a.json"},
        "transcript_manifest": {"locator": "b.json"},
        "report_manifest": {"locator": "c.json"},
    }


class RealChainCorrectTests(unittest.TestCase):
    def test_invalid_mode_refuses_before_io(self):
        with self.assertRaises(ModuleError) as raised:
            Implementation({}).invoke("run_bound", args("llm", "not-used"), FakeApi())
        self.assertEqual(raised.exception.error["category"], "UnsupportedCoverage")

    def test_legacy_path_refuses_before_missing_check(self):
        raw = str(Path(__file__).resolve().parents[1] / "compat/legacy/missing-real-chain-correct.json")
        with self.assertRaises(ModuleError) as raised:
            Implementation({}).invoke("run_bound", args("rules", raw), FakeApi())
        self.assertEqual(raised.exception.error["category"], "PermissionDenied")

    def test_missing_raw_refuses_without_baseline(self):
        raw = str(Path(__file__).resolve().parents[1] / "runs/experiments/missing-real-chain-correct.json")
        with self.assertRaises(ModuleError) as raised:
            Implementation({}).invoke("run_bound", args("rules", raw), FakeApi())
        self.assertEqual(raised.exception.error["category"], "MissingInput")

    def test_fixture_coverage(self):
        ids = [case["id"] for case in CASES]
        self.assertEqual(ids, ["changed", "unchanged", "failed", "empty", "blank", "mixed"])
        failed = CASES[2]["rows"][0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(CASES[3]["rows"][0]["text"], "")
        source = Path("modules/temporary/real_chain_correct/operations.py").read_text(encoding="utf-8")
        self.assertNotIn("radio-54-togeradi", source)
        self.assertNotIn("transcript_segments.json", source)


if __name__ == "__main__":
    unittest.main()
