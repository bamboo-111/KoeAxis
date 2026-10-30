"""Offline checks for real Gemini stage entry. No network."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/temporary/real_llm_stages/operations.py"


def load():
    spec = importlib.util.spec_from_file_location("real_llm_stages_operations", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load()

    def test_online_grant_requires_both_halves(self):
        mod = self.mod
        closed = {"permissions": {"offline": {"network": False, "credentials": False},
                                  "online": {"network": False, "endpoints": "Gemini"}}}
        self.assertFalse(mod.online_granted(closed, "1"))
        self.assertFalse(mod.online_granted(mod.load_spec(), ""))
        self.assertTrue(mod.online_granted(mod.load_spec(), "1"))

    def test_alias_and_endpoint_and_scrub(self):
        mod = self.mod
        self.assertEqual(mod.correct_gemini_model("gemini-3.8-flash"), "gemini-3.8-flash-high")
        self.assertEqual(mod.correct_gemini_model(""), "gemini-3.8-flash-high")
        self.assertEqual(mod.correct_gemini_model("other-model"), "other-model")
        view = mod.endpoint_view("https://203.0.113.10/v1beta/models/x")
        self.assertEqual(view["host_kind"], "numeric")
        self.assertEqual(view["path_class"], "native-versioned")
        self.assertNotIn("203.0.113.10", str(view))
        text = mod.scrub("key sk-secret url https://203.0.113.10/v1", ["sk-secret"])
        self.assertNotIn("sk-secret", text)
        self.assertNotIn("203.0.113.10", text)
        self.assertEqual(mod.classify_message("HTTP 401: Invalid API Key")["class"], "auth")
        self.assertEqual(mod.classify_message("HTTP 400: unknown provider for model gemini-3.8-flash")["class"],
                         "api_model_route_rejected")

    def test_wording_tolerance_and_hard_count(self):
        mod = self.mod
        allowed, reason = mod.wording_allowed("今天天气清楚", "今天天气晴朗", "きょう")
        self.assertTrue(allowed)
        self.assertEqual(reason, "wording")
        rejected, why = mod.wording_allowed("一共 12 人", "一共 13 人", "12")
        self.assertFalse(rejected)
        self.assertEqual(why, "numbers")
        same, same_why = mod.wording_allowed("一共 99 人", "一共 99 人", "没有数字")
        self.assertFalse(same)
        self.assertEqual(same_why, "new_number")
        kept, kept_why = mod.wording_allowed("一共 12 人", "一共 12 人", "12")
        self.assertTrue(kept)
        self.assertEqual(kept_why, "equal")
        quotes = mod.wording_assessment("甲「好」乙", "甲好乙。", "甲好乙")
        self.assertNotIn("marks", quotes["reasons"])
        self.assertEqual(quotes["bracket_pairing"], "NOT_ASSESSED")
        self.assertNotEqual(quotes["proper_nouns"], "PASS")
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cue = {"start_time": 0, "end_time": 1, "original_subtitle": "あ", "translated_subtitle": "啊", "speaker": ""}
            other = dict(cue, translated_subtitle="呀")
            (root / "a.json").write_text(json.dumps({"1": cue}), encoding="utf-8")
            (root / "b.json").write_text(json.dumps({"1": other}), encoding="utf-8")
            pair = mod.compare_manifests("translate", root / "a.json", root / "b.json")
            self.assertEqual(pair["verdict"], "NOT_PASSED")
            self.assertEqual(pair["cap_30"], "NOT_PASSED")
            self.assertGreater(pair["wording_diff_rate"], 0.30)
            self.assertEqual(pair["migration_equivalence_assessment"]["verdict"], "NOT_ASSESSED")
            self.assertEqual(pair["noise_plus_10pp"], "NOT_ASSESSED")
            stable = {"1": cue, "2": cue, "3": cue, "4": cue}
            drifted = {"1": other, "2": cue, "3": cue, "4": cue}
            (root / "c.json").write_text(json.dumps(stable), encoding="utf-8")
            (root / "d.json").write_text(json.dumps(drifted), encoding="utf-8")
            under = mod.compare_manifests("translate", root / "c.json", root / "d.json")
            self.assertEqual(under["cap_30"], "PASS")
            self.assertEqual(under["verdict"], "NOT_ASSESSED")
            self.assertEqual(under["proper_nouns"], "NOT_ASSESSED")
            self.assertEqual(under["bracket_pairing"], "NOT_ASSESSED")
            (root / "e.json").write_text(json.dumps(stable), encoding="utf-8")
            identical = mod.compare_manifests("translate", root / "c.json", root / "e.json")
            self.assertEqual(identical["pair_repeatability_verdict"], "PASS")
            self.assertEqual(identical["migration_equivalence_assessment"]["verdict"], "NOT_ASSESSED")
            introduced = dict(cue, translated_subtitle="一共 99 人", original_subtitle="没有数字")
            (root / "f.json").write_text(json.dumps({"1": introduced}), encoding="utf-8")
            (root / "g.json").write_text(json.dumps({"1": introduced}), encoding="utf-8")
            leaked = mod.compare_manifests("translate", root / "f.json", root / "g.json")
            self.assertEqual(leaked["verdict"], "NOT_PASSED")
            self.assertEqual(leaked["rejected_by_reason"].get("new_number"), 1)
            detailed = mod.compare_manifests("translate", root / "f.json", root / "g.json", detail=True)
            self.assertEqual(len(detailed["item_reasons"]), 1)
            self.assertIn("new_number", detailed["item_reasons"][0]["reasons"])
            shifted = dict(cue, start_time=2, end_time=3)
            (root / "h.json").write_text(json.dumps({"1": cue}), encoding="utf-8")
            (root / "i.json").write_text(json.dumps({"1": shifted}), encoding="utf-8")
            times = mod.compare_manifests("split", root / "h.json", root / "i.json")
            self.assertIn("start_time", times["hard_mismatch_fields"])
            self.assertIn("end_time", times["hard_mismatch_fields"])
            self.assertEqual(times["verdict"], "NOT_PASSED")
            (root / "j.json").write_text(json.dumps({"1": cue, "2": cue}), encoding="utf-8")
            broken = mod.compare_manifests("split", root / "h.json", root / "j.json")
            self.assertIn("ids", broken["hard_mismatch_fields"])
            self.assertIsNone(broken["wording_diff_rate"])
            self.assertEqual(broken["wording_quality"], "NOT_ASSESSED")
            self.assertFalse(broken["positional_zip"])
            http = mod.compare_request_sequence(
                [{"model": "gemini-3.8-flash-high", "prompt_sha256": "a"}] * 34,
                [{"model": "gemini-3.8-flash-high", "prompt_sha256": "b"}] * 35,
            )
            self.assertEqual(http["http_requests"], "NOT_PASSED")
            self.assertEqual(http["model"], "PASS")
            self.assertEqual(http["temperature"], "NOT_ASSESSED")
            self.assertEqual(http["retry"], "NOT_ASSESSED")
            self.assertEqual(http["fallback"], "NOT_ASSESSED")
            changed = mod.compare_request_sequence(
                [{"model": "gemini-3.8-flash-high"}],
                [{"model": "other-model"}],
            )
            self.assertEqual(changed["model"], "NOT_PASSED")
            criteria = mod.build_criteria("", [
                {"stage": "translate", "repeat": "a", "status": "passed", "attempts": 34, "artifact": ""},
                {"stage": "translate", "repeat": "b", "status": "passed", "attempts": 35, "artifact": ""},
            ])
            self.assertEqual(criteria["stages"]["translate"]["verdict"], "NOT_PASSED")
            self.assertIn("http_requests", criteria["stages"]["translate"]["hard_mismatch_fields"])
            self.assertEqual(criteria["migration_equivalence_assessment"]["verdict"], "NOT_ASSESSED")
            self.assertFalse(criteria["episode_pass"])

    def test_root_calls_stage_runs_offline_without_argv(self):
        mod = self.mod

        class Fake:
            def __init__(self):
                self.calls = []
            def environment(self, name, default=None):
                return "" if default is None else default
            def snapshot(self):
                return {"environment": {
                    "plan": "experiments/specifications/real-llm-stages.json",
                    "network": False,
                    "output": "runs/experiments/REAL-LLM-STAGES-NOT-CREATED",
                }}
            def call(self, target, arguments, artifacts=None, resources=None):
                self.calls.append((target["module_id"], target["operation_id"], arguments))
                role = {
                    "stage.split": "split",
                    "stage.translate": "translate",
                    "stage.text_proofread": "text",
                }[target["module_id"]]
                value, effects = mod.Implementation({"role": role}).invoke(target["operation_id"], arguments, self)
                return {"outcome": "returned", "returned": value, "effects": effects}

        fake = Fake()
        value, effects = mod.Implementation({"role": "suite"}).invoke(
            "run", {"plan_id": "REAL-LLM-STAGES-001"}, fake)
        self.assertEqual(effects, [])
        report = value["report"]
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["live"])
        self.assertFalse(report["episode_pass"])
        self.assertFalse(report["argv_returned"])
        self.assertEqual(report["executed_operations"], [
            "stage.split/run", "stage.translate/run", "stage.text_proofread/run",
        ])
        self.assertEqual(report["mimo"], "blocked_auth")
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(all(item[1] == "run" for item in fake.calls))
        self.assertTrue(all(item[2]["mode"] == "offline" for item in fake.calls))
        for child in report["calls"]:
            self.assertIsNone(child["report"]["argv"])
            self.assertEqual(child["report"]["attempts"], 0)
            self.assertEqual(child["report"]["operation"].split("/")[-1], "run")

    def test_online_without_flag_or_key_does_not_execute(self):
        mod = self.mod

        class Api:
            def __init__(self, flag):
                self.flag = flag
            def environment(self, name, default=None):
                if name == "REAL_LLM_STAGES_ONLINE":
                    return self.flag
                return "" if default is None else default

        blocked = mod.Implementation({"role": "split"}).invoke(
            "run", {"mode": "online", "repeat": "a", "output": "runs/experiments/REAL-LLM-STAGES-NOT-CREATED"},
            Api(""),
        )[0]["report"]
        self.assertEqual(blocked["status"], "blocked_declaration")
        self.assertEqual(blocked["attempts"], 0)
        self.assertFalse(blocked["live"])
        missing = mod.Implementation({"role": "translate"}).invoke(
            "run", {"mode": "online", "repeat": "a", "output": "runs/experiments/REAL-LLM-STAGES-NOT-CREATED"},
            Api("1"),
        )[0]["report"]
        self.assertEqual(missing["status"], "blocked_credentials")
        self.assertEqual(missing["attempts"], 0)
        self.assertFalse(missing["live"])


if __name__ == "__main__":
    unittest.main()
