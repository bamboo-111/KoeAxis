"""Offline checks for the MiMo audio-review entry. No network."""
import importlib.util
import json
import pathlib
import shutil
import socket
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modules/temporary/real_mimo/operations.py"


def load():
    spec = importlib.util.spec_from_file_location("real_mimo_operations", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MimoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load()

    def test_online_grant_requires_both_halves(self):
        mod = self.mod
        closed = {"permissions": {"offline": {"network": False, "credentials": False},
                                  "online": {"network": False, "endpoints": "Token Plan OpenAI Anthropic"}}}
        self.assertFalse(mod.online_granted(closed, "1"))
        self.assertFalse(mod.online_granted(mod.load_spec(), ""))
        self.assertTrue(mod.online_granted(mod.load_spec(), "1"))

    def test_endpoint_scrub_and_audio_class(self):
        mod = self.mod
        view = mod.endpoint_view("https://token-plan-cn.xiaomimimo.com/v1")
        self.assertEqual(view["host_kind"], "named")
        self.assertEqual(view["path_class"], "openai-v1")
        self.assertNotIn("xiaomimimo", str(view))
        anthropic = mod.endpoint_view("https://token-plan-cn.xiaomimimo.com/anthropic")
        self.assertEqual(anthropic["path_class"], "anthropic-base")
        self.assertNotIn("token-plan", str(anthropic))
        text = mod.scrub("key sk-secret url https://token-plan-cn.xiaomimimo.com/v1/chat", ["sk-secret"])
        self.assertNotIn("sk-secret", text)
        self.assertNotIn("xiaomimimo", text)
        self.assertEqual(mod.classify_message("HTTP 401: Invalid API Key")["class"], "auth")
        self.assertIn(mod.classify_message("HTTP 400: unsupported content type input_audio")["class"], mod.DETERMINISTIC)
        self.assertEqual(mod.classify_message("Error code: 400 - input_audio is not supported")["class"], "audio_unsupported")

    def test_samples_cover_three_classes_without_positional_zip(self):
        mod = self.mod
        translated = {
            "3": {"original_subtitle": "こんにちは", "translated_subtitle": "你好", "start_time": 0, "end_time": 1500},
            "1": {"original_subtitle": "短い", "translated_subtitle": "短", "start_time": 0, "end_time": 800},
            "9": {"original_subtitle": "長い文", "translated_subtitle": "很长", "start_time": 0, "end_time": 9000},
            "4": {"original_subtitle": "12人", "translated_subtitle": "12人", "start_time": 0, "end_time": 2000},
        }
        samples = mod.select_samples(translated)
        kinds = [item["kind"] for item in samples]
        self.assertEqual(kinds, ["ordinary", "number_or_name", "longer_audio"])
        self.assertEqual([item["id"] for item in samples], [1, 4, 9])
        wording = mod.wording_assessment("一共 12 人", "一共 13 人", "12")
        self.assertIn("numbers", wording["hard"])
        self.assertFalse(wording["positional_zip"])
        self.assertEqual(wording["proper_nouns"], "NOT_ASSESSED")
        self.assertEqual(wording["bracket_pairing"], "NOT_ASSESSED")
        introduced = mod.wording_assessment("一共 99 人", "一共", "没有数字")
        self.assertIn("new_number", introduced["reasons"])

    def test_anthropic_body_keeps_audio(self):
        mod = self.mod
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "same prompt"},
                {"type": "input_audio", "input_audio": {"data": "QQQQ", "format": "wav"}},
            ],
        }]
        body = mod.anthropic_body(messages)
        content = body["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "same prompt")
        self.assertEqual(content[1]["type"], "input_audio")
        self.assertEqual(content[1]["input_audio"]["data"], "QQQQ")
        self.assertEqual(body["model"], "mimo-v2.6-pro")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        with self.assertRaises(RuntimeError):
            mod.anthropic_body([{"role": "user", "content": [{"type": "text", "text": "only"}]}])

    def test_root_calls_real_entries_offline_without_socket_or_key(self):
        mod = self.mod
        guarded = {"connect": 0}

        class Guard(socket.socket):
            def connect(self, *args, **kwargs):
                guarded["connect"] += 1
                raise AssertionError("offline opened a socket")

        class Fake:
            def __init__(self):
                self.calls = []
                self.reads = []

            def environment(self, name, default=None):
                self.reads.append(name)
                if name == "MIMO_API_KEY":
                    raise AssertionError("offline read the key")
                return "" if default is None else default

            def snapshot(self):
                return {"environment": {
                    "plan": "experiments/specifications/real-mimo.json",
                    "network": False,
                    "output": "runs/experiments/REAL-MIMO-NOT-CREATED",
                }}

            def call(self, target, arguments, artifacts=None, resources=None):
                self.calls.append((target["module_id"], target["operation_id"], arguments))
                role = {"stage.audio_review": "audio", "provider.audio_llm": "provider"}[target["module_id"]]
                operation = target["operation_id"]
                value, effects = mod.Implementation({"role": role}).invoke(operation, arguments, self)
                return {"outcome": "returned", "returned": value, "effects": effects}

        fake = Fake()
        original = socket.socket
        socket.socket = Guard
        try:
            value, effects = mod.Implementation({"role": "suite"}).invoke(
                "run", {"plan_id": "REAL-MIMO-001"}, fake)
        finally:
            socket.socket = original
        self.assertEqual(effects, [])
        self.assertEqual(guarded["connect"], 0)
        self.assertNotIn("MIMO_API_KEY", fake.reads)
        report = value["report"]
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["live"])
        self.assertFalse(report["episode_pass"])
        self.assertFalse(report["argv_returned"])
        self.assertEqual(report["migration_equivalence"], "NOT_ASSESSED")
        self.assertFalse(report["text_probe_used_as_audio"])
        self.assertEqual(report["executed_operations"], [
            "stage.audio_review/run",
            "provider.audio_llm/compare",
        ])
        self.assertEqual(
            [(item[0], item[1]) for item in fake.calls],
            [("stage.audio_review", "run"), ("provider.audio_llm", "compare")],
        )
        self.assertTrue(all(item[2]["mode"] == "offline" for item in fake.calls))
        for child in report["calls"]:
            self.assertIsNone(child["report"]["argv"])
            self.assertEqual(child["report"]["attempts"], 0)

    def test_online_without_flag_does_not_execute(self):
        mod = self.mod

        class Api:
            def environment(self, name, default=None):
                if name == "MIMO_API_KEY":
                    raise AssertionError("closed grant read the key")
                return "" if default is None else default

        blocked = mod.Implementation({"role": "audio"}).invoke(
            "run",
            {"mode": "online", "phase": "first-batch", "output": "runs/experiments/REAL-MIMO-NOT-CREATED"},
            Api(),
        )[0]["report"]
        self.assertEqual(blocked["status"], "blocked_declaration")
        self.assertEqual(blocked["attempts"], 0)
        self.assertFalse(blocked["live"])
        self.assertIsNone(blocked["argv"])


    def test_metadata_difference_is_not_a_subtitle_number_error(self):
        mod = self.mod
        left = _fence([_cue()])
        right_item = _cue()
        del right_item["confidence"]
        result = mod.assess_subtitle_pair(left, _fence([right_item]), expected_id="16")
        self.assertEqual(result["subtitle_text"]["verdict"], "PASS")
        self.assertFalse(result["metadata"]["counted_as_subtitle_error"])
        self.assertEqual(result["metadata"]["differences"][0]["fields"], ["confidence"])
        self.assertNotIn("numbers", result["subtitle_text"]["cues"][0]["wording"]["reasons"])
        self.assertFalse(result["positional_zip"])
        self.assertEqual(result["audio_consumption"], "NOT_PROVEN")
        self.assertEqual(result["audio_effectiveness"], "NOT_ASSESSED")

    def test_subtitle_number_change_is_reported(self):
        mod = self.mod
        result = mod.assess_subtitle_pair(
            _fence([_cue(suggested_translation="一共 12 人")]),
            _fence([_cue(suggested_translation="一共 13 人")]),
            expected_id="16",
        )
        self.assertEqual(result["subtitle_text"]["verdict"], "NOT_PASSED")
        self.assertIn("numbers", result["subtitle_text"]["cues"][0]["wording"]["hard"])

    def test_missing_duplicate_and_parse_failure_do_not_pass(self):
        mod = self.mod
        missing = mod.assess_subtitle_pair(_fence([_cue(id="16")]), _fence([_cue(id="99")]), expected_id="16")
        self.assertEqual(missing["subtitle_text"]["verdict"], "NOT_ASSESSED")
        self.assertEqual(missing["subtitle_text"]["reason"], "missing_cue")
        self.assertFalse(missing["positional_zip"])
        duplicate = mod.assess_subtitle_pair(_fence([_cue(id="16"), _cue(id="16")]), _fence([_cue(id="16")]), expected_id="16")
        self.assertEqual(duplicate["subtitle_text"]["verdict"], "NOT_ASSESSED")
        self.assertEqual(duplicate["subtitle_text"]["reason"], "id_conflict")
        self.assertFalse(duplicate["subtitle_text"]["positional_zip"])
        broken = mod.assess_subtitle_pair("not json", _fence([_cue()]), expected_id="16")
        self.assertEqual(broken["subtitle_text"]["verdict"], "NOT_ASSESSED")
        self.assertEqual(broken["subtitle_text"]["reason"], "parse_failed")

    def test_truncated_stored_text_is_not_assessed(self):
        mod = self.mod
        full = _fence([_cue(confidence=0.95)])
        result = mod.assess_subtitle_pair(full, full[:20], expected_id="16", anthropic_stored_chars=len(full))
        self.assertEqual(result["subtitle_text"]["verdict"], "NOT_ASSESSED")
        self.assertEqual(result["subtitle_text"]["reason"], "truncated_stored_text")
        self.assertNotIn("cues", result["subtitle_text"])

    def test_public_result_keeps_the_full_safe_text(self):
        sent = {"ok": True, "text": "x" * 600, "failure_class": "none", "http_status": 200, "elapsed_s": 1, "model": "mimo-v2.6-pro", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        public = self.mod._public_result(sent, [])
        self.assertEqual(public["text_chars"], 600)
        self.assertEqual(len(public["text"]), 600)
        self.assertEqual(len(public["text_preview"]), 500)

    def test_phase_call_records_do_not_overwrite(self):
        mod = self.mod
        output = "runs/experiments/REAL-MIMO-R1PATH"
        root = mod.ROOT / output
        if root.exists():
            shutil.rmtree(root)
        try:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            first = mod._Capture(output, "audio-openai", "first-batch", [], archive=False)
            full = mod._Capture(output, "audio-openai", "full", [], archive=False)
            first._finish(True, "none", "mimo-v2.6-pro", {}, "prompt", "first-body", 0.0, usage)
            full._finish(True, "none", "mimo-v2.6-pro", {}, "prompt", "full-body", 0.0, usage)
            first_path = root / "calls" / "audio-openai" / "first-batch" / "0001.json"
            full_path = root / "calls" / "audio-openai" / "full" / "0001.json"
            self.assertTrue(first_path.is_file())
            self.assertTrue(full_path.is_file())
            self.assertNotEqual(first_path.read_text(encoding="utf-8"), full_path.read_text(encoding="utf-8"))
            self.assertFalse((root / "calls" / "audio-openai" / "0001.json").exists())
        finally:
            if root.exists():
                shutil.rmtree(root)


def _fence(items):
    return "```json\n" + json.dumps(items, ensure_ascii=False) + "\n```"


def _cue(**overrides):
    item = {
        "id": "16",
        "error_type": "asr_suspect",
        "original": "9時30分",
        "translation": "9点30分",
        "suggested_original": "9時30分",
        "suggested_translation": "9点30分",
        "asr_suspect": True,
        "needs_audio_review": False,
        "reason": "数字",
        "confidence": 0.9,
    }
    item.update(overrides)
    return item


if __name__ == "__main__":
    unittest.main()
