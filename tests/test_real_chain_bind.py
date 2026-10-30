"""Host-level checks for explicit upstream run_bound. No network."""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import socket
import sys
import threading
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = pathlib.Path(__file__).resolve().parents[1] / "compat" / "legacy"
# A distinct synthetic comparison baseline, not an unpublished episode artifact.
BASELINE = None
SCRATCH = pathlib.Path(os.environ.get("PI_SCRATCH_DIR") or (ROOT / "runs" / "experiments"))
ENV_KEYS = (
    "REAL_LLM_STAGES_ONLINE",
    "REAL_LLM_RECOVERY_ONLINE",
    "REAL_MIMO_ONLINE",
    "REAL_CHAIN_BIND_UPSTREAM",
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
    "GEMINI_MODEL",
    "MIMO_API_KEY",
    "MIMO_BASE_URL",
)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = 16000 * 4
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * frames)


def completion(text):
    message = type("Message", (), {"content": text})()
    choice = type("Choice", (), {"message": message})()
    usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2})()
    return type("Completion", (), {"choices": [choice], "usage": usage})()


def payload_text(payload):
    chunks = []
    if not isinstance(payload, dict):
        return ""
    system = payload.get("systemInstruction")
    if isinstance(system, dict):
        for part in system.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    for item in payload.get("contents") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def fixed_gemini(text):
    import json as _json
    import re
    if "Prefer merged text" in text:
        ids = [int(item) for item in re.findall(r'"id":(\d+)', text)]
        groups = [[item] for item in ids] or [[1]]
        return _json.dumps({"groups": groups}, ensure_ascii=False)
    if "待校对字幕" in text:
        section = text.split("待校对字幕：", 1)[-1].split("后文", 1)[0]
        ids = re.findall(r'"id":"([^"]+)"', section)
        body = [{"id": item, "verdict": "ok"} for item in ids] or [{"id": "1", "verdict": "ok"}]
        return _json.dumps(body, ensure_ascii=False)
    if "<input_subtitle>{}</input_subtitle>" in text:
        return "{}"
    match = re.search(r"<input_subtitle>(\{.*?\})</input_subtitle>", text, re.S)
    if match:
        payload = _json.loads(match.group(1))
        if isinstance(payload, dict) and payload:
            return _json.dumps({key: "合成简体甲" for key in payload}, ensure_ascii=False)
        return "{}"
    return _json.dumps({"1": "合成简体甲"}, ensure_ascii=False)


class Binder:
    def __init__(self):
        self.prompts = []
        self.calls = 0
        self.connects = []
        self.lock = threading.Lock()
        self._gemini = None
        self._chat = None
        self._connect = None
        self._create = None

    def install(self):
        if str(OLD) not in sys.path:
            sys.path.insert(0, str(OLD))
        import optimizer.gemini_native as gemini_native
        import qwen_asr.mimo_requests as requests

        def gemini(**kwargs):
            text = payload_text(kwargs.get("payload"))
            with self.lock:
                self.calls += 1
                self.prompts.append(text)
            return completion(fixed_gemini(text))

        def chat(**kwargs):
            chunks = []
            for message in kwargs.get("messages") or []:
                content = message.get("content") if isinstance(message, dict) else ""
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            chunks.append(str(part.get("text") or ""))
            text = "\n".join(chunks)
            with self.lock:
                self.calls += 1
                self.prompts.append(text)
            return completion("[]")

        def connect(sock, address):
            self.connects.append(repr(address))
            raise AssertionError("network connect blocked: %s" % (address,))

        self._gemini = gemini_native.generate_content
        self._chat = requests.chat_completion_create
        self._connect = socket.socket.connect
        self._create = socket.create_connection
        gemini_native.generate_content = gemini
        requests.chat_completion_create = chat
        socket.socket.connect = connect
        socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("create_connection blocked"))

    def restore(self):
        if str(OLD) not in sys.path:
            sys.path.insert(0, str(OLD))
        import optimizer.gemini_native as gemini_native
        import qwen_asr.mimo_requests as requests
        if self._gemini is not None:
            gemini_native.generate_content = self._gemini
        if self._chat is not None:
            requests.chat_completion_create = self._chat
        if self._connect is not None:
            socket.socket.connect = self._connect
        if self._create is not None:
            socket.create_connection = self._create


def manifests(directory, marker, *, aligned_id=None, split_id=None, cover=True, bad=None):
    directory.mkdir(parents=True, exist_ok=True)
    wav = directory / "synthetic.wav"
    write_wav(wav)
    write_json(directory / "lineage.json", {"source": "synthetic", "note": "not this-round upstream"})
    segment_id = aligned_id or "segmentbind001"
    split_segment_id = split_id or segment_id
    text = "こんにちは" + marker
    aligned = [{
        "segment_id": segment_id,
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 4.0,
        "text": text,
        "tokens": [
            {"text": "こんにちは", "start_time": 0.2, "end_time": 0.8},
            {"text": marker, "start_time": 0.8, "end_time": 1.4},
        ],
    }]
    transcript = [{
        "segment_id": "segmentbind001",
        "text": text,
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 4.0,
    }]
    split = {
        "1": {
            "start_time": 500,
            "end_time": 1500,
            "original_subtitle": text,
            "translated_subtitle": "",
            "segment_id": split_segment_id,
        }
    }
    translated = {
        "1": {
            "start_time": 500,
            "end_time": 1500,
            "original_subtitle": text,
            "translated_subtitle": "你好" + marker,
            "needs_audio_review": True,
            "suspect_confidence": 0.2,
            "suspect_types": ["proper_noun"],
        }
    }
    segments = [{
        "segment_id": "segmentbind001",
        "audio_path": str(wav),
        "source_audio_path": str(wav),
        "global_start_time": 0.0 if cover else 20.0,
        "global_end_time": 4.0 if cover else 24.0,
        "text": text,
        "language": "Japanese",
    }]
    files = {
        "aligned_segments.json": aligned,
        "transcript_segments.json": transcript,
        "split_segments.json": split,
        "translated_segments.json": translated,
        "segments.json": segments,
    }
    for name, payload in files.items():
        path = directory / name
        if bad == name:
            path.write_text("{", encoding="utf-8", newline="\n")
        else:
            write_json(path, payload)
    return directory


class BindTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = load("real_llm_stages_bind", "modules/temporary/real_llm_stages/operations.py")
        cls.mimo = load("real_mimo_bind", "modules/temporary/real_mimo/operations.py")
        cls.saved = {key: os.environ.get(key) for key in ENV_KEYS}
        cls.outputs = []
        global BASELINE
        BASELINE = manifests(SCRATCH / "bind-comparison-baseline", "BASELINEDIFFERENT")

    @classmethod
    def tearDownClass(cls):
        for key, value in cls.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for output in cls.outputs:
            shutil.rmtree(ROOT / output, ignore_errors=True)

    def setUp(self):
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def _host(self, plan_name, plan_id, output_id, upstream, online_flag, key_name, base_name):
        import host
        from host.runtime import Host

        output = "runs/experiments/" + output_id
        self.outputs.append(output)
        target = ROOT / output
        if target.exists():
            shutil.rmtree(target)
        registration = [item for item in host._plan_registry["plans"] if item["plan"].endswith(plan_name)]
        self.assertEqual(len(registration), 1)
        if upstream is not None:
            os.environ["REAL_CHAIN_BIND_UPSTREAM"] = str(upstream)
        if online_flag:
            os.environ[online_flag[0]] = "1"
            os.environ[key_name] = "bind-test-key"
            os.environ[base_name] = "https://127.0.0.1/v1"
        host_obj = Host(ROOT, output, registration[0])
        try:
            result = host_obj.invoke(host_obj.envelope(
                {"run_id": "bootstrap-run", "binding_id": "bootstrap"},
                registration[0]["entry"],
                {"plan_id": plan_id},
            ))
        finally:
            host_obj.end_runs()
        self.assertEqual(result["outcome"], "returned", result)
        return result["returned"]["report"], target

    def test_default_root_stays_on_run_without_selector(self):
        report, _output = self._host(
            "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-OFFLINE-LLM", None, None, "", "",
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["case_count"], 3)
        self.assertEqual(report["executed_operations"], [
            "stage.split/run", "stage.translate/run", "stage.text_proofread/run",
        ])
        self.assertFalse(report["chain_passed"])
        self.assertTrue(all(item["report"]["attempts"] == 0 for item in report["calls"]))
        mimo, _output = self._host(
            "real-mimo.json", "REAL-MIMO-001", "BIND-OFFLINE-MIMO", None, None, "", "",
        )
        self.assertEqual(mimo["status"], "passed")
        self.assertEqual(mimo["failed"], 0)
        self.assertEqual(mimo["executed_operations"], [
            "stage.audio_review/run", "provider.audio_llm/compare",
        ])
        self.assertFalse(mimo["chain_passed"])

    def test_missing_and_bad_input_do_not_fall_back(self):
        binder = Binder()
        binder.install()
        try:
            empty = SCRATCH / "bind-empty"
            if empty.exists():
                shutil.rmtree(empty)
            empty.mkdir(parents=True)
            write_json(empty / "lineage.json", {"source": "synthetic"})
            report, _output = self._host(
                "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-MISS-LLM",
                empty, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
            )
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["chain_passed"], False)
            self.assertEqual(binder.calls, 0)
            self.assertEqual(binder.connects, [])
            blob = json.dumps(report, ensure_ascii=False)
            baseline_sha = sha_file(BASELINE / "aligned_segments.json")
            self.assertNotIn(baseline_sha, blob)
            self.assertTrue(all(item["report"]["read_baseline_fallback"] is False for item in report["calls"]))
            self.assertTrue(all(item["report"]["failure_class"] == "missing_upstream" for item in report["calls"]))
            self.assertTrue(all(item["report"]["attempts"] == 0 for item in report["calls"]))

            bad = manifests(SCRATCH / "bind-bad", "BINDUPBAD", bad="aligned_segments.json")
            (bad / "translated_segments.json").write_text("{", encoding="utf-8", newline="\n")
            (bad / "segments.json").write_text("{", encoding="utf-8", newline="\n")
            (bad / "split_segments.json").write_text("{", encoding="utf-8", newline="\n")
            (bad / "transcript_segments.json").write_text("{", encoding="utf-8", newline="\n")
            before = binder.calls
            parsed, _output = self._host(
                "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-PARSE-LLM",
                bad, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
            )
            self.assertEqual(binder.calls, before)
            self.assertTrue(all(item["report"]["failure_class"] == "parse_failed" for item in parsed["calls"]))

            mismatch = manifests(SCRATCH / "bind-id", "BINDUPID", aligned_id="segmentother", split_id="missingcue", cover=False)
            identified, _output = self._host(
                "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-ID-LLM",
                mismatch, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
            )
            classes = {item["report"]["stage"]: item["report"]["failure_class"] for item in identified["calls"]}
            self.assertEqual(classes["split"], "id_mismatch")
            self.assertEqual(classes["translate"], "id_mismatch")
            self.assertEqual(classes["text"], "id_mismatch")
            self.assertEqual(binder.calls, before)
        finally:
            binder.restore()

    def test_supplied_bytes_reach_the_business_call(self):
        binder = Binder()
        binder.install()
        try:
            first = manifests(SCRATCH / "bind-alpha", "BINDUPALPHA")
            report, output = self._host(
                "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-ALPHA-LLM",
                first, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
            )
            self.assertEqual(report["status"], "partial")
            self.assertFalse(report["chain_passed"])
            self.assertFalse(report["text_proofread_in_baseline_order"])
            self.assertGreater(binder.calls, 0)
            self.assertEqual(binder.connects, [])
            self.assertTrue(any("BINDUPALPHA" in item for item in binder.prompts))
            self.assertNotIn("bind-test-key", json.dumps(report))
            by_stage = {item["report"]["stage"]: item["report"] for item in report["calls"]}
            self.assertEqual(
                [by_stage[name]["operation"] for name in ("split", "translate", "text")],
                ["stage.split/run_bound", "stage.translate/run_bound", "stage.text_proofread/run_bound"],
            )
            self.assertEqual(by_stage["text"]["in_baseline_order"], False)
            for name, filename in (
                ("split", "aligned_segments.json"),
                ("translate", "split_segments.json"),
                ("text", "translated_segments.json"),
            ):
                self.assertEqual(by_stage[name]["inputs"][filename], sha_file(first / filename))
                self.assertNotEqual(by_stage[name]["inputs"][filename], sha_file(BASELINE / filename))
                self.assertEqual(by_stage[name]["source_label"], "synthetic")
                self.assertFalse(by_stage[name]["claimed_this_round"])
                self.assertEqual(by_stage[name]["status"], "partial")
            copied = output / "stages" / "split-a" / "aligned_segments.json"
            self.assertEqual(copied.read_bytes(), (first / "aligned_segments.json").read_bytes())

            second = manifests(SCRATCH / "bind-beta", "BINDUPBETA")
            previous = list(binder.prompts)
            changed, _output = self._host(
                "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-BETA-LLM",
                second, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
            )
            fresh = binder.prompts[len(previous):]
            self.assertTrue(any("BINDUPBETA" in item for item in fresh))
            self.assertFalse(any("BINDUPBETA" in item for item in previous))
            self.assertEqual(changed["calls"][0]["report"]["inputs"]["aligned_segments.json"], sha_file(second / "aligned_segments.json"))
            self.assertNotEqual(
                changed["calls"][0]["report"]["inputs"]["transcript_segments.json"],
                report["calls"][0]["report"]["inputs"]["transcript_segments.json"],
            )

            audio = manifests(SCRATCH / "bind-mimo", "BINDUPMIMO")
            before = len(binder.prompts)
            mimo, mimo_output = self._host(
                "real-mimo.json", "REAL-MIMO-001", "BIND-MIMO",
                audio, ("REAL_MIMO_ONLINE",), "MIMO_API_KEY", "MIMO_BASE_URL",
            )
            self.assertEqual(mimo["status"], "partial")
            self.assertFalse(mimo["chain_passed"])
            self.assertEqual(mimo["migration_equivalence"], "NOT_ASSESSED")
            self.assertEqual(mimo["executed_operations"], ["stage.audio_review/run_bound"])
            child = mimo["calls"][0]["report"]
            self.assertEqual(child["operation"], "stage.audio_review/run_bound")
            self.assertEqual(child["inputs"]["translated_segments.json"], sha_file(audio / "translated_segments.json"))
            self.assertNotEqual(child["inputs"]["translated_segments.json"], sha_file(BASELINE / "translated_segments.json"))
            self.assertFalse(child["audio_copied"])
            self.assertEqual(child["source_label"], "synthetic")
            self.assertGreater(binder.calls, before)
            self.assertTrue(any("BINDUPMIMO" in item for item in binder.prompts[before:]))
            self.assertEqual(binder.connects, [])
            copied_segments = mimo_output / "stages" / "audio-openai" / "segments.json"
            self.assertEqual(copied_segments.read_bytes(), (audio / "segments.json").read_bytes())
            self.assertEqual(pathlib.Path(json.loads(copied_segments.read_text(encoding="utf-8"))[0]["audio_path"]), audio / "synthetic.wav")
        finally:
            binder.restore()

    def test_repair_stage_attributed_ab_and_mimo_rejections(self):
        binder = Binder()
        binder.install()
        evidence = {"stages": {}, "mimo_failures": {}}
        try:
            cases = []
            for suffix, marker in (("A", "REPAIRALPHA"), ("B", "REPAIRBETA")):
                source = manifests(SCRATCH / ("bind-repair-" + suffix.lower()), marker)
                llm, llm_out = self._host(
                    "real-llm-stages.json", "REAL-LLM-STAGES-001", "BIND-REPAIR-LLM-" + suffix,
                    source, ("REAL_LLM_STAGES_ONLINE",), "GEMINI_API_KEY", "GEMINI_BASE_URL",
                )
                mimo, mimo_out = self._host(
                    "real-mimo.json", "REAL-MIMO-001", "BIND-REPAIR-MIMO-" + suffix,
                    source, ("REAL_MIMO_ONLINE",), "MIMO_API_KEY", "MIMO_BASE_URL",
                )
                self.assertEqual(llm["status"], "partial")
                self.assertEqual(mimo["status"], "partial")
                children = {x["report"]["stage"]: x["report"] for x in llm["calls"]}
                children["audio"] = mimo["calls"][0]["report"]
                run = {}
                for stage, filename, out, calls_dir in (
                    ("split", "aligned_segments.json", llm_out, "calls/split-a"),
                    ("translate", "split_segments.json", llm_out, "calls/translate-a"),
                    ("text", "translated_segments.json", llm_out, "calls/text-a"),
                    ("audio", "translated_segments.json", mimo_out, "calls/audio-openai/first-batch"),
                ):
                    child = children[stage]
                    self.assertEqual(child["status"], "partial", child)
                    self.assertFalse(child["read_baseline_fallback"])
                    self.assertEqual(child["inputs"][filename], sha_file(source / filename))
                    self.assertNotEqual(child["inputs"][filename], sha_file(BASELINE / filename))
                    artifact = out / child["artifact"]
                    self.assertTrue(artifact.is_file(), str(artifact))
                    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / calls_dir).glob("*.json"))]
                    self.assertEqual(len(records), child["attempts"])
                    matching = [r for r in records if marker in r["prompt"]]
                    self.assertTrue(matching, (stage, marker, [r["prompt"][:500] for r in records]))
                    self.assertTrue(all(hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest() == r["prompt_sha256"] for r in records))
                    run[stage] = {"marker": marker, "input_sha256": child["inputs"][filename],
                                  "prompt_sha256": matching[0]["prompt_sha256"], "requests": len(records),
                                  "artifact_sha256": sha_file(artifact)}
                cases.append(run)
            for stage in ("split", "translate", "text", "audio"):
                self.assertNotEqual(cases[0][stage]["input_sha256"], cases[1][stage]["input_sha256"])
                self.assertNotEqual(cases[0][stage]["prompt_sha256"], cases[1][stage]["prompt_sha256"])
                evidence["stages"][stage] = {"A": cases[0][stage], "B": cases[1][stage]}

            for kind, expected in (("missing", "missing_upstream"), ("parse", "parse_failed"), ("id", "id_mismatch")):
                source = manifests(SCRATCH / ("bind-repair-fail-" + kind), "REPAIRFAIL", cover=kind != "id")
                if kind == "missing":
                    (source / "split_segments.json").unlink()
                elif kind == "parse":
                    (source / "translated_segments.json").write_text("{", encoding="utf-8", newline="\n")
                before = binder.calls
                result, out = self._host(
                    "real-mimo.json", "REAL-MIMO-001", "BIND-REPAIR-FAIL-" + kind.upper(),
                    source, ("REAL_MIMO_ONLINE",), "MIMO_API_KEY", "MIMO_BASE_URL",
                )
                child = result["calls"][0]["report"]
                self.assertEqual(result["status"], "failed")
                self.assertEqual(child["operation"], "stage.audio_review/run_bound")
                self.assertEqual(child["failure_class"], expected)
                self.assertEqual(child["attempts"], 0)
                self.assertFalse(child["read_baseline_fallback"])
                self.assertEqual(binder.calls, before)
                self.assertFalse((out / "stages" / "audio-openai").exists())
                self.assertFalse((out / "calls" / "audio-openai").exists())
                progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["status"], "failed")
                self.assertEqual(progress["completed"], 0)
                evidence["mimo_failures"][kind] = {"failure_class": expected, "attempts": 0, "progress": progress["status"]}
            self.assertEqual(binder.connects, [])
            evidence["socket_connects"] = 0
            print("REPAIR_EVIDENCE " + json.dumps(evidence, ensure_ascii=True, sort_keys=True), flush=True)
        finally:
            binder.restore()


if __name__ == "__main__":
    unittest.main()
