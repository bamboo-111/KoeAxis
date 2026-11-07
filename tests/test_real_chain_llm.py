"""Offline checks for real-chain-llm run_chain. No real service and no GPU."""
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRATCH = pathlib.Path(os.environ.get("PI_SCRATCH_DIR") or (ROOT / "runs" / "experiments"))
REAL_ALIGNED = ROOT / "runs/experiments/REAL-CHAIN-ALIGN-ON/work/aligned_segments.json"
REAL_TRANSCRIPT = ROOT / "runs/experiments/REAL-CHAIN-CORRECT-001/cpu-rules/work/transcript_segments.json"
ENV_KEYS = (
    "REAL_LLM_STAGES_ONLINE",
    "REAL_LLM_RECOVERY_ONLINE",
    "REAL_MIMO_ONLINE",
    "REAL_CHAIN_BIND_UPSTREAM",
    "REAL_CHAIN_LLM_ALIGNED",
    "REAL_CHAIN_LLM_TRANSCRIPT",
    "REAL_CHAIN_LLM_EXECUTE_REAL",
    "REAL_CHAIN_LLM_SPEAKER_ENROLL",
    "REAL_CHAIN_LLM_SPEAKER_NPZ",
    "REAL_CHAIN_LLM_SOURCE_AUDIO",
    "REAL_CHAIN_LLM_REUSE_SPLIT",
    "REAL_CHAIN_LLM_REUSE_CALLS",
    "REAL_CHAIN_LLM_REUSE_SPLIT_SHA",
    "REAL_CHAIN_LLM_REUSE_TRANSLATE",
    "REAL_CHAIN_ALIGN_MODE",
    "REAL_ALIGN_RECOVERY_MODE",
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
    "GEMINI_MODEL",
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
    import re
    if "Prefer merged text" in text:
        ids = [int(item) for item in re.findall(r'"id":(\d+)', text)]
        groups = [[item] for item in ids] or [[1]]
        return json.dumps({"groups": groups}, ensure_ascii=False)
    match = re.search(r"<input_subtitle>(\{.*?\})</input_subtitle>", text, re.S)
    if match:
        payload = json.loads(match.group(1))
        if isinstance(payload, dict) and payload:
            return json.dumps({key: "合成简体甲" for key in payload}, ensure_ascii=False)
        return "{}"
    return json.dumps({"1": "合成简体甲"}, ensure_ascii=False)


class Binder:
    def __init__(self, responder=None):
        self.prompts = []
        self.calls = 0
        self.connects = []
        self.lock = threading.Lock()
        self.responder = responder or fixed_gemini
        self._gemini = None
        self._client = None
        self._connect = None
        self._create = None

    def install(self):
        old = pathlib.Path(__file__).resolve().parents[1] / "compat" / "legacy"
        if str(old) not in sys.path:
            sys.path.insert(0, str(old))
        import optimizer.gemini_native as gemini_native
        import optimizer.llm_client as llm_client

        def gemini(**kwargs):
            text = payload_text(kwargs.get("payload"))
            with self.lock:
                self.calls += 1
                self.prompts.append(text)
            body = self.responder(text)
            message = type("Message", (), {"content": body})()
            choice = type("Choice", (), {"message": message})()
            usage = type("Usage", (), {"prompt_tokens": 4, "completion_tokens": 2})()
            return type("Completion", (), {"choices": [choice], "usage": usage})()

        def connect(sock, address):
            self.connects.append(repr(address))
            raise AssertionError("network connect blocked: %s" % (address,))

        self._gemini = gemini_native.generate_content
        self._client = llm_client.gemini_generate_content
        self._connect = socket.socket.connect
        self._create = socket.create_connection
        gemini_native.generate_content = gemini
        llm_client.gemini_generate_content = gemini
        socket.socket.connect = connect
        socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("create_connection blocked"))

    def restore(self):
        old = pathlib.Path(__file__).resolve().parents[1] / "compat" / "legacy"
        if str(old) not in sys.path:
            sys.path.insert(0, str(old))
        import optimizer.gemini_native as gemini_native
        import optimizer.llm_client as llm_client
        if self._gemini is not None:
            gemini_native.generate_content = self._gemini
        if self._client is not None:
            llm_client.gemini_generate_content = self._client
        if self._connect is not None:
            socket.socket.connect = self._connect
        if self._create is not None:
            socket.create_connection = self._create


def manifests(directory, marker, *, lineage="synthetic", aligned_id="segmentchain001", transcript_id=None, end=4.0, bad=None, speaker=False):
    directory.mkdir(parents=True, exist_ok=True)
    text = "こんにちは" + marker
    aligned = [{
        "segment_id": aligned_id,
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": end,
        "text": text,
        "tokens": [
            {"text": "こんにちは", "start_time": 0.2, "end_time": 0.8},
            {"text": marker, "start_time": 0.8, "end_time": 1.4},
        ],
    }]
    transcript = [{
        "segment_id": transcript_id or aligned_id,
        "text": text,
        "status": "completed",
        "audio_path": "E:/missing/chain-not-opened.wav",
        "global_start_time": 0.0,
        "global_end_time": 4.0,
        "speaker": "should-not-bind",
    }]
    if not speaker:
        transcript[0].pop("speaker")
    write_json(directory / "lineage.json", {"source": lineage, "note": "test input"})
    files = {"aligned_segments.json": aligned, "transcript_segments.json": transcript}
    for name, payload in files.items():
        path = directory / name
        if bad == name:
            path.write_text("{", encoding="utf-8", newline="\n")
        else:
            write_json(path, payload)
    return directory


def many_manifests(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    tokens = []
    parts = []
    for index in range(count):
        if index == count - 1:
            text = "独立末批边界文字十二" + "BATCHB"
        else:
            text = "".join(chr(0x4E10 + ((index * 18 + offset) % 800)) for offset in range(18))
        start = index * 3.0
        tokens.append({"text": text, "start_time": start, "end_time": start + 0.8})
        parts.append(text)
    end = (count - 1) * 3.0 + 0.8
    joined = "".join(parts)
    aligned = [{
        "segment_id": "segmentchainmany",
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": end,
        "text": joined,
        "tokens": tokens,
    }]
    transcript = [{
        "segment_id": "segmentchainmany",
        "text": joined,
        "status": "completed",
        "audio_path": "E:/missing/chain-not-opened.wav",
        "global_start_time": 0.0,
        "global_end_time": end,
    }]
    write_json(directory / "lineage.json", {"source": "synthetic", "note": "many cues"})
    write_json(directory / "aligned_segments.json", aligned)
    write_json(directory / "transcript_segments.json", transcript)
    return directory


class _Attempts:
    def __init__(self, completed, failures):
        self.circuit_open = False
        self.completed = completed
        self.failures = failures


class ChainLlmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load("real_chain_llm_ops", "modules/temporary/real_llm_stages/operations.py")
        cls.saved = {key: os.environ.get(key) for key in ENV_KEYS}
        cls.outputs = []

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
        self._connect = socket.socket.connect
        self._create = socket.create_connection
        socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("socket connect blocked"))
        socket.create_connection = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("create_connection blocked"))

    def tearDown(self):
        socket.socket.connect = self._connect
        socket.create_connection = self._create

    def _host(self, plan_name, plan_id, output_id):
        import host
        from host.runtime import Host

        output = "runs/experiments/" + output_id
        self.outputs.append(output)
        target = ROOT / output
        if target.exists():
            shutil.rmtree(target)
        registration = [item for item in host._plan_registry["plans"] if item["plan"].endswith(plan_name)]
        self.assertEqual(len(registration), 1)
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

    def _records(self, output, stage):
        folder = output / "calls" / (stage + "-a")
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]

    def test_lineage_labels_and_reuse_guard(self):
        mod = self.mod
        base = SCRATCH / "chain-llm-labels"
        if base.exists():
            shutil.rmtree(base)
        for label in ("synthetic", "baseline", "B", "current-run"):
            directory = base / label
            directory.mkdir(parents=True)
            write_json(directory / "lineage.json", {"source": label})
            self.assertEqual(mod._source_label(directory), label)
        bad = base / "bad"
        bad.mkdir()
        write_json(bad / "lineage.json", {"source": "nope"})
        with self.assertRaises(ValueError):
            mod._source_label(bad)
        stale = base / "stale"
        stale.mkdir()
        write_json(stale / "handoff" / "lineage.json", {"inputs": {"aligned_segments.json": "old", "transcript_segments.json": "old"}, "source": "current-run"})
        (stale / "handoff" / "translated_segments.json").write_text('{"1":{"translated_subtitle":"OLD"}}\n', encoding="utf-8", newline="\n")
        decision = mod.consume_previous_handoff(stale, {"aligned_segments.json": "new", "transcript_segments.json": "new"})
        self.assertEqual(decision, "reject-input-changed")
        self.assertFalse((stale / "handoff").exists())
        same = base / "same"
        same.mkdir()
        hashes = {"aligned_segments.json": "aa", "transcript_segments.json": "bb"}
        write_json(same / "handoff" / "lineage.json", {"inputs": hashes, "source": "synthetic"})
        self.assertEqual(mod.consume_previous_handoff(same, hashes), "recompute-same-input")
        self.assertFalse((same / "handoff").exists())
        split = base / "split.json"
        write_json(split, {"1": {"original_subtitle": "x", "segment_id": "other"}})
        self.assertIn("segment_id", mod.split_product_error(split, [{"segment_id": "segmentchain001"}]))
        shutil.rmtree(base, ignore_errors=True)

    def test_offline_default_does_not_call_translate(self):
        report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-OFF-DEFAULT")
        # The audit-only precheck must not be silently marked passed in a release.
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed"], 2)
        self.assertEqual(report["executed_operations"], ["stage.split/run_bound"])
        self.assertFalse(report["translate_called"])
        self.assertEqual(report["provider_attempts"], {"split": 0, "translate": 0})
        self.assertFalse(report["handoff_complete"])
        self.assertFalse(report["chain_passed"])
        self.assertEqual(report["migration_equivalence"], "NOT_ASSESSED")
        self.assertFalse(report["real_service_executed"])
        self.assertFalse(report["text_proofread_called"])
        self.assertFalse((output / "handoff" / "lineage.json").exists())
        self.assertNotIn("business_complete", json.dumps(report, ensure_ascii=False))
        self.assertTrue((output / "evidence" / "criteria.json").is_file())
        self.assertEqual({item["id"] for item in report["checks"] if not item["passed"]},
                         {"real-aligned-sha", "real-transcript-sha"})

    def test_default_run_root_unchanged(self):
        report, _output = self._host("real-llm-stages.json", "REAL-LLM-STAGES-001", "CHAIN-LLM-OLD-RUN")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["executed_operations"], [
            "stage.split/run", "stage.translate/run", "stage.text_proofread/run",
        ])
        self.assertTrue(all(item["report"]["attempts"] == 0 for item in report["calls"]))
        self.assertFalse(report["chain_passed"])

    def test_bad_inputs_make_zero_provider_calls(self):
        binder = Binder()
        binder.install()
        try:
            os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
            os.environ["GEMINI_API_KEY"] = "chain-test-key"
            os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"
            missing = SCRATCH / "chain-llm-missing"
            if missing.exists():
                shutil.rmtree(missing)
            missing.mkdir(parents=True)
            write_json(missing / "lineage.json", {"source": "synthetic"})
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(missing)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(missing)
            report, _output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-MISS")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure_class"], "missing_upstream")
            self.assertEqual(report["executed_operations"], [])
            self.assertEqual(binder.calls, 0)
            self.assertFalse(report["handoff_complete"])

            bad = manifests(SCRATCH / "chain-llm-bad", "CHAINBAD", bad="aligned_segments.json")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(bad / "aligned_segments.json")
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(bad / "transcript_segments.json")
            parsed, _output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-BAD")
            self.assertEqual(parsed["failure_class"], "parse_failed")
            self.assertEqual(parsed["executed_operations"], [])
            self.assertEqual(binder.calls, 0)

            mismatch = manifests(SCRATCH / "chain-llm-id", "CHAINID", transcript_id="segmentother")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(mismatch)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(mismatch)
            identified, _output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-ID")
            self.assertEqual(identified["failure_class"], "id_mismatch")
            self.assertEqual(identified["executed_operations"], [])

            timed = manifests(SCRATCH / "chain-llm-time", "CHAINTIME", end=9.0)
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(timed)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(timed)
            shifted, _output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-TIME")
            self.assertEqual(shifted["failure_class"], "time_mismatch")
            self.assertFalse(shifted["translate_called"])
            self.assertEqual(binder.calls, 0)
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

    def test_success_handoff_reads_this_split(self):
        binder = Binder()
        binder.install()
        evidence = {}
        try:
            os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
            os.environ["GEMINI_API_KEY"] = "chain-test-key"
            os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"
            first = manifests(SCRATCH / "chain-llm-alpha", "CHAINALPHA", speaker=True)
            write_json(first / "split_segments.json", {"1": {"original_subtitle": "DECOY_SPLIT_NOT_USED", "segment_id": "segmentchain001"}})
            write_json(first / "translated_segments.json", {"1": {"original_subtitle": "DECOY", "translated_subtitle": "DECOY_TRANSLATED_NOT_USED"}})
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(first)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(first)
            before = binder.calls
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-OK")
            self.assertEqual(report["status"], "partial", report.get("detail"))
            self.assertEqual(report["failed"], 0)
            self.assertFalse(report["chain_passed"])
            self.assertEqual(report["migration_equivalence"], "NOT_ASSESSED")
            self.assertFalse(report["real_service_executed"])
            self.assertFalse(report["text_proofread_called"])
            self.assertEqual(report["executed_operations"], ["stage.split/run_bound", "stage.translate/run_bound"])
            self.assertGreater(report["provider_attempts"]["split"], 0)
            self.assertGreater(report["provider_attempts"]["translate"], 0)
            self.assertEqual(binder.calls - before, report["provider_attempts"]["split"] + report["provider_attempts"]["translate"])
            self.assertEqual(binder.connects, [])
            self.assertNotIn("chain-test-key", json.dumps(report))
            split_records = self._records(output, "split")
            translate_records = self._records(output, "translate")
            self.assertEqual(len(split_records), report["provider_attempts"]["split"])
            self.assertEqual(len(translate_records), report["provider_attempts"]["translate"])
            self.assertTrue(any("CHAINALPHA" in item["prompt"] for item in split_records))
            self.assertTrue(any("CHAINALPHA" in item["prompt"] for item in translate_records))
            self.assertTrue(all(hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest() == item["prompt_sha256"] for item in split_records))
            self.assertTrue(all(hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest() == item["prompt_sha256"] for item in translate_records))
            self.assertFalse(any("DECOY_SPLIT_NOT_USED" in item["prompt"] for item in translate_records))
            split_child, translate_child = report["calls"]
            self.assertEqual(split_child["operation"], "stage.split/run_bound")
            self.assertEqual(translate_child["operation"], "stage.translate/run_bound")
            self.assertEqual(split_child["source_label"], "synthetic")
            produced = output / split_child["artifact"]
            self.assertEqual(translate_child["inputs"]["split_segments.json"], sha_file(produced))
            self.assertNotEqual(translate_child["inputs"]["split_segments.json"], sha_file(first / "split_segments.json"))
            handoff = output / "handoff"
            lineage = json.loads((handoff / "lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(lineage["source"], "synthetic")
            self.assertEqual(lineage["producer"], "real-chain-llm")
            self.assertEqual(set(lineage["files"]), {
                "aligned_segments.json", "transcript_segments.json", "segments.json",
                "split_segments.json", "translated_segments.json",
            })
            self.assertNotIn("lineage.json", lineage["files"])
            for name, digest in lineage["files"].items():
                self.assertEqual(digest, sha_file(handoff / name))
            self.assertEqual(lineage["inputs"]["aligned_segments.json"], sha_file(first / "aligned_segments.json"))
            self.assertEqual(lineage["inputs"]["transcript_segments.json"], sha_file(first / "transcript_segments.json"))
            produced_split = json.loads(produced.read_bytes().decode("utf-8"))
            handoff_split = json.loads((handoff / "split_segments.json").read_text(encoding="utf-8"))
            self.assertEqual(set(produced_split), set(handoff_split))
            for key, item in produced_split.items():
                bound = handoff_split[key]
                self.assertEqual(bound["original_subtitle"], item["original_subtitle"])
                self.assertEqual(bound["segment_id"], "segmentchain001")
                self.assertLess(float(bound["start_time"]), float(bound["end_time"]))
            self.assertNotIn(b"DECOY_SPLIT_NOT_USED", (handoff / "split_segments.json").read_bytes())
            self.assertNotIn(b"DECOY_TRANSLATED_NOT_USED", (handoff / "translated_segments.json").read_bytes())
            segments = json.loads((handoff / "segments.json").read_text(encoding="utf-8"))
            self.assertNotIn("speaker", segments[0])
            self.assertEqual(segments[0]["audio_path"], "E:/missing/chain-not-opened.wav")
            self.assertEqual(segments[0]["segment_id"], "segmentchain001")
            evidence["alpha"] = {
                "split_prompt_sha256": next(item["prompt_sha256"] for item in split_records if "CHAINALPHA" in item["prompt"]),
                "translate_prompt_sha256": next(item["prompt_sha256"] for item in translate_records if "CHAINALPHA" in item["prompt"]),
                "split_attempts": report["provider_attempts"]["split"],
                "translate_attempts": report["provider_attempts"]["translate"],
                "handoff_split_sha256": lineage["files"]["split_segments.json"],
                "handoff_translated_sha256": lineage["files"]["translated_segments.json"],
                "input_transcript_sha256": lineage["inputs"]["transcript_segments.json"],
            }

            second = manifests(SCRATCH / "chain-llm-beta", "CHAINBETA", lineage="current-run")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(second)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(second)
            changed, changed_output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-BETA")
            self.assertEqual(changed["status"], "partial")
            self.assertEqual(changed["calls"][0]["source_label"], "current-run")
            beta_lineage = json.loads((changed_output / "handoff" / "lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(beta_lineage["source"], "current-run")
            self.assertNotEqual(beta_lineage["inputs"]["transcript_segments.json"], evidence["alpha"]["input_transcript_sha256"])
            beta_split = self._records(changed_output, "split")
            beta_translate = self._records(changed_output, "translate")
            self.assertTrue(any("CHAINBETA" in item["prompt"] for item in beta_split))
            self.assertTrue(any("CHAINBETA" in item["prompt"] for item in beta_translate))
            self.assertFalse(any("CHAINALPHA" in item["prompt"] for item in beta_split + beta_translate))
            self.assertNotEqual(beta_split[0]["prompt_sha256"], evidence["alpha"]["split_prompt_sha256"])
            evidence["beta_split_prompt_sha256"] = next(item["prompt_sha256"] for item in beta_split if "CHAINBETA" in item["prompt"])
            evidence["socket_connects"] = len(binder.connects)
            print("CHAIN_EVIDENCE " + json.dumps(evidence, ensure_ascii=True, sort_keys=True), flush=True)
        finally:
            binder.restore()

    def test_split_failure_does_not_call_translate(self):
        def fail_split(text):
            if "Prefer merged text" in text:
                raise RuntimeError("HTTP 400 bad request")
            return fixed_gemini(text)

        binder = Binder(fail_split)
        binder.install()
        try:
            os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
            os.environ["GEMINI_API_KEY"] = "chain-test-key"
            os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"
            source = manifests(SCRATCH / "chain-llm-split-fail", "CHAINSPLITFAIL")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-SPLIT-FAIL")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["executed_operations"], ["stage.split/run_bound"])
            self.assertFalse(report["translate_called"])
            self.assertGreater(report["provider_attempts"]["split"], 0)
            self.assertEqual(report["provider_attempts"]["translate"], 0)
            self.assertFalse(report["handoff_complete"])
            self.assertFalse((output / "handoff").exists())
            self.assertFalse((output / "calls" / "translate-a").exists())
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

    def test_translate_failure_has_no_success_handoff(self):
        def fail_translate(text):
            if "<input_subtitle>" in text and "Prefer merged text" not in text:
                raise RuntimeError("HTTP 400 bad request")
            return fixed_gemini(text)

        binder = Binder(fail_translate)
        binder.install()
        try:
            os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
            os.environ["GEMINI_API_KEY"] = "chain-test-key"
            os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"
            source = manifests(SCRATCH / "chain-llm-tr-fail", "CHAINTRFAIL")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-TR-FAIL")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["executed_operations"], ["stage.split/run_bound", "stage.translate/run_bound"])
            self.assertTrue(report["translate_called"])
            self.assertGreater(report["provider_attempts"]["split"], 0)
            self.assertGreater(report["provider_attempts"]["translate"], 0)
            self.assertFalse(report["handoff_complete"])
            self.assertFalse((output / "handoff" / "lineage.json").exists())
            self.assertTrue(self._records(output, "split"))
            self.assertTrue(self._records(output, "translate"))
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

    @unittest.skipUnless(REAL_ALIGNED.is_file() and REAL_TRANSCRIPT.is_file(),
                         'local historical episode artifacts are not part of the release')
    def test_real_files_do_not_start_stages(self):
        binder = Binder()
        binder.install()
        try:
            os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
            os.environ["GEMINI_API_KEY"] = "chain-test-key"
            os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(REAL_ALIGNED)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(REAL_TRANSCRIPT)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REAL")
            self.assertEqual(report["failure_class"], "real_input_not_executed")
            self.assertTrue(report["online_refused"])
            self.assertEqual(report["executed_operations"], [])
            self.assertEqual(report["provider_attempts"], {"split": 0, "translate": 0})
            self.assertFalse(report["handoff_complete"])
            self.assertFalse(report["chain_passed"])
            self.assertEqual(binder.calls, 0)
            self.assertEqual(binder.connects, [])
            self.assertEqual(report["aligned_precheck_sha256"], sha_file(REAL_ALIGNED))
            self.assertEqual(report["transcript_precheck_sha256"], sha_file(REAL_TRANSCRIPT))
            self.assertFalse((output / "chain-input").exists())
        finally:
            binder.restore()


    def _grant_online(self):
        os.environ["REAL_LLM_STAGES_ONLINE"] = "1"
        os.environ["GEMINI_API_KEY"] = "chain-test-key"
        os.environ["GEMINI_BASE_URL"] = "https://127.0.0.1/v1"

    def test_coverage_helpers_reject_incomplete_products(self):
        mod = self.mod
        base = SCRATCH / "chain-llm-cover"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)
        aligned = [{"segment_id": "segmentchain001", "text": "甲乙"}]
        write_json(base / "aligned_segments.json", aligned)
        self.assertFalse(mod.split_covers_aligned(base / "missing-split.json", aligned))
        partial = base / "partial-split.json"
        write_json(partial, {"1": {"original_subtitle": "甲", "segment_id": "segmentchain001"}})
        self.assertEqual(mod.split_product_error(partial, aligned), "")
        self.assertFalse(mod.split_covers_aligned(partial, aligned))
        self.assertFalse(mod.stage_business_complete("split", partial, _Attempts(1, 0)))
        broken = base / "broken-split.json"
        write_json(broken, {"1": {"segment_id": "segmentchain001"}})
        self.assertIn("original_subtitle", mod.split_product_error(broken, aligned))
        full = base / "split_segments.json"
        write_json(full, {"1": {"original_subtitle": "甲 乙", "segment_id": "segmentchain001"}})
        self.assertTrue(mod.split_covers_aligned(full, aligned))
        self.assertTrue(mod.stage_business_complete("split", full, _Attempts(3, 3)))
        blank = base / "blank-translated.json"
        write_json(blank, {"1": {"original_subtitle": "甲乙", "translated_subtitle": "  "}})
        self.assertFalse(mod.translation_covers_split(full, blank))
        self.assertFalse(mod.stage_business_complete("translate", blank, _Attempts(2, 1)))
        done = base / "translated_segments.json"
        write_json(done, {"1": {"original_subtitle": "甲乙", "translated_subtitle": "译甲"}})
        self.assertTrue(mod.translation_covers_split(full, done))
        self.assertTrue(mod.stage_business_complete("translate", done, _Attempts(2, 1)))
        self.assertFalse(mod.stage_business_complete("translate", done, _Attempts(0, 0)))
        shutil.rmtree(base, ignore_errors=True)

    def test_adapter_guard_accepts_fallback_and_boundary_but_rejects_dropped_text(self):
        mod = self.mod
        base = SCRATCH / "chain-llm-adapter-guard"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)

        def write_pair(directory, aligned, transcript, split):
            directory.mkdir(parents=True)
            write_json(directory / "aligned_segments.json", aligned)
            write_json(directory / "transcript_segments.json", transcript)
            write_json(directory / "split_segments.json", split)

        fallback = base / "fallback"
        write_pair(
            fallback,
            [{
                "segment_id": "segfb",
                "status": "completed",
                "global_start_time": 0.0,
                "global_end_time": 2.0,
                "text": "甲乙",
                "tokens": [{"text": "甲乙", "start_time": 0.2, "end_time": 1.6}],
            }],
            [{
                "segment_id": "segfb",
                "status": "completed",
                "text": "甲乙丙",
                "global_start_time": 0.0,
                "global_end_time": 2.0,
                "audio_path": "E:/missing/not-opened.wav",
            }],
            {"1": {"original_subtitle": "甲乙丙", "segment_id": "segfb", "start_time": 0, "end_time": 2000}},
        )
        kept = mod.adapter_split_guard(
            fallback / "split_segments.json",
            fallback / "aligned_segments.json",
            fallback / "transcript_segments.json",
        )
        self.assertTrue(kept["passed"], kept)
        self.assertEqual(kept["content"], "pass")
        self.assertEqual(kept["short_response"], "pass")
        self.assertFalse(kept["legacy_aligned_text_cover"])
        self.assertGreater(kept["source_key_len"], 0)
        self.assertEqual(kept["source_key_sha256"], kept["result_key_sha256"])
        write_json(fallback / "split_segments.json", {
            "1": {"original_subtitle": "甲乙", "segment_id": "segfb", "start_time": 0, "end_time": 2000},
        })
        dropped = mod.adapter_split_guard(
            fallback / "split_segments.json",
            fallback / "aligned_segments.json",
            fallback / "transcript_segments.json",
        )
        self.assertFalse(dropped["passed"])
        self.assertEqual(dropped["content"], "fail")
        self.assertNotEqual(dropped["source_key_len"], dropped["result_key_len"])
        write_json(fallback / "split_segments.json", {"1": {"segment_id": "segfb"}})
        missing = mod.adapter_split_guard(
            fallback / "split_segments.json",
            fallback / "aligned_segments.json",
            fallback / "transcript_segments.json",
        )
        self.assertFalse(missing["passed"])
        self.assertIn("original_subtitle", missing["error"])

        boundary = base / "boundary"
        write_pair(
            boundary,
            [
                {
                    "segment_id": "sega",
                    "status": "completed",
                    "global_start_time": 0.0,
                    "global_end_time": 2.0,
                    "text": "あいう",
                    "tokens": [{"text": "あいう", "start_time": 0.2, "end_time": 1.8}],
                },
                {
                    "segment_id": "segb",
                    "status": "completed",
                    "global_start_time": 1.0,
                    "global_end_time": 3.0,
                    "text": "あいうか",
                    "tokens": [
                        {"text": "あ", "start_time": 1.1, "end_time": 1.4},
                        {"text": "いう", "start_time": 1.4, "end_time": 1.7},
                        {"text": "か", "start_time": 1.7, "end_time": 2.8},
                    ],
                },
            ],
            [
                {"segment_id": "sega", "status": "completed", "text": "あいう", "global_start_time": 0.0, "global_end_time": 2.0, "audio_path": "E:/missing/not-opened.wav"},
                {"segment_id": "segb", "status": "completed", "text": "あいうか", "global_start_time": 1.0, "global_end_time": 3.0, "audio_path": "E:/missing/not-opened.wav"},
            ],
            {"1": {"original_subtitle": "あいうか", "segment_id": "sega", "start_time": 200, "end_time": 2800}},
        )
        deduped = mod.adapter_split_guard(
            boundary / "split_segments.json",
            boundary / "aligned_segments.json",
            boundary / "transcript_segments.json",
        )
        self.assertTrue(deduped["passed"], deduped)
        self.assertFalse(deduped["legacy_aligned_text_cover"])
        calls = boundary / "calls"
        calls.mkdir()
        write_json(calls / "0001.json", {"ok": True, "stage": "split"})
        digest = sha_file(boundary / "split_segments.json")
        with self.assertRaises(ValueError):
            mod._accept_reused_split(
                boundary / "split_segments.json",
                calls,
                "0" * 64,
                boundary / "aligned_segments.json",
                boundary / "transcript_segments.json",
            )
        accepted = mod._accept_reused_split(
            boundary / "split_segments.json",
            calls,
            digest,
            boundary / "aligned_segments.json",
            boundary / "transcript_segments.json",
        )
        self.assertEqual(accepted["calls"], 1)
        self.assertTrue(accepted["guard"]["passed"])
        shutil.rmtree(base, ignore_errors=True)

    def test_mixed_translate_batch_does_not_handoff(self):
        def reject_last_batch(text):
            if "BATCHB" in text and "Prefer merged text" not in text:
                raise RuntimeError("batch rejected")
            return fixed_gemini(text)

        binder = Binder(reject_last_batch)
        binder.install()
        try:
            self._grant_online()
            source = many_manifests(SCRATCH / "chain-llm-mixed", 21)
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-MIXED")
            split_file = output / "stages" / "split-a" / "split_segments.json"
            translated = output / "stages" / "translate-a" / "translated_segments.json"
            split_payload = json.loads(split_file.read_text(encoding="utf-8"))
            self.assertEqual(len(split_payload), 21, sorted(split_payload))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failure_class"], "incomplete_product")
            self.assertEqual(report["executed_operations"], ["stage.split/run_bound", "stage.translate/run_bound"])
            self.assertTrue(report["translate_called"])
            self.assertFalse(report["handoff_complete"])
            self.assertFalse(report["chain_passed"])
            self.assertFalse((output / "handoff" / "lineage.json").exists())
            self.assertTrue(translated.is_file())
            self.assertFalse(self.mod.translation_covers_split(split_file, translated))
            records = self._records(output, "translate")
            self.assertTrue(any(item["ok"] and "BATCHB" not in item["prompt"] for item in records))
            self.assertTrue(any((not item["ok"]) and "BATCHB" in item["prompt"] for item in records))
            self.assertNotIn("business_complete", report["calls"][0])
            self.assertNotIn("business_complete", report["calls"][1])
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

    def test_retry_then_complete_still_writes_handoff(self):
        seen = {"merge": 0, "translate": 0}

        def retry_merge(text):
            if "Prefer merged text" in text:
                seen["merge"] += 1
                if seen["merge"] == 1:
                    raise RuntimeError("batch rejected")
            return fixed_gemini(text)

        binder = Binder(retry_merge)
        binder.install()
        try:
            self._grant_online()
            source = manifests(SCRATCH / "chain-llm-merge-retry", "CHAINMERGERETRY")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-MERGE-RETRY")
            split_records = self._records(output, "split")
            self.assertTrue(any(not item["ok"] for item in split_records))
            self.assertTrue(any(item["ok"] for item in split_records))
            self.assertGreater(report["provider_attempts"]["split"], 1)
            self.assertTrue(report["translate_called"])
            self.assertEqual(report["failure_class"], "none")
            self.assertTrue(report["handoff_complete"])
            self.assertTrue((output / "handoff" / "lineage.json").is_file())
            self.assertFalse(report["chain_passed"])
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

        def retry_translate(text):
            if "Prefer merged text" not in text:
                seen["translate"] += 1
                if seen["translate"] == 1:
                    raise RuntimeError("batch rejected")
            return fixed_gemini(text)

        binder = Binder(retry_translate)
        binder.install()
        try:
            self._grant_online()
            source = manifests(SCRATCH / "chain-llm-tr-retry", "CHAINTRRETRY")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-TR-RETRY")
            records = self._records(output, "translate")
            self.assertGreater(seen["translate"], 1)
            self.assertTrue(any(not item["ok"] for item in records))
            self.assertTrue(any(item["ok"] for item in records))
            self.assertEqual(report["failure_class"], "none")
            self.assertTrue(report["handoff_complete"])
            self.assertTrue((output / "handoff" / "lineage.json").is_file())
            self.assertEqual(report["status"], "partial")
            self.assertFalse(report["chain_passed"])
            self.assertEqual(binder.connects, [])
        finally:
            binder.restore()

    def test_execute_real_flag_reaches_offline_split(self):
        source = manifests(SCRATCH / "execute-flag-synthetic", "SYNTHETICFLAG")
        os.environ["REAL_CHAIN_LLM_EXECUTE_REAL"] = "1"
        os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source / "aligned_segments.json")
        os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source / "transcript_segments.json")
        report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REAL-FLAG")
        self.assertNotEqual(report["failure_class"], "real_input_not_executed")
        self.assertFalse(report["online_refused"])
        self.assertEqual(report["executed_operations"], ["stage.split/run_bound"])
        self.assertEqual(report["provider_attempts"], {"split": 0, "translate": 0})
        self.assertFalse(report["translate_called"])
        self.assertFalse(report["handoff_complete"])
        self.assertFalse(report["real_service_executed"])
        self.assertFalse((output / "handoff" / "lineage.json").exists())

    def test_execute_real_without_speaker_does_not_call(self):
        self._grant_online()
        source = manifests(SCRATCH / "execute-flag-synthetic", "SYNTHETICFLAG")
        os.environ["REAL_CHAIN_LLM_EXECUTE_REAL"] = "1"
        os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source / "aligned_segments.json")
        os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source / "transcript_segments.json")
        report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REAL-NOSPEAKER")
        self.assertEqual(report["failure_class"], "speaker_resource_missing")
        self.assertEqual(report["executed_operations"], [])
        self.assertEqual(report["provider_attempts"], {"split": 0, "translate": 0})
        self.assertFalse(report["real_service_executed"])
        self.assertFalse(report["handoff_complete"])
        self.assertFalse((output / "chain-input").exists())

    def test_translate_reuse_matches_prompt_not_index(self):
        mod = self.mod
        base = SCRATCH / "chain-llm-reuse-id"
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)

        def record(name, prompt, response, ok=True, sha_ok=True):
            response_sha = mod._sha_text(response)
            if not sha_ok:
                response_sha = "0" * 64
            write_json(base / name, {
                "stage": "translate",
                "ok": ok,
                "model": mod.RELAY_GEMINI_MODEL,
                "prompt": prompt,
                "response": response,
                "prompt_sha256": mod._sha_text(prompt),
                "response_sha256": response_sha,
            })

        record("0002.json", "PROMPT-A", "RESPONSE-A")
        record("0001.json", "PROMPT-B", "RESPONSE-B")
        record("0003.json", "PROMPT-EMPTY", "")
        record("0004.json", "PROMPT-FAIL", "RESPONSE-FAIL", ok=False)
        record("0005.json", "PROMPT-BADSHA", "RESPONSE-BAD", sha_ok=False)
        loaded = mod.load_translate_reuse(base)
        table = loaded["by_prompt"]
        self.assertEqual(table[mod._sha_text("PROMPT-A")]["response"], "RESPONSE-A")
        self.assertEqual(table[mod._sha_text("PROMPT-A")]["source"], "0002.json")
        self.assertEqual(table[mod._sha_text("PROMPT-B")]["source"], "0001.json")
        self.assertNotIn(mod._sha_text("PROMPT-EMPTY"), table)
        self.assertNotIn(mod._sha_text("PROMPT-FAIL"), table)
        self.assertNotIn(mod._sha_text("PROMPT-BADSHA"), table)
        record("0006.json", "PROMPT-A", "RESPONSE-OTHER")
        conflict = mod.load_translate_reuse(base)
        self.assertNotIn(mod._sha_text("PROMPT-A"), conflict["by_prompt"])
        self.assertGreaterEqual(conflict["ambiguous"], 1)
        shutil.rmtree(base, ignore_errors=True)

    def test_cached_translate_is_not_a_new_network_attempt(self):
        binder = Binder()
        binder.install()
        try:
            self._grant_online()
            source = manifests(SCRATCH / "chain-llm-reuse-src", "REUSEGAP")
            os.environ["REAL_CHAIN_LLM_ALIGNED"] = str(source)
            os.environ["REAL_CHAIN_LLM_TRANSCRIPT"] = str(source)
            report, output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REUSE-SRC")
            self.assertTrue(report["handoff_complete"], report.get("detail"))
            records = [item for item in self._records(output, "translate") if item.get("ok")]
            self.assertTrue(records)
            cached = SCRATCH / "chain-llm-reuse-cache"
            if cached.exists():
                shutil.rmtree(cached)
            cached.mkdir()
            for index, item in enumerate(records, 1):
                write_json(cached / ("%04d.json" % index), item)
            os.environ["REAL_CHAIN_LLM_REUSE_TRANSLATE"] = str(cached)

            def only_split(text):
                if "<input_subtitle>" in text and "Prefer merged text" not in text:
                    raise RuntimeError("cached translate prompt reached the network")
                return fixed_gemini(text)

            binder.restore()
            second = Binder(only_split)
            second.install()
            try:
                again, again_output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REUSE-HIT")
            finally:
                second.restore()
            self.assertEqual(again["provider_attempts"]["translate"], 0)
            self.assertGreater(again["translate_responses_reused"], 0)
            self.assertTrue(again["handoff_complete"], again.get("detail"))
            self.assertEqual(again["failure_class"], "none")
            reused = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((again_output / "calls" / "translate-a" / "reused").glob("*.json"))
            ]
            self.assertTrue(reused)
            self.assertTrue(all(item["reused"] is True and item["network"] is False for item in reused))
            self.assertTrue(all(item["usage"]["prompt_tokens"] == 0 and item["usage"]["completion_tokens"] == 0 for item in reused))
            self.assertFalse((again_output / "calls" / "translate-a").glob("*.json") and any(
                not json.loads(path.read_text(encoding="utf-8")).get("reused", False)
                for path in (again_output / "calls" / "translate-a").glob("*.json")
            ))
            self.assertEqual(second.connects, [])

            gap = SCRATCH / "chain-llm-reuse-gap"
            if gap.exists():
                shutil.rmtree(gap)
            gap.mkdir()
            broken = dict(records[0])
            broken["response"] = json.dumps({"1": "かな"}, ensure_ascii=False)
            broken["response_sha256"] = hashlib.sha256(broken["response"].encode("utf-8")).hexdigest()
            write_json(gap / "0009.json", broken)
            os.environ["REAL_CHAIN_LLM_REUSE_TRANSLATE"] = str(gap)
            third = Binder()
            third.install()
            try:
                filled, filled_output = self._host("real-chain-llm.json", "REAL-CHAIN-LLM-001", "CHAIN-LLM-REUSE-GAP")
            finally:
                third.restore()
            self.assertGreater(filled["provider_attempts"]["translate"], 0)
            self.assertGreater(filled["translate_responses_reused"], 0)
            self.assertTrue(filled["handoff_complete"], filled.get("detail"))
            network = self._records(filled_output, "translate")
            self.assertEqual(filled["provider_attempts"]["translate"], len(network))
            self.assertTrue(all(item.get("network") is True and item.get("reused") is False for item in network))
            self.assertTrue(any("Error:" in item["prompt"] or "かな" in item["prompt"] for item in network))
            self.assertEqual(third.connects, [])
        finally:
            binder.restore()

    def test_bind_keeps_crossing_cue_times_and_rejects_uncovered_start(self):
        segments = [
            {"segment_id": "a", "global_start_time": 0.0, "global_end_time": 2.0},
            {"segment_id": "b", "global_start_time": 1.5, "global_end_time": 4.0},
        ]
        split = {"1": {"start_time": 1800, "end_time": 2600, "original_subtitle": "跨"}}
        translated = {"1": {"start_time": 1800, "end_time": 2600, "original_subtitle": "跨", "translated_subtitle": "跨"}}
        bound = self.mod.bind_split_segment_ids(split, translated, segments)
        self.assertEqual(bound["1"]["segment_id"], "a")
        self.assertEqual((bound["1"]["start_time"], bound["1"]["end_time"]), (1800, 2600))
        self.assertEqual((translated["1"]["start_time"], translated["1"]["end_time"]), (1800, 2600))
        outside = {"2": {"start_time": 5000, "end_time": 5200, "original_subtitle": "外"}}
        other = {"2": {"start_time": 5000, "end_time": 5200, "original_subtitle": "外", "translated_subtitle": "外"}}
        with self.assertRaisesRegex(ValueError, "outside every segment"):
            self.mod.bind_split_segment_ids(outside, other, segments)

if __name__ == "__main__":
    unittest.main()
