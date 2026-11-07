"""Episode 54 Gemini stages executed through the host, not a second pipeline.

The root run calls stage.split/run, stage.translate/run, and stage.text_proofread/run.
run_chain calls stage.split/run_bound and then stage.translate/run_bound.
Those operations bind the read-only qwen_asr stage functions and pass credentials
as arguments. They do not call qwen_asr.cli.main and they do not build argv.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

OLD = Path(__file__).resolve().parents[3] / "compat" / "legacy"
BASELINE = OLD / "workspaces" / "radio-54-togeradi-20261018"
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments/specifications/real-llm-stages.json"
VERSION = "0.1.0-real-llm-stages"
TICKET = "REAL-LLM-STAGES-001"
REJECTED_GEMINI_MODEL = "gemini-3.8-flash"
RELAY_GEMINI_MODEL = "gemini-3.8-flash-high"
BASELINE_MODEL = "gemini-3.8-flash"
WORKERS = 4
GLOSSARY = OLD / "samples" / "翻译对照.xlsx"
DETERMINISTIC = frozenset({
    "api_model_route_rejected",
    "api_request_rejected",
    "auth",
    "cli_parameter_rejected",
})
STAGE_ORDER = ("split", "translate", "text")
STAGE_OPERATION = {
    "split": "stage.split/run",
    "translate": "stage.translate/run",
    "text": "stage.text_proofread/run",
}
STAGE_BOUND_OPERATION = {
    "split": "stage.split/run_bound",
    "translate": "stage.translate/run_bound",
    "text": "stage.text_proofread/run_bound",
}
BOUND_FILES = {
    "split": ("aligned_segments.json", "transcript_segments.json"),
    "translate": ("split_segments.json", "aligned_segments.json"),
    "text": ("translated_segments.json", "segments.json"),
}
BIND_ENV = "REAL_CHAIN_BIND_UPSTREAM"
ENV_NAMES = (
    ["REAL_LLM_STAGES_ONLINE", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL", "GEMINI_CONCURRENCY"]
    + [f"GEMINI_API_KEY_{i}" for i in range(1, 10)]
    + [f"GEMINI_BASE_URL_{i}" for i in range(1, 10)]
    + [f"GEMINI_MODEL_{i}" for i in range(1, 10)]
    + [f"GEMINI_CONCURRENCY_{i}" for i in range(1, 10)]
)


class CircuitOpen(BaseException):
    """Stops a stage immediately. BaseException so legacy `except Exception` cannot retry it."""

    def __init__(self, failure_class):
        super().__init__(failure_class)
        self.failure_class = failure_class


def scrub(text, secrets=None):
    out = "" if text is None else str(text)
    for secret in secrets or []:
        if secret and secret in out:
            out = out.replace(secret, "[REDACTED]")
    out = re.sub(r"https?://\S+", "[url]", out)
    out = re.sub(
        r"(?i)(x-goog-api-key|authorization|api[_-]?key)(['\"=: ]+)\S+",
        r"\1\2[REDACTED]",
        out,
    )
    return out


def endpoint_view(url):
    parsed = urlparse(str(url or ""))
    host = parsed.hostname or ""
    path = parsed.path or ""
    if re.fullmatch(r"\d+(?:\.\d+){3}", host) or ":" in host:
        kind = "numeric"
    elif host:
        kind = "named"
    else:
        kind = "absent"
    if "/v1beta" in path or "/v1alpha" in path:
        path_class = "native-versioned"
    elif path.rstrip("/").endswith("/v1"):
        path_class = "openai-v1"
    elif path:
        path_class = "other"
    else:
        path_class = "empty"
    return {"scheme": parsed.scheme or "absent", "host_kind": kind, "path_class": path_class}


def gemini_root(base):
    raw = str(base or "").strip().rstrip("/")
    if not raw:
        return ""
    lowered = raw.lower()
    if "/v1beta" in lowered or "/v1alpha" in lowered:
        return raw
    if lowered.endswith("/v1"):
        raw = raw[: -len("/v1")]
    return raw + "/v1beta"


def correct_gemini_model(name):
    text = str(name or "").strip()
    if not text or text == REJECTED_GEMINI_MODEL:
        return RELAY_GEMINI_MODEL
    return text


def classify_message(text):
    raw = text or ""
    lowered = raw.lower()
    if "unrecognized arguments" in lowered:
        return {"class": "cli_parameter_rejected", "http_status": None, "layer": "cli_parser"}
    status = None
    match = re.search(r"HTTP(?:/1\.[01])?\s+(\d{3})", raw)
    if match is None:
        match = re.search(r"(?i)error code:\s*(\d{3})", raw)
    if match:
        status = int(match.group(1))
    if "unknown provider for model" in lowered:
        return {"class": "api_model_route_rejected", "http_status": status or 400, "layer": "http_api"}
    if status in (401, 403) or "invalid api key" in lowered or "authenticationerror" in lowered or "llmautherror" in lowered:
        return {"class": "auth", "http_status": status or 401, "layer": "http_api"}
    if any(token in lowered for token in ("connecterror", "proxyerror", "connection refused", "getaddrinfo", "name or service not known")):
        return {"class": "transport_connect", "http_status": status, "layer": "transport"}
    if "timed out" in lowered or "timeout" in lowered or "请求超时" in raw:
        return {"class": "timeout", "http_status": status, "layer": "transport"}
    if status == 429 or (status is not None and status >= 500) or "ratelimit" in lowered:
        return {"class": "transient_upstream", "http_status": status, "layer": "http_api"}
    if status == 400 or "bad request" in lowered:
        return {"class": "api_request_rejected", "http_status": 400, "layer": "http_api"}
    if status is not None:
        return {"class": "http_other", "http_status": status, "layer": "http_api"}
    return {"class": "other", "http_status": None, "layer": "unknown"}


def load_spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def online_granted(spec, flag):
    """Online sockets require the explicit flag and an affirmative spec grant.

    A plan whose online permission says network false must not send, even if the
    process flag is set. The host's hardcoded environment.network field is not
    this grant and is not a substitute for it.
    """
    if str(flag or "") != "1":
        return False
    permissions = spec.get("permissions") if isinstance(spec, dict) else None
    if not isinstance(permissions, dict):
        return False
    offline = permissions.get("offline") if isinstance(permissions.get("offline"), dict) else {}
    online = permissions.get("online") if isinstance(permissions.get("online"), dict) else {}
    if offline.get("network") is not False or offline.get("credentials") is not False:
        return False
    if online.get("network") is not True:
        return False
    if "Gemini" not in str(online.get("endpoints") or ""):
        return False
    return True


def _stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _output_dir(output):
    if not output or ".." in str(output).split("/"):
        raise RuntimeError("refusing output path")
    path = (ROOT / str(output)).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise RuntimeError("refusing output outside the project")
    return path


def _read_progress(output):
    path = _output_dir(output) / "progress.json"
    if not path.is_file():
        return {
            "ticket": TICKET,
            "stage": "none",
            "repeat": "none",
            "completed": 0,
            "total": 0,
            "total_known": False,
            "failure_class": "none",
            "stages": {},
        }
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    loaded.setdefault("ticket", TICKET)
    loaded.setdefault("stages", {})
    return loaded


def update_progress(output, stage, repeat, completed, total, failure_class, status, total_known=None):
    if not output:
        return {}
    directory = _output_dir(output)
    if not directory.is_dir():
        return {}
    payload = _read_progress(output)
    known = bool(total_known) if total_known is not None else bool(total)
    slot = {
        "repeat": repeat,
        "completed": int(completed),
        "total": int(total),
        "total_known": known,
        "failure_class": failure_class or "none",
        "status": status,
    }
    payload["stages"][stage] = slot
    payload.update({
        "ticket": TICKET,
        "updated_at": _stamp(),
        "stage": stage,
        "repeat": repeat,
        "completed": int(completed),
        "total": int(total),
        "total_known": known,
        "failure_class": failure_class or "none",
        "last_progress_at": _stamp(),
        "host_call_boundary": "sequential; host invoke is not thread-safe",
        "batch_workers": WORKERS,
    })
    _write_json(directory / "progress.json", payload)
    return payload


def _inside(path, root):
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise RuntimeError("refusing to write outside the run directory")
    return resolved


def _copy_inputs(dest, names, source_dir):
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in names:
        src = source_dir / name
        if not src.is_file():
            raise FileNotFoundError(name)
        target = dest / name
        shutil.copyfile(src, target)
        copied[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return copied


def _resolve_upstream(raw):
    text = str(raw or "").strip()
    if not text or ".." in text.replace("\\", "/").split("/"):
        raise ValueError("upstream path is empty or unsafe")
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    known = {name for names in BOUND_FILES.values() for name in names}
    if path.is_file():
        if path.name not in known:
            raise ValueError("upstream file is not a required manifest: " + path.name)
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError("upstream path does not exist: " + str(path))


def _load_bound_json(source, name):
    file = source / name
    if not file.is_file():
        raise FileNotFoundError("missing upstream file: " + name)
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("parse failed: %s: %s" % (name, exc.msg)) from exc


def _source_label(source):
    label_path = source / "lineage.json"
    if not label_path.is_file():
        return "unlabeled-supplied"
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("parse failed: lineage.json: %s" % exc.msg) from exc
    if not isinstance(payload, dict):
        raise ValueError("parse failed: lineage.json: object required")
    label = str(payload.get("source") or "")
    if label not in {"synthetic", "baseline", "B", "current-run"}:
        raise ValueError("lineage.json source must be synthetic, baseline, B, or current-run")
    return label


def _segment_ids(items, name):
    if not isinstance(items, list) or not items:
        raise ValueError("%s must be a non-empty list" % name)
    found = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("segment_id") or ""):
            raise ValueError("%s item missing segment_id" % name)
        found.append(str(item["segment_id"]))
    if len(found) != len(set(found)):
        raise ValueError("duplicate segment_id in " + name)
    return found


def _cue_covered(item, segments):
    try:
        start = int(item.get("start_time", 0))
        end = int(item.get("end_time", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("translated cue has invalid timing") from exc
    for segment in segments:
        try:
            seg_start = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
            seg_end = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
        except (TypeError, ValueError) as exc:
            raise ValueError("segments.json item has invalid timing") from exc
        if end > seg_start and start < seg_end:
            return True
    return False


def _validate_bound(stage, source):
    names = BOUND_FILES[stage]
    parsed = {name: _load_bound_json(source, name) for name in names}
    label = _source_label(source)
    if stage == "split":
        aligned_ids = _segment_ids(parsed["aligned_segments.json"], "aligned_segments.json")
        transcript_ids = set(_segment_ids(parsed["transcript_segments.json"], "transcript_segments.json"))
        missing = [item for item in aligned_ids if item not in transcript_ids]
        if missing:
            raise ValueError("id mismatch: aligned segment_id not in transcript: " + ",".join(missing))
    elif stage == "translate":
        split = parsed["split_segments.json"]
        if not isinstance(split, dict) or not split:
            raise ValueError("split_segments.json must be a non-empty object")
        aligned_ids = set(_segment_ids(parsed["aligned_segments.json"], "aligned_segments.json"))
        missing = []
        for key, item in split.items():
            if not isinstance(item, dict) or "original_subtitle" not in item:
                raise ValueError("split_segments.json %s missing original_subtitle" % key)
            sid = item.get("segment_id")
            if sid and str(sid) not in aligned_ids:
                missing.append("%s:%s" % (key, sid))
        if missing:
            raise ValueError("id mismatch: split segment_id not in aligned: " + ",".join(missing))
    else:
        translated = parsed["translated_segments.json"]
        segments = parsed["segments.json"]
        if not isinstance(translated, dict) or not translated:
            raise ValueError("translated_segments.json must be a non-empty object")
        _segment_ids(segments, "segments.json")
        uncovered = []
        for key, item in translated.items():
            if not isinstance(item, dict):
                raise ValueError("translated_segments.json %s is not an object" % key)
            if not _cue_covered(item, segments):
                uncovered.append(str(key))
        if uncovered:
            raise ValueError("id dependency: translated cue not covered by segments.json: " + ",".join(uncovered))
    return label


def _bound_fields(stage, source, label):
    return {
        "operation": STAGE_BOUND_OPERATION[stage],
        "lineage": "supplied",
        "source_label": label,
        "upstream": str(source),
        "read_baseline_fallback": False,
        "explicit_path_is_baseline_tree": source.resolve() == (BASELINE / "manifests").resolve(),
        "claimed_this_round": False,
        "chain_pass": False,
        "chain_passed": False,
        "migration_equivalence": "NOT_ASSESSED",
        "coverage": "explicit_input_only",
        "required_files": list(BOUND_FILES[stage]),
        "optional_files": ["lineage.json"],
        "in_baseline_order": stage != "text",
        "audio_copied": False,
    }


def _legacy():
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    import optimizer.gemini_native as gemini_native
    import optimizer.llm_client as llm_client
    import qwen_asr.gemini_pool as gemini_pool
    import qwen_asr.models as models
    import qwen_asr.optimizer_bridge_stages as stages
    import qwen_asr.text_proofread as text_proofread
    return {
        "gemini_native": gemini_native,
        "llm_client": llm_client,
        "gemini_pool": gemini_pool,
        "models": models,
        "stages": stages,
        "text_proofread": text_proofread,
    }


def _payload_text(payload):
    chunks = []
    if not isinstance(payload, dict):
        return ""
    system = payload.get("systemInstruction")
    if isinstance(system, dict):
        for part in system.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    elif isinstance(system, str):
        chunks.append(system)
    for item in payload.get("contents") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("parts") or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


def _response_text(response):
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return "" if content is None else str(content)


def _usage_counts(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }

def _usage_known(response):
    usage = getattr(response, "usage", None)
    return usage is not None and all(
        isinstance(getattr(usage, key, None), int) and not isinstance(getattr(usage, key, None), bool)
        for key in ("prompt_tokens", "completion_tokens")
    )


class _Capture:
    def __init__(self, output, stage, repeat, secrets):
        self.output = output
        self.stage = stage
        self.repeat = repeat
        self.secrets = [item for item in secrets if item]
        self.calls = []
        self.completed = 0
        self.failures = 0
        self.failure_class = "none"
        self.circuit_open = False
        self.transient_run = 0
        self.total = 0
        self.total_known = False
        self.tokens_in = 0
        self.tokens_out = 0
        self.lock = threading.Lock()
        self._originals = []
        self.reused = []
        self.reused_count = 0
        self.reuse_by_prompt = {}

    def set_total(self, total):
        with self.lock:
            self.total = int(total)
            self.total_known = True
        update_progress(
            self.output, self.stage, self.repeat, self.completed, self.total,
            self.failure_class, "running", total_known=True,
        )

    def install(self, modules):
        native = modules["gemini_native"]
        client = modules["llm_client"]
        self._originals = [
            (native, "generate_content", native.generate_content),
            (client, "gemini_generate_content", client.gemini_generate_content),
        ]
        native.generate_content = self._wrap(native.generate_content)
        client.gemini_generate_content = native.generate_content

    def restore(self):
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals = []

    def set_reuse(self, table):
        self.reuse_by_prompt = dict(table or {})

    def _reused_completion(self, text, model):
        message = SimpleNamespace(content=text, role="assistant")
        choice = SimpleNamespace(index=0, finish_reason="stop", message=message)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return SimpleNamespace(id="reused-translate", model=model, choices=[choice], usage=usage)

    def _note_reuse(self, hit, model, view, prompt):
        with self.lock:
            self.reused_count += 1
            index = self.reused_count
            record = {
                "index": index,
                "ok": True,
                "reused": True,
                "network": False,
                "failure_class": "none",
                "model": model,
                "source": hit.get("source") or "",
                "prompt_sha256": _sha_text(prompt),
                "response_sha256": hit.get("response_sha256") or _sha_text(hit.get("response") or ""),
            }
            self.reused.append(record)
        directory = _output_dir(self.output)
        _write_json(directory / "calls" / ("%s-%s" % (self.stage, self.repeat)) / "reused" / ("%04d.json" % index), {
            "index": index,
            "stage": self.stage,
            "repeat": self.repeat,
            "reused": True,
            "network": False,
            "new_not_baseline": True,
            "ok": True,
            "failure_class": "none",
            "model": model,
            "endpoint": view,
            "source": hit.get("source") or "",
            "prompt_sha256": record["prompt_sha256"],
            "response_sha256": record["response_sha256"],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "request_started_unix_s": None,
            "request_finished_unix_s": None,
            "request_duration_s": None,
            "usage_known": False,
            "sdk_retry_count": "unknown",
            "prompt": prompt,
            "response": hit.get("response") or "",
        })

    def _wrap(self, original):
        def wrapped(**kwargs):
            model = str(kwargs.get("model") or "")
            view = endpoint_view(kwargs.get("base_url"))
            prompt = scrub(_payload_text(kwargs.get("payload")), self.secrets)
            hit = self.reuse_by_prompt.get(_sha_text(prompt))
            if hit is not None and hit.get("model") == model and str(hit.get("response") or "").strip():
                self._note_reuse(hit, model, view, prompt)
                return self._reused_completion(hit["response"], model)
            if self.circuit_open:
                raise CircuitOpen(self.failure_class)
            request_started_unix_s = time.time()
            started = time.perf_counter()
            response = None
            error = None
            try:
                response = original(**kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                finished = time.time()
                duration = time.perf_counter() - started
                kind = classify_message(scrub(str(error), self.secrets)) if error is not None else {"class": "none"}
                self._finish(
                    error is None, kind["class"], model, view, prompt,
                    scrub(_response_text(response), self.secrets) if error is None else "",
                    started, _usage_counts(response) if error is None else {"prompt_tokens": 0, "completion_tokens": 0},
                    http_status=kind.get("http_status"), error_name=type(error).__name__ if error is not None else "",
                    timing=(request_started_unix_s, finished, duration), usage_known=_usage_known(response),
                )
                if error is not None and isinstance(error, Exception):
                    stop = kind["class"] in DETERMINISTIC and not (
                        self.stage == "translate" and kind["class"] == "api_request_rejected"
                    )
                    if stop:
                        self.circuit_open = True
                        raise CircuitOpen(kind["class"]) from None
                    self.transient_run += 1
                    if self.transient_run >= 6:
                        self.circuit_open = True
                        self.failure_class = kind["class"]
                        raise CircuitOpen(kind["class"]) from None
            self.transient_run = 0
            return response
        return wrapped

    def _finish(self, ok, failure_class, model, view, prompt, response, started, usage, http_status=None, error_name="", timing=None, usage_known=False):
        with self.lock:
            self.completed += 1
            index = self.completed
            if not ok:
                self.failures += 1
                self.failure_class = failure_class
            self.tokens_in += int(usage.get("prompt_tokens") or 0)
            self.tokens_out += int(usage.get("completion_tokens") or 0)
            record = {
                "index": index,
                "ok": ok,
                "failure_class": failure_class,
                "model": model,
                "request_started_unix_s": timing[0] if timing else None,
                "request_finished_unix_s": timing[1] if timing else None,
                "request_duration_s": timing[2] if timing else None,
                "usage_known": usage_known,
                "sdk_retry_count": "unknown",
                "elapsed_s": round(time.perf_counter() - started, 3),
                "prompt_sha256": _sha_text(prompt),
                "response_sha256": _sha_text(response),
                "prompt_chars": len(prompt),
                "response_chars": len(response),
                "http_status": http_status,
                "error_name": error_name,
            }
            self.calls.append(record)
            completed = self.completed
            failure = self.failure_class
            total = self.total if self.total_known else completed
        directory = _output_dir(self.output)
        call_dir = directory / "calls" / f"{self.stage}-{self.repeat}"
        _write_json(call_dir / f"{index:04d}.json", {
            "index": index,
            "stage": self.stage,
            "repeat": self.repeat,
            "new_not_baseline": True,
            "ok": ok,
            "failure_class": failure_class,
            "model": model,
            "endpoint": view,
            "elapsed_s": record["elapsed_s"],
            "request_started_unix_s": record["request_started_unix_s"],
            "request_finished_unix_s": record["request_finished_unix_s"],
            "request_duration_s": record["request_duration_s"],
            "usage_known": record["usage_known"],
            "sdk_retry_count": "unknown",
            "prompt_sha256": record["prompt_sha256"],
            "response_sha256": record["response_sha256"],
            "usage": usage,
            "prompt": prompt,
            "response": response,
            "http_status": http_status,
            "error_name": error_name,
            "network": True,
            "reused": False,
        })
        update_progress(
            self.output, self.stage, self.repeat, completed, total, failure, "running",
            total_known=self.total_known,
        )


class _RedactLog(logging.Filter):
    def __init__(self, secrets):
        super().__init__()
        self.secrets = secrets

    def filter(self, record):
        try:
            text = record.getMessage()
        except (TypeError, ValueError):
            text = str(record.msg)
        record.msg = scrub(text, self.secrets)
        record.args = ()
        return True


def _credentials(api):
    slots = []
    suffixes = [""] + [f"_{index}" for index in range(1, 10)]
    for suffix in suffixes:
        key = str(api.environment(f"GEMINI_API_KEY{suffix}", "") or "").strip()
        base_name = "GEMINI_BASE_URL" if suffix == "" else f"GEMINI_BASE_URL{suffix}"
        base = gemini_root(str(api.environment(base_name, "") or ""))
        if key and base:
            slots.append({"name": f"gemini{suffix}", "key": key, "base": base})
    before = str(api.environment("GEMINI_MODEL", "") or "").strip()
    return slots, before


def _blank(stage, mode, status, failure_class, detail=""):
    return {
        "stage": stage,
        "operation": STAGE_OPERATION[stage],
        "mode": mode,
        "status": status,
        "live": False,
        "attempts": 0,
        "failures": 0,
        "failure_class": failure_class,
        "circuit_open": False,
        "model_sent": RELAY_GEMINI_MODEL,
        "model_before": "",
        "baseline_model": BASELINE_MODEL,
        "alias_equivalence": False,
        "episode_pass": False,
        "chain_pass": False,
        "argv": None,
        "detail": detail,
        "endpoint": endpoint_view(""),
        "inputs": {},
        "counts": {},
        "calls": [],
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "elapsed_s": 0.0,
        "artifact": "",
        "new_not_baseline": True,
        "host_isolation": "host does not enforce network isolation; environment.network is a report field",
    }


def _prepare(output, stage, repeat, names, source_dir, origin_extra=None):
    directory = _output_dir(output)
    dest = _inside(directory / "stages" / f"{stage}-{repeat}", directory)
    if dest.exists():
        shutil.rmtree(dest)
    copied = _copy_inputs(dest, names, source_dir)
    origin = {
        "new_not_baseline": True,
        "stage": stage,
        "repeat": repeat,
        "model_sent": RELAY_GEMINI_MODEL,
        "baseline_model": BASELINE_MODEL,
        "alias_equivalence": False,
        "inputs": copied,
        "note": "produced by this run; not a baseline B file",
    }
    if origin_extra:
        origin.update(origin_extra)
    _write_json(dest / "origin.json", origin)
    return dest, copied


def _run_split(modules, work, slot, speaker=None):
    if speaker is not None:
        _install_speaker_cache(work, speaker)
    modules["stages"].run_split_stage(
        work_paths=modules["models"].WorkPaths.from_workdir(work),
        optimizer_root=OLD / "optimizer",
        llm_model=RELAY_GEMINI_MODEL,
        base_url=slot["base"],
        api_key=slot["key"],
        thread_num=WORKERS,
        max_word_count_cjk=18,
        max_word_count_english=18,
        prompt_limit_ratio=0.8,
        disable_thinking=False,
        llm_extra_body=None,
        timeout=180.0,
        split_mode="rule",
        split_boundaries="bunsetsu",
        speaker_enroll=None if speaker is None else speaker["enroll"],
    )


def _content_context(api):
    selected = str(api.environment("OPT54_CONTENT_CONTEXT", "") or "").strip()
    if not selected:
        return ""
    programs = json.loads((ROOT / "experiments/references/program-content/v1.json").read_text(encoding="utf-8"))["programs"]
    if selected not in programs:
        raise ValueError("unknown OPT54_CONTENT_CONTEXT")
    binding = programs[selected]
    raw = (ROOT / binding["terms"]).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    terms = [e["term"] for e in data["entries"] if not e["ambiguous"] and not e["force_replace"]]
    return (binding["context_prefix"] + binding["context_separator"].join(terms)
            + binding["context_digest_prefix"] + hashlib.sha256(raw).hexdigest()
            + binding["context_digest_suffix"])


def _run_translate(modules, work, slot, context=""):
    modules["stages"].run_translate_stage(
        work_paths=modules["models"].WorkPaths.from_workdir(work),
        target_language="简体中文",
        llm_model=RELAY_GEMINI_MODEL,
        base_url=slot["base"],
        api_key=slot["key"],
        optimizer_root=OLD / "optimizer",
        thread_num=WORKERS,
        batch_num=20,
        custom_prompt="",
        glossary_xlsx=GLOSSARY if GLOSSARY.is_file() else None,
        disable_thinking=True,
        llm_extra_body=None,
        timeout=180.0,
        prompt_version="v1",
        context_lines=0,
        program_context=context,
    )


def _run_text(modules, work, slots, capture, context=""):
    pool = modules["gemini_pool"]
    proof = modules["text_proofread"]
    endpoints = [
        pool.GeminiEndpoint(
            name=slot["name"], base_url=slot["base"], api_key=slot["key"],
            model=RELAY_GEMINI_MODEL, max_inflight=WORKERS,
        )
        for slot in slots
    ]
    client = pool.GeminiRelayPool(endpoints, timeout=600.0)
    paths = modules["models"].WorkPaths.from_workdir(work)
    try:
        requests = proof.build_requests(
            json.loads(paths.translated_manifest.read_text(encoding="utf-8")),
            json.loads(paths.segments_manifest.read_text(encoding="utf-8")),
            proof.load_glossary(GLOSSARY) if GLOSSARY.is_file() else [],
            proof.TextProofreadConfig(model=RELAY_GEMINI_MODEL, program="日语电台节目", program_context=context),
        )
        capture.set_total(len(requests))
        config = proof.TextProofreadConfig(
            effort="medium",
            model=RELAY_GEMINI_MODEL,
            concurrency=WORKERS,
            timeout=600.0,
            program="日语电台节目",
            program_context=context,
            glossary_xlsx=str(GLOSSARY) if GLOSSARY.is_file() else None,
            resume=False,
            max_attempts=3,
        )
        return proof.run_text_proofread(
            paths, config, client=client,
            progress=lambda done, total: capture.set_total(total),
        )
    finally:
        client.close()


def _count_manifest(path):
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return 0


def _execute(stage, output, repeat, slots, before, source_dir=None, bound=False, source_label="", speaker=None, translate_reuse=None, context=""):
    started = time.perf_counter()
    secrets = [slot["key"] for slot in slots]
    capture = _Capture(output, stage, repeat, secrets)
    if translate_reuse:
        capture.set_reuse(translate_reuse.get("by_prompt") or {})
    modules = _legacy()
    log_filter = _RedactLog(secrets)
    logging.getLogger().addFilter(log_filter)
    for name in ("httpx", "httpcore", "openai", "optimizer.llm_client"):
        logging.getLogger(name).setLevel(logging.ERROR)
    update_progress(output, stage, repeat, 0, 0, "none", "running", total_known=False)
    work = None
    manifest = None
    copied = {}
    extra = {}
    source = source_dir if bound else (BASELINE / "manifests")
    origin_extra = None
    if bound:
        origin_extra = {
            "lineage": "supplied",
            "source_label": source_label,
            "upstream": str(source),
            "read_baseline_fallback": False,
            "claimed_this_round": False,
            "note": "caller-supplied bytes; not a this-round upstream claim",
            "in_baseline_order": stage != "text",
        }

    def _finish(report):
        if bound:
            report.update(_bound_fields(stage, source, source_label))
            if report.get("status") == "partial" and not str(report.get("detail") or "").strip():
                report["detail"] = "explicit upstream consumed; not a chain pass"
        return report

    try:
        capture.install(modules)
        if stage == "split":
            work, copied = _prepare(
                output, stage, repeat,
                ["aligned_segments.json", "transcript_segments.json"],
                source,
                origin_extra,
            )
            _run_split(modules, work, slots[0], speaker)
            manifest = work / "split_segments.json"
            extra["content_guard"] = manifest.is_file()
            if speaker is not None:
                extra["speaker"] = speaker.get("public") or {}
        elif stage == "translate":
            work, copied = _prepare(
                output, stage, repeat,
                ["split_segments.json", "aligned_segments.json"],
                source,
                origin_extra,
            )
            total = _count_manifest(work / "split_segments.json")
            capture.set_total((total + 19) // 20)
            _run_translate(modules, work, slots[0], context)
            manifest = work / "translated_segments.json"
        else:
            work, copied = _prepare(
                output, stage, repeat,
                ["translated_segments.json", "segments.json"],
                source,
                origin_extra,
            )
            text_report = _run_text(modules, work, slots, capture, context)
            manifest = work / "experiments" / "text-proofread" / "text_proofread_segments.json"
            if not manifest.is_file():
                manifest = modules["models"].WorkPaths.from_workdir(work).text_proofread_manifest
            extra["text_status"] = text_report.get("status") if isinstance(text_report, dict) else "missing"
            extra["text_failed"] = int(text_report.get("failed_count") or 0) if isinstance(text_report, dict) else 0
        counts = {"items": _count_manifest(manifest), "requests": capture.completed}
        status = "passed"
        if capture.circuit_open or (capture.completed and capture.failures == capture.completed):
            status = "failed"
        elif capture.failures or extra.get("text_status") == "FAIL":
            status = "partial"
        elif counts["items"] <= 0 or (capture.completed <= 0 and capture.reused_count <= 0):
            status = "failed"
        failure = capture.failure_class if status != "passed" else "none"
        if bound and status == "passed":
            status = "partial"
        update_progress(
            output, stage, repeat, capture.completed,
            capture.total if capture.total_known else capture.completed,
            failure, status, total_known=True,
        )
        report = _blank(stage, "online", status, failure)
        report.update({
            "live": True,
            "attempts": capture.completed,
            "failures": capture.failures,
            "circuit_open": capture.circuit_open,
            "model_before": before,
            "endpoint": endpoint_view(slots[0]["base"]),
            "inputs": copied,
            "counts": counts,
            "calls": capture.calls,
            "tokens": {"prompt_tokens": capture.tokens_in, "completion_tokens": capture.tokens_out},
            "elapsed_s": round(time.perf_counter() - started, 3),
            "artifact": str(manifest.relative_to(_output_dir(output))).replace("\\", "/") if manifest.is_file() else "",
            "repeat": repeat,
            "business_complete": stage_business_complete(stage, manifest, capture),
            "reused_responses": capture.reused_count,
        })
        report.update(extra)
        return _finish(report)
    except CircuitOpen as exc:
        failure = exc.failure_class or "circuit"
        update_progress(output, stage, repeat, capture.completed, capture.completed, failure, "failed", total_known=True)
        report = _blank(stage, "online", "failed", failure, "deterministic stop")
        report.update({
            "live": True,
            "attempts": capture.completed,
            "failures": capture.failures,
            "circuit_open": True,
            "model_before": before,
            "endpoint": endpoint_view(slots[0]["base"]),
            "inputs": copied,
            "calls": capture.calls,
            "tokens": {"prompt_tokens": capture.tokens_in, "completion_tokens": capture.tokens_out},
            "elapsed_s": round(time.perf_counter() - started, 3),
            "repeat": repeat,
            "business_complete": False,
            "reused_responses": capture.reused_count,
        })
        return _finish(report)
    except Exception as exc:
        failure = classify_message(scrub(str(exc), secrets))["class"]
        update_progress(output, stage, repeat, capture.completed, capture.completed, failure, "failed", total_known=True)
        report = _blank(stage, "online", "failed", failure, scrub(str(exc), secrets)[:300])
        report.update({
            "live": capture.completed > 0,
            "attempts": capture.completed,
            "failures": max(capture.failures, 1),
            "circuit_open": capture.circuit_open,
            "model_before": before,
            "endpoint": endpoint_view(slots[0]["base"]) if slots else endpoint_view(""),
            "inputs": copied,
            "calls": capture.calls,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "repeat": repeat,
            "business_complete": False,
            "reused_responses": capture.reused_count,
        })
        return _finish(report)
    finally:
        capture.restore()
        logging.getLogger().removeFilter(log_filter)


def run_stage(stage, arguments, api):
    mode = arguments["mode"]
    repeat = arguments["repeat"]
    output = arguments["output"]
    flag = str(api.environment("REAL_LLM_STAGES_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    if mode != "online" or not granted:
        report = _blank(stage, "offline", "offline", "none", "offline grant; no socket")
        report["repeat"] = repeat
        if mode == "online" and not granted:
            report["status"] = "blocked_declaration"
            report["failure_class"] = "online_not_granted"
            report["detail"] = "explicit online grant is closed; no socket"
        return report
    slots, before = _credentials(api)
    if not slots:
        report = _blank(stage, "online", "blocked_credentials", "missing_credentials", "GEMINI_API_KEY or GEMINI_BASE_URL unset")
        report["repeat"] = repeat
        report["model_before"] = before
        return report
    sent = correct_gemini_model(before)
    if sent != RELAY_GEMINI_MODEL:
        report = _blank(stage, "online", "blocked_model", "model_not_verified", "refusing an unverified model name")
        report["repeat"] = repeat
        report["model_before"] = before
        report["model_sent"] = sent
        return report
    try:
        translate_reuse = _optional_translate_reuse(api) if stage == "translate" else None
        context = _content_context(api) if stage in ("translate", "text") else ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = _blank(stage, "online", "failed", "invalid_upstream", str(exc)[:300])
        report["repeat"] = repeat
        return report
    return _execute(stage, output, repeat, slots, before, translate_reuse=translate_reuse, context=context)


def _load_manifest(path):
    if path is None or not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _ordered(payload):
    def key(item):
        return (int(item[0]) if str(item[0]).isdigit() else 10**12, str(item[0]))
    return sorted(payload.items(), key=key)


def _numbers(text):
    return re.findall(r"\d+", text or "")


def _language_shares(text):
    raw = text or ""
    total = max(1, len(raw))
    kana = len(re.findall(r"[\u3040-\u30ff]", raw))
    han = len(re.findall(r"[\u4e00-\u9fff]", raw))
    return {"kana": kana / total, "han": han / total}


def text_marks(text):
    raw = text or ""
    return {
        "sha256": _sha_text(raw),
        "length": len(raw),
        "numbers": _numbers(raw),
        "negations": len(re.findall(r"没有|不是|不会|不能|不要|不可|不得|不|没|未|无|ない|ません", raw)),
        "quotes": sum(raw.count(mark) for mark in "「」『』\"'（）()"),
        "language": _language_shares(raw),
    }


def _migration_gap():
    """Two runs of this binding are not a D01 migration comparison."""
    return {
        "verdict": "NOT_ASSESSED",
        "reason": "pair is two runs of this binding; old reference-A/reference-B and a same-condition candidate were not compared",
        "noise_plus_10pp": "NOT_ASSESSED",
        "retroactive_old_not_passed": False,
    }


@lru_cache(maxsize=1)
def _evaluation():
    """Load experiment-only code from bytes pinned by this entrypoint identity."""
    from host.runtime import capture_source, execute_source
    path = Path(__file__).with_name("evaluation.py")
    source = capture_source(path, "6f4609a7b1f4119aeb4d7825f227b8ecf564a1f93ea5b1473bffca5cf218b8e8")
    return execute_source("_llm_wording_evaluation", path, source)


def wording_assessment(candidate, reference, source):
    # Compatibility boundary: resolve legacy hooks at call time, not import time.
    return _evaluation().wording_assessment(
        candidate, reference, source, text_marks=text_marks, numbers=_numbers)


def wording_allowed(candidate, reference, source):
    item = wording_assessment(candidate, reference, source)
    return item["implemented_allowed"], item["reason"]


def _pair_verdict(hard, rejected, compared, rate, wording_quality, unassessed_wording):
    if hard or rejected:
        return "NOT_PASSED"
    if wording_quality == "comparable" and rate is not None and rate > 0.30:
        return "NOT_PASSED"
    if compared == 0 or wording_quality != "comparable":
        return "NOT_ASSESSED"
    if unassessed_wording:
        return "NOT_ASSESSED"
    return "PASS"


def compare_manifests(kind, left_path, right_path, detail=False):
    """Compare one same-binding pair. This is not a migration-equivalence result."""
    left = _load_manifest(left_path)
    right = _load_manifest(right_path)
    if not left or not right:
        return {
            "verdict": "NOT_ASSESSED",
            "pair_repeatability_verdict": "NOT_ASSESSED",
            "migration_equivalence_assessment": _migration_gap(),
            "reason": "missing manifest",
            "items": 0,
            "cap_30": "NOT_ASSESSED",
            "noise_plus_10pp": "NOT_ASSESSED",
            "proper_nouns": "NOT_ASSESSED",
            "bracket_pairing": "NOT_ASSESSED",
        }
    text_field = "original_subtitle" if kind == "split" else "translated_subtitle"
    hard_fields = ["start_time", "end_time", "original_subtitle", "speaker"]
    left_items = _ordered(left)
    right_items = _ordered(right)
    left_ids = [item[0] for item in left_items]
    right_ids = [item[0] for item in right_items]
    left_map = dict(left_items)
    right_map = dict(right_items)
    shared = [item for item in left_ids if item in right_map]
    hard_mismatch = []
    if left_ids != right_ids:
        hard_mismatch.append("ids")
    if len(left_ids) != len(right_ids):
        hard_mismatch.append("cue_count")
    comparable = left_ids == right_ids
    rejected_by_reason = {}
    wording_diffs = 0
    wording_rejected = 0
    quote_char_mismatch = 0
    unassessed_wording = 0
    compared = 0
    boundary_fields = {}
    items = []
    for lid in shared:
        litem = left_map[lid]
        ritem = right_map[lid]
        if not isinstance(litem, dict) or not isinstance(ritem, dict):
            hard_mismatch.append("shape")
            continue
        compared += 1
        row_hard = []
        for field in hard_fields:
            if litem.get(field) != ritem.get(field):
                row_hard.append(field)
                if comparable:
                    hard_mismatch.append(field)
                else:
                    boundary_fields[field] = boundary_fields.get(field, 0) + 1
        if comparable:
            assessment = wording_assessment(
                str(litem.get(text_field) or ""),
                str(ritem.get(text_field) or ""),
                str(litem.get("original_subtitle") or ""),
            )
            if assessment["quote_char_mismatch"]:
                quote_char_mismatch += 1
            if not assessment["texts_equal"]:
                wording_diffs += 1
                if assessment["proper_nouns"] == "NOT_ASSESSED" or assessment["bracket_pairing"] == "NOT_ASSESSED":
                    unassessed_wording += 1
            if not assessment["implemented_allowed"]:
                wording_rejected += 1
                for reason in assessment["reasons"]:
                    rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
            if detail and (row_hard or not assessment["implemented_allowed"] or not assessment["texts_equal"]):
                items.append({
                    "id": lid,
                    "hard_fields": row_hard,
                    "reasons": assessment["reasons"],
                    "reason": assessment["reason"],
                    "length_ratio": assessment["length_ratio"],
                    "numbers_a": assessment["numbers_a"],
                    "numbers_b": assessment["numbers_b"],
                    "negations_a": assessment["negations_a"],
                    "negations_b": assessment["negations_b"],
                    "quote_char_mismatch": assessment["quote_char_mismatch"],
                    "proper_nouns": assessment["proper_nouns"],
                    "bracket_pairing": assessment["bracket_pairing"],
                    "language_a": assessment["language_a"],
                    "language_b": assessment["language_b"],
                    "sha_a": assessment["sha_a"],
                    "sha_b": assessment["sha_b"],
                })
        elif detail and row_hard:
            items.append({"id": lid, "hard_fields": row_hard, "quality": "NOT_ASSESSED"})
    if detail:
        for lid in left_ids:
            if lid not in right_map:
                items.append({"id": lid, "side": "only_a", "quality": "NOT_ASSESSED"})
        for lid in right_ids:
            if lid not in left_map:
                items.append({"id": lid, "side": "only_b", "quality": "NOT_ASSESSED"})
    joined_left = "".join(str(item.get("original_subtitle") or "") for _, item in left_items if isinstance(item, dict))
    joined_right = "".join(str(item.get("original_subtitle") or "") for _, item in right_items if isinstance(item, dict))
    content_equal = joined_left == joined_right
    if kind == "split" and not content_equal:
        hard_mismatch.append("joined_original")
    wording_quality = "comparable" if comparable else "NOT_ASSESSED"
    rate = (wording_diffs / compared) if compared and comparable else None
    if wording_quality != "comparable":
        cap = "NOT_ASSESSED"
    elif rate is None:
        cap = "NOT_ASSESSED"
    elif rate > 0.30:
        cap = "NOT_PASSED"
    else:
        cap = "PASS"
    verdict = _pair_verdict(
        hard_mismatch, wording_rejected, compared if comparable else 0, rate, wording_quality, unassessed_wording,
    )
    result = {
        "verdict": verdict,
        "pair_repeatability_verdict": verdict,
        "migration_equivalence_assessment": _migration_gap(),
        "items": compared if comparable else 0,
        "shared_ids": len(shared),
        "cue_count_a": len(left_items),
        "cue_count_b": len(right_items),
        "only_a": len(set(left_ids) - set(right_ids)),
        "only_b": len(set(right_ids) - set(left_ids)),
        "hard_mismatch_fields": sorted(set(hard_mismatch)),
        "boundary_field_counts": boundary_fields,
        "content_joined_equal": content_equal,
        "positional_zip": False,
        "wording_quality": wording_quality,
        "wording_diff_rate": None if rate is None else round(rate, 4),
        "wording_rejected": wording_rejected if comparable else 0,
        "rejected_by_reason": rejected_by_reason,
        "quote_char_mismatch": quote_char_mismatch,
        "cap_30": cap,
        "noise_plus_10pp": "NOT_ASSESSED",
        "proper_nouns": "NOT_ASSESSED" if unassessed_wording or not comparable else "same_text",
        "bracket_pairing": "NOT_ASSESSED" if unassessed_wording or not comparable else "same_text",
        "note": "A and B are two runs of this binding. pair_repeatability_verdict is not migration equivalence. Quote character totals are not bracket pairing.",
    }
    if not comparable:
        result["wording_quality_reason"] = "cue ids differ; same-id text is a boundary shift, not a positional wording-quality score"
    if detail:
        result["item_reasons"] = items
    return result


def compare_request_sequence(left, right):
    """HTTP request-count is a hard difference. Missing temperature, retry, and fallback stay NOT_ASSESSED."""
    left = [item for item in (left or []) if isinstance(item, dict)]
    right = [item for item in (right or []) if isinstance(item, dict)]
    if not left and not right:
        count_verdict = "NOT_ASSESSED"
    elif len(left) != len(right):
        count_verdict = "NOT_PASSED"
    else:
        count_verdict = "PASS"

    def column(rows, key):
        if not rows or any(key not in row for row in rows):
            return None
        return [row.get(key) for row in rows]

    def _norm(value):
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    def same(key):
        got_left = column(left, key)
        got_right = column(right, key)
        if got_left is None or got_right is None:
            return "NOT_ASSESSED"
        if got_left == got_right:
            return "PASS"
        # A longer call list is an HTTP-count failure, not a second model or endpoint.
        if { _norm(item) for item in got_left } == { _norm(item) for item in got_right }:
            return "PASS"
        return "NOT_PASSED"

    matched = 0
    if column(left, "prompt_sha256") is not None and column(right, "prompt_sha256") is not None:
        for index in range(min(len(left), len(right))):
            if left[index].get("prompt_sha256") == right[index].get("prompt_sha256"):
                matched += 1
    return {
        "count_a": len(left),
        "count_b": len(right),
        "http_requests": count_verdict,
        "model": same("model"),
        "endpoint": same("endpoint"),
        "temperature": same("temperature"),
        "retry": same("retry"),
        "fallback": same("fallback"),
        "prompt_sha256_equal_positions": matched,
        "template_separated_from_input": "NOT_ASSESSED",
    }


def _http_from_reports(left, right):
    left_calls = left.get("calls") if isinstance(left.get("calls"), list) else []
    right_calls = right.get("calls") if isinstance(right.get("calls"), list) else []
    if left_calls or right_calls:
        return compare_request_sequence(left_calls, right_calls)
    left_attempts = left.get("attempts")
    right_attempts = right.get("attempts")
    if left_attempts is None or right_attempts is None:
        return compare_request_sequence([], [])
    return compare_request_sequence([{}] * int(left_attempts), [{}] * int(right_attempts))


def _apply_http(pair, http):
    pair["http_requests"] = http
    if http.get("http_requests") == "NOT_PASSED":
        fields = list(pair.get("hard_mismatch_fields") or [])
        if "http_requests" not in fields:
            fields.append("http_requests")
        pair["hard_mismatch_fields"] = sorted(set(fields))
        pair["verdict"] = "NOT_PASSED"
        pair["pair_repeatability_verdict"] = "NOT_PASSED"
    elif http.get("http_requests") == "NOT_ASSESSED" and pair.get("verdict") == "PASS":
        pair["verdict"] = "NOT_ASSESSED"
        pair["pair_repeatability_verdict"] = "NOT_ASSESSED"
        pair["reason"] = "http request count was not assessed"
    for key in ("model", "temperature", "retry", "fallback"):
        if http.get(key) == "NOT_PASSED":
            fields = list(pair.get("hard_mismatch_fields") or [])
            fields.append(key)
            pair["hard_mismatch_fields"] = sorted(set(fields))
            pair["verdict"] = "NOT_PASSED"
            pair["pair_repeatability_verdict"] = "NOT_PASSED"
        elif http.get(key) == "NOT_ASSESSED" and pair.get("verdict") == "PASS":
            pair["verdict"] = "NOT_ASSESSED"
            pair["pair_repeatability_verdict"] = "NOT_ASSESSED"
    return pair


def _artifact(output, report):
    relative = report.get("artifact") or ""
    if not relative or not output:
        return None
    return _output_dir(output) / relative


def build_criteria(output, reports):
    criteria = {
        "document": "docs/88-d01-online-tolerance.md",
        "retroactive_old_not_passed": False,
        "alias_equivalence": False,
        "baseline_model": BASELINE_MODEL,
        "model_sent": RELAY_GEMINI_MODEL,
        "episode_pass": False,
        "pair_repeatability_verdict": "NOT_ASSESSED",
        "migration_equivalence_assessment": _migration_gap(),
        "stages": {},
    }
    by_key = {}
    for report in reports:
        if isinstance(report, dict) and report.get("repeat") in {"a", "b"}:
            by_key[(report.get("stage"), report.get("repeat"))] = report
    verdicts = []
    for stage in STAGE_ORDER:
        left = by_key.get((stage, "a"))
        right = by_key.get((stage, "b"))
        if not left or not right or left.get("status") == "offline":
            criteria["stages"][stage] = {
                "verdict": "NOT_ASSESSED",
                "pair_repeatability_verdict": "NOT_ASSESSED",
                "migration_equivalence_assessment": _migration_gap(),
                "reason": "pair not executed",
            }
            verdicts.append("NOT_ASSESSED")
            continue
        pair = compare_manifests(stage, _artifact(output, left), _artifact(output, right))
        pair = _apply_http(pair, _http_from_reports(left, right))
        criteria["stages"][stage] = pair
        verdicts.append(pair.get("verdict"))
    if any(item == "NOT_PASSED" for item in verdicts):
        criteria["pair_repeatability_verdict"] = "NOT_PASSED"
    elif any(item == "NOT_ASSESSED" for item in verdicts):
        criteria["pair_repeatability_verdict"] = "NOT_ASSESSED"
    elif verdicts and all(item == "PASS" for item in verdicts):
        criteria["pair_repeatability_verdict"] = "PASS"
    return criteria


def summarize_call_file(path):
    """Drop prompt and response text. Keep only non-secret structure."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else ""
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "index": payload.get("index"),
        "ok": payload.get("ok"),
        "failure_class": payload.get("failure_class"),
        "model": payload.get("model"),
        "endpoint": payload.get("endpoint"),
        "prompt_sha256": payload.get("prompt_sha256"),
        "response_sha256": payload.get("response_sha256"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "retry_marker": "Fix the errors above" in prompt,
        "empty_input_marker": "<input_subtitle>{}</input_subtitle>" in prompt,
        "has_temperature": "temperature" in payload,
    }


def summarize_call_dir(path):
    folder = Path(path)
    if not folder.is_dir():
        return []
    return [summarize_call_file(item) for item in sorted(folder.glob("*.json"))]


def assess_saved_run(run_dir):
    """Re-judge saved manifests and call metadata. Does not read or write provider text out."""
    run = Path(run_dir)
    manifests = {
        "split": (
            run / "stages/split-a/split_segments.json",
            run / "stages/split-b/split_segments.json",
        ),
        "translate": (
            run / "stages/translate-a/translated_segments.json",
            run / "stages/translate-b/translated_segments.json",
        ),
        "text": (
            run / "stages/text-a/experiments/text-proofread/text_proofread_segments.json",
            run / "stages/text-b/experiments/text-proofread/text_proofread_segments.json",
        ),
    }
    stages = {}
    for stage, (left, right) in manifests.items():
        pair = compare_manifests(stage, left, right, detail=True)
        left_calls = summarize_call_dir(run / "calls" / f"{stage}-a")
        right_calls = summarize_call_dir(run / "calls" / f"{stage}-b")
        http = compare_request_sequence(left_calls, right_calls)
        http["retry_marker_a"] = sum(1 for item in left_calls if item["retry_marker"])
        http["retry_marker_b"] = sum(1 for item in right_calls if item["retry_marker"])
        http["retry_marker_indexes_a"] = [item["index"] for item in left_calls if item["retry_marker"]]
        http["retry_marker_indexes_b"] = [item["index"] for item in right_calls if item["retry_marker"]]
        http["empty_input_a"] = sum(1 for item in left_calls if item["empty_input_marker"])
        http["empty_input_b"] = sum(1 for item in right_calls if item["empty_input_marker"])
        http["empty_input_indexes_a"] = [item["index"] for item in left_calls if item["empty_input_marker"]]
        http["empty_input_indexes_b"] = [item["index"] for item in right_calls if item["empty_input_marker"]]
        http["completion_tokens_eq_1_a"] = sum(1 for item in left_calls if item["completion_tokens"] == 1)
        http["completion_tokens_eq_1_b"] = sum(1 for item in right_calls if item["completion_tokens"] == 1)
        http["temperature"] = "NOT_ASSESSED"
        http["retry"] = "NOT_ASSESSED"
        http["fallback"] = "NOT_ASSESSED"
        if http["retry_marker_a"] != http["retry_marker_b"]:
            http["validation_retry_markers"] = "NOT_PASSED"
        else:
            http["validation_retry_markers"] = "observed_equal_count"
        pair = _apply_http(pair, http)
        if http["validation_retry_markers"] == "NOT_PASSED":
            fields = list(pair.get("hard_mismatch_fields") or [])
            fields.append("validation_retry_markers")
            pair["hard_mismatch_fields"] = sorted(set(fields))
            pair["verdict"] = "NOT_PASSED"
            pair["pair_repeatability_verdict"] = "NOT_PASSED"
        stages[stage] = pair
    verdicts = [item["verdict"] for item in stages.values()]
    if any(item == "NOT_PASSED" for item in verdicts):
        pair_verdict = "NOT_PASSED"
    elif any(item == "NOT_ASSESSED" for item in verdicts):
        pair_verdict = "NOT_ASSESSED"
    else:
        pair_verdict = "PASS"
    return {
        "pair_repeatability_verdict": pair_verdict,
        "migration_equivalence_assessment": _migration_gap(),
        "stages": stages,
    }


def _child_report(result):
    if result.get("outcome") != "returned":
        raised = result.get("raised") or {}
        return {
            "outcome": "raised",
            "category": raised.get("category"),
            "message": scrub(str(raised.get("message") or ""))[:300],
        }
    report = (result.get("returned") or {}).get("report") or {}
    return {"outcome": "returned", "report": report}


def _bound_failure(stage, output, repeat, failure_class, detail):
    report = _blank(stage, "online", "failed", failure_class, detail[:300])
    report["repeat"] = repeat
    report["live"] = False
    report["attempts"] = 0
    report["operation"] = STAGE_BOUND_OPERATION[stage]
    report["lineage"] = "supplied"
    report["read_baseline_fallback"] = False
    report["claimed_this_round"] = False
    report["chain_pass"] = False
    report["chain_passed"] = False
    report["migration_equivalence"] = "NOT_ASSESSED"
    report["coverage"] = "explicit_input_only"
    report["required_files"] = list(BOUND_FILES[stage])
    report["optional_files"] = ["lineage.json"]
    report["in_baseline_order"] = stage != "text"
    report["audio_copied"] = False
    try:
        if output and _output_dir(output).is_dir():
            update_progress(output, stage, repeat, 0, 0, failure_class, "failed", total_known=True)
    except (OSError, RuntimeError, ValueError):
        pass
    return report


def _failure_class_for_bound(exc):
    message = str(exc)
    if isinstance(exc, FileNotFoundError) or message.startswith("missing upstream") or message.startswith("upstream path does not exist"):
        return "missing_upstream"
    if message.startswith("parse failed"):
        return "parse_failed"
    if message.startswith("id mismatch") or message.startswith("id dependency"):
        return "id_mismatch"
    return "invalid_upstream"


def run_stage_bound(stage, arguments, api):
    mode = arguments["mode"]
    repeat = arguments["repeat"]
    output = arguments["output"]
    try:
        source = _resolve_upstream(arguments["upstream"])
        label = _validate_bound(stage, source)
    except (FileNotFoundError, ValueError) as exc:
        return _bound_failure(stage, output, repeat, _failure_class_for_bound(exc), str(exc))
    flag = str(api.environment("REAL_LLM_STAGES_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    if mode != "online" or not granted:
        report = _blank(stage, "offline", "offline", "none", "offline grant; no socket")
        report["repeat"] = repeat
        if mode == "online" and not granted:
            report["status"] = "blocked_declaration"
            report["failure_class"] = "online_not_granted"
            report["detail"] = "explicit online grant is closed; no socket"
        report.update(_bound_fields(stage, source, label))
        return report
    slots, before = _credentials(api)
    if not slots:
        report = _blank(stage, "online", "blocked_credentials", "missing_credentials", "GEMINI_API_KEY or GEMINI_BASE_URL unset")
        report["repeat"] = repeat
        report["model_before"] = before
        report.update(_bound_fields(stage, source, label))
        return report
    sent = correct_gemini_model(before)
    if sent != RELAY_GEMINI_MODEL:
        report = _blank(stage, "online", "blocked_model", "model_not_verified", "refusing an unverified model name")
        report["repeat"] = repeat
        report["model_before"] = before
        report["model_sent"] = sent
        report.update(_bound_fields(stage, source, label))
        return report
    speaker = None
    explicit_speaker = arguments.get('speaker_resources')
    if stage == "split" and (explicit_speaker is not None or _speaker_requested(api)):
        try:
            speaker = _speaker_binding(api, explicit_speaker)
        except ValueError as exc:
            return _bound_failure(stage, output, repeat, "speaker_resource_missing", str(exc))
    try:
        translate_reuse = _optional_translate_reuse(api) if stage == "translate" else None
        context = _content_context(api) if stage in ("translate", "text") else ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _bound_failure(stage, output, repeat, "invalid_upstream", str(exc))
    return _execute(stage, output, repeat, slots, before, source_dir=source, bound=True, source_label=label, speaker=speaker, translate_reuse=translate_reuse, context=context)


def run_bind_suite(upstream, api):
    from host.contracts import ref

    flag = str(api.environment("REAL_LLM_STAGES_ONLINE", "") or "")
    spec = load_spec()
    granted = online_granted(spec, flag)
    mode = "online" if granted else "offline"
    snap = api.snapshot()["environment"]
    output = str(snap.get("output") or "")
    names = {"split": "stage.split", "translate": "stage.translate", "text": "stage.text_proofread"}
    calls = []
    for stage in STAGE_ORDER:
        target = ref(names[stage], "run_bound", VERSION)
        child = _child_report(api.call(target, {
            "mode": mode,
            "repeat": "a",
            "output": output or "offline-placeholder",
            "upstream": upstream,
        }))
        calls.append(child)
    reports = [item["report"] for item in calls if item.get("outcome") == "returned" and isinstance(item.get("report"), dict)]
    failed = sum(1 for item in calls if item.get("outcome") != "returned")
    failed += sum(1 for report in reports if report.get("status") not in {"partial", "offline"})
    summary = {
        "plan_id": TICKET,
        "status": "partial" if failed == 0 else "failed",
        "passed": len(calls) - failed,
        "failed": failed,
        "case_count": len(calls),
        "live": any(report.get("live") for report in reports),
        "episode_pass": False,
        "chain_passed": False,
        "chain_pass": False,
        "models_run": False,
        "mimo": "not_called",
        "migration_equivalence": "NOT_ASSESSED",
        "coverage": "explicit_input_only",
        "text_proofread_in_baseline_order": False,
        "read_baseline_fallback": False,
        "claimed_this_round": False,
        "executed_operations": [STAGE_BOUND_OPERATION[stage] for stage in STAGE_ORDER],
        "argv_returned": False,
        "mode": mode,
        "online_granted": granted,
        "bind_upstream": upstream,
        "host_network_flag": snap.get("network"),
        "host_plan": snap.get("plan"),
        "calls": calls,
    }
    if output and (ROOT / output).is_dir():
        directory = _output_dir(output)
        _write_json(directory / "evidence" / "report.json", summary)
        _write_json(directory / "evidence" / "criteria.json", {
            "migration_equivalence_assessment": {"verdict": "NOT_ASSESSED"},
            "chain_passed": False,
            "episode_pass": False,
            "coverage": "explicit_input_only",
            "text_proofread_in_baseline_order": False,
        })
    return summary


CHAIN_ID = "REAL-CHAIN-LLM-001"
CHAIN_PRODUCER = "real-chain-llm"
CHAIN_ALIGNED_ENV = "REAL_CHAIN_LLM_ALIGNED"
CHAIN_TRANSCRIPT_ENV = "REAL_CHAIN_LLM_TRANSCRIPT"
EXECUTE_REAL_ENV = "REAL_CHAIN_LLM_EXECUTE_REAL"
SPEAKER_ENROLL_ENV = "REAL_CHAIN_LLM_SPEAKER_ENROLL"
SPEAKER_NPZ_ENV = "REAL_CHAIN_LLM_SPEAKER_NPZ"
SOURCE_AUDIO_ENV = "REAL_CHAIN_LLM_SOURCE_AUDIO"
SPEAKER_RESOURCE_ENVS = (SPEAKER_ENROLL_ENV, SPEAKER_NPZ_ENV, SOURCE_AUDIO_ENV)
REUSE_SPLIT_ENV = "REAL_CHAIN_LLM_REUSE_SPLIT"
REUSE_CALLS_ENV = "REAL_CHAIN_LLM_REUSE_CALLS"
REUSE_SHA_ENV = "REAL_CHAIN_LLM_REUSE_SPLIT_SHA"
REUSE_TRANSLATE_ENV = "REAL_CHAIN_LLM_REUSE_TRANSLATE"


def load_translate_reuse(directory):
    """Index saved translate responses by exact prompt sha. Call order is not an identity."""
    path = Path(str(directory or "").strip())
    if not path.is_dir():
        raise ValueError("translate reuse calls missing")
    grouped = {}
    for file in sorted(item for item in path.glob("*.json") if item.is_file()):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("reused") is True:
            continue
        if payload.get("stage") not in (None, "translate"):
            continue
        prompt = payload.get("prompt")
        response = payload.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        prompt_sha = payload.get("prompt_sha256")
        response_sha = payload.get("response_sha256")
        if _sha_text(prompt) != prompt_sha or _sha_text(response) != response_sha:
            continue
        if payload.get("ok") is not True or not response.strip():
            continue
        if payload.get("model") != RELAY_GEMINI_MODEL:
            continue
        grouped.setdefault(prompt_sha, []).append({
            "response": response,
            "response_sha256": response_sha,
            "source": file.name,
            "model": payload.get("model"),
        })
    accepted = {}
    ambiguous = 0
    for key, items in grouped.items():
        if len({item["response_sha256"] for item in items}) != 1:
            ambiguous += 1
            continue
        accepted[key] = items[-1]
    return {"by_prompt": accepted, "ambiguous": ambiguous, "files": sum(len(items) for items in grouped.values())}


def _optional_translate_reuse(api):
    raw = str(api.environment(REUSE_TRANSLATE_ENV, "") or "").strip()
    if not raw:
        return None
    return load_translate_reuse(raw)


CHAIN_SYNTHETIC = ROOT / "modules/temporary/real_llm_stages/chain_synthetic"
REAL_ALIGNED_REL = "runs/experiments/REAL-CHAIN-ALIGN-ON/work/aligned_segments.json"
REAL_TRANSCRIPT_REL = "runs/experiments/REAL-CHAIN-CORRECT-001/cpu-rules/work/transcript_segments.json"
REAL_ALIGNED_SHA = "2cdd86f4e31946a931541db7b407e7fbc22bbd5451dcd589007996f41b9d71c8"
REAL_TRANSCRIPT_SHA = "b76058c6f4330c577c25eef4e82a91800682a5c31c4a181db4a23c34cb56d2b2"
HANDOFF_NAMES = (
    "aligned_segments.json",
    "transcript_segments.json",
    "segments.json",
    "split_segments.json",
    "translated_segments.json",
)
ATTEMPTS_SEMANTICS = "provider-boundary invocations, including stubs; not a confirmed remote service"
CHAIN_CRITERIA = {
    "migration_equivalence_assessment": {"verdict": "NOT_ASSESSED"},
    "chain_passed": False,
    "episode_pass": False,
    "coverage": "explicit_input_only",
    "real_service_executed": False,
    "text_proofread_in_run_chain": False,
}


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precheck_real():
    def one(relative, expected):
        path = ROOT / relative
        if not path.is_file():
            return False, ""
        digest = _sha_file(path)
        return digest == expected, digest

    aligned_ok, aligned_sha = one(REAL_ALIGNED_REL, REAL_ALIGNED_SHA)
    transcript_ok, transcript_sha = one(REAL_TRANSCRIPT_REL, REAL_TRANSCRIPT_SHA)
    return {
        "aligned_ok": aligned_ok,
        "aligned_sha256": aligned_sha,
        "transcript_ok": transcript_ok,
        "transcript_sha256": transcript_sha,
    }


def _known_real(path):
    targets = [(ROOT / REAL_ALIGNED_REL).resolve(), (ROOT / REAL_TRANSCRIPT_REL).resolve()]
    try:
        return path.resolve() in targets
    except OSError:
        return False


def _execute_real(api):
    return str(api.environment(EXECUTE_REAL_ENV, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _resource_path(api, name):
    raw = str(api.environment(name, "") or "").strip()
    if not raw:
        return None
    text = raw.replace("\\", "/")
    if ".." in text.split("/"):
        raise ValueError("resource path is unsafe: " + name)
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _speaker_requested(api):
    return any(str(api.environment(name, "") or "").strip() for name in SPEAKER_RESOURCE_ENVS)


def _speaker_binding(api, explicit=None):
    """Bind explicit speaker inputs or preserve the older environment interface."""
    if explicit is None:
        enroll = _resource_path(api, SPEAKER_ENROLL_ENV)
        npz = _resource_path(api, SPEAKER_NPZ_ENV)
        audio = _resource_path(api, SOURCE_AUDIO_ENV)
    else:
        if not isinstance(explicit, dict) or set(explicit) != {'enroll', 'npz', 'audio'}:
            raise ValueError('speaker_resources requires enroll, npz and audio')
        if any(not isinstance(explicit[key], str) or not Path(explicit[key]).is_absolute() for key in explicit):
            raise ValueError('speaker resource paths must be absolute')
        enroll, npz, audio = (Path(explicit[key]).resolve() for key in ('enroll', 'npz', 'audio'))
    missing = [
        name for name, path in (
            (SPEAKER_ENROLL_ENV, enroll),
            (SPEAKER_NPZ_ENV, npz),
            (SOURCE_AUDIO_ENV, audio),
        )
        if path is None or not path.is_file()
    ]
    if missing:
        raise ValueError("speaker resource missing: " + ",".join(missing))
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    import numpy as np
    from qwen_asr.speaker.voiceprint import load_enrollment
    enrollment = load_enrollment(enroll)
    with np.load(npz, allow_pickle=False) as packed:
        meta = json.loads(str(packed["meta"]))
    key = meta.get("key") if isinstance(meta, dict) else None
    if not isinstance(key, dict):
        raise ValueError("speaker cache has no key")
    stat = audio.stat()
    if int(key.get("audio_size") or -1) != stat.st_size:
        raise ValueError("speaker cache audio size does not match the source audio")
    if str(key.get("enrollment") or "") != enrollment.digest:
        raise ValueError("speaker cache enrollment digest does not match")
    if key.get("speakers") not in ([], None):
        raise ValueError("speaker cache was built for an explicit speaker subset")
    if int(key.get("version") or 0) != 2:
        raise ValueError("speaker cache version is not reusable")
    return {
        "enroll": str(enroll),
        "npz": str(npz),
        "audio": str(audio),
        "audio_sha256": _sha_file(audio),
        "npz_sha256": _sha_file(npz),
        "enrollment_digest": enrollment.digest,
        "cache_mtime_ns": int(key.get("audio_mtime_ns") or 0),
    }


def _install_speaker_cache(work, binding):
    """Place the cached track where the legacy split reads it. Do not recompute scores."""
    audio_dest = work / "audio.wav"
    npz_dest = work / "speaker_scores.npz"
    shutil.copy2(binding["audio"], audio_dest)
    shutil.copyfile(binding["npz"], npz_dest)
    if _sha_file(audio_dest) != binding["audio_sha256"]:
        raise ValueError("staged speaker audio hash differs from the cached source")
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    from qwen_asr.speaker.voiceprint import SpeakerTrack, cache_key, load_enrollment
    enrollment = load_enrollment(binding["enroll"])
    if enrollment.digest != binding["enrollment_digest"]:
        raise ValueError("speaker enrollment digest changed")
    key = cache_key(audio_dest, enrollment, None)
    retargeted = False
    track = SpeakerTrack.load(npz_dest, key)
    if track is None:
        loose = SpeakerTrack.load(npz_dest, None)
        if loose is None:
            raise ValueError("speaker cache unreadable")
        loose.save(npz_dest, key)
        retargeted = True
        track = SpeakerTrack.load(npz_dest, key)
    if track is None:
        raise ValueError("speaker cache did not bind to the staged audio")
    public = {
        "reused": True,
        "retargeted": retargeted,
        "members": [str(item) for item in track.members],
        "npz_sha256": binding["npz_sha256"],
        "audio_sha256": binding["audio_sha256"],
        "enrollment_digest": enrollment.digest,
        "baseline_used_speaker": True,
        "gpu_used": False,
    }
    binding["public"] = public
    return public



def _chain_role_path(raw, filename):
    if not str(raw or "").strip():
        return (CHAIN_SYNTHETIC / filename).resolve()
    text = str(raw).strip()
    if ".." in text.replace("\\", "/").split("/"):
        raise ValueError("upstream path is empty or unsafe")
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.is_dir():
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError("missing upstream file: " + filename)
    return path


def _read_json_value(path):
    if not path.is_file():
        return None, ("missing_upstream", "missing file: " + path.name)
    try:
        return json.loads(path.read_text(encoding="utf-8")), ("", "")
    except json.JSONDecodeError as exc:
        return None, ("parse_failed", "parse failed: %s: %s" % (path.name, exc.msg))
    except OSError as exc:
        return None, ("missing_upstream", str(exc))


def _same_time(left, right):
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _chain_dependency_error(aligned, transcript):
    try:
        aligned_ids = _segment_ids(aligned, "aligned_segments.json")
        transcript_ids = _segment_ids(transcript, "transcript_segments.json")
    except ValueError as exc:
        return "id_mismatch", str(exc)
    missing = [item for item in aligned_ids if item not in set(transcript_ids)]
    if missing:
        return "id_mismatch", "aligned segment_id not in transcript: " + ",".join(missing)
    by_id = {str(item["segment_id"]): item for item in transcript}
    for item in aligned:
        other = by_id[str(item["segment_id"])]
        if not _same_time(item.get("global_start_time"), other.get("global_start_time")):
            return "time_mismatch", "global_start_time mismatch for " + str(item["segment_id"])
        if not _same_time(item.get("global_end_time"), other.get("global_end_time")):
            return "time_mismatch", "global_end_time mismatch for " + str(item["segment_id"])
    return "", ""


def _chain_label(source_dir, explicit):
    if not explicit:
        return "synthetic"
    path = source_dir / "lineage.json"
    if not path.is_file():
        return "current-run"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "current-run"
    if isinstance(payload, dict) and payload.get("source") in {"synthetic", "current-run"}:
        return str(payload["source"])
    return "current-run"


def _derive_segments(items, source_audio=""):
    derived = []
    for item in items:
        row = {
            "segment_id": str(item["segment_id"]),
            "global_start_time": item.get("global_start_time"),
            "global_end_time": item.get("global_end_time"),
        }
        if "audio_path" in item and item.get("audio_path") is not None:
            row["audio_path"] = item["audio_path"]
        if source_audio:
            row["source_audio_path"] = source_audio
        derived.append(row)
    return derived


def consume_previous_handoff(directory, input_hashes):
    """Drop any previous handoff. Never treat it as this run's success."""
    path = directory / "handoff" / "lineage.json"
    if not path.is_file():
        return "absent"
    decision = "recompute-same-input"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        decision = "reject-unreadable"
        payload = None
    if not isinstance(payload, dict) or payload.get("inputs") != input_hashes:
        decision = "reject-input-changed"
    shutil.rmtree(directory / "handoff", ignore_errors=True)
    return decision


def split_product_error(split_file, aligned_payload):
    if not split_file.is_file():
        return "split artifact missing"
    try:
        payload = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "split artifact parse failed"
    if not isinstance(payload, dict) or not payload:
        return "split artifact empty"
    try:
        aligned_ids = set(_segment_ids(aligned_payload, "aligned_segments.json"))
    except ValueError as exc:
        return str(exc)
    for key, item in payload.items():
        if not isinstance(item, dict) or "original_subtitle" not in item:
            return "split item missing original_subtitle: " + str(key)
        sid = item.get("segment_id")
        if sid and str(sid) not in aligned_ids:
            return "split segment_id not in aligned: " + str(sid)
    return ""


def _norm_text(value):
    return re.sub(r"\s+", "", str(value or ""))


def _ordered_items(payload):
    def sort_key(key):
        text = str(key)
        return (0, int(text)) if text.isdigit() else (1, text)

    return [payload[key] for key in sorted(payload, key=sort_key)]


def split_covers_aligned(split_file, aligned_payload):
    if not split_file.is_file() or not isinstance(aligned_payload, list):
        return False
    try:
        payload = json.loads(split_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    produced = "".join(
        _norm_text(item.get("original_subtitle"))
        for item in _ordered_items(payload)
        if isinstance(item, dict)
    )
    expected = "".join(_norm_text(item.get("text")) for item in aligned_payload if isinstance(item, dict))
    return bool(expected) and produced == expected


class _GuardCue:
    def __init__(self, text, start, end):
        self.text = str(text or "")
        self.start_time = start
        self.end_time = end



def adapter_split_guard(split_file, aligned_file, transcript_file):
    """Old adapter source plus the old content and short-response guards.

    Equality with aligned[].text is recorded and is not the decision.
    """
    split_file = Path(split_file)
    aligned_file = Path(aligned_file)
    transcript_file = Path(transcript_file)
    report = {
        "passed": False,
        "error": "",
        "content": "not_run",
        "short_response": "not_run",
        "source_key_len": 0,
        "result_key_len": 0,
        "source_key_sha256": "",
        "result_key_sha256": "",
        "legacy_aligned_text_cover": False,
    }
    try:
        aligned_payload = json.loads(aligned_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report["error"] = "split guard input unreadable"
        return report
    report["legacy_aligned_text_cover"] = split_covers_aligned(split_file, aligned_payload)
    product_error = split_product_error(split_file, aligned_payload)
    if product_error:
        report["error"] = product_error
        report["content"] = "fail"
        return report
    if aligned_file.name != "aligned_segments.json" or transcript_file.name != "transcript_segments.json":
        report["error"] = "adapter guard requires aligned_segments.json and transcript_segments.json"
        return report
    if aligned_file.parent.resolve() != transcript_file.parent.resolve():
        report["error"] = "adapter guard requires aligned and transcript in one directory"
        return report
    try:
        split_payload = json.loads(split_file.read_text(encoding="utf-8"))
        if str(OLD) not in sys.path:
            sys.path.insert(0, str(OLD))
        from qwen_asr.models import WorkPaths
        from qwen_asr.optimizer_bridge_adapter import _normalize_content, aligned_manifest_to_asr_data
        from qwen_asr.optimizer_bridge_guards import (
            validate_split_content_preserved,
            validate_split_short_responses_preserved,
        )
        from qwen_asr.optimizer_bridge_stages import load_optimizer_types
        source = aligned_manifest_to_asr_data(
            WorkPaths.from_workdir(aligned_file.parent),
            *load_optimizer_types(OLD / "optimizer")[:2],
        )
        cues = [
            _GuardCue(item.get("original_subtitle"), item.get("start_time"), item.get("end_time"))
            for item in _ordered_items(split_payload)
            if isinstance(item, dict)
        ]
        source_key = _normalize_content("".join(str(segment.text) for segment in source.segments))
        result_key = _normalize_content("".join(cue.text for cue in cues))
        report["source_key_len"] = len(source_key)
        report["result_key_len"] = len(result_key)
        report["source_key_sha256"] = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
        report["result_key_sha256"] = hashlib.sha256(result_key.encode("utf-8")).hexdigest()
        validate_split_content_preserved(source.segments, cues, normalize_content=_normalize_content)
        report["content"] = "pass"
        validate_split_short_responses_preserved(source.segments, cues, normalize_content=_normalize_content)
        report["short_response"] = "pass"
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)[:300]
        if report["content"] != "pass":
            report["content"] = "fail"
        else:
            report["short_response"] = "fail"
    return report


def _accept_reused_split(split_raw, calls_raw, expected_sha, aligned_file, transcript_file):
    """Bind a finished split by hash, call records, and the old guards."""
    split_path = Path(str(split_raw or "").strip())
    if split_path.is_dir():
        split_path = split_path / "split_segments.json"
    if not split_path.is_file():
        raise ValueError("reused split file missing")
    actual = _sha_file(split_path)
    if not expected_sha or actual != str(expected_sha).strip():
        raise ValueError("reused split sha mismatch")
    work = split_path.parent
    if _sha_file(work / "aligned_segments.json") != _sha_file(aligned_file):
        raise ValueError("reused split aligned input does not match this chain")
    if _sha_file(work / "transcript_segments.json") != _sha_file(transcript_file):
        raise ValueError("reused split transcript input does not match this chain")
    calls_dir = Path(str(calls_raw or "").strip())
    records = sorted(path for path in calls_dir.glob("*.json") if path.is_file()) if calls_dir.is_dir() else []
    if not records:
        raise ValueError("reused split calls missing")
    for path in records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("stage") != "split":
            raise ValueError("reused split call is not a successful split record")
    guard = adapter_split_guard(split_path, work / "aligned_segments.json", work / "transcript_segments.json")
    if not guard["passed"]:
        raise ValueError(guard["error"] or "reused split failed the adapter guard")
    return {"split_file": split_path, "calls": len(records), "sha256": actual, "guard": guard}


def translation_covers_split(split_file, translated_file):
    try:
        split_payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        translated = json.loads(Path(translated_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(split_payload, dict) or not split_payload or not isinstance(translated, dict):
        return False
    for key, item in split_payload.items():
        original = _norm_text(item.get("original_subtitle") if isinstance(item, dict) else "")
        out = translated.get(str(key))
        if not isinstance(out, dict):
            return False
        if original and not _norm_text(out.get("translated_subtitle")):
            return False
    return True


def stage_business_complete(stage, manifest, capture):
    """Partial delivery is not business completion. Earlier failed attempts do not veto a finished product."""
    if capture.circuit_open or manifest is None or not getattr(manifest, "is_file", lambda: False)():
        return False
    if capture.completed <= 0 and getattr(capture, "reused_count", 0) <= 0:
        return False
    if stage == "split":
        aligned_path = manifest.parent / "aligned_segments.json"
        transcript_path = manifest.parent / "transcript_segments.json"
        try:
            aligned_payload = json.loads(aligned_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if split_product_error(manifest, aligned_payload):
            return False
        if transcript_path.is_file() and Path(manifest).name == "split_segments.json":
            return adapter_split_guard(manifest, aligned_path, transcript_path)["passed"]
        return split_covers_aligned(manifest, aligned_payload)
    if stage == "translate":
        return translation_covers_split(manifest.parent / "split_segments.json", manifest)
    return capture.failures == 0


def _prepare_chain_input(directory, aligned_file, transcript_file, label):
    dest = _inside(directory / "chain-input", directory)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copyfile(aligned_file, dest / "aligned_segments.json")
    shutil.copyfile(transcript_file, dest / "transcript_segments.json")
    _write_json(dest / "lineage.json", {
        "source": label,
        "producer": CHAIN_PRODUCER,
        "note": "chain input label; not a quality claim",
    })
    return dest


def _call_bound(api, stage, mode, output, upstream):
    from host.contracts import ref

    names = {"split": "stage.split", "translate": "stage.translate"}
    return _child_report(api.call(ref(names[stage], "run_bound", VERSION), {
        "mode": mode,
        "repeat": "a",
        "output": output,
        "upstream": str(upstream),
    }))


def _public_child(child):
    report = child.get("report") if isinstance(child, dict) else None
    if not isinstance(report, dict):
        return {"outcome": "raised", "failure_class": "raised", "attempts": 0, "status": "failed"}
    return {
        "outcome": child.get("outcome") or "returned",
        "stage": report.get("stage"),
        "operation": report.get("operation"),
        "status": report.get("status"),
        "attempts": int(report.get("attempts") or 0),
        "failure_class": report.get("failure_class"),
        "source_label": report.get("source_label"),
        "read_baseline_fallback": report.get("read_baseline_fallback"),
        "claimed_this_round": report.get("claimed_this_round"),
        "chain_passed": False,
        "in_baseline_order": report.get("in_baseline_order"),
        "artifact": report.get("artifact") or "",
        "inputs": report.get("inputs") or {},
        "live": bool(report.get("live")),
    }


def bind_split_segment_ids(split_payload, translated_payload, segments):
    """Add the parent id old MiMo selects from the cue start. Never change times.

    qwen_asr.mimo_audio.segment_for_subtitle_id returns the first segment, in
    the given list order, whose window contains the cue start. It does not
    require the cue end to stay inside that window and it does not clamp.
    A start outside every segment is a real mismatch and is rejected.
    """
    if not isinstance(split_payload, dict) or not isinstance(translated_payload, dict):
        raise ValueError("split segment binding payload")
    parents = []
    for item in segments:
        if not isinstance(item, dict) or not item.get("segment_id"):
            raise ValueError("split segment binding parent")
        parents.append((
            str(item["segment_id"]),
            float(item["global_start_time"]) * 1000,
            float(item["global_end_time"]) * 1000,
        ))
    if not parents:
        raise ValueError("split segment binding has no parents")
    for key, item in split_payload.items():
        if not isinstance(item, dict):
            raise ValueError("split segment binding item")
        start = float(item["start_time"])
        end = float(item["end_time"])
        if not start < end:
            raise ValueError("split cue has no positive duration")
        covering = [row for row in parents if row[1] <= start < row[2]]
        if not covering:
            raise ValueError("split cue start is outside every segment")
        other = translated_payload.get(str(key))
        if not isinstance(other, dict):
            raise ValueError("translated cue missing for split binding")
        if float(other["start_time"]) != start or float(other["end_time"]) != end:
            raise ValueError("translated timing differs from split")
        item["segment_id"] = covering[0][0]
    return split_payload


def _write_handoff(directory, aligned_file, transcript_file, transcript_payload, split_file, translated_file, label, input_hashes, source_audio=""):
    handoff = directory / "handoff"
    if handoff.exists():
        shutil.rmtree(handoff)
    handoff.mkdir(parents=True)
    shutil.copyfile(aligned_file, handoff / "aligned_segments.json")
    shutil.copyfile(transcript_file, handoff / "transcript_segments.json")
    segments = _derive_segments(transcript_payload, source_audio)
    split_payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    translated_payload = json.loads(Path(translated_file).read_text(encoding="utf-8"))
    bind_split_segment_ids(split_payload, translated_payload, segments)
    _write_json(handoff / "segments.json", segments)
    _write_json(handoff / "split_segments.json", split_payload)
    _write_json(handoff / "translated_segments.json", translated_payload)
    lineage = {
        "source": "synthetic" if label == "synthetic" else "current-run",
        "producer": CHAIN_PRODUCER,
        "files": {name: _sha_file(handoff / name) for name in HANDOFF_NAMES},
        "inputs": {
            "aligned_segments.json": input_hashes["aligned_segments.json"],
            "transcript_segments.json": input_hashes["transcript_segments.json"],
        },
    }
    _write_json(handoff / "lineage.json", lineage)
    return lineage


def _reference_checks(explicit, executed, translate_called, provider_attempts, handoff_complete, calls, reused, pre):
    baseline = all(not item.get("read_baseline_fallback") for item in calls)
    return [
        {"id": "real-aligned-sha", "passed": bool(pre["aligned_ok"])},
        {"id": "real-transcript-sha", "passed": bool(pre["transcript_ok"])},
        {"id": "synthetic-selected", "passed": not explicit},
        {"id": "split-called", "passed": executed == [STAGE_BOUND_OPERATION["split"]]},
        {"id": "translate-not-called", "passed": translate_called is False},
        {"id": "attempts-zero", "passed": provider_attempts == {"split": 0, "translate": 0}},
        {"id": "handoff-incomplete", "passed": handoff_complete is False},
        {"id": "chain-passed-false", "passed": True},
        {"id": "migration-not-assessed", "passed": True},
        {"id": "text-not-called", "passed": STAGE_BOUND_OPERATION["text"] not in executed},
        {"id": "no-baseline-fallback", "passed": baseline},
        {"id": "previous-success-not-reused", "passed": reused != "accepted"},
    ]


def _split_labels_speaker(split_file):
    try:
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and any(
        isinstance(item, dict) and "speaker" in item for item in payload.values()
    )


def run_chain(arguments, api):
    if arguments["plan_id"] != CHAIN_ID:
        raise KeyError(arguments["plan_id"])
    flag = str(api.environment("REAL_LLM_STAGES_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    execute_real = _execute_real(api)
    snap = api.snapshot()["environment"]
    output = str(snap.get("output") or "")
    directory = _output_dir(output)
    pre = _precheck_real()
    aligned_raw = str(api.environment(CHAIN_ALIGNED_ENV, "") or "").strip()
    transcript_raw = str(api.environment(CHAIN_TRANSCRIPT_ENV, "") or "").strip()
    explicit = bool(aligned_raw or transcript_raw)
    failure_class = "none"
    detail = ""
    calls = []
    executed = []
    translate_called = False
    handoff_complete = False
    handoff_lineage = None
    reused = "absent"
    input_source = "synthetic" if not explicit else "explicit"
    provider_attempts = {"split": 0, "translate": 0}
    translate_reused = 0
    online_refused = False
    speaker_binding = None
    split_reused = False
    reused_split_calls = 0
    adapter_guard = None
    reuse_split_raw = str(api.environment(REUSE_SPLIT_ENV, "") or "").strip()
    source_audio = ""
    label = "synthetic"
    input_hashes = {}
    aligned_file = None
    transcript_file = None
    aligned_payload = None
    transcript_payload = None
    synthetic_aligned = _sha_file(CHAIN_SYNTHETIC / "aligned_segments.json")
    synthetic_transcript = _sha_file(CHAIN_SYNTHETIC / "transcript_segments.json")
    if explicit and (not aligned_raw or not transcript_raw):
        failure_class = "missing_upstream"
        detail = "aligned and transcript must both be set or both left unset"
    else:
        try:
            aligned_file = _chain_role_path(aligned_raw, "aligned_segments.json")
            transcript_file = _chain_role_path(transcript_raw, "transcript_segments.json")
        except FileNotFoundError as exc:
            failure_class = "missing_upstream"
            detail = str(exc)
        except ValueError as exc:
            failure_class = "invalid_upstream"
            detail = str(exc)
    known_real = aligned_file is not None and transcript_file is not None and (_known_real(aligned_file) or _known_real(transcript_file))
    if known_real and not execute_real:
        online_refused = True
        failure_class = "real_input_not_executed"
        detail = "known real aligned/transcript files are hash-checked only unless REAL_CHAIN_LLM_EXECUTE_REAL is set"
        aligned_payload = None
    elif failure_class == "none":
        input_hashes = {
            "aligned_segments.json": _sha_file(aligned_file),
            "transcript_segments.json": _sha_file(transcript_file),
        }
        reused = consume_previous_handoff(directory, input_hashes)
        aligned_payload, error = _read_json_value(aligned_file)
        if error[0]:
            failure_class, detail = error
        else:
            transcript_payload, error = _read_json_value(transcript_file)
            if error[0]:
                failure_class, detail = error
            else:
                failure_class, detail = _chain_dependency_error(aligned_payload, transcript_payload)
                if not failure_class:
                    failure_class = "none"
    if failure_class == "none" and not online_refused and execute_real and granted:
        try:
            speaker_binding = _speaker_binding(api)
            source_audio = speaker_binding["audio"]
        except ValueError as exc:
            failure_class = "speaker_resource_missing"
            detail = str(exc)
    split_file = None
    if failure_class == "none" and not online_refused and reuse_split_raw:
        if not (execute_real and granted):
            failure_class = "invalid_upstream"
            detail = "split reuse requires the explicit real switch and an online grant"
        else:
            try:
                accepted = _accept_reused_split(
                    reuse_split_raw,
                    str(api.environment(REUSE_CALLS_ENV, "") or ""),
                    str(api.environment(REUSE_SHA_ENV, "") or ""),
                    aligned_file,
                    transcript_file,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failure_class = "incomplete_product"
                detail = str(exc)[:300]
            else:
                split_reused = True
                reused_split_calls = accepted["calls"]
                adapter_guard = accepted["guard"]
                split_file = accepted["split_file"]
                label = _chain_label(aligned_file.parent, explicit)
                _prepare_chain_input(directory, aligned_file, transcript_file, label)
    elif failure_class == "none" and not online_refused:
        label = _chain_label(aligned_file.parent, explicit)
        prepared = _prepare_chain_input(directory, aligned_file, transcript_file, label)
        mode = "online" if granted else "offline"
        split_child = _call_bound(api, "split", mode, output, prepared)
        public_split = _public_child(split_child)
        calls.append(public_split)
        executed.append(STAGE_BOUND_OPERATION["split"])
        provider_attempts["split"] = public_split["attempts"]
        if mode == "online":
            split_report = split_child.get("report") if isinstance(split_child, dict) else {}
            artifact = str((split_report or {}).get("artifact") or "")
            candidate = directory / artifact if artifact else directory / "missing-split.json"
            product_error = ""
            if public_split["outcome"] != "returned":
                product_error = "split call rejected"
            elif split_report.get("business_complete") is not True:
                product_error = "split business incomplete"
            elif Path(artifact).name != "split_segments.json":
                product_error = "split artifact missing"
            else:
                product_error = split_product_error(candidate, aligned_payload)
                guard = adapter_split_guard(
                    candidate,
                    candidate.parent / "aligned_segments.json",
                    candidate.parent / "transcript_segments.json",
                )
                adapter_guard = guard
                if not guard["passed"]:
                    product_error = guard["error"] or "split adapter guard rejected"
                if not product_error and speaker_binding and not _split_labels_speaker(candidate):
                    product_error = "split dropped speaker labels"
            if product_error:
                if public_split["status"] in {"partial", "passed"}:
                    failure_class = "incomplete_product"
                else:
                    failure_class = str(public_split["failure_class"] or "split_failed")
                detail = product_error
            else:
                lineage_src = prepared / "lineage.json"
                if lineage_src.is_file():
                    shutil.copyfile(lineage_src, candidate.parent / "lineage.json")
                split_file = candidate
    if failure_class == "none" and split_file is not None:
        translate_called = True
        translate_child = _call_bound(api, "translate", "online", output, split_file.parent)
        public_translate = _public_child(translate_child)
        calls.append(public_translate)
        executed.append(STAGE_BOUND_OPERATION["translate"])
        provider_attempts["translate"] = public_translate["attempts"]
        translate_report = translate_child.get("report") if isinstance(translate_child, dict) else {}
        translate_reused = int((translate_report or {}).get("reused_responses") or 0)
        translated_artifact = str((translate_report or {}).get("artifact") or "")
        translated_file = directory / translated_artifact if translated_artifact else directory / "missing-translated.json"
        split_sha = _sha_file(split_file)
        read_split = (public_translate.get("inputs") or {}).get("split_segments.json")
        translate_error = ""
        if public_translate["outcome"] != "returned":
            translate_error = "translate call rejected"
        elif translate_report.get("business_complete") is not True:
            translate_error = "translate business incomplete"
        elif Path(translated_artifact).name != "translated_segments.json" or not translated_file.is_file():
            translate_error = "translate artifact missing"
        elif read_split != split_sha:
            translate_error = "translate did not read this split product"
        elif not translation_covers_split(split_file, translated_file):
            translate_error = "translate product does not cover this split"
        if translate_error:
            if public_translate["status"] in {"partial", "passed"}:
                failure_class = "incomplete_product"
            else:
                failure_class = str(public_translate["failure_class"] or "translate_failed")
            detail = translate_error
        else:
            handoff_lineage = _write_handoff(
                directory, aligned_file, transcript_file, transcript_payload,
                split_file, translated_file, label, input_hashes, source_audio,
            )
            handoff_complete = True
    default_offline = (not explicit) and (not online_refused) and failure_class == "none" and not granted
    if default_offline:
        checks = _reference_checks(explicit, executed, translate_called, provider_attempts, handoff_complete, calls, reused, pre)
    elif handoff_complete:
        checks = [
            {"id": "split-then-translate", "passed": executed == [STAGE_BOUND_OPERATION["split"], STAGE_BOUND_OPERATION["translate"]] or (split_reused and executed == [STAGE_BOUND_OPERATION["translate"]] and provider_attempts["split"] == 0)},
            {"id": "handoff-complete", "passed": True},
            {"id": "chain-passed-false", "passed": True},
            {"id": "text-not-called", "passed": True},
            {"id": "real-service-not-claimed", "passed": True},
        ]
    else:
        checks = [
            {"id": "translate-stopped", "passed": (not translate_called) or failure_class != "none"},
            {"id": "no-success-handoff", "passed": handoff_complete is False},
            {"id": "chain-passed-false", "passed": True},
            {"id": "text-not-called", "passed": STAGE_BOUND_OPERATION["text"] not in executed},
        ]
    failed = sum(1 for item in checks if not item["passed"])
    if default_offline and failed == 0:
        status = "passed"
    elif handoff_complete and failed == 0:
        status = "partial"
    else:
        status = "failed"
    summary = {
        "plan_id": CHAIN_ID,
        "status": status,
        "passed": len(checks) - failed,
        "failed": failed,
        "case_count": len(checks),
        "delivery_status": "partial",
        "chain_passed": False,
        "chain_pass": False,
        "episode_pass": False,
        "migration_equivalence": "NOT_ASSESSED",
        "coverage": "explicit_input_only",
        "mode": "offline" if not granted or online_refused else "online",
        "online_granted": bool(granted),
        "online_refused": online_refused,
        "input_source": input_source,
        "source_label": label if failure_class == "none" or calls else "",
        "failure_class": failure_class,
        "detail": detail[:300],
        "executed_operations": executed,
        "translate_called": translate_called,
        "text_proofread_called": False,
        "provider_attempts": provider_attempts,
        "attempts_semantics": ATTEMPTS_SEMANTICS,
        "real_service_executed": False,
        "models_run": False,
        "gpu_used": False,
        "mimo": "not_called",
        "handoff_complete": handoff_complete,
        "handoff_lineage": handoff_lineage,
        "reused_previous_success": False,
        "reuse_decision": reused,
        "read_baseline_fallback": any(item.get("read_baseline_fallback") for item in calls),
        "claimed_this_round": False,
        "aligned_precheck_sha256": pre["aligned_sha256"],
        "transcript_precheck_sha256": pre["transcript_sha256"],
        "synthetic_aligned_sha256": synthetic_aligned,
        "synthetic_transcript_sha256": synthetic_transcript,
        "input_hashes": input_hashes,
        "checks": checks,
        "calls": calls,
    }
    service_executed = bool(
        execute_real and granted and not online_refused
        and (provider_attempts["split"] + provider_attempts["translate"]) > 0
    )
    if not default_offline:
        summary["real_service_executed"] = service_executed
        summary["live"] = bool(service_executed and any(item.get("live") for item in calls))
        summary["split_reused"] = split_reused
        summary["reused_split_calls"] = reused_split_calls
        summary["split_requests_added"] = provider_attempts["split"]
        summary["translate_responses_reused"] = translate_reused
        summary["translate_requests_added"] = provider_attempts["translate"]
        summary["gpu_used_measured"] = bool(speaker_binding and speaker_binding.get("public"))
        if adapter_guard is not None:
            summary["adapter_split_guard"] = {
                "passed": adapter_guard["passed"],
                "content": adapter_guard["content"],
                "short_response": adapter_guard["short_response"],
                "source_key_len": adapter_guard["source_key_len"],
                "result_key_len": adapter_guard["result_key_len"],
                "source_key_sha256": adapter_guard["source_key_sha256"],
                "result_key_sha256": adapter_guard["result_key_sha256"],
                "legacy_aligned_text_cover": adapter_guard["legacy_aligned_text_cover"],
            }
        if speaker_binding and speaker_binding.get("public"):
            summary["speaker"] = speaker_binding["public"]
            summary["gpu_used"] = bool(speaker_binding["public"].get("gpu_used"))
    evidence = directory / "evidence"
    _write_json(evidence / "report.json", summary)
    _write_json(evidence / "cases.json", {
        "mode": summary["mode"],
        "input_source": input_source,
        "aligned_precheck_sha256": pre["aligned_sha256"],
        "transcript_precheck_sha256": pre["transcript_sha256"],
        "synthetic_aligned_sha256": synthetic_aligned,
        "synthetic_transcript_sha256": synthetic_transcript,
        "executed_operations": executed,
        "translate_called": translate_called,
        "provider_attempts": provider_attempts,
        "handoff_complete": handoff_complete,
        "chain_passed": False,
        "migration_equivalence": "NOT_ASSESSED",
        "real_service_executed": False if default_offline else summary["real_service_executed"],
        "text_proofread_called": False,
        "read_baseline_fallback": summary["read_baseline_fallback"],
        "reused_previous_success": False,
        "checks": checks,
    })
    _write_json(evidence / "criteria.json", CHAIN_CRITERIA)
    return summary


class Implementation:
    def __init__(self, settings):
        self.settings = settings or {}

    def invoke(self, operation, arguments, api):
        from host.contracts import ref

        role = self.settings.get("role")
        if operation == "run" and role in STAGE_ORDER:
            return {"report": run_stage(role, arguments, api)}, []
        if operation == "run_bound" and role in STAGE_ORDER:
            return {"report": run_stage_bound(role, arguments, api)}, []
        if operation == "run_chain" and role == "suite":
            return {"report": run_chain(arguments, api)}, []
        if operation != "run" or role != "suite":
            raise KeyError(operation)
        upstream = str(api.environment(BIND_ENV, "") or "").strip()
        if upstream:
            return {"report": run_bind_suite(upstream, api)}, []
        flag = str(api.environment("REAL_LLM_STAGES_ONLINE", "") or "")
        spec = load_spec()
        granted = online_granted(spec, flag)
        mode = "online" if granted else "offline"
        snap = api.snapshot()["environment"]
        output = str(snap.get("output") or "")
        repeats = ("a", "b") if mode == "online" else ("a",)
        calls = []
        stopped = set()
        for stage in STAGE_ORDER:
            target = ref(
                {"split": "stage.split", "translate": "stage.translate", "text": "stage.text_proofread"}[stage],
                "run",
                VERSION,
            )
            for repeat in repeats:
                if stage in stopped:
                    report = _blank(stage, mode, "skipped_after_circuit", "circuit", "previous repeat stopped the stage")
                    report["repeat"] = repeat
                    calls.append({"outcome": "returned", "report": report})
                    continue
                child = _child_report(api.call(target, {"mode": mode, "repeat": repeat, "output": output}))
                calls.append(child)
                report = child.get("report") or {}
                if report.get("circuit_open") or report.get("status") in {"failed", "blocked_credentials", "blocked_declaration", "blocked_model"}:
                    stopped.add(stage)
        returned = [item for item in calls if item["outcome"] == "returned"]
        reports = [item["report"] for item in returned if isinstance(item.get("report"), dict)]
        failed = len(calls) - len(returned)
        failed += sum(1 for report in reports if report.get("status") not in {"offline", "passed", "skipped_after_circuit"})
        criteria = build_criteria(output, reports) if mode == "online" else {
            "verdict": "NOT_ASSESSED",
            "reason": "offline replay has no provider text",
            "retroactive_old_not_passed": False,
        }
        live = any(report.get("live") for report in reports)
        if mode == "offline" and failed == 0:
            status = "passed"
        elif failed == 0 and reports and all(report.get("status") in {"passed", "skipped_after_circuit"} for report in reports):
            status = "passed"
        elif any(report.get("status") == "passed" for report in reports):
            status = "partial"
        else:
            status = "failed" if mode == "online" else "failed"
        if mode == "offline":
            status = "passed" if failed == 0 else "failed"
        summary = {
            "plan_id": TICKET,
            "status": status,
            "passed": len(calls) - failed,
            "failed": failed,
            "case_count": len(calls),
            "live": live,
            "episode_pass": False,
            "chain_passed": False,
            "models_run": False,
            "mimo": "blocked_auth",
            "executed_operations": [STAGE_OPERATION[stage] for stage in STAGE_ORDER],
            "argv_returned": False,
            "mode": mode,
            "online_granted": granted,
            "declared_online_network": (spec.get("permissions") or {}).get("online", {}).get("network") is True,
            "host_network_flag": snap.get("network"),
            "host_plan": snap.get("plan"),
            "host_isolation": "not enforced; host environment.network stays false and is reported separately",
            "model_sent": RELAY_GEMINI_MODEL,
            "baseline_model": BASELINE_MODEL,
            "alias_equivalence": False,
            "stage_parallelism": "host invoke is single-threaded, so stage calls are sequential; batches inside a stage use the legacy pool",
            "calls": calls,
            "criteria": criteria,
        }
        if output and (ROOT / output).is_dir():
            directory = _output_dir(output)
            _write_json(directory / "evidence" / "report.json", summary)
            if mode == "offline":
                update_progress(output, "offline", "a", 0, 0, "none", "offline", total_known=True)
            _write_json(directory / "evidence" / "criteria.json", criteria)
        return {"report": summary}, []
