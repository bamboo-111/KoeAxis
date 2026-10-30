"""Token Plan MiMo audio review through the host, plus a limited protocol comparison.

The root calls stage.audio_review/run and provider.audio_llm/compare. Those
operations call the read-only qwen_asr audio-review functions with an explicit
base URL. They do not call qwen_asr.cli.main and they do not return argv.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

OLD = Path(__file__).resolve().parents[3] / "compat" / "legacy"
BASELINE = OLD / "workspaces" / "radio-54-togeradi-20261018"
GLOSSARY = OLD / "samples" / "翻译对照.xlsx"
ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments/specifications/real-mimo.json"
ARCHIVE = ROOT / "runs/experiments/REAL-MIMO-001"
VERSION = "0.1.0-real-mimo"
TICKET = "REAL-MIMO-001"
MODEL = "mimo-v2.6-pro"
OPENAI_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
ANTHROPIC_BASE = "https://token-plan-cn.xiaomimimo.com/anthropic"
WORKERS = 6
BATCH_SIZE = 3
MAX_TOKENS = 4096
TEMPERATURE = 0.0
TIMEOUT_S = 240.0
DETERMINISTIC = frozenset({
    "api_request_rejected",
    "audio_unsupported",
    "auth",
    "forbidden",
})
STAGE_INPUT_NAMES = {
    "manifests/segments.json": "segments.json",
    "manifests/translated_segments.json": "translated_segments.json",
    "manifests/split_segments.json": "split_segments.json",
}
BOUND_OPERATION = "stage.audio_review/run_bound"
BOUND_FILES = ("segments.json", "translated_segments.json", "split_segments.json")
BIND_ENV = "REAL_CHAIN_BIND_UPSTREAM"
REUSE_CALLS_ENV = "REAL_CHAIN_MIMO_REUSE_CALLS"


class CircuitOpen(BaseException):
    """Stops one protocol immediately. BaseException so legacy except Exception cannot retry it."""

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
        r"(?i)(x-api-key|authorization|api[_-]?key)(['\"=: ]+)\S+",
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
    elif path.rstrip("/").endswith("/anthropic"):
        path_class = "anthropic-base"
    elif path:
        path_class = "other"
    else:
        path_class = "empty"
    return {"scheme": parsed.scheme or "absent", "host_kind": kind, "path_class": path_class}


def classify_message(text, status=None):
    raw = text or ""
    lowered = raw.lower()
    if status is None:
        match = re.search(r"HTTP(?:/1\.[01])?\s+(\d{3})", raw)
        if match is None:
            match = re.search(r"(?i)error code:\s*(\d{3})", raw)
        if match:
            status = int(match.group(1))
    if status in (401, 403) or "invalid api key" in lowered or "authentication" in lowered:
        return {"class": "auth" if status != 403 else "forbidden", "http_status": status or 401, "layer": "http_api"}
    audio_tokens = ("input_audio", "audio content", "unsupported content", "media_type", "invalid content type", "content block")
    if any(token in lowered for token in audio_tokens) and (status in (400, 404, 415, 422) or "audio" in lowered):
        return {"class": "audio_unsupported", "http_status": status or 400, "layer": "protocol"}
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
    """Sockets require the explicit flag and an affirmative spec grant.

    The host environment.network field is not this grant.
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
    endpoints = str(online.get("endpoints") or "")
    if "Token Plan" not in endpoints or "OpenAI" not in endpoints or "Anthropic" not in endpoints:
        return False
    return True


def _stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

class _ReusedChoice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class _ReusedResponse:
    def __init__(self, content):
        self.choices = [_ReusedChoice(content)]
        self.usage = None


def load_mimo_reuse(root):
    """Index saved calls by exact public-prompt SHA. Ambiguous or failed rows are excluded."""
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError("mimo reuse calls missing")
    grouped = {}
    for path in sorted(directory.rglob("*.json")):
        if path.parent.name == "reused":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("ok") is not True or data.get("failure_class") not in (None, "none"):
            continue
        prompt = data.get("prompt")
        response = data.get("response")
        if not isinstance(prompt, str) or not prompt or not isinstance(response, str) or not response.strip():
            continue
        if data.get("model") != MODEL:
            continue
        digest = _sha_text(prompt)
        if digest != data.get("prompt_sha256"):
            continue
        response_sha = _sha_text(response)
        bucket = grouped.setdefault(digest, {"response": response, "response_sha": response_sha, "ambiguous": False})
        if bucket["response_sha"] != response_sha:
            bucket["ambiguous"] = True
    return {digest: item["response"] for digest, item in grouped.items() if not item["ambiguous"]}


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


def _inside(path, root):
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise RuntimeError("refusing to write outside the run directory")
    return resolved

def call_artifact_path(output, stage, phase, index):
    """Phase-scoped call record. first-batch and full must not share a filename."""
    directory = _output_dir(output)
    phase_name = str(phase or "unphased")
    stage_name = str(stage or "unspecified")
    if phase_name in {"", ".", ".."} or "/" in phase_name or "\\" in phase_name:
        raise RuntimeError("refusing call phase")
    if stage_name in {"", ".", ".."} or "/" in stage_name or "\\" in stage_name:
        raise RuntimeError("refusing call stage")
    return _inside(directory / "calls" / stage_name / phase_name / f"{int(index):04d}.json", directory)


def update_progress(output, stage, phase, completed, total, failure_class, status, total_known=False, archive=False):
    payload = {
        "ticket": TICKET,
        "stage": stage,
        "phase": phase,
        "completed": int(completed),
        "total": int(total),
        "total_known": bool(total_known),
        "failure_class": failure_class or "none",
        "status": status,
        "updated": _stamp(),
    }
    if output:
        _write_json(_output_dir(output) / "progress.json", payload)
    if archive:
        _write_json(ARCHIVE / "progress.json", payload)
    return payload


def _blank(operation, mode, status, failure_class, detail=""):
    return {
        "operation": operation,
        "mode": mode,
        "status": status,
        "live": False,
        "attempts": 0,
        "failures": 0,
        "failure_class": failure_class,
        "circuit_open": False,
        "model_sent": MODEL,
        "model_replaced": False,
        "episode_pass": False,
        "chain_pass": False,
        "migration_equivalence": "NOT_ASSESSED",
        "argv": None,
        "detail": detail,
        "endpoint": endpoint_view(""),
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
        "elapsed_s": 0.0,
        "calls": [],
        "host_isolation": "host does not enforce network isolation; environment.network is a report field",
    }


def _credentials(api):
    key = str(api.environment("MIMO_API_KEY", "") or "").strip()
    base_state = "set" if str(api.environment("MIMO_BASE_URL", "") or "").strip() else "unset"
    return key, base_state


def _legacy():
    if str(OLD) not in sys.path:
        sys.path.insert(0, str(OLD))
    import qwen_asr.mimo_proofread as proof
    import qwen_asr.mimo_requests as requests
    return {"proof": proof, "requests": requests}


def _copy_inputs(dest):
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for source_name, flat_name in STAGE_INPUT_NAMES.items():
        src = BASELINE / source_name
        if not src.is_file():
            raise FileNotFoundError(source_name)
        target = dest / flat_name
        shutil.copyfile(src, target)
        copied[flat_name] = _sha_bytes(target.read_bytes())
    return copied


def _resolve_upstream(raw):
    text = str(raw or "").strip()
    if not text or ".." in text.replace("\\", "/").split("/"):
        raise ValueError("upstream path is empty or unsafe")
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path.is_file():
        if path.name not in BOUND_FILES:
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
        raise ValueError("unsupported lineage source")
    return label


def _validate_audio_bound(source):
    parsed = {name: _load_bound_json(source, name) for name in BOUND_FILES}
    label = _source_label(source)
    segments = parsed["segments.json"]
    translated = parsed["translated_segments.json"]
    split = parsed["split_segments.json"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments.json must be a non-empty list")
    seen = []
    for item in segments:
        if not isinstance(item, dict) or not str(item.get("segment_id") or ""):
            raise ValueError("segments.json item missing segment_id")
        seen.append(str(item["segment_id"]))
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate segment_id in segments.json")
    if not isinstance(translated, dict) or not translated:
        raise ValueError("translated_segments.json must be a non-empty object")
    if not isinstance(split, dict) or not split:
        raise ValueError("split_segments.json must be a non-empty object")
    split_keys = {str(key) for key in split}
    translated_keys = {str(key) for key in translated}
    missing = sorted(split_keys - translated_keys, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    blank = sorted(
        (
            key for key in split_keys & translated_keys
            if not isinstance(translated.get(key), dict)
            or not str(translated[key].get("translated_subtitle") or "").strip()
        ),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if missing or blank:
        details = []
        if missing:
            details.append("missing_keys=" + ",".join(missing[:10]))
        if blank:
            details.append("blank_translation_keys=" + ",".join(blank[:10]))
        raise ValueError("id mismatch: translated_segments.json is incomplete for split_segments.json; " + "; ".join(details))
    uncovered = []
    for key, item in translated.items():
        if not isinstance(item, dict):
            raise ValueError("translated_segments.json %s is not an object" % key)
        try:
            start = int(item.get("start_time", 0))
            end = int(item.get("end_time", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("translated cue has invalid timing") from exc
        covered = False
        for segment in segments:
            try:
                seg_start = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
                seg_end = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
            except (TypeError, ValueError) as exc:
                raise ValueError("segments.json item has invalid timing") from exc
            if end > seg_start and start < seg_end:
                covered = True
                break
        if not covered:
            uncovered.append(str(key))
    if uncovered:
        raise ValueError("id dependency: translated cue not covered by segments.json: " + ",".join(uncovered))
    return label, parsed


def _copy_bound_inputs(dest, source):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in BOUND_FILES:
        src = source / name
        target = dest / name
        shutil.copyfile(src, target)
        copied[name] = _sha_bytes(target.read_bytes())
    return copied


def _bound_fields(source, label, copied):
    return {
        "operation": BOUND_OPERATION,
        "lineage": "supplied",
        "source_label": label,
        "upstream": str(source),
        "inputs": copied,
        "read_baseline_fallback": False,
        "explicit_path_is_baseline_tree": source.resolve() == (BASELINE / "manifests").resolve(),
        "claimed_this_round": False,
        "chain_pass": False,
        "chain_passed": False,
        "migration_equivalence": "NOT_ASSESSED",
        "coverage": "explicit_input_only",
        "required_files": list(BOUND_FILES),
        "optional_files": ["lineage.json", "audio path referenced by segments.json"],
        "audio_copied": False,
        "in_baseline_order": True,
    }


def _scrub_tree(root, secrets):
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".srt", ".log", ".md"}:
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            continue
        cleaned = scrub(text, secrets)
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8", newline="\n")


class _Capture:
    def __init__(self, output, stage, phase, secrets, archive):
        self.output = output
        self.stage = stage
        self.phase = phase
        self.secrets = [item for item in secrets if item]
        self.archive = archive
        self.calls = []
        self.completed = 0
        self.index_base = 0
        if (_output_dir(output) / "chain-state.json").is_file():
            calls_dir = _output_dir(output) / "calls" / stage / phase
            self.index_base = max((int(p.stem) for p in calls_dir.glob("*.json") if p.stem.isdigit()), default=0)
        self.failures = 0
        self.failure_class = "none"
        self.circuit_open = False
        self.transient_run = 0
        self.total = 0
        self.total_known = False
        self.tokens_in = 0
        self.tokens_out = 0
        self.reused = 0
        self.reuse = None
        self.reuse_loaded = False
        self.lock = threading.Lock()
        self._original = None
        self._module = None

    def set_reuse(self, index):
        self.reuse = index if isinstance(index, dict) else {}
        self.reuse_loaded = True

    def set_total(self, total):
        with self.lock:
            self.total = int(total)
            self.total_known = True
            completed = self.completed
            failure = self.failure_class
        update_progress(
            self.output, self.stage, self.phase, completed, total, failure, "running",
            total_known=True, archive=self.archive,
        )

    def install(self, module):
        self._module = module
        self._original = module.chat_completion_create
        module.chat_completion_create = self._wrap(module.chat_completion_create)

    def restore(self):
        if self._module is not None and self._original is not None:
            self._module.chat_completion_create = self._original
        self._module = None
        self._original = None

    def _wrap(self, original):
        def wrapped(**kwargs):
            if self.circuit_open:
                raise CircuitOpen(self.failure_class or "circuit")
            started = time.perf_counter()
            model = str(getattr(kwargs.get("config"), "model", "") or "")
            base = str(getattr(kwargs.get("config"), "base_url", "") or "")
            view = endpoint_view(base)
            prompt = _public_prompt(kwargs.get("messages"), self.secrets)
            cached = None if self.reuse is None or model != MODEL else self.reuse.get(_sha_text(prompt))
            if isinstance(cached, str) and cached:
                self._note_reuse(model, view, prompt, cached)
                return _ReusedResponse(cached)
            request_started_unix_s = time.time()
            request_started = time.perf_counter()
            response = None
            error = None
            try:
                response = original(**kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                finished = time.time()
                duration = time.perf_counter() - request_started
                kind = classify_message(scrub(str(error), self.secrets), getattr(error, "status_code", None)) if error is not None else {"class": "none"}
                usage_obj = getattr(response, "usage", None)
                usage_known = usage_obj is not None and all(
                    isinstance(getattr(usage_obj, key, None), int) and not isinstance(getattr(usage_obj, key, None), bool)
                    for key in ("prompt_tokens", "completion_tokens")
                )
                self._finish(error is None, kind["class"], model, view, prompt,
                             scrub(_response_text(response), self.secrets) if error is None else "", started,
                             _usage_counts(response) if error is None else {"prompt_tokens": 0, "completion_tokens": 0},
                             timing=(request_started_unix_s, finished, duration), usage_known=usage_known)
                if error is not None and isinstance(error, Exception):
                    if kind["class"] in DETERMINISTIC:
                        self.circuit_open = True
                        raise CircuitOpen(kind["class"]) from None
                    self.transient_run += 1
                    if self.transient_run >= 3:
                        self.circuit_open = True
                        self.failure_class = kind["class"]
                        raise CircuitOpen(kind["class"]) from None
            self.transient_run = 0
            return response
        return wrapped

    def _finish(self, ok, failure_class, model, view, prompt, response, started, usage, timing=None, usage_known=False):
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
                "endpoint": view,
            }
            self.calls.append({key: value for key, value in record.items() if key != "endpoint"})
            completed = self.completed
            failure = self.failure_class
            total = self.total if self.total_known else completed
        _write_json(call_artifact_path(self.output, self.stage, self.phase, self.index_base + index), {
            "index": self.index_base + index,
            "stage": self.stage,
            "phase": self.phase,
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
            "network": True,
            "reused": False,
            "prompt_sha256": record["prompt_sha256"],
            "response_sha256": record["response_sha256"],
            "usage": usage,
            "prompt": prompt,
            "response": response,
        })
        update_progress(
            self.output, self.stage, self.phase, completed, total, failure, "running",
            total_known=self.total_known, archive=self.archive,
        )

    def _note_reuse(self, model, view, prompt, response):
        with self.lock:
            self.reused += 1
            index = self.reused
        directory = _output_dir(self.output) / "calls" / self.stage / self.phase / "reused"
        _write_json(directory / ("%04d.json" % index), {
            "index": index,
            "stage": self.stage,
            "phase": self.phase,
            "reused": True,
            "network": False,
            "ok": True,
            "failure_class": "none",
            "model": model,
            "endpoint": view,
            "elapsed_s": 0.0,
            "prompt_sha256": _sha_text(prompt),
            "response_sha256": _sha_text(response),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "request_started_unix_s": None,
            "request_finished_unix_s": None,
            "request_duration_s": None,
            "usage_known": False,
            "sdk_retry_count": "unknown",
            "prompt": prompt,
            "response": response,
        })


def _public_prompt(messages, secrets):
    chunks = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text") or ""))
            elif isinstance(part, dict) and part.get("type") == "input_audio":
                audio = part.get("input_audio") or {}
                data = str(audio.get("data") or "")
                chunks.append("[audio sha256=%s bytes=%s format=%s]" % (
                    _sha_text(data), len(data), audio.get("format") or "",
                ))
    return scrub("\n".join(chunks), secrets)


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


def _config(modules, key):
    proof = modules["proof"]
    extra = proof._parse_extra_body('{"thinking":{"type":"disabled"}}')
    return proof.MiMoConfig(
        base_url=OPENAI_BASE,
        api_key=key,
        model=MODEL,
        timeout=TIMEOUT_S,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        disable_thinking=True,
        extra_body=extra,
        compact_output=False,
    )


def _client(modules, key):
    from openai import OpenAI
    proof = modules["proof"]
    return OpenAI(
        base_url=proof._normalize_base_url(OPENAI_BASE),
        api_key=key,
        timeout=TIMEOUT_S,
        max_retries=0,
    )


def _namespace(**values):
    class Namespace:
        pass
    item = Namespace()
    for key, value in values.items():
        setattr(item, key, value)
    return item


def _stage_args(resume):
    return _namespace(
        resume=resume,
        segment_limit=0,
        stage1_confidence_threshold=0.75,
        stage1_apply_threshold=0.8,
        stage2_apply_threshold=0.9,
        workers=WORKERS,
        nearby_audio_workers=WORKERS,
        nearby_batch_size=BATCH_SIZE,
        nearby_batch_max_gap_s=8.0,
        nearby_padding_s=1.5,
        nearby_context_subtitles=1,
        audio_review_scope="suspects",
        max_glossary_entries=80,
        max_retries=2,
        retry_base_delay=2.0,
        retry_max_delay=8.0,
        keep_raw=False,
        diagnostic_all=False,
    )


def _prepare_stage(output):
    directory = _output_dir(output)
    work = _inside(directory / "stages" / "audio-openai", directory)
    if not (work / "translated_segments.json").is_file():
        if work.exists():
            shutil.rmtree(work)
        _copy_inputs(work)
    return work


def _content_glossary(api):
    selected = str(api.environment("OPT54_CONTENT_CONTEXT", "") or "").strip()
    if not selected:
        return []
    programs = json.loads((ROOT / "experiments/references/program-content/v1.json").read_text(encoding="utf-8"))["programs"]
    if selected not in programs:
        raise ValueError("unknown OPT54_CONTENT_CONTEXT")
    binding = programs[selected]
    raw = (ROOT / binding["terms"]).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    note = binding["glossary_note_prefix"] + hashlib.sha256(raw).hexdigest()[:16]
    return [{"group": binding["glossary_group"], "source": e["term"],
             "target": binding["glossary_target"], "note": note}
            for e in data["entries"] if not e["ambiguous"] and not e["force_replace"]]


def _load_stage_inputs(modules, work):
    proof = modules["proof"]
    segments, translated = proof._load_pipeline_inputs(work)
    glossary = proof._load_glossary(GLOSSARY if GLOSSARY.is_file() else None)
    return segments, translated, glossary


def _review_batches(modules, segments, translated, args):
    proof = modules["proof"]
    stage1 = proof._build_manifest_suspect_report(
        translated, confidence_threshold=args.stage1_confidence_threshold,
    )
    suspect_ids = proof._collect_stage1_suspects(stage1)
    return stage1, suspect_ids, proof._build_nearby_audio_batches(
        subtitle_ids=suspect_ids,
        segments=segments,
        translated=translated,
        context_subtitles=args.nearby_context_subtitles,
        batch_size=args.nearby_batch_size,
        max_gap_s=args.nearby_batch_max_gap_s,
    )


def _run_one_batch(modules, work, client, config, args, segments, translated, glossary, stage1, batch):
    proof = modules["proof"]
    output_dir = work / "experiments" / "mimo-proofread"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "mimo_proofread_segments.json"
    report_path = output_dir / "mimo_proofread_report.json"
    stage1_path = output_dir / "mimo_stage1_text_suspect_report.json"
    stage2_path = output_dir / "mimo_stage2_nearby_audio_report.json"
    srt_path = output_dir / "subtitles.mimo-proofread.srt"
    branch = proof._load_existing_branch(manifest_path, translated)
    stage2 = proof._load_existing_report(stage2_path)
    results = proof._process_stage2_nearby_audio_batch_task(
        target_ids=list(batch),
        client=client,
        config=config,
        segments=segments,
        translated=translated,
        glossary=glossary,
        clips_dir=output_dir / "nearby-audio-clips",
        context_subtitles=args.nearby_context_subtitles,
        padding_s=args.nearby_padding_s,
        max_glossary_entries=args.max_glossary_entries,
        max_retries=args.max_retries,
        base_delay=args.retry_base_delay,
        max_delay=args.retry_max_delay,
        keep_raw=False,
        apply_confidence_threshold=args.stage2_apply_threshold,
    )
    for result in results:
        report_id = str(result.report_item.get("id", result.segment_id))
        stage2 = proof._replace_report_item(stage2, report_id, result.report_item, key="id")
        if result.report_item.get("status") == "completed":
            proof._apply_branch_updates(branch, result.updates, source="mimo-nearby-audio")
    proof._write_two_stage_outputs(
        manifest_path, report_path, stage1_path, stage2_path, srt_path,
        branch, stage1, stage2, started=time.monotonic(), translated=translated,
    )
    return results


def _execute_stage(phase, output, key, source_dir=None, bound=False, source_label="", context_glossary=None):
    started = time.perf_counter()
    modules = _legacy()
    copied = {}
    archive = not bound

    def _finish(report):
        if not bound:
            return report
        if report.get("status") == "passed":
            report["status"] = "partial"
        report.update(_bound_fields(source_dir, source_label, copied))
        if report.get("status") == "partial":
            detail = str(report.get("detail") or "").strip()
            if "explicit upstream consumed" not in detail:
                report["detail"] = (detail + "; " if detail else "") + "explicit upstream consumed; not a chain pass"
        return report

    capture = _Capture(output, "audio-openai", phase, [key], archive=archive)
    reuse_raw = str(os.environ.get(REUSE_CALLS_ENV, "") or "").strip()
    if reuse_raw:
        capture.set_reuse(load_mimo_reuse(reuse_raw))
    log_filter = _RedactLog([key])
    logging.getLogger().addFilter(log_filter)
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)
    work = None
    try:
        capture.install(modules["requests"])
        if bound:
            directory = _output_dir(output)
            work = _inside(directory / "stages" / "audio-openai", directory)
            chain_state = directory / "chain-state.json"
            if chain_state.is_file() and work.is_dir():
                copied.update({name: _sha_bytes((work / name).read_bytes()) for name in BOUND_FILES})
                expected = {name: _sha_bytes((source_dir / name).read_bytes()) for name in BOUND_FILES}
                if copied != expected:
                    raise ValueError("bound work input mismatch")
            else:
                copied.update(_copy_bound_inputs(work, source_dir))
            _write_json(work / "origin.json", {
                "lineage": "supplied",
                "source_label": source_label,
                "upstream": str(source_dir),
                "inputs": copied,
                "read_baseline_fallback": False,
                "claimed_this_round": False,
                "audio_copied": False,
                "note": "caller-supplied bytes; not a this-round upstream claim",
            })
        else:
            work = _prepare_stage(output)
        segments, translated, glossary = _load_stage_inputs(modules, work)
        if context_glossary:
            glossary = context_glossary + glossary
        args = _stage_args(resume=phase == "full")
        stage1, suspect_ids, batches = _review_batches(modules, segments, translated, args)
        capture.set_total(len(batches) if phase == "full" else min(1, len(batches)))
        client = _client(modules, key)
        config = _config(modules, key)
        if phase == "first-batch":
            if not batches:
                if bound:
                    raise RuntimeError("no nearby-audio batch in the supplied manifests")
                raise RuntimeError("no real nearby-audio batch in the baseline suspects")
            results = _run_one_batch(modules, work, client, config, args, segments, translated, glossary, stage1, batches[0])
            detail = "first nearby-audio batch ids=%s" % ",".join(batches[0])
        else:
            output_dir = work / "experiments" / "mimo-proofread"
            proof = modules["proof"]
            code = proof._run_two_stage_nearby(
                args=args,
                client=client,
                config=config,
                audio_config=config,
                segments=segments,
                translated=translated,
                glossary=glossary,
                output_dir=output_dir,
                manifest_path=output_dir / "mimo_proofread_segments.json",
                report_path=output_dir / "mimo_proofread_report.json",
                srt_path=output_dir / "subtitles.mimo-proofread.srt",
            )
            detail = "full two-stage-nearby resume exit=%s suspects=%s batches=%s" % (code, len(suspect_ids), len(batches))
        status = "passed"
        business_ok = (not capture.circuit_open and not capture.failures and
                       (capture.completed > 0 or capture.reused > 0 or (phase == "full" and code == 0)))
        if phase == "first-batch":
            business_ok = business_ok and bool(results) and all(
                item.report_item.get("status") == "completed" for item in results)
        else:
            stage2_report = proof._load_existing_report(output_dir / "mimo_stage2_nearby_audio_report.json")
            business_ok = business_ok and code == 0 and not proof._pending_review_ids(suspect_ids, stage2_report, resume=True)
        status = "passed" if business_ok else "failed"
        failure = "none" if business_ok else (capture.failure_class if capture.failure_class != "none" else "business_failure")
        if not bound or not (_output_dir(output) / "chain-state.json").is_file():
            status = "passed"
            if capture.circuit_open or capture.completed <= 0:
                status = "failed"
            elif capture.failures:
                status = "partial"
            if bound and status == "passed":
                status = "partial"
            failure = capture.failure_class if status != "passed" else "none"
        report = _blank("stage.audio_review/run", "online", status, failure, detail)
        report["business_ok"] = business_ok
        report["successful_requests"] = capture.completed - capture.failures
        report.update({
            "live": capture.completed > 0,
            "phase": phase,
            "attempts": capture.completed,
            "failures": capture.failures,
            "circuit_open": capture.circuit_open,
            "endpoint": endpoint_view(OPENAI_BASE),
            "endpoint_source": "ticket-token-plan-openai",
            "env_base_url": "recorded-by-caller",
            "tokens": {"prompt_tokens": capture.tokens_in, "completion_tokens": capture.tokens_out},
            "elapsed_s": round(time.perf_counter() - started, 3),
            "calls": capture.calls,
            "batch_count": len(batches),
            "suspect_count": len(suspect_ids),
            "cue_count": len(translated) if isinstance(translated, dict) else 0,
            "artifact": "stages/audio-openai/experiments/mimo-proofread/mimo_proofread_segments.json",
            "argv": None,
        })
        if capture.reuse_loaded:
            report["responses_reused"] = capture.reused
            report["requests_added"] = capture.completed
        update_progress(
            output, "audio-openai", phase, capture.completed,
            capture.total if capture.total_known else capture.completed,
            failure, status, total_known=True, archive=archive,
        )
        return _finish(report)
    except CircuitOpen as exc:
        failure = exc.failure_class or "circuit"
        report = _blank("stage.audio_review/run", "online", "failed", failure, "deterministic stop")
        report.update({
            "live": capture.completed > 0,
            "phase": phase,
            "attempts": capture.completed,
            "failures": capture.failures,
            "circuit_open": True,
            "endpoint": endpoint_view(OPENAI_BASE),
            "endpoint_source": "ticket-token-plan-openai",
            "tokens": {"prompt_tokens": capture.tokens_in, "completion_tokens": capture.tokens_out},
            "elapsed_s": round(time.perf_counter() - started, 3),
            "calls": capture.calls,
            "argv": None,
        })
        update_progress(output, "audio-openai", phase, capture.completed, capture.completed, failure, "failed", total_known=True, archive=archive)
        return _finish(report)
    except Exception as exc:
        failure = classify_message(scrub(str(exc), [key]))["class"]
        report = _blank("stage.audio_review/run", "online", "failed", failure, scrub(str(exc), [key])[:300])
        report.update({
            "live": capture.completed > 0,
            "phase": phase,
            "attempts": capture.completed,
            "failures": max(capture.failures, 1),
            "circuit_open": capture.circuit_open,
            "endpoint": endpoint_view(OPENAI_BASE),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "calls": capture.calls,
            "argv": None,
        })
        update_progress(output, "audio-openai", phase, capture.completed, capture.completed, failure, "failed", total_known=True, archive=archive)
        return _finish(report)
    finally:
        capture.restore()
        logging.getLogger().removeFilter(log_filter)
        if work is not None:
            _scrub_tree(work, [key])
        _scrub_tree(_output_dir(output) / "calls", [key])


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


def _duration_ms(item):
    try:
        return int(item.get("end_time") or 0) - int(item.get("start_time") or 0)
    except (TypeError, ValueError):
        return -1


def _row_text(item):
    return str(item.get("original_subtitle") or "") + "\n" + str(item.get("translated_subtitle") or "")


def select_samples(translated):
    """Deterministic ordinary / number-or-name / longer-audio ids. No network."""
    rows = []
    if not isinstance(translated, dict):
        return []
    for key, item in translated.items():
        if not str(key).isdigit() or not isinstance(item, dict):
            continue
        rows.append((int(key), item))
    rows.sort(key=lambda pair: pair[0])
    ordinary = None
    numbered = None
    longer = None
    longer_ms = -1
    for key, item in rows:
        text = _row_text(item)
        duration = _duration_ms(item)
        flagged = bool(item.get("asr_suspect") or item.get("needs_audio_review"))
        if ordinary is None and not flagged and not re.search(r"\d", text) and 0 < duration <= 4000:
            ordinary = key
        if numbered is None and re.search(r"\d", text):
            numbered = key
        if duration > longer_ms:
            longer_ms = duration
            longer = key
    chosen = []
    for kind, key in (("ordinary", ordinary), ("number_or_name", numbered), ("longer_audio", longer)):
        if key is None or key in {item["id"] for item in chosen}:
            continue
        chosen.append({"kind": kind, "id": key})
    return chosen


def anthropic_body(messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Map one OpenAI audio message onto an Anthropic messages body without dropping audio."""
    content = []
    saw_audio = False
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        parts = message.get("content")
        if isinstance(parts, str):
            content.append({"type": "text", "text": parts})
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                content.append({"type": "text", "text": str(part.get("text") or "")})
            elif part.get("type") == "input_audio":
                audio = part.get("input_audio") or {}
                data = str(audio.get("data") or "")
                if not data:
                    raise RuntimeError("refusing an audio block with no bytes")
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": audio.get("format") or "wav"},
                })
                saw_audio = True
    if not saw_audio:
        raise RuntimeError("refusing a text-only body as an audio comparison")
    return {
        "model": MODEL,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": content}],
    }


def _audio_sha_from_messages(messages):
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_audio":
                data = str((part.get("input_audio") or {}).get("data") or "")
                return _sha_text(data), len(data)
    return "", 0


def _capture_audio_messages(modules, client, config, sample_id, translated, segments, glossary, clips_dir):
    proof = modules["proof"]
    requests = modules["requests"]
    captured = {}

    def trap(**kwargs):
        captured["messages"] = kwargs.get("messages")
        raise CircuitOpen("captured")

    original = requests.chat_completion_create
    requests.chat_completion_create = trap
    try:
        try:
            proof._process_stage2_nearby_audio_batch_task(
                target_ids=[str(sample_id)],
                client=client,
                config=config,
                segments=segments,
                translated=translated,
                glossary=glossary,
                clips_dir=clips_dir,
                context_subtitles=1,
                padding_s=1.5,
                max_glossary_entries=80,
                max_retries=1,
                base_delay=0.0,
                max_delay=0.0,
                keep_raw=False,
                apply_confidence_threshold=0.9,
            )
        except CircuitOpen as exc:
            if exc.failure_class != "captured":
                raise
    finally:
        requests.chat_completion_create = original
    messages = captured.get("messages")
    if not messages:
        raise RuntimeError("audio message was not built")
    return messages


def _send_openai_messages(client, messages):
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return {
        "ok": True,
        "failure_class": "none",
        "http_status": 200,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "text": _response_text(response),
        "usage": _usage_counts(response),
        "model": MODEL,
    }


def _send_anthropic(key, body):
    import anthropic
    from anthropic.types.message import Message
    client = anthropic.Anthropic(base_url=ANTHROPIC_BASE, api_key=key, max_retries=0, timeout=TIMEOUT_S)
    started = time.perf_counter()
    try:
        message = client.post("/v1/messages", cast_to=Message, body=body)
    except anthropic.APIStatusError as exc:
        kind = classify_message(scrub(str(exc), [key]), getattr(exc, "status_code", None))
        return {
            "ok": False,
            "failure_class": kind["class"],
            "http_status": kind["http_status"],
            "elapsed_s": round(time.perf_counter() - started, 3),
            "text": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": MODEL,
            "detail": scrub(str(exc), [key])[:240],
        }
    text = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text.append(getattr(block, "text", "") or "")
    usage = getattr(message, "usage", None)
    return {
        "ok": True,
        "failure_class": "none",
        "http_status": 200,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "text": "\n".join(text),
        "usage": {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        },
        "model": str(getattr(message, "model", "") or MODEL),
    }


def _numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text or "")


def _language_shares(text):
    value = text or ""
    if not value:
        return {"ja": 0.0, "zh": 0.0}
    ja = len(re.findall(r"[\u3040-\u30ff]", value))
    zh = len(re.findall(r"[\u4e00-\u9fff]", value))
    total = max(1, len(value))
    return {"ja": round(ja / total, 4), "zh": round(zh / total, 4)}


@lru_cache(maxsize=1)
def _evaluation():
    """Load experiment-only code from bytes pinned by this entrypoint identity."""
    from host.runtime import capture_source, execute_source
    path = Path(__file__).with_name("evaluation.py")
    source = capture_source(path, "d33157b2e984c68850114f55a35c14a838f177024a58bb03cb687cc9f30fd970")
    return execute_source("_mimo_wording_evaluation", path, source)


def wording_assessment(candidate, reference, source):
    # Compatibility boundary: preserve the operations-module monkeypatch hooks.
    return _evaluation().wording_assessment(
        candidate, reference, source, numbers=_numbers, language_shares=_language_shares)


def _load_baseline_translated():
    path = BASELINE / "manifests" / "translated_segments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("translated manifest is not an object")
    return payload


def _execute_compare(output, key, base_state):
    started = time.perf_counter()
    modules = _legacy()
    translated = _load_baseline_translated()
    segments, _ignored, glossary = _load_stage_inputs(modules, _prepare_readonly_view())
    samples = select_samples(translated)
    update_progress(output, "protocol-compare", "limited", 0, len(samples), "none", "running", total_known=True, archive=True)
    client = _client(modules, key)
    config = _config(modules, key)
    clips = _inside(_output_dir(output) / "compare" / "clips", _output_dir(output))
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    anthropic_open = True
    openai_open = True
    tokens_in = 0
    tokens_out = 0
    attempts = 0
    failures = 0
    failure_class = "none"
    for index, sample in enumerate(samples, start=1):
        messages = _capture_audio_messages(
            modules, client, config, sample["id"], translated, segments, glossary, clips,
        )
        body = anthropic_body(messages)
        audio_sha, audio_chars = _audio_sha_from_messages(messages)
        prompt = _public_prompt(messages, [key])
        row = {
            "kind": sample["kind"],
            "id": sample["id"],
            "audio_sha256": audio_sha,
            "audio_b64_chars": audio_chars,
            "prompt_sha256": _sha_text(prompt),
            "same_audio_bytes": True,
            "same_prompt_text": True,
            "openai": None,
            "anthropic": None,
        }
        if openai_open:
            try:
                sent = _send_openai_messages(client, messages)
            except Exception as exc:
                kind = classify_message(scrub(str(exc), [key]), getattr(exc, "status_code", None))
                sent = {
                    "ok": False, "failure_class": kind["class"], "http_status": kind["http_status"],
                    "elapsed_s": 0.0, "text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "model": MODEL, "detail": scrub(str(exc), [key])[:240],
                }
            attempts += 1
            tokens_in += int(sent["usage"]["prompt_tokens"])
            tokens_out += int(sent["usage"]["completion_tokens"])
            if not sent["ok"]:
                failures += 1
                failure_class = sent["failure_class"]
                if sent["failure_class"] in DETERMINISTIC:
                    openai_open = False
            row["openai"] = _public_result(sent, [key])
        else:
            row["openai"] = {"status": "skipped_after_circuit"}
        if anthropic_open:
            sent = _send_anthropic(key, body)
            attempts += 1
            tokens_in += int(sent["usage"]["prompt_tokens"])
            tokens_out += int(sent["usage"]["completion_tokens"])
            if not sent["ok"]:
                failures += 1
                failure_class = sent["failure_class"]
                if sent["failure_class"] in DETERMINISTIC or sent["failure_class"] == "audio_unsupported":
                    anthropic_open = False
            row["anthropic"] = _public_result(sent, [key])
        else:
            row["anthropic"] = {"status": "skipped_after_circuit"}
        row["quality"] = _pair_quality(row.get("openai"), row.get("anthropic"), _row_text(translated[str(sample["id"])]))
        rows.append(row)
        _write_json(_output_dir(output) / "compare" / f"{index:02d}-{sample['kind']}.json", row)
        update_progress(
            output, "protocol-compare", "limited", index, len(samples), failure_class, "running",
            total_known=True, archive=True,
        )
    explicit_rejection = any((row.get("anthropic") or {}).get("failure_class") == "audio_unsupported" for row in rows)
    incompatible = explicit_rejection
    both_http = [row for row in rows if (row.get("openai") or {}).get("ok") and (row.get("anthropic") or {}).get("ok")]
    if explicit_rejection:
        capability = "audio_incompatible"
    elif rows and len(both_http) == len(rows):
        capability = "http_ok"
    elif both_http:
        capability = "partial_http"
    else:
        capability = "incomplete"
    quality = "NOT_ASSESSED"
    status = "passed" if attempts and failures == 0 else ("partial" if any((row.get("openai") or {}).get("ok") or (row.get("anthropic") or {}).get("ok") for row in rows) else "failed")
    if incompatible and any((row.get("openai") or {}).get("ok") for row in rows):
        status = "partial"
    report = _blank("provider.audio_llm/compare", "online", status, failure_class if status != "passed" else "none")
    report.update({
        "live": attempts > 0,
        "attempts": attempts,
        "failures": failures,
        "circuit_open": not anthropic_open or not openai_open,
        "endpoint": {
            "openai": endpoint_view(OPENAI_BASE),
            "anthropic": endpoint_view(ANTHROPIC_BASE),
        },
        "endpoint_source": "ticket-token-plan",
        "env_mimo_base_url": base_state,
        "model_sent": MODEL,
        "shared_parameters": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "thinking": "disabled",
            "same_prompt_text": True,
            "same_audio_bytes": True,
        },
        "unmapped": {
            "openai_extra_body": "mapped to Anthropic thinking.type=disabled",
            "openai_input_audio": "sent as the same input_audio block; not rewritten into a text probe",
            "system_message": "unused on both",
        },
        "capability": capability,
        "quality": quality,
        "audio_consumption": "NOT_PROVEN",
        "audio_effectiveness": "NOT_ASSESSED",
        "migration_equivalence": "NOT_ASSESSED",
        "samples": rows,
        "tokens": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
        "elapsed_s": round(time.perf_counter() - started, 3),
        "text_probe_used_as_audio": False,
        "argv": None,
    })
    _scrub_tree(_output_dir(output) / "compare", [key])
    update_progress(
        output, "protocol-compare", "limited", len(samples), len(samples),
        report["failure_class"], status, total_known=True, archive=True,
    )
    return report


def _prepare_readonly_view():
    """Stage inputs already live in the baseline. Comparison reads that tree through a copied manifest view only when the stage has not prepared one."""
    return BASELINE


BUSINESS_FIELDS = ("original", "translation", "suggested_original", "suggested_translation")
METADATA_FIELDS = ("error_type", "asr_suspect", "needs_audio_review", "reason", "confidence")
CONTRACT_FIELDS = ("id",) + BUSINESS_FIELDS + METADATA_FIELDS


def _public_result(sent, secrets):
    text = scrub(sent.get("text") or "", secrets)
    return {
        "ok": bool(sent.get("ok")),
        "failure_class": sent.get("failure_class"),
        "http_status": sent.get("http_status"),
        "elapsed_s": sent.get("elapsed_s"),
        "model": sent.get("model"),
        "usage": sent.get("usage"),
        "text_sha256": _sha_text(text),
        "text_chars": len(text),
        "text_preview": text[:500],
        "text": text,
        "detail": sent.get("detail") or "",
    }


def _strip_fence(text):
    raw = (text or "").strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if match:
        return match.group(1).strip(), True
    return raw, False


def parse_subtitle_completion(text):
    """Parse one completion. A failed parse is never an empty pass."""
    body, fenced = _strip_fence(text)
    if not body:
        return {"status": "empty", "fenced": fenced, "items": None}
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "parse_failed", "fenced": fenced, "items": None}
    if not isinstance(payload, list):
        return {"status": "not_array", "fenced": fenced, "items": None}
    return {"status": "parsed", "fenced": fenced, "items": payload}


def index_cues(items):
    by_id = {}
    duplicates = []
    missing_id = 0
    non_objects = 0
    for item in items:
        if not isinstance(item, dict):
            non_objects += 1
            continue
        cue = item.get("id")
        if cue is None or str(cue).strip() == "":
            missing_id += 1
            continue
        key = str(cue).strip()
        if key in by_id:
            duplicates.append(key)
            continue
        by_id[key] = item
    status = "ok" if not (duplicates or missing_id or non_objects) else "id_conflict"
    return {
        "status": status,
        "by_id": by_id,
        "duplicate_ids": duplicates,
        "missing_id_count": missing_id,
        "non_object_count": non_objects,
        "positional_zip": False,
    }


def _business_text(item):
    return "\n".join(str(item.get(key) or "") for key in BUSINESS_FIELDS)


def _stored_truncated(text, stored_chars):
    if stored_chars is None:
        return False
    return len(text or "") < int(stored_chars)


def assess_subtitle_pair(openai_text, anthropic_text, source="", expected_id=None, openai_stored_chars=None, anthropic_stored_chars=None):
    """Compare parsed subtitle fields by cue id. Metadata never enters number or length checks."""
    result = {
        "audio_consumption": "NOT_PROVEN",
        "audio_effectiveness": "NOT_ASSESSED",
        "migration_equivalence": "NOT_ASSESSED",
        "positional_zip": False,
        "parse": {},
        "contract": {},
        "metadata": {"differences": [], "counted_as_subtitle_error": False},
        "subtitle_text": {"verdict": "NOT_ASSESSED", "reason": "not_run", "positional_zip": False},
    }
    truncated = []
    if _stored_truncated(openai_text, openai_stored_chars):
        truncated.append("openai")
    if _stored_truncated(anthropic_text, anthropic_stored_chars):
        truncated.append("anthropic")
    if truncated:
        result["parse"] = {"status": "truncated_stored_text", "sides": truncated}
        result["subtitle_text"] = {"verdict": "NOT_ASSESSED", "reason": "truncated_stored_text", "positional_zip": False}
        return result
    left = parse_subtitle_completion(openai_text)
    right = parse_subtitle_completion(anthropic_text)
    result["parse"] = {
        "openai": left["status"],
        "anthropic": right["status"],
        "openai_fenced": left["fenced"],
        "anthropic_fenced": right["fenced"],
    }
    if left["status"] != "parsed" or right["status"] != "parsed":
        result["subtitle_text"] = {"verdict": "NOT_ASSESSED", "reason": "parse_failed", "positional_zip": False}
        return result
    left_index = index_cues(left["items"])
    right_index = index_cues(right["items"])
    result["parse"]["duplicate_ids"] = {"openai": left_index["duplicate_ids"], "anthropic": right_index["duplicate_ids"]}
    result["parse"]["ids"] = {"openai": sorted(left_index["by_id"]), "anthropic": sorted(right_index["by_id"])}
    if left_index["status"] != "ok" or right_index["status"] != "ok":
        result["subtitle_text"] = {
            "verdict": "NOT_ASSESSED",
            "reason": "id_conflict",
            "positional_zip": False,
            "duplicate_ids": result["parse"]["duplicate_ids"],
            "missing_id_count": {"openai": left_index["missing_id_count"], "anthropic": right_index["missing_id_count"]},
        }
        return result
    expected = None if expected_id is None else str(expected_id).strip()
    if expected and (expected not in left_index["by_id"] or expected not in right_index["by_id"]):
        result["contract"] = {"missing_cue": {"id": expected, "openai": expected not in left_index["by_id"], "anthropic": expected not in right_index["by_id"]}}
        result["subtitle_text"] = {"verdict": "NOT_ASSESSED", "reason": "missing_cue", "expected_id": expected, "positional_zip": False}
        return result
    shared = [expected] if expected else sorted(set(left_index["by_id"]) & set(right_index["by_id"]))
    metadata = []
    contract = []
    cues = []
    hard = False
    soft = False
    for cue in shared:
        openai_item = left_index["by_id"][cue]
        anthropic_item = right_index["by_id"][cue]
        known = set(CONTRACT_FIELDS)
        gaps = {
            "openai": [key for key in CONTRACT_FIELDS if key not in openai_item],
            "anthropic": [key for key in CONTRACT_FIELDS if key not in anthropic_item],
        }
        extra = {
            "openai": sorted(key for key in openai_item if key not in known),
            "anthropic": sorted(key for key in anthropic_item if key not in known),
        }
        if gaps["openai"] or gaps["anthropic"] or extra["openai"] or extra["anthropic"]:
            contract.append({"id": cue, "missing_keys": gaps, "extra_keys": extra})
        fields = []
        for key in METADATA_FIELDS:
            if (key in openai_item) != (key in anthropic_item) or openai_item.get(key) != anthropic_item.get(key):
                fields.append(key)
        if fields:
            metadata.append({"id": cue, "fields": fields})
        missing_business = [key for key in BUSINESS_FIELDS if key not in openai_item or key not in anthropic_item]
        if missing_business:
            result["contract"] = {"missing_keys": contract, "missing_business_fields": missing_business, "unmapped_extra_keys_not_used": True}
            result["metadata"] = {"differences": metadata, "counted_as_subtitle_error": False}
            result["subtitle_text"] = {"verdict": "NOT_ASSESSED", "reason": "incomplete_business_fields", "positional_zip": False}
            return result
        wording = wording_assessment(_business_text(anthropic_item), _business_text(openai_item), source)
        cues.append({"id": cue, "wording": wording})
        if wording["hard"]:
            hard = True
        if any(reason not in {"equal"} for reason in wording["reasons"]):
            soft = True
    result["metadata"] = {"differences": metadata, "counted_as_subtitle_error": False}
    result["contract"] = {"missing_keys": contract}
    if hard:
        verdict, reason = "NOT_PASSED", "hard_text"
    elif soft:
        verdict, reason = "NOT_PASSED", "wording"
    else:
        verdict, reason = "PASS", "equal_or_within_checks"
    result["subtitle_text"] = {
        "verdict": verdict,
        "reason": reason,
        "positional_zip": False,
        "cues": cues,
        "proper_nouns": "NOT_ASSESSED",
        "bracket_pairing": "NOT_ASSESSED",
    }
    return result


def _pair_quality(openai_row, anthropic_row, source):
    if not openai_row or not anthropic_row:
        return {"verdict": "NOT_ASSESSED", "reason": "missing side", "audio_consumption": "NOT_PROVEN", "audio_effectiveness": "NOT_ASSESSED", "migration_equivalence": "NOT_ASSESSED", "positional_zip": False}
    if not openai_row.get("ok") or not anthropic_row.get("ok"):
        reason = "explicit_audio_rejection" if anthropic_row.get("failure_class") == "audio_unsupported" else "side_failed"
        return {"verdict": "NOT_ASSESSED", "reason": reason, "audio_consumption": "NOT_PROVEN", "audio_effectiveness": "NOT_ASSESSED", "migration_equivalence": "NOT_ASSESSED", "positional_zip": False}
    assessment = assess_subtitle_pair(
        openai_row.get("text") or "",
        anthropic_row.get("text") or "",
        source=source or "",
        openai_stored_chars=openai_row.get("text_chars"),
        anthropic_stored_chars=anthropic_row.get("text_chars"),
    )
    subtitle = assessment["subtitle_text"]
    return {
        "verdict": subtitle.get("verdict"),
        "reason": subtitle.get("reason"),
        "wording": subtitle.get("cues"),
        "dimensions": assessment,
        "positional_zip": False,
        "audio_consumption": "NOT_PROVEN",
        "audio_effectiveness": "NOT_ASSESSED",
        "migration_equivalence": "NOT_ASSESSED",
    }


def reassess_stored_compare(compare_dir):
    """Recompute pair judgments from already stored compare files. No network."""
    directory = Path(compare_dir)
    pairs = []
    openai_requests = 0
    anthropic_requests = 0
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if "openai" not in row or "anthropic" not in row:
            continue
        openai_row = row.get("openai") or {}
        anthropic_row = row.get("anthropic") or {}
        if openai_row.get("ok"):
            openai_requests += 1
        if anthropic_row.get("ok"):
            anthropic_requests += 1
        assessment = assess_subtitle_pair(
            openai_row.get("text") or "",
            anthropic_row.get("text") or "",
            expected_id=row.get("id"),
            openai_stored_chars=openai_row.get("text_chars"),
            anthropic_stored_chars=anthropic_row.get("text_chars"),
        )
        pairs.append({
            "file": path.name,
            "kind": row.get("kind"),
            "id": row.get("id"),
            "http": {
                "openai": {"ok": openai_row.get("ok"), "http_status": openai_row.get("http_status"), "failure_class": openai_row.get("failure_class"), "elapsed_s": openai_row.get("elapsed_s"), "usage": openai_row.get("usage"), "text_chars": openai_row.get("text_chars"), "stored_chars": len(openai_row.get("text") or "")},
                "anthropic": {"ok": anthropic_row.get("ok"), "http_status": anthropic_row.get("http_status"), "failure_class": anthropic_row.get("failure_class"), "elapsed_s": anthropic_row.get("elapsed_s"), "usage": anthropic_row.get("usage"), "text_chars": anthropic_row.get("text_chars"), "stored_chars": len(anthropic_row.get("text") or "")},
            },
            "first_round_machine_verdict": (row.get("quality") or {}).get("verdict"),
            "first_round_machine_reason": (row.get("quality") or {}).get("reason"),
            "assessment": assessment,
        })
    return {
        "audio_consumption": "NOT_PROVEN",
        "audio_effectiveness": "NOT_ASSESSED",
        "migration_equivalence": "NOT_ASSESSED",
        "positional_zip": False,
        "request_accounting": {
            "compare_requests": openai_requests + anthropic_requests,
            "openai_compare_requests": openai_requests,
            "anthropic_compare_requests": anthropic_requests,
            "shape": "3 samples x 2 protocols" if openai_requests == 3 and anthropic_requests == 3 else "counted from stored files",
            "not_anthropic_only": anthropic_requests != (openai_requests + anthropic_requests),
        },
        "pairs": pairs,
    }


def run_stage(arguments, api):
    mode = arguments["mode"]
    phase = arguments["phase"]
    output = arguments["output"]
    flag = str(api.environment("REAL_MIMO_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    if mode != "online" or not granted:
        report = _blank("stage.audio_review/run", "offline", "offline", "none", "offline grant; no socket")
        report["phase"] = phase
        if mode == "online" and not granted:
            report["status"] = "blocked_declaration"
            report["failure_class"] = "online_not_granted"
            report["detail"] = "explicit online grant is closed; no socket"
        return report
    key, base_state = _credentials(api)
    if not key:
        report = _blank("stage.audio_review/run", "online", "blocked_credentials", "missing_credentials", "MIMO_API_KEY unset")
        report["phase"] = phase
        report["env_mimo_base_url"] = base_state
        return report
    report = _execute_stage(phase, output, key, context_glossary=_content_glossary(api))
    report["env_mimo_base_url"] = base_state
    return report


def run_compare(arguments, api):
    mode = arguments["mode"]
    output = arguments["output"]
    flag = str(api.environment("REAL_MIMO_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    if mode != "online" or not granted:
        report = _blank("provider.audio_llm/compare", "offline", "offline", "none", "offline grant; no socket")
        report["text_probe_used_as_audio"] = False
        report["capability"] = "not_sent"
        if mode == "online" and not granted:
            report["status"] = "blocked_declaration"
            report["failure_class"] = "online_not_granted"
        return report
    key, base_state = _credentials(api)
    if not key:
        report = _blank("provider.audio_llm/compare", "online", "blocked_credentials", "missing_credentials", "MIMO_API_KEY unset")
        report["env_mimo_base_url"] = base_state
        report["text_probe_used_as_audio"] = False
        return report
    return _execute_compare(output, key, base_state)


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


def _audio_failure_class(exc):
    message = str(exc)
    if isinstance(exc, FileNotFoundError) or message.startswith("missing upstream") or message.startswith("upstream path does not exist"):
        return "missing_upstream"
    if message.startswith("parse failed"):
        return "parse_failed"
    if message.startswith("id mismatch") or message.startswith("id dependency"):
        return "id_mismatch"
    return "invalid_upstream"


def _audio_bound_failure(output, phase, failure_class, detail):
    report = _blank(BOUND_OPERATION, "online", "failed", failure_class, detail[:300])
    report["phase"] = phase
    report["live"] = False
    report["attempts"] = 0
    report["lineage"] = "supplied"
    report["read_baseline_fallback"] = False
    report["claimed_this_round"] = False
    report["chain_pass"] = False
    report["chain_passed"] = False
    report["migration_equivalence"] = "NOT_ASSESSED"
    report["coverage"] = "explicit_input_only"
    report["required_files"] = list(BOUND_FILES)
    report["optional_files"] = ["lineage.json", "audio path referenced by segments.json"]
    report["audio_copied"] = False
    try:
        if output and _output_dir(output).is_dir():
            update_progress(output, "audio-openai", phase, 0, 0, failure_class, "failed", total_known=True, archive=False)
    except (OSError, RuntimeError, ValueError):
        pass
    return report


def run_stage_bound(arguments, api):
    mode = arguments["mode"]
    phase = arguments["phase"]
    output = arguments["output"]
    try:
        source = _resolve_upstream(arguments["upstream"])
        label, _parsed = _validate_audio_bound(source)
    except (FileNotFoundError, ValueError) as exc:
        return _audio_bound_failure(output, phase, _audio_failure_class(exc), str(exc))
    flag = str(api.environment("REAL_MIMO_ONLINE", "") or "")
    granted = online_granted(load_spec(), flag)
    if mode != "online" or not granted:
        report = _blank(BOUND_OPERATION, "offline", "offline", "none", "offline grant; no socket")
        report["phase"] = phase
        if mode == "online" and not granted:
            report["status"] = "blocked_declaration"
            report["failure_class"] = "online_not_granted"
            report["detail"] = "explicit online grant is closed; no socket"
        report.update(_bound_fields(source, label, {}))
        return report
    key, base_state = _credentials(api)
    if not key:
        report = _blank(BOUND_OPERATION, "online", "blocked_credentials", "missing_credentials", "MIMO_API_KEY unset")
        report["phase"] = phase
        report["env_mimo_base_url"] = base_state
        report.update(_bound_fields(source, label, {}))
        return report
    report = _execute_stage(phase, output, key, source_dir=source, bound=True, source_label=label, context_glossary=_content_glossary(api))
    report["env_mimo_base_url"] = base_state
    return report


def run_bind_suite(upstream, api):
    from host.contracts import ref

    flag = str(api.environment("REAL_MIMO_ONLINE", "") or "")
    spec = load_spec()
    granted = online_granted(spec, flag)
    mode = "online" if granted else "offline"
    snap = api.snapshot()["environment"]
    output = str(snap.get("output") or "")
    phase = "first-batch" if mode == "online" else "offline"
    target = ref("stage.audio_review", "run_bound", VERSION)
    child = _child_report(api.call(target, {
        "mode": mode,
        "phase": phase,
        "output": output or "offline-placeholder",
        "upstream": upstream,
    }))
    calls = [child]
    reports = [child["report"]] if child.get("outcome") == "returned" and isinstance(child.get("report"), dict) else []
    failed = 0 if reports and reports[0].get("status") in {"partial", "offline"} else 1
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
        "gpu_used": False,
        "migration_equivalence": "NOT_ASSESSED",
        "coverage": "explicit_input_only",
        "read_baseline_fallback": False,
        "claimed_this_round": False,
        "audio_copied": False,
        "executed_operations": [BOUND_OPERATION],
        "argv_returned": False,
        "mode": mode,
        "online_granted": granted,
        "bind_upstream": upstream,
        "text_probe_used_as_audio": False,
        "host_network_flag": snap.get("network"),
        "host_plan": snap.get("plan"),
        "calls": calls,
    }
    if output and (ROOT / output).is_dir():
        directory = _output_dir(output)
        _write_json(directory / "evidence" / "report.json", summary)
        _write_json(directory / "evidence" / "criteria.json", {
            "migration_equivalence": "NOT_ASSESSED",
            "chain_passed": False,
            "episode_pass": False,
            "coverage": "explicit_input_only",
            "text_probe_used_as_audio": False,
        })
    return summary


CHAIN_FILES = ("aligned_segments.json", "transcript_segments.json", "segments.json", "split_segments.json", "translated_segments.json")


def _validate_chain(source):
    lineage = _load_bound_json(source, "lineage.json")
    if not isinstance(lineage, dict) or lineage.get("source") not in {"current-run", "synthetic", "baseline", "B"}:
        raise ValueError("invalid lineage source")
    if lineage.get("source") == "current-run" and lineage.get("producer") != "real-chain-llm":
        raise ValueError("invalid lineage producer")
    files = {name: _load_bound_json(source, name) for name in CHAIN_FILES}
    hashes = {name: _sha_bytes((source / name).read_bytes()) for name in CHAIN_FILES}
    if lineage.get("files") != hashes or lineage.get("inputs") != {name: hashes[name] for name in CHAIN_FILES[:2]}:
        raise ValueError("lineage hash mismatch")
    _validate_audio_bound(source)
    def index(items, name):
        if not isinstance(items, list) or not items:
            raise ValueError("id dependency: " + name)
        result = {}
        for item in items:
            if not isinstance(item, dict) or not item.get("segment_id") or str(item["segment_id"]) in result:
                raise ValueError("id dependency: " + name)
            result[str(item["segment_id"])] = item
        return result
    aligned = index(files["aligned_segments.json"], "aligned")
    transcript = index(files["transcript_segments.json"], "transcript")
    segments = index(files["segments.json"], "segments")
    if set(aligned) != set(transcript) or set(segments) != set(transcript):
        raise ValueError("id dependency: segment sets differ")
    for key in transcript:
        for field in ("global_start_time", "global_end_time"):
            if any(field not in group[key] for group in (aligned, transcript, segments)):
                raise ValueError("id dependency: missing segment time")
            if len({float(group[key][field]) for group in (aligned, transcript, segments)}) != 1:
                raise ValueError("id dependency: segment time mismatch")
        if float(transcript[key]["global_end_time"]) <= float(transcript[key]["global_start_time"]):
            raise ValueError("id dependency: invalid segment time")
    ordered_segments = files["segments.json"]
    for key, item in files["split_segments.json"].items():
        if not isinstance(item, dict) or str(item.get("segment_id") or "") not in segments:
            raise ValueError("id dependency: split segment")
        try:
            start, end = float(item["start_time"]), float(item["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("id dependency: split timing") from exc
        if not start < end:
            raise ValueError("id dependency: split timing")
        covering = []
        for segment in ordered_segments:
            try:
                left = float(segment["global_start_time"]) * 1000
                right = float(segment["global_end_time"]) * 1000
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("id dependency: split timing") from exc
            if left <= start < right:
                covering.append(str(segment["segment_id"]))
        if not covering or str(item["segment_id"]) != covering[0]:
            raise ValueError("id dependency: split cue start outside segment")
        other = files["translated_segments.json"].get(key)
        if not isinstance(other, dict) or any(item.get(f) != other.get(f) for f in ("start_time", "end_time", "original_subtitle")):
            raise ValueError("id dependency: translated cue")
    for segment in segments.values():
        for field in ("audio_path", "source_audio_path"):
            raw = str(segment.get(field) or "")
            if not raw:
                if field == "audio_path":
                    raise ValueError("missing explicit audio_path")
                continue
            audio = Path(raw)
            if not audio.is_file():
                if field == "source_audio_path":
                    continue
                raise FileNotFoundError("explicit audio_path missing")
            label = "audio:" + str(audio.resolve())
            if label not in hashes:
                h = hashlib.sha256()
                with audio.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(chunk)
                hashes[label] = h.hexdigest()
    hashes["lineage.json"] = _sha_bytes((source / "lineage.json").read_bytes())
    return hashes


def _sha_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_current_finish(api, source, host_directory):
    """Publish a ready finish handoff only after the MiMo report itself is complete."""
    raw = str(api.environment("REAL_CHAIN_FINISH_HANDOFF", "") or "").strip()
    if not raw:
        return "not_requested"
    workdir = (Path(host_directory) / "stages" / "audio-openai").resolve()
    if not workdir.is_dir():
        return "mimo_workdir_missing"
    report_path = workdir / "experiments" / "mimo-proofread" / "mimo_proofread_report.json"
    manifest_path = workdir / "experiments" / "mimo-proofread" / "mimo_proofread_segments.json"
    if not report_path.is_file() or not manifest_path.is_file():
        return "mimo_product_missing"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    complete = (
        report.get("mode") == "two-stage-nearby"
        and report.get("stage1_failed") == 0
        and report.get("stage2_failed") == 0
        and report.get("audio_review_unresolved_count") == 0
        and report.get("stage2_completed") == report.get("audio_review_candidate_count")
    )
    if not complete:
        return "mimo_not_complete"
    source = Path(source)
    for name in ("aligned_segments.json", "transcript_segments.json"):
        dest = workdir / name
        src = source / name
        if dest.is_file():
            continue
        if not src.is_file():
            return "missing_" + name
        shutil.copyfile(src, dest)
    segments = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    audio_raw = ""
    if isinstance(segments, list) and segments and isinstance(segments[0], dict):
        audio_raw = str(segments[0].get("source_audio_path") or "")
    audio_path = Path(audio_raw) if audio_raw else None
    if audio_path is None or not audio_path.is_file():
        return "source_audio_missing"
    relative_files = [
        "aligned_segments.json",
        "transcript_segments.json",
        "segments.json",
        "split_segments.json",
        "translated_segments.json",
        "experiments/mimo-proofread/mimo_proofread_segments.json",
        "experiments/mimo-proofread/mimo_proofread_report.json",
    ]
    files = {}
    for name in relative_files:
        path = workdir / name
        if not path.is_file():
            return "missing_" + name
        files[name] = _sha_path(path)
    resources = []
    for env_name, purpose in (
        ("REAL_CHAIN_LLM_SPEAKER_ENROLL", "speaker-enrollment"),
        ("REAL_CHAIN_LLM_SPEAKER_NPZ", "speaker-score-cache"),
        ("REAL_CHAIN_LLM_SOURCE_AUDIO", "source-audio"),
    ):
        value = str(api.environment(env_name, "") or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            return "resource_missing:" + env_name
        resources.append({"path": str(path.resolve()), "sha256": _sha_path(path), "purpose": purpose})
    publish_finish_handoff(raw, {
        "schema": "real-chain-finish-handoff-v1",
        "status": "ready",
        "workdir": str(workdir),
        "files": files,
        "source_audio": {"path": str(audio_path.resolve()), "sha256": _sha_path(audio_path)},
        "lineage": {
            "llm": {"handoff": str(source.resolve()), "producer": "real-chain-llm"},
            "mimo": {"output": str(Path(host_directory).resolve()), "workdir": str(workdir)},
        },
        "resources": resources,
    })
    return "ready"

def run_chain(api):
    from host.contracts import ref
    snap = api.snapshot()["environment"]
    host_output = str(snap.get("output") or "")
    host_directory = _output_dir(host_output)
    resume_output = str(api.environment("REAL_MIMO_CHAIN_RESUME_OUTPUT", "") or "").strip()
    output = resume_output or host_output
    directory = _output_dir(output)
    upstream = str(api.environment("REAL_MIMO_CHAIN_UPSTREAM", "") or "").strip()
    granted = online_granted(load_spec(), str(api.environment("REAL_MIMO_ONLINE", "") or ""))
    summary = {"plan_id": "REAL-CHAIN-MIMO-FULL-001", "status": "failed", "passed": 0, "failed": 1,
               "case_count": 1, "chain_passed": False, "episode_pass": False, "coverage": "partial",
               "migration_equivalence": "NOT_ASSESSED", "models_run": False, "gpu_used": False,
               "read_baseline_fallback": False, "calls": [], "live": False}
    if not upstream and not granted:
        summary.update(status="passed", passed=1, failed=0, mode="offline", detail="offline entry; no provider call")
    else:
        try:
            if resume_output and (not resume_output.startswith("runs/experiments/REAL-CHAIN-MIMO-FULL-") or not (directory / "chain-state.json").is_file()):
                raise ValueError("resume output is not a previous MiMo chain run")
            source = _resolve_upstream(upstream)
            hashes = _validate_chain(source)
            digest = _sha_text(json.dumps(hashes, sort_keys=True, separators=(",", ":")))
            state_path = directory / "chain-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
            if state is not None and state.get("input_sha256") != digest:
                raise ValueError("input digest differs from existing progress")
            if not granted:
                raise ValueError("online declaration closed")
            summary["inputs"] = hashes
            summary["input_sha256"] = digest
            if state is None:
                state = {"input_sha256": digest, "first_success": False, "full_success": False}
                _write_json(state_path, state)
            target = ref("stage.audio_review", "run_bound", VERSION)
            if not state.get("first_success"):
                child = _child_report(api.call(target, {"mode": "online", "phase": "first-batch", "output": output, "upstream": str(source)}))
                summary["calls"].append(child)
                report = child.get("report") or {}
                if child.get("outcome") != "returned" or not report.get("business_ok"):
                    raise ValueError("first batch failed")
                if _validate_chain(source) != hashes:
                    raise ValueError("input changed after first batch")
                state["first_success"] = True
                _write_json(state_path, state)
            if not state.get("full_success"):
                child = _child_report(api.call(target, {"mode": "online", "phase": "full", "output": output, "upstream": str(source)}))
                summary["calls"].append(child)
                report = child.get("report") or {}
                if child.get("outcome") != "returned" or not report.get("business_ok"):
                    raise ValueError("full stage failed")
                if _validate_chain(source) != hashes:
                    raise ValueError("input changed after full stage")
                state["full_success"] = True
                _write_json(state_path, state)
            summary.update(status="partial", passed=1, failed=0, mode="online", full_success=True)
            try:
                summary["finish_handoff"] = _publish_current_finish(api, source, directory)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                summary["finish_handoff"] = "publish_failed"
                summary["finish_handoff_detail"] = scrub(str(exc))[:300]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            summary["detail"] = scrub(str(exc))[:300]
    summary["live"] = any((x.get("report") or {}).get("live") for x in summary["calls"])
    if summary["live"]:
        summary["real_service_executed"] = True
    reused_reports = [x.get("report") or {} for x in summary["calls"] if "responses_reused" in (x.get("report") or {})]
    if reused_reports:
        summary["mimo_responses_reused"] = sum(int(item.get("responses_reused") or 0) for item in reused_reports)
        summary["mimo_requests_added"] = sum(int(item.get("requests_added") or 0) for item in reused_reports)
    summary["work_output"] = output
    _write_json(host_directory / "evidence" / "report.json", summary)
    _write_json(host_directory / "evidence" / "criteria.json", {"chain_passed": False, "episode_pass": False, "migration_equivalence": "NOT_ASSESSED", "coverage": "partial"})
    return summary


def publish_finish_handoff(path, payload):
    """Atomically publish the downstream handoff. Caller decides ready versus not."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target

class Implementation:
    def __init__(self, settings):
        self.settings = settings or {}

    def invoke(self, operation, arguments, api):
        from host.contracts import ref
        role = self.settings.get("role")
        if operation == "run" and role == "audio":
            return {"report": run_stage(arguments, api)}, []
        if operation == "run_bound" and role == "audio":
            return {"report": run_stage_bound(arguments, api)}, []
        if operation == "compare" and role == "provider":
            return {"report": run_compare(arguments, api)}, []
        if operation == "run_chain" and role == "suite":
            return {"report": run_chain(api)}, []
        if operation != "run" or role != "suite":
            raise KeyError(operation)
        upstream = str(api.environment(BIND_ENV, "") or "").strip()
        if upstream:
            return {"report": run_bind_suite(upstream, api)}, []
        flag = str(api.environment("REAL_MIMO_ONLINE", "") or "")
        spec = load_spec()
        granted = online_granted(spec, flag)
        mode = "online" if granted else "offline"
        snap = api.snapshot()["environment"]
        output = str(snap.get("output") or "")
        calls = []
        audio_target = ref("stage.audio_review", "run", VERSION)
        provider_target = ref("provider.audio_llm", "compare", VERSION)
        first = _child_report(api.call(audio_target, {"mode": mode, "phase": "first-batch" if mode == "online" else "offline", "output": output or "offline-placeholder"}))
        calls.append(first)
        first_report = first.get("report") or {}
        continue_full = mode == "online" and first_report.get("status") == "passed" and not first_report.get("circuit_open")
        if mode == "online":
            if continue_full:
                second = _child_report(api.call(audio_target, {"mode": "online", "phase": "full", "output": output}))
            else:
                skipped = _blank("stage.audio_review/run", "online", "skipped_after_circuit", first_report.get("failure_class") or "circuit", "first real batch did not pass")
                skipped["phase"] = "full"
                second = {"outcome": "returned", "report": skipped}
            calls.append(second)
        compare = _child_report(api.call(provider_target, {"mode": mode, "output": output or "offline-placeholder"}))
        calls.append(compare)
        reports = [item.get("report") for item in calls if item.get("outcome") == "returned" and isinstance(item.get("report"), dict)]
        failed = sum(1 for item in calls if item.get("outcome") != "returned")
        failed += sum(1 for report in reports if report.get("status") not in {"offline", "passed", "skipped_after_circuit"})
        live = any(report.get("live") for report in reports)
        if mode == "offline":
            status = "passed" if failed == 0 else "failed"
        elif any(report.get("status") == "passed" for report in reports) and failed:
            status = "partial"
        elif failed == 0 and reports and all(report.get("status") in {"passed", "skipped_after_circuit"} for report in reports):
            status = "passed"
        else:
            status = "failed" if not any(report.get("status") == "passed" for report in reports) else "partial"
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
            "gpu_used": False,
            "migration_equivalence": "NOT_ASSESSED",
            "executed_operations": [
                "stage.audio_review/run",
                "provider.audio_llm/compare",
            ],
            "argv_returned": False,
            "mode": mode,
            "online_granted": granted,
            "declared_online_network": (spec.get("permissions") or {}).get("online", {}).get("network") is True,
            "host_network_flag": snap.get("network"),
            "host_plan": snap.get("plan"),
            "host_isolation": "not enforced; host environment.network stays false and is reported separately",
            "model_sent": MODEL,
            "model_replaced": False,
            "stage_kind": "real_audio_review" if mode == "online" else "offline_entry",
            "comparison_kind": "limited_same_audio" if mode == "online" else "not_sent",
            "text_probe_used_as_audio": False,
            "calls": calls,
        }
        if output and (ROOT / output).is_dir():
            directory = _output_dir(output)
            _write_json(directory / "evidence" / "report.json", summary)
            _write_json(directory / "evidence" / "criteria.json", {
                "document": "docs/88-d01-online-tolerance.md",
                "migration_equivalence": "NOT_ASSESSED",
                "reason": "no paired old reference-A/reference-B in this ticket",
                "episode_pass": False,
                "chain_pass": False,
                "positional_zip": False,
                "text_probe_used_as_audio": False,
            })
            if mode == "offline":
                update_progress(output, "offline", "offline", 0, 0, "none", "offline", total_known=True, archive=False)
        return {"report": summary}, []
