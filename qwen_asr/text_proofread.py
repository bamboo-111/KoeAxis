"""Text-only proofread of the Chinese subtitles (stage ``text-proofread``).

Runs after the MiMo audio proofread (or straight after translate) and only rewrites
``translated_subtitle``: Japanese text and timing are never touched, so no realign is needed.

* every subtitle is checked by exactly one request (the audio segment it starts in); segment-boundary
  overlaps are sent as read-only context, together with ``CONTEXT_LINES`` lines on each side;
* the model must give a verdict for every target line (prompt v6, see ``build_prompt``); the line just
  before / after the targets may be edited too, but only as the other half of a ``duplicate`` fix;
* a suggestion is applied only with confidence >= the threshold, a known non-style error type, no
  Japanese kana and no copy of a neighbouring line;
* requests go to every configured Gemini relay through :class:`GeminiRelayPool`; finished requests are
  checkpointed by prompt hash, so a rerun only sends what failed or changed.

The result is a separate manifest (``experiments/text-proofread``); normalize/export prefer it over the
MiMo branch. Evaluated on episode 54 (2026-10-19): ~4 min at medium effort with two relays.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_asr.defaults import DEFAULT_LLM_MODEL, default_gemini_base_url
from qwen_asr.gemini_pool import GeminiEndpoint, GeminiRelayPool
from qwen_asr.mimo_audio import filter_glossary, load_glossary, subtitle_entries_for_segment
from qwen_asr.mimo_outputs import to_srt
from qwen_asr.mimo_requests import is_transient_error, parse_suggestions, prompt_json
from qwen_asr.models import WorkPaths
from qwen_asr.program_context import program_context_block, program_context_from_args
from qwen_asr.performance import record_network_request, usage_cached_tokens, usage_tokens
from qwen_asr.progress import write_progress
from qwen_asr.storage import read_json, write_json_atomic

LOGGER = logging.getLogger(__name__)

PROMPT_VERSION = "v6"
DEFAULT_EFFORT = "medium"
DEFAULT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_CONCURRENCY = 4
DEFAULT_APPLY_THRESHOLD = 0.8
CONTEXT_LINES = 3
GLOSSARY_LIMIT = 80
HINT_CHARS = 80
EFFORTS = ("low", "medium", "high")
# Relay slots read from the environment: GEMINI_BASE_URL / GEMINI_API_KEY, then the _1 ... _9 variants.
ENV_SUFFIXES = ("",) + tuple(f"_{index}" for index in range(1, 10))
APPLY_ERROR_TYPES = frozenset({"mistranslation", "omission", "addition", "duplicate", "term", "kana"})
KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa]")

_PROMPT_BODY = """输入是一个片段内的字幕行：speaker、日文原文 ja（ASR 识别结果，不修改）、当前简体中文译文 zh。部分行带有 hint：翻译阶段自动标出的疑点，可能不准，只用来提示重点核对哪里。
前文/后文仅作上下文，一般不要输出；唯一例外：前文最后一行或后文第一行与待校对行构成 duplicate 的一对时（如删掉本行重复内容，需要在那一行补回），可以一并给出那一行的修改。

对每一条待校对字幕都要给出结论。先逐个检查本行日文的实词（名词、谓语动词、数字、否定、疑问/反问）在本行译文里
是否都有对应且义项符合语境，再检查本行译文有没有本行日文里没有的内容。只修正确定的错误，error_type 取下列之一：
- mistranslation：意思译错——主宾或施受关系弄反、否定/肯定弄反、反问或疑问译成陈述、数字日期错、
  名词选了不符合语境的义项（外来语、服饰、道具、器材、乐队术语等）。
- omission：漏译了有实际意义的内容，尤其是谓语动词。口头禅、语气词、填充词（あの、なんか、えー 等）不算。
- addition：译文多出了本行日文里没有的内容。
- duplicate：一句话被拆成多行时，只修两种情况：①同一内容在相邻两行的译文里都出现了（重复）；
  ②某行日文的内容在本行和相邻行的译文里都找不到（遗漏）。修法是删掉重复部分或补回遗漏部分，涉及的待校对行都要给出。
  仅仅为了中文语序把内容放在相邻行、整体不重不漏的，不是错误，不要重排。
- term：人名、栏目名、作品名、乐队术语与术语表不符或前后不一致。专名常被 ASR 听成同音词；
  只有术语表或上下文里有能对上的词条时，才按正确词翻译。
- kana：译文残留日文假名，改为中文或拉丁字母。

保持原译（verdict 为 ok）的情况：
- 只是措辞、语序、标点、语气上的偏好；原译可以接受。
- 日文本身残缺或疑似听错：不要按字面去改一个符合上下文的原译，也不要猜测原意。
- 口号、梗、听众笔名等无法从术语表或上下文确认的内容。
- 修改不得改变整句意思：日文的否定、疑问常在下一行句尾，不能因为拆行而丢失或反转；也不要把相邻两行改成相同的文字。

confidence：确信原译有错且新译正确才给 ≥0.8，否则 ≤0.6。
只输出 JSON 数组（不要 markdown 或其他文字），每条待校对字幕一项，按 id 顺序：
无需修改：{"id": "字幕ID", "verdict": "ok"}
需要修改：{"id": "字幕ID", "verdict": "fix", "error_type": "上面之一", "suggested_translation": "修正后的完整本行译文",
"reason": "不超过20字", "confidence": 0到1}
"""


@dataclass(frozen=True)
class TextProofreadConfig:
    effort: str = DEFAULT_EFFORT
    model: str = DEFAULT_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    timeout: float = 600.0
    apply_threshold: float = DEFAULT_APPLY_THRESHOLD
    program: str = "日语节目"
    hosts: str = ""
    program_context: str = ""  # qwen_asr.program_context block text (--program-context files + episode title)
    glossary_xlsx: str | None = None
    resume: bool = True
    max_attempts: int = 3
    retry_delay_s: float = 10.0


def build_prompt_header(program: str, hosts: str) -> str:
    header = f"你是{program.strip() or '日语节目'}的中文字幕校对编辑。"
    if hosts.strip():
        header += f"主持成员：{hosts.strip()}。"
    return header + "\n"


def build_prompt(
    config: TextProofreadConfig,
    *,
    glossary: list[dict[str, str]],
    before: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> str:
    return (
        build_prompt_header(config.program, config.hosts)
        + program_context_block(config.program_context)
        + _PROMPT_BODY
        + f"术语表 JSON：{prompt_json(glossary)}\n"
        + f"前文（只读）：{prompt_json(before)}\n"
        + f"待校对字幕：{prompt_json(targets)}\n"
        + f"后文（只读）：{prompt_json(after)}"
    )


def endpoints_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    default_model: str = DEFAULT_MODEL,
    default_concurrency: int = DEFAULT_CONCURRENCY,
) -> list[GeminiEndpoint]:
    """One endpoint per configured relay slot: ``GEMINI_API_KEY{s}`` plus ``GEMINI_BASE_URL{s}``.

    ``GEMINI_MODEL{s}`` and ``GEMINI_CONCURRENCY{s}`` override the model and in-flight cap per relay
    (a relay that only serves e.g. ``gemini-3.8-flash-high`` needs its own model name).
    """
    env = os.environ if environ is None else environ
    endpoints: list[GeminiEndpoint] = []
    seen: set[tuple[str, str]] = set()
    for suffix in ENV_SUFFIXES:
        api_key = str(env.get(f"GEMINI_API_KEY{suffix}", "") or "").strip()
        if not api_key:
            continue
        base_url = default_gemini_base_url(str(env.get(f"GEMINI_BASE_URL{suffix}", "") or ""))
        if (base_url, api_key) in seen:
            continue
        seen.add((base_url, api_key))
        model = str(env.get(f"GEMINI_MODEL{suffix}", "") or "").strip() or default_model
        try:
            concurrency = int(str(env.get(f"GEMINI_CONCURRENCY{suffix}", "") or "").strip() or default_concurrency)
        except ValueError:
            concurrency = default_concurrency
        endpoints.append(GeminiEndpoint(
            name=f"gemini{suffix}", base_url=base_url, api_key=api_key, model=model,
            max_inflight=max(1, concurrency)))
    return endpoints


def subtitle_order(manifest: dict[str, Any]) -> list[str]:
    def key(subtitle_id: str) -> tuple[int, int, str]:
        item = manifest[subtitle_id]
        number = int(subtitle_id) if subtitle_id.isdigit() else 0
        return int(item.get("start_time", 0) or 0), number, subtitle_id

    return sorted((sid for sid, item in manifest.items() if isinstance(item, dict)), key=key)


def owner_map(segments: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, str]:
    """Each subtitle belongs to the last segment (in time order) that starts at or before it starts."""
    owner: dict[str, str] = {}
    ordered = sorted(segments, key=lambda seg: float(seg.get("global_start_time", 0.0) or 0.0))
    for segment in ordered:
        segment_start_ms = float(segment.get("global_start_time", 0.0) or 0.0) * 1000
        for subtitle_id in subtitle_entries_for_segment(manifest, segment):
            subtitle_start = float(manifest[subtitle_id].get("start_time", 0) or 0)
            if subtitle_id not in owner or subtitle_start >= segment_start_ms:
                owner[subtitle_id] = str(segment.get("segment_id"))
    return owner


def _prompt_row(manifest: dict[str, Any], subtitle_id: str, *, with_hint: bool) -> dict[str, Any]:
    item = manifest[subtitle_id]
    row = {
        "id": subtitle_id,
        "speaker": item.get("speaker", ""),
        "ja": item.get("original_subtitle", ""),
        "zh": item.get("translated_subtitle", ""),
    }
    hint = str(item.get("suspect_reason") or "").strip()
    if with_hint and hint:
        row["hint"] = hint[:HINT_CHARS]
    return row


@dataclass(frozen=True)
class ProofreadRequest:
    segment_id: str
    target_ids: list[str]
    neighbor_ids: list[str]
    prompt: str

    @property
    def key(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:24]


def build_requests(
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    glossary: list[dict[str, str]],
    config: TextProofreadConfig,
) -> list[ProofreadRequest]:
    order = subtitle_order(manifest)
    position = {subtitle_id: index for index, subtitle_id in enumerate(order)}
    owners = owner_map(segments, manifest)
    requests: list[ProofreadRequest] = []
    for segment in segments:
        segment_id = str(segment.get("segment_id"))
        entries = {
            subtitle_id: item
            for subtitle_id, item in subtitle_entries_for_segment(manifest, segment).items()
            if owners.get(subtitle_id) == segment_id
        }
        if not entries:
            continue
        target_ids = sorted(entries, key=position.__getitem__)
        first, last = position[target_ids[0]], position[target_ids[-1]]
        before_ids = order[max(0, first - CONTEXT_LINES):first]
        after_ids = order[last + 1:last + 1 + CONTEXT_LINES]
        prompt = build_prompt(
            config,
            glossary=filter_glossary(glossary, entries, GLOSSARY_LIMIT),
            before=[_prompt_row(manifest, sid, with_hint=False) for sid in before_ids],
            targets=[_prompt_row(manifest, sid, with_hint=True) for sid in target_ids],
            after=[_prompt_row(manifest, sid, with_hint=False) for sid in after_ids],
        )
        neighbor_ids = [*before_ids[-1:], *after_ids[:1]]
        requests.append(ProofreadRequest(segment_id, target_ids, neighbor_ids, prompt))
    return requests


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def select_edits(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    apply_threshold: float = DEFAULT_APPLY_THRESHOLD,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Pick at most one accepted edit per subtitle; everything else is returned as rejected with a reason."""
    order = subtitle_order(manifest)
    position = {subtitle_id: index for index, subtitle_id in enumerate(order)}
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for result in results:
        targets = set(result.get("target_ids", []))
        neighbors = set(result.get("neighbor_ids", []))
        for suggestion in result.get("suggestions", []):
            if not isinstance(suggestion, dict):
                continue
            subtitle_id = str(suggestion.get("id", "")).strip()
            if str(suggestion.get("verdict", "fix")).strip().lower() == "ok" or subtitle_id not in manifest:
                continue
            current = str(manifest[subtitle_id].get("translated_subtitle", "") or "").strip()
            new = str(suggestion.get("suggested_translation", "") or "").strip()
            if not new or new == current:
                continue
            edit = {
                "id": subtitle_id,
                "segment_id": result.get("segment_id"),
                "error_type": str(suggestion.get("error_type", "") or "").strip(),
                "confidence": _coerce_confidence(suggestion.get("confidence")),
                "original": manifest[subtitle_id].get("original_subtitle", ""),
                "old": current,
                "new": new,
                "reason": str(suggestion.get("reason", "") or "").strip(),
                "owner": subtitle_id in targets,
            }
            reason = _rejection_reason(edit, manifest, order, position, targets, neighbors, apply_threshold)
            if reason:
                rejected.append({**edit, "rejected": reason})
                continue
            previous = accepted.get(subtitle_id)
            if previous is None:
                accepted[subtitle_id] = edit
                continue
            keep, drop = (edit, previous) if (edit["confidence"], edit["owner"]) > (
                previous["confidence"], previous["owner"]) else (previous, edit)
            accepted[subtitle_id] = keep
            rejected.append({**drop, "rejected": "superseded"})
    return accepted, rejected


def _rejection_reason(
    edit: dict[str, Any],
    manifest: dict[str, Any],
    order: list[str],
    position: dict[str, int],
    targets: set[str],
    neighbors: set[str],
    apply_threshold: float,
) -> str:
    subtitle_id = edit["id"]
    if subtitle_id not in targets:
        if subtitle_id not in neighbors:
            return "not_in_request"
        if edit["error_type"] != "duplicate":
            return "neighbor_non_duplicate"
    if edit["error_type"] not in APPLY_ERROR_TYPES:
        return "error_type"
    if edit["confidence"] < apply_threshold:
        return "low_confidence"
    if KANA.search(edit["new"]):
        return "kana"
    index = position[subtitle_id]
    for neighbor_index in (index - 1, index + 1):
        if 0 <= neighbor_index < len(order):
            neighbor = manifest[order[neighbor_index]]
            if edit["new"] == str(neighbor.get("translated_subtitle", "") or "").strip():
                return "same_as_neighbor"
    return ""


def apply_edits(manifest: dict[str, Any], accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    branch = copy.deepcopy(manifest)
    for subtitle_id, edit in accepted.items():
        item = branch[subtitle_id]
        item["translated_subtitle"] = edit["new"]
        item["text_proofread"] = {
            "previous_translation": edit["old"],
            "error_type": edit["error_type"],
            "reason": edit["reason"],
            "confidence": edit["confidence"],
            "prompt_version": PROMPT_VERSION,
        }
    return branch


def resolve_input(work_paths: WorkPaths) -> tuple[str, Path]:
    if work_paths.mimo_proofread_manifest.exists():
        return "mimo", work_paths.mimo_proofread_manifest
    return "translated", work_paths.translated_manifest


def _usage_dict(usage: Any) -> dict[str, int]:
    input_tokens, output_tokens, reasoning_tokens = usage_tokens(usage)
    return {
        "prompt_tokens": int(input_tokens or 0),
        "completion_tokens": int(output_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
    }


def _call(client: Any, config: TextProofreadConfig, request: ProofreadRequest) -> dict[str, Any]:
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, max(1, config.max_attempts) + 1):
        request_started = time.perf_counter()
        usage = None
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=0.0,
                extra_body={"reasoning_effort": config.effort},
            )
            usage = getattr(response, "usage", None)
            content = response.choices[0].message.content or ""
            suggestions = parse_suggestions(content)
        except Exception as exc:  # noqa: BLE001 - recorded, retried, reported
            last_error = exc
            _record(config, request_started, usage, exc)
            retryable = is_transient_error(exc) or isinstance(exc, ValueError)
            if attempt >= config.max_attempts or not retryable:
                break
            time.sleep(config.retry_delay_s * attempt)
            continue
        _record(config, request_started, usage, None)
        served = client.last_served() if hasattr(client, "last_served") else {}
        return {
            "segment_id": request.segment_id, "key": request.key, "target_ids": request.target_ids,
            "neighbor_ids": request.neighbor_ids, "suggestions": suggestions, "usage": _usage_dict(usage),
            "elapsed_s": round(time.monotonic() - started, 2), "attempts": attempt, "served": served, "error": "",
        }
    return {
        "segment_id": request.segment_id, "key": request.key, "target_ids": request.target_ids,
        "neighbor_ids": request.neighbor_ids, "suggestions": [], "usage": {},
        "elapsed_s": round(time.monotonic() - started, 2), "attempts": config.max_attempts, "served": {},
        "error": f"{type(last_error).__name__}: {last_error}"[:300],
    }


def _record(config: TextProofreadConfig, started: float, usage: Any, error: Exception | None) -> None:
    input_tokens, output_tokens, reasoning_tokens = usage_tokens(usage)
    message = str(error or "").lower()
    record_network_request(
        provider="gemini-relay-pool",
        model=config.model,
        duration_s=time.perf_counter() - started,
        success=error is None,
        attempts=1,
        timeout="timeout" in message or "timed out" in message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        thinking_disabled=False,
        no_think_sent=False,
        extra_body_sent=True,
        error_type=type(error).__name__ if error is not None else None,
        cached_tokens=usage_cached_tokens(usage),
    )


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and not row.get("error") and row.get("key"):
            cached[str(row["key"])] = row
    return cached


def run_text_proofread(
    work_paths: WorkPaths,
    config: TextProofreadConfig,
    *,
    client: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    input_name, input_path = resolve_input(work_paths)
    manifest = read_json(input_path, default={})
    segments = read_json(work_paths.segments_manifest, default=[])
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"text-proofread input is missing or empty: {input_path}")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError(f"text-proofread needs the segments manifest: {work_paths.segments_manifest}")
    glossary = load_glossary(Path(config.glossary_xlsx)) if config.glossary_xlsx else []
    requests = build_requests(manifest, segments, glossary, config)

    work_paths.text_proofread_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_paths.text_proofread_dir / "requests.jsonl"
    cached = _load_checkpoint(checkpoint_path) if config.resume else {}
    if not config.resume:
        checkpoint_path.unlink(missing_ok=True)
    pending = [request for request in requests if request.key not in cached]
    results = [cached[request.key] for request in requests if request.key in cached]

    own_client = client is None
    relays: dict[str, Any] = {}
    if pending:
        if client is None:
            endpoints = endpoints_from_env(default_model=config.model, default_concurrency=config.concurrency)
            if not endpoints:
                raise RuntimeError("text-proofread needs a Gemini relay: set GEMINI_API_KEY (and GEMINI_BASE_URL)")
            client = GeminiRelayPool(endpoints, timeout=config.timeout)
        workers = int(getattr(client, "capacity", 0) or config.concurrency)
        done = len(results)
        try:
            with ThreadPoolExecutor(max(1, workers)) as executor, checkpoint_path.open("a", encoding="utf-8") as fh:
                futures = [executor.submit(_call, client, config, request) for request in pending]
                for future in as_completed(futures):
                    row = future.result()
                    results.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    if progress is not None:
                        progress(done, len(requests))
        finally:
            if hasattr(client, "stats"):
                relays = client.stats()
            if own_client and hasattr(client, "close"):
                client.close()

    order = {request.key: index for index, request in enumerate(requests)}
    results.sort(key=lambda row: order.get(row.get("key"), len(order)))
    succeeded = [row for row in results if not row.get("error")]
    failed = [row for row in results if row.get("error")]
    accepted, rejected = select_edits(manifest, succeeded, apply_threshold=config.apply_threshold)
    branch = apply_edits(manifest, accepted)

    if requests and not succeeded:
        status = "FAIL"
    elif failed:
        status = "WARN"
    else:
        status = "PASS"
    latencies = sorted(float(row.get("elapsed_s", 0.0) or 0.0) for row in succeeded) or [0.0]
    report = {
        "status": status,
        "prompt_version": PROMPT_VERSION,
        "effort": config.effort,
        "input": input_name,
        "input_path": str(input_path),
        "apply_threshold": config.apply_threshold,
        "request_count": len(requests),
        "cached_count": len(requests) - len(pending),
        "failed_count": len(failed),
        "failed_segments": [row.get("segment_id") for row in failed],
        "failures": [{"segment_id": row.get("segment_id"), "error": row.get("error")} for row in failed],
        "subtitle_count": len(branch),
        "applied_count": len(accepted),
        "rejected_count": len(rejected),
        "applied_by_type": _count_by(accepted.values(), "error_type"),
        "rejected_by_reason": _count_by(rejected, "rejected"),
        "wall_s": round(time.monotonic() - started, 1),
        "latency_p50_s": latencies[len(latencies) // 2],
        "latency_max_s": latencies[-1],
        "prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0) or 0) for row in succeeded),
        "reasoning_tokens": sum(int((row.get("usage") or {}).get("reasoning_tokens", 0) or 0) for row in succeeded),
        "relays": relays,
        "applied": sorted(accepted.values(), key=lambda edit: _sort_key(edit["id"])),
        "rejected": sorted(rejected, key=lambda edit: _sort_key(edit["id"])),
    }
    if status != "FAIL":
        write_json_atomic(work_paths.text_proofread_manifest, branch)
        work_paths.text_proofread_srt.write_text(to_srt(branch), encoding="utf-8")
    write_json_atomic(work_paths.text_proofread_report, report)
    return report


def _count_by(items: Any, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key, "") or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _sort_key(subtitle_id: str) -> tuple[int, str]:
    return (int(subtitle_id) if subtitle_id.isdigit() else 0, subtitle_id)


def config_from_args(args: Any) -> TextProofreadConfig:
    effort = str(getattr(args, "text_proofread_effort", DEFAULT_EFFORT) or DEFAULT_EFFORT)
    if effort not in EFFORTS:
        raise ValueError(f"--text-proofread-effort must be one of {', '.join(EFFORTS)}")
    return TextProofreadConfig(
        effort=effort,
        model=str(getattr(args, "text_proofread_model", "") or DEFAULT_MODEL),
        concurrency=max(1, int(getattr(args, "text_proofread_concurrency", DEFAULT_CONCURRENCY) or DEFAULT_CONCURRENCY)),
        timeout=max(30.0, float(getattr(args, "text_proofread_timeout", 600.0) or 600.0)),
        apply_threshold=min(1.0, max(0.0, float(
            getattr(args, "text_proofread_apply_threshold", DEFAULT_APPLY_THRESHOLD) or DEFAULT_APPLY_THRESHOLD))),
        program=str(getattr(args, "text_proofread_program", "") or "日语节目"),
        hosts=str(getattr(args, "text_proofread_hosts", "") or ""),
        glossary_xlsx=getattr(args, "glossary_xlsx", None) or None,
        resume=bool(getattr(args, "resume", True)) and not bool(getattr(args, "force", False)),
    )


def cmd_text_proofread(args: Any, work_paths: WorkPaths) -> int:
    from qwen_asr.artifact_state import ArtifactState

    config = config_from_args(args)
    config = dataclasses.replace(config, program_context=program_context_from_args(args, work_paths))
    state = ArtifactState(work_paths)
    previous = read_json(work_paths.text_proofread_report, default={})
    if (
        config.resume
        and state.is_complete("text-proofread")
        and not state.is_outdated("text-proofread")
        and isinstance(previous, dict)
        and previous.get("status") == "PASS"
        and previous.get("effort") == config.effort
    ):
        write_progress(work_paths, stage="text-proofread", status="skipped",
                       summary=f"text-proofread already complete: {previous.get('applied_count', 0)} lines changed")
        return 0

    def progress(done: int, total: int) -> None:
        write_progress(work_paths, stage="text-proofread", status="running", done=done, total=total,
                       summary=f"text-proofread {done}/{total}")

    report = run_text_proofread(work_paths, config, progress=progress)
    if report["status"] != "FAIL":
        # normalize / export skip when their outputs exist; they must be rebuilt from the new branch.
        state.delete_downstream_outputs("text-proofread")
    write_progress(
        work_paths,
        stage="text-proofread",
        status="completed" if report["status"] != "FAIL" else "failed",
        done=report["request_count"] - report["failed_count"],
        total=report["request_count"],
        summary=(
            f"text-proofread {report['status']}：改 {report['applied_count']} 行，"
            f"{report['failed_count']}/{report['request_count']} 个请求失败"
        ),
    )
    return 0 if report["status"] != "FAIL" else 1
