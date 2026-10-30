from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qwen_asr.commands import cmd_align, cmd_prepare, cmd_split, cmd_transcribe
from qwen_asr.glossary import (
    GlossaryEntry,
    write_canonical_glossary_xlsx,
    write_normalized_glossary_xlsx,
)
from qwen_asr import history_glossary_ass as _ass
from qwen_asr import history_glossary_matching as _matching
from qwen_asr import history_glossary_rules as _rules
from qwen_asr.models import AlignedSegment, AlignedToken, AudioSegment, WorkPaths
from qwen_asr.optimizer_bridge import load_specific_asr_data
from qwen_asr.storage import append_jsonl, ensure_directory, read_json, write_json_atomic
from optimizer.llm_client import call_llm

LOGGER = logging.getLogger(__name__)

EPISODE_PATTERN = re.compile(r"#\s*(\d+)", re.IGNORECASE)
ASS_DIALOGUE_PATTERN = _ass.ASS_DIALOGUE_PATTERN
JP_TERMS = _rules.JP_TERMS
CN_TERMS = _rules.CN_TERMS
JP_CURATION_HINTS = _rules.JP_CURATION_HINTS
CN_CURATION_HINTS = _rules.CN_CURATION_HINTS
_normalize_glossary_text = _rules.normalize_glossary_text
_looks_like_glossary_candidate = _rules.looks_like_glossary_candidate
_guess_glossary_group = _rules.guess_glossary_group
_has_cjk = _rules.has_cjk
_has_kana = _rules.has_kana
_contains_ascii_word = _rules.contains_ascii_word
_is_glossary_like_pair = _rules.is_glossary_like_pair
_is_curated_priority = _rules.is_curated_priority
_is_llm_glossary_entry_allowed = _rules.is_llm_glossary_entry_allowed
_looks_like_sentence_text = _rules.looks_like_sentence_text
_looks_like_contextual_role_phrase = _rules.looks_like_contextual_role_phrase
_score_to_level = _rules.score_to_level
_clean_ass_text = _ass.clean_ass_text
_escape_ass_text = _ass.escape_ass_text
_ass_time_to_ms = _ass.ass_time_to_ms
_ms_to_ass_time = _ass.ms_to_ass_time
_interval_overlap_score = _matching.interval_overlap_score
_boundary_score = _matching.boundary_score
_overlap_ms = _matching.overlap_ms


@dataclass(frozen=True)
class HistoryEpisodePair:
    episode_id: str
    media_path: Path
    ass_path: Path


@dataclass(frozen=True)
class AssDialogue:
    start_ms: int
    end_ms: int
    style: str
    text: str


@dataclass(frozen=True)
class MatchResult:
    episode_id: str
    media_path: str
    ass_path: str
    ass_start_ms: int
    ass_end_ms: int
    ass_text: str
    source_text: str
    source_kind: str
    source_start_ms: int
    source_end_ms: int
    matched_segment_count: int
    score: float
    level: str
    time_overlap_score: float
    boundary_score: float
    length_ratio_score: float
    merge_penalty: float
    token_coverage_score: float
    reasons: list[str]


def cmd_history_glossary(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    history_dir = Path(args.history_dir).resolve()
    output_xlsx = Path(args.output_xlsx).resolve()
    review_ass = Path(args.review_ass).resolve() if args.review_ass else work_paths.workdir / "review.ass"
    export_matches = Path(args.export_matches).resolve() if args.export_matches else work_paths.workdir / "matches.jsonl"
    summary_path = work_paths.workdir / "match-summary.json"

    ensure_directory(work_paths.workdir)
    pairs = discover_history_pairs(history_dir, episode_filter=args.episode_filter)
    if not pairs:
        raise RuntimeError(f"No MP3 + ASS pairs found in: {history_dir}")

    if export_matches.exists():
        export_matches.unlink()

    all_matches: list[MatchResult] = []
    episode_summaries: list[dict[str, object]] = []

    for pair in pairs:
        LOGGER.info("Processing episode %s", pair.episode_id)
        episode_workdir = work_paths.workdir / f"episode-{pair.episode_id}"
        ensure_directory(episode_workdir)
        episode_paths = WorkPaths.from_workdir(episode_workdir)
        _run_episode_pipeline(pair, args, episode_paths)

        dialogues = parse_ass_dialogues(pair.ass_path)
        split_data = load_specific_asr_data(episode_paths, source="split", optimizer_root=Path(args.optimizer_root))
        if split_data is None or not getattr(split_data, "segments", None):
            raise RuntimeError(f"split stage did not produce usable data for episode {pair.episode_id}")
        aligned_segments = _load_aligned_segments(episode_paths)
        episode_matches = match_dialogues_to_asr(
            episode_id=pair.episode_id,
            media_path=pair.media_path,
            ass_path=pair.ass_path,
            dialogues=dialogues,
            split_segments=list(split_data.segments),
            aligned_segments=aligned_segments,
            min_match_score=float(args.min_match_score),
        )
        for item in episode_matches:
            append_jsonl(export_matches, item.__dict__)
        all_matches.extend(episode_matches)
        episode_summaries.append(_build_episode_summary(pair, episode_paths, episode_matches))

    if args.extractor_mode == "llm":
        if not args.llm_model or not args.llm_base_url or not args.llm_api_key:
            raise RuntimeError("LLM extractor mode requires --llm-model, --llm-base-url, and --llm-api-key")
        glossary_entries = extract_glossary_entries_with_llm(
            all_matches,
            min_match_score=float(args.min_match_score),
            llm_model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            disable_thinking=args.disable_thinking,
            llm_extra_body_json=args.llm_extra_body_json,
            timeout=args.timeout,
        )
    else:
        glossary_entries = extract_glossary_entries(
            all_matches,
            min_match_score=float(args.min_match_score),
            min_term_frequency=int(args.min_term_frequency),
        )
    glossary_result = write_canonical_glossary_xlsx(glossary_entries, output_xlsx)
    normalized_output_path = ""
    if glossary_result.entry_count > 0:
        normalized_result = write_normalized_glossary_xlsx(glossary_result.output_path)
        normalized_output_path = str(normalized_result.output_path)
    export_review_ass(review_ass, all_matches)

    summary_payload = {
        "history_dir": str(history_dir),
        "episode_count": len(pairs),
        "match_count": len(all_matches),
        "glossary_entries": glossary_result.entry_count,
        "output_xlsx": str(glossary_result.output_path),
        "normalized_output_xlsx": normalized_output_path,
        "review_ass": str(review_ass),
        "matches_jsonl": str(export_matches),
        "episodes": episode_summaries,
    }
    write_json_atomic(summary_path, summary_payload)
    LOGGER.info("History glossary written: %s", glossary_result.output_path)
    if normalized_output_path:
        LOGGER.info("Normalized glossary written: %s", normalized_output_path)
    else:
        LOGGER.info("No glossary entries passed filters; normalized glossary skipped.")
    LOGGER.info("Review ASS written: %s", review_ass)
    return 0


def discover_history_pairs(history_dir: Path, episode_filter: str | None = None) -> list[HistoryEpisodePair]:
    if not history_dir.exists():
        raise FileNotFoundError(f"History directory not found: {history_dir}")

    media_by_episode: dict[str, Path] = {}
    ass_by_episode: dict[str, Path] = {}
    for path in history_dir.iterdir():
        if not path.is_file():
            continue
        episode_id = _extract_episode_id(path.name)
        if not episode_id:
            continue
        if episode_filter and episode_filter not in path.name and episode_filter not in episode_id:
            continue
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            media_by_episode[episode_id] = path.resolve()
        elif suffix == ".ass":
            ass_by_episode[episode_id] = path.resolve()

    pairs = [
        HistoryEpisodePair(
            episode_id=episode_id,
            media_path=media_by_episode[episode_id],
            ass_path=ass_by_episode[episode_id],
        )
        for episode_id in sorted(set(media_by_episode) & set(ass_by_episode), key=lambda item: int(item))
    ]
    return pairs


def parse_ass_dialogues(path: Path) -> list[AssDialogue]:
    return _ass.parse_ass_dialogues(path, AssDialogue)


def match_dialogues_to_asr(
    *,
    episode_id: str,
    media_path: Path,
    ass_path: Path,
    dialogues: list[AssDialogue],
    split_segments: list[object],
    aligned_segments: list[AlignedSegment],
    min_match_score: float,
) -> list[MatchResult]:
    token_pool = [
        token
        for segment in aligned_segments
        if segment.status == "completed"
        for token in segment.tokens
        if str(token.text).strip()
    ]
    matches: list[MatchResult] = []
    for dialogue in dialogues:
        best = _best_split_match(
            episode_id=episode_id,
            media_path=media_path,
            ass_path=ass_path,
            dialogue=dialogue,
            split_segments=split_segments,
        )
        if best is None or best.score < 0.5:
            fallback = _match_from_tokens(
                episode_id=episode_id,
                media_path=media_path,
                ass_path=ass_path,
                dialogue=dialogue,
                tokens=token_pool,
            )
            if fallback is not None and (best is None or fallback.score > best.score):
                best = fallback
        if best is None:
            best = MatchResult(
                episode_id=episode_id,
                media_path=str(media_path),
                ass_path=str(ass_path),
                ass_start_ms=dialogue.start_ms,
                ass_end_ms=dialogue.end_ms,
                ass_text=dialogue.text,
                source_text="",
                source_kind="missing",
                source_start_ms=dialogue.start_ms,
                source_end_ms=dialogue.end_ms,
                matched_segment_count=0,
                score=0.0,
                level="low",
                time_overlap_score=0.0,
                boundary_score=0.0,
                length_ratio_score=0.0,
                merge_penalty=0.4,
                token_coverage_score=0.0,
                reasons=["no candidate"],
            )
        level = _score_to_level(best.score, min_match_score)
        matches.append(
            MatchResult(
                **{
                    **best.__dict__,
                    "level": level,
                }
            )
        )
    return matches


def extract_glossary_entries(
    matches: Iterable[MatchResult],
    *,
    min_match_score: float,
    min_term_frequency: int,
) -> list[GlossaryEntry]:
    grouped: dict[str, list[MatchResult]] = defaultdict(list)
    for item in matches:
        if item.level != "high" or item.score < min_match_score:
            continue
        source = _normalize_glossary_text(item.source_text)
        target = _normalize_glossary_text(item.ass_text)
        if not _looks_like_glossary_candidate(source, target, item):
            continue
        if not _is_glossary_like_pair(source, target):
            continue
        grouped[source].append(
            MatchResult(
                **{
                    **item.__dict__,
                    "source_text": source,
                    "ass_text": target,
                }
            )
        )

    entries: list[GlossaryEntry] = []
    for source, items in sorted(grouped.items()):
        target_counter = Counter(item.ass_text for item in items if item.ass_text)
        if not target_counter:
            continue
        target, frequency = target_counter.most_common(1)[0]
        curated = _is_curated_priority(source, target, items)
        if frequency < min_term_frequency and not curated:
            continue
        alternates = [candidate for candidate, _ in target_counter.most_common() if candidate != target]
        notes = [
            f"freq={frequency}",
            f"episodes={','.join(sorted({item.episode_id for item in items}))}",
        ]
        if curated:
            notes.append("curated=priority")
        if alternates:
            notes.append("alts=" + " | ".join(alternates[:3]))
        entries.append(
            GlossaryEntry(
                group=_guess_glossary_group(source, target),
                source=source,
                target=target,
                note="; ".join(notes),
            )
        )
    return entries


def extract_glossary_entries_with_llm(
    matches: Iterable[MatchResult],
    *,
    min_match_score: float,
    llm_model: str,
    base_url: str,
    api_key: str,
    disable_thinking: bool,
    llm_extra_body_json: str | None,
    timeout: float,
) -> list[GlossaryEntry]:
    candidates: list[dict[str, object]] = []
    for item in matches:
        if item.level not in {"high", "medium"}:
            continue
        if item.score < max(0.6, min_match_score - 0.12):
            continue
        source = _normalize_glossary_text(item.source_text)
        target = _normalize_glossary_text(item.ass_text)
        if not source or not target:
            continue
        candidates.append(
            {
                "episode_id": item.episode_id,
                "score": item.score,
                "source": source,
                "target": target,
                "reasons": item.reasons,
            }
        )

    if not candidates:
        return []

    prompt = _build_glossary_extraction_prompt(candidates[:160])
    extra_body = _parse_history_llm_extra_body(llm_extra_body_json)
    response = call_llm(
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract a reusable translation glossary from matched Japanese/Chinese subtitle pairs. "
                    "Keep only names, show titles, segment titles, fixed slogans, role labels, and highly reusable short phrases. "
                    "Reject ordinary full-sentence translations, context-specific lines, and noisy ASR fragments. "
                    "Return strict JSON with shape {\"entries\": [{\"group\": str, \"source\": str, \"target\": str, \"note\": str}]}."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=llm_model,
        temperature=0.1,
        base_url=base_url,
        api_key=api_key,
        disable_thinking=disable_thinking,
        require_json=True,
        llm_extra_body=extra_body,
        timeout=timeout,
    )
    payload = json.loads(response.choices[0].message.content)
    rows = payload.get("entries", []) if isinstance(payload, dict) else []
    entries: list[GlossaryEntry] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = GlossaryEntry(
            group=str(row.get("group", "fixed_phrases")).strip() or "fixed_phrases",
            source=_normalize_glossary_text(str(row.get("source", ""))),
            target=_normalize_glossary_text(str(row.get("target", ""))),
            note=str(row.get("note", "")).strip(),
        )
        if not entry.source or not entry.target:
            continue
        if not _is_llm_glossary_entry_allowed(entry):
            continue
        key = (entry.group, entry.source, entry.target)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    return entries


def export_review_ass(path: Path, matches: Iterable[MatchResult]) -> None:
    _ass.export_review_ass(path, matches, ensure_directory=ensure_directory)


def _run_episode_pipeline(pair: HistoryEpisodePair, args: argparse.Namespace, work_paths: WorkPaths) -> None:
    prepare_args = argparse.Namespace(
        workdir=str(work_paths.workdir),
        log_level=args.log_level,
        resume=True,
        force=False,
        skip_preflight=False,
        dry_run_check=False,
        media=str(pair.media_path),
        video=None,
        denoise=args.denoise,
        denoise_level=args.denoise_level,
        denoise_backend=getattr(args, "denoise_backend", "mdx_net"),
        denoise_profile=getattr(args, "denoise_profile", "strong"),
        mdx_model=getattr(args, "mdx_model", "UVR-MDX-NET-Inst_HQ_3.onnx"),
        mdx_model_dir=getattr(args, "mdx_model_dir", None),
        vad_backend=getattr(args, "vad_backend", "pyannote_onnx_v3"),
        vad_threshold=getattr(args, "vad_threshold", 0.5),
        vad_onset=getattr(args, "vad_onset", 0.5),
        vad_offset=getattr(args, "vad_offset", 0.35),
        vad_min_speech_ms=getattr(args, "vad_min_speech_ms", 180),
        vad_min_silence_ms=getattr(args, "vad_min_silence_ms", 250),
        vad_speech_pad_ms=getattr(args, "vad_speech_pad_ms", 120),
        pyannote_onnx_model=getattr(args, "pyannote_onnx_model", "segmentation-3.0"),
        max_segment_seconds=args.max_segment_seconds,
        min_segment_seconds=args.min_segment_seconds,
        preferred_silence_ms=args.preferred_silence_ms,
        min_silence_ms=args.min_silence_ms,
        padding_ms=args.padding_ms,
        overlap_ms=args.overlap_ms,
        eager_segment_export=False,
    )
    cmd_prepare(prepare_args, work_paths)

    transcribe_args = argparse.Namespace(
        workdir=str(work_paths.workdir),
        log_level=args.log_level,
        resume=True,
        force=False,
        skip_preflight=False,
        dry_run_check=False,
        model_cache_dir=args.model_cache_dir,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_raw_model_output=False,
        local_files_only=args.local_files_only,
        model=args.model,
        batch_size=None,
        batch_mode="adaptive",
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
        profile_batches=True,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
    )
    cmd_transcribe(transcribe_args, work_paths)

    align_args = argparse.Namespace(
        workdir=str(work_paths.workdir),
        log_level=args.log_level,
        resume=True,
        force=False,
        skip_preflight=False,
        dry_run_check=False,
        model_cache_dir=args.model_cache_dir,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_raw_model_output=False,
        local_files_only=args.local_files_only,
        model=args.align_model,
        cleanup_interval=args.align_cleanup_interval,
    )
    cmd_align(align_args, work_paths)

    split_args = argparse.Namespace(
        workdir=str(work_paths.workdir),
        log_level=args.log_level,
        resume=True,
        force=False,
        skip_preflight=False,
        dry_run_check=False,
        optimizer_root=args.optimizer_root,
        thread_num=1,
        max_word_count_cjk=args.max_word_count_cjk,
        max_word_count_english=args.max_word_count_english,
        prompt_limit_ratio=args.prompt_limit_ratio,
        split_mode=getattr(args, "split_mode", "rule"),
        llm_model=None,
        llm_base_url=None,
        llm_api_key=None,
        disable_thinking=False,
        llm_extra_body_json=None,
        timeout=args.timeout,
    )
    cmd_split(split_args, work_paths)


def _build_episode_summary(
    pair: HistoryEpisodePair,
    work_paths: WorkPaths,
    matches: list[MatchResult],
) -> dict[str, object]:
    segment_payload = read_json(work_paths.segments_manifest, default=[])
    segments = [AudioSegment(**item) for item in segment_payload] if isinstance(segment_payload, list) else []
    warnings = summarize_segment_warnings(segments)
    profile = read_json(work_paths.transcribe_profile_path, default={})
    if isinstance(profile, dict):
        oom_retry_count = int(((profile.get("summary") or {}) if isinstance(profile.get("summary"), dict) else {}).get("oom_retry_count", 0))
        if oom_retry_count > 0:
            warnings.append(f"oom retries={oom_retry_count}")
    counts = Counter(item.level for item in matches)
    return {
        "episode_id": pair.episode_id,
        "media_path": str(pair.media_path),
        "ass_path": str(pair.ass_path),
        "matches": len(matches),
        "high": counts.get("high", 0),
        "medium": counts.get("medium", 0),
        "low": counts.get("low", 0),
        "warnings": warnings,
        "workdir": str(work_paths.workdir),
    }


def summarize_segment_warnings(segments: list[AudioSegment]) -> list[str]:
    if not segments:
        return ["no segments"]
    durations = [max(0.0, float(segment.duration)) for segment in segments]
    total_duration = sum(durations)
    long_share = sum(1 for value in durations if value >= 100.0) / len(durations)
    near_limit_share = sum(1 for value in durations if value >= 108.0) / len(durations)
    audio_minutes = total_duration / 60.0 if total_duration > 0 else 0.0
    density = len(durations) / audio_minutes if audio_minutes > 0 else 0.0
    warnings: list[str] = []
    if long_share >= 0.25:
        warnings.append(f"long segment share={long_share:.2f}")
    if near_limit_share >= 0.15:
        warnings.append(f"near limit share={near_limit_share:.2f}")
    if density >= 4.0:
        warnings.append(f"segment density high={density:.2f}/min")
    elif density > 0 and density <= 0.35:
        warnings.append(f"segment density low={density:.2f}/min")
    return warnings


def _load_aligned_segments(work_paths: WorkPaths) -> list[AlignedSegment]:
    payload = read_json(work_paths.aligned_manifest, default=[])
    if not isinstance(payload, list):
        return []
    result: list[AlignedSegment] = []
    for item in payload:
        tokens = [AlignedToken(**token) for token in item.get("tokens", [])]
        clone = dict(item)
        clone["tokens"] = tokens
        result.append(AlignedSegment(**clone))
    return result


def _best_split_match(
    *,
    episode_id: str,
    media_path: Path,
    ass_path: Path,
    dialogue: AssDialogue,
    split_segments: list[object],
) -> MatchResult | None:
    positions = [
        index
        for index, segment in enumerate(split_segments)
        if getattr(segment, "end_time", 0) >= dialogue.start_ms - 2500
        and getattr(segment, "start_time", 0) <= dialogue.end_ms + 2500
    ]
    if not positions:
        return None

    best: MatchResult | None = None
    max_window = min(4, len(positions))
    for start_offset, _position in enumerate(positions):
        for window_size in range(1, max_window + 1):
            window_positions = positions[start_offset : start_offset + window_size]
            if len(window_positions) < window_size:
                continue
            if any(window_positions[index + 1] != window_positions[index] + 1 for index in range(len(window_positions) - 1)):
                continue
            window = [split_segments[index] for index in window_positions]
            candidate = _score_candidate(
                episode_id=episode_id,
                media_path=media_path,
                ass_path=ass_path,
                dialogue=dialogue,
                source_text=" ".join(str(getattr(item, "text", "")).strip() for item in window).strip(),
                source_kind="split",
                source_start_ms=int(getattr(window[0], "start_time", dialogue.start_ms)),
                source_end_ms=int(getattr(window[-1], "end_time", dialogue.end_ms)),
                matched_segment_count=len(window),
                covered_duration=sum(
                    _overlap_ms(
                        dialogue.start_ms,
                        dialogue.end_ms,
                        int(getattr(item, "start_time", dialogue.start_ms)),
                        int(getattr(item, "end_time", dialogue.end_ms)),
                    )
                    for item in window
                ),
            )
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def _match_from_tokens(
    *,
    episode_id: str,
    media_path: Path,
    ass_path: Path,
    dialogue: AssDialogue,
    tokens: list[AlignedToken],
) -> MatchResult | None:
    matched = [
        token
        for token in tokens
        if int(round(token.end_time * 1000)) >= dialogue.start_ms - 800
        and int(round(token.start_time * 1000)) <= dialogue.end_ms + 800
    ]
    if not matched:
        return None
    source_text = "".join(token.text for token in matched).strip()
    source_start_ms = int(round(matched[0].start_time * 1000))
    source_end_ms = int(round(matched[-1].end_time * 1000))
    return _score_candidate(
        episode_id=episode_id,
        media_path=media_path,
        ass_path=ass_path,
        dialogue=dialogue,
        source_text=source_text,
        source_kind="aligned_tokens",
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        matched_segment_count=len(matched),
        covered_duration=sum(
            _overlap_ms(
                dialogue.start_ms,
                dialogue.end_ms,
                int(round(token.start_time * 1000)),
                int(round(token.end_time * 1000)),
            )
            for token in matched
        ),
    )


def _score_candidate(
    *,
    episode_id: str,
    media_path: Path,
    ass_path: Path,
    dialogue: AssDialogue,
    source_text: str,
    source_kind: str,
    source_start_ms: int,
    source_end_ms: int,
    matched_segment_count: int,
    covered_duration: int,
) -> MatchResult:
    scoring = _matching.score_candidate_payload(
        dialogue=dialogue,
        source_text=source_text,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        matched_segment_count=matched_segment_count,
        covered_duration=covered_duration,
        normalize_text=_normalize_glossary_text,
    )
    return MatchResult(
        episode_id=episode_id,
        media_path=str(media_path),
        ass_path=str(ass_path),
        ass_start_ms=dialogue.start_ms,
        ass_end_ms=dialogue.end_ms,
        ass_text=dialogue.text,
        source_text=source_text,
        source_kind=source_kind,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        matched_segment_count=matched_segment_count,
        score=scoring["score"],
        level="low",
        time_overlap_score=scoring["time_overlap_score"],
        boundary_score=scoring["boundary_score"],
        length_ratio_score=scoring["length_ratio_score"],
        merge_penalty=scoring["merge_penalty"],
        token_coverage_score=scoring["token_coverage_score"],
        reasons=scoring["reasons"],
    )


def _extract_episode_id(name: str) -> str | None:
    match = EPISODE_PATTERN.search(name)
    if not match:
        return None
    return str(int(match.group(1)))


def _length_ratio_score(chinese_text: str, source_text: str) -> float:
    return _matching.length_ratio_score(chinese_text, source_text, normalize_text=_normalize_glossary_text)


def _build_glossary_extraction_prompt(candidates: list[dict[str, object]]) -> str:
    return (
        "From the following candidate subtitle matches, extract only reusable glossary entries.\n"
        "Allowed groups: names, show_terms, fixed_phrases.\n"
        "Prefer entries with high score and short reusable wording.\n"
        "Do not include ordinary sentence translations.\n"
        "Keep only stable entities or stable labels: person names, band names, show titles, segment titles, venue names, song titles, hashtags, and highly repeated fixed catchphrases.\n"
        "Do not output contextual role-address phrases or temporary descriptions such as '导演朱李', '今天的XX', '受伤的手指', or sentence fragments that only work in one line.\n"
        "If the Chinese target contains grammatical sentence structure, discard it.\n"
        "Prefer the base term itself over decorated variants with particles or modifiers.\n"
        "Return as few entries as necessary; precision is more important than recall.\n\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def _parse_history_llm_extra_body(value: str | None) -> dict | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("llm_extra_body_json must be a JSON object.")
    return parsed


__all__ = [
    "AssDialogue",
    "HistoryEpisodePair",
    "MatchResult",
    "cmd_history_glossary",
    "discover_history_pairs",
    "export_review_ass",
    "extract_glossary_entries",
    "match_dialogues_to_asr",
    "parse_ass_dialogues",
    "summarize_segment_warnings",
]
