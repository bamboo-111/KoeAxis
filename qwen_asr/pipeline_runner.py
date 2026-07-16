from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qwen_asr.artifact_state import ArtifactState
from qwen_asr.content_quality import evaluate_content_conservation
from qwen_asr.final_quality import evaluate_final_quality
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress, write_progress
from qwen_asr.quality_suspects import apply_quality_diff_suspects_to_translated, apply_quality_suspects_to_translated
from qwen_asr.stages import stage_names_for_run
from qwen_asr.storage import read_json, write_json_atomic


StageHandler = Callable[[argparse.Namespace, WorkPaths], int]


@dataclass(frozen=True, slots=True)
class StageInvocation:
    name: str
    args: argparse.Namespace


class PipelineRunner:
    def __init__(self, work_paths: WorkPaths, handlers: dict[str, StageHandler]) -> None:
        self.work_paths = work_paths
        self.handlers = handlers
        self.state = ArtifactState(work_paths)

    def build_invocations(self, args: argparse.Namespace) -> list[StageInvocation]:
        names = stage_names_for_run(
            with_correct=bool(args.with_correct),
            with_align=bool(args.with_align),
            with_split=bool(args.with_split),
            with_translate=bool(args.with_translate),
            with_mimo_proofread=bool(getattr(args, "with_mimo_proofread", False)),
            with_normalize=bool(args.with_normalize),
        )
        invocations: list[StageInvocation] = []
        for name in names:
            stage_args = argparse.Namespace(**vars(args))
            if name == "correct":
                stage_args.batch_num = getattr(args, "correct_batch_num", 8)
            elif name == "align":
                stage_args.model = args.align_model
                stage_args.asr_reference_model = getattr(args, "model", None)
                stage_args.asr_reference_max_new_tokens = getattr(args, "max_new_tokens", 512)
                stage_args.asr_reference_language = getattr(args, "language", None)
                stage_args.cleanup_interval = getattr(args, "align_cleanup_interval", 4)
            elif name == "normalize":
                stage_args.source = getattr(args, "normalize_source", "auto")
            elif name == "quality-gate":
                stage_args.include_export = False
            invocations.append(StageInvocation(name=name, args=stage_args))
        return invocations

    def run(self, args: argparse.Namespace) -> int:
        for invocation in self.build_invocations(args):
            if invocation.name == "prepare" and not args.force and self.state.is_complete("prepare"):
                write_progress(
                    self.work_paths,
                    stage="prepare",
                    status="skipped",
                    summary="prepare artifacts already complete",
                )
                continue
            write_progress(
                self.work_paths,
                stage=invocation.name,
                status="running",
                summary=f"{invocation.name} started",
            )
            if invocation.name == "quality-gate":
                status = self._run_final_gate(include_export=False, require_srt=False)
            else:
                handler = self.handlers[invocation.name]
                status = handler(invocation.args, self.work_paths)
            if status == 0 and invocation.name == "translate":
                status = self._apply_quality_suspects_if_requested(args)
            if status == 0 and invocation.name == "mimo-proofread" and not self.state.is_complete("mimo-proofread"):
                status = 1
                write_progress(
                    self.work_paths,
                    stage=invocation.name,
                    status="failed",
                    summary="mimo-proofread quality gate failed",
                )
            if status == 0 and invocation.name == "export":
                status = self._run_final_gate(
                    include_export=True,
                    require_srt=str(getattr(args, "format", "srt")) in {"srt", "both"},
                )
            existing = read_progress(self.work_paths) or {}
            write_progress(
                self.work_paths,
                stage=invocation.name,
                status="completed" if status == 0 else "failed",
                done=existing.get("done"),
                total=existing.get("total"),
                current=existing.get("current", ""),
                summary=existing.get("summary") or f"{invocation.name} {'completed' if status == 0 else 'failed'}",
            )
            if status != 0:
                return status
            if invocation.name in {"align", "split", "mimo-proofread", "proofread-realign"}:
                self._record_content_gate(include_export=False)
        return 0

    def _apply_quality_suspects_if_requested(self, args: argparse.Namespace) -> int:
        if not bool(getattr(args, "with_mimo_proofread", False)):
            return 0
        ass_report_paths, diff_report_paths = self._resolve_quality_suspect_reports(args)
        if not ass_report_paths and not diff_report_paths:
            return 0
        translated = read_json(self.work_paths.translated_manifest, default={})
        if not isinstance(translated, dict) or not translated:
            write_progress(
                self.work_paths,
                stage="quality-suspects",
                status="failed",
                summary="translated_segments.json is missing or empty",
            )
            return 1
        aggregate = {
            "translated": translated,
            "report": {"source": "quality-suspects", "candidate_count": 0, "applied_count": 0, "reports": []},
        }
        for report_path in ass_report_paths:
            ass_report = read_json(report_path, default={})
            if not isinstance(ass_report, dict):
                return self._quality_suspect_failure(f"ASS quality report is invalid: {report_path}")
            result = apply_quality_suspects_to_translated(
                aggregate["translated"],
                ass_report,
                max_distance_ms=int(getattr(args, "quality_suspect_max_distance_ms", 8000)),
            )
            aggregate = self._merge_quality_suspect_result(
                aggregate,
                result,
                report_path=report_path,
                source="ass-quality",
            )
        for report_path in diff_report_paths:
            diff_report = read_json(report_path, default={})
            if not isinstance(diff_report, dict):
                return self._quality_suspect_failure(f"ASS quality diff report is invalid: {report_path}")
            result = apply_quality_diff_suspects_to_translated(
                aggregate["translated"],
                diff_report,
                max_distance_ms=int(getattr(args, "quality_suspect_max_distance_ms", 8000)),
            )
            aggregate = self._merge_quality_suspect_result(
                aggregate,
                result,
                report_path=report_path,
                source="ass-quality-diff",
            )
        write_json_atomic(self.work_paths.translated_manifest, aggregate["translated"])
        output_path = Path(
            str(getattr(args, "quality_suspect_report_output", "") or "")
            or self.work_paths.workdir / "reports" / "quality_suspects.json"
        )
        write_json_atomic(output_path, aggregate["report"])
        write_progress(
            self.work_paths,
            stage="quality-suspects",
            status="completed",
            summary=(
                "\u8d28\u91cf\u95e8\u7591\u70b9\u6ce8\u5165\uff1a"
                f"{aggregate['report']['applied_count']}/{aggregate['report']['candidate_count']}"
            ),
        )
        return 0

    def _quality_suspect_failure(self, summary: str) -> int:
        write_progress(
            self.work_paths,
            stage="quality-suspects",
            status="failed",
            summary=summary,
        )
        return 1

    @staticmethod
    def _merge_quality_suspect_result(
        aggregate: dict,
        result: dict,
        *,
        report_path: Path,
        source: str,
    ) -> dict:
        stage_report = dict(result.get("report", {}))
        stage_report["path"] = str(report_path)
        stage_report["source"] = source
        aggregate["translated"] = result["translated"]
        aggregate_report = aggregate["report"]
        aggregate_report["candidate_count"] = int(aggregate_report.get("candidate_count", 0)) + int(
            stage_report.get("candidate_count", 0)
        )
        aggregate_report["applied_count"] = int(aggregate_report.get("applied_count", 0)) + int(
            stage_report.get("applied_count", 0)
        )
        reports = aggregate_report.setdefault("reports", [])
        if isinstance(reports, list):
            reports.append(stage_report)
        return aggregate

    def _resolve_quality_suspect_reports(self, args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
        ass_reports: list[Path] = []
        diff_reports: list[Path] = []
        explicit_diff_report = str(getattr(args, "quality_suspect_ass_diff_report", "") or "").strip()
        if explicit_diff_report:
            diff_reports.append(Path(explicit_diff_report))
        explicit_report = str(getattr(args, "quality_suspect_ass_report", "") or "").strip()
        if explicit_report:
            ass_reports.append(Path(explicit_report))
        generated_reports = self._generate_quality_ass_reports(args)
        if generated_reports and bool(getattr(args, "quality_suspect_include_main_ass_report", True)):
            ass_reports.append(generated_reports[0])
        generated_diff = self._generate_quality_ass_diff_report(args, generated_reports)
        if generated_diff is not None:
            diff_reports.append(generated_diff)
        return ass_reports, diff_reports

    def _generate_quality_ass_reports(self, args: argparse.Namespace) -> list[Path]:
        from qwen_asr.ass_quality import build_ass_quality_report, render_markdown_report

        quality_ass = str(getattr(args, "quality_ass", "") or "").strip()
        if not quality_ass:
            return []
        source_values = [str(getattr(args, "quality_ass_source", "translated") or "translated")]
        diff_sources = [
            value.strip()
            for value in str(getattr(args, "quality_ass_diff_sources", "") or "").split(",")
            if value.strip()
        ]
        for source in diff_sources:
            if source not in source_values:
                source_values.append(source)
        output_paths: list[Path] = []
        for source in source_values:
            report = build_ass_quality_report(
                self.work_paths,
                ass_path=Path(quality_ass),
                source=source,
                offset_ms=getattr(args, "quality_ass_offset_ms", None),
                window_ms=int(getattr(args, "quality_ass_window_ms", 1200)),
                diagnostic_window_ms=int(getattr(args, "quality_ass_diagnostic_window_ms", 8000)),
                low_score_threshold=float(getattr(args, "quality_ass_low_score_threshold", 0.45)),
                fail_score_threshold=float(getattr(args, "quality_ass_fail_score_threshold", 0.20)),
                max_cases=int(getattr(args, "quality_ass_max_cases", 30)),
            )
            output_path = self.work_paths.workdir / "reports" / f"ass_quality.{source}.quality_suspects.json"
            write_json_atomic(output_path, report)
            markdown_path = output_path.with_suffix(".md")
            markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
            output_paths.append(output_path)
        return output_paths

    def _generate_quality_ass_diff_report(
        self,
        args: argparse.Namespace,
        generated_reports: list[Path],
    ) -> Path | None:
        diff_sources = [
            value.strip()
            for value in str(getattr(args, "quality_ass_diff_sources", "") or "").split(",")
            if value.strip()
        ]
        if len(diff_sources) < 2:
            return None
        from qwen_asr.ass_quality_diff import build_ass_quality_diff_report, render_markdown_report

        report_by_source = {path.name.split(".")[1]: path for path in generated_reports if path.name.startswith("ass_quality.")}
        report_specs: list[tuple[str | None, Path]] = []
        for source in diff_sources:
            path = report_by_source.get(source)
            if path is not None:
                report_specs.append((source, path))
        if len(report_specs) < 2:
            return None
        report = build_ass_quality_diff_report(
            report_specs,
            max_cases=int(getattr(args, "quality_ass_max_cases", 30)),
        )
        output_path = self.work_paths.workdir / "reports" / "ass_quality.diff.quality_suspects.json"
        write_json_atomic(output_path, report)
        markdown_path = output_path.with_suffix(".md")
        markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
        return output_path

    def _run_final_gate(self, *, include_export: bool, require_srt: bool) -> int:
        report = evaluate_final_quality(
            self.work_paths,
            include_export=include_export,
            require_srt=require_srt,
        )
        status = str(report.get("status", "WARN") or "WARN")
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        write_progress(
            self.work_paths,
            stage="quality-gate",
            status="completed" if status != "FAIL" else "failed",
            done=int(summary.get("pass_count", 0) or 0),
            total=(
                int(summary.get("pass_count", 0) or 0)
                + int(summary.get("warn_count", 0) or 0)
                + int(summary.get("fail_count", 0) or 0)
            ),
            summary=(
                f"聚合质量门 {status}："
                f"{summary.get('fail_count', 0)} FAIL，{summary.get('warn_count', 0)} WARN"
            ),
        )
        return 0 if status != "FAIL" else 1

    def _record_content_gate(self, *, include_export: bool) -> None:
        report = evaluate_content_conservation(self.work_paths, include_export=include_export)
        write_progress(
            self.work_paths,
            stage="content-quality",
            status="completed" if report["status"] != "FAIL" else "failed",
            summary=(
                f"内容守恒质量门 {report['status']}："
                f"{report['summary']['fail_count']} FAIL，{report['summary']['warn_count']} WARN"
            ),
        )
