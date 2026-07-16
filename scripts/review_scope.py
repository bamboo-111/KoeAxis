from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "reports"

COMMAND_TEST_MAP: dict[str, tuple[str, ...]] = {
    "qwen_asr/commands/align.py": (
        "tests/test_pipeline_runner.py",
        "tests/test_align_cleanup.py",
        "tests/test_align_quality.py",
    ),
    "qwen_asr/commands/correct.py": (
        "tests/test_cli_help.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/export.py": (
        "tests/test_export_paths.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/normalize.py": (
        "tests/test_export_paths.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/preflight.py": (
        "tests/test_cli_help.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/prepare.py": (
        "tests/test_cli_help.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/quality.py": (
        "tests/test_final_quality.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/run.py": (
        "tests/test_cli_help.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/split.py": (
        "tests/test_optimizer_bridge_timing.py",
        "tests/test_pipeline_runner.py",
    ),
    "qwen_asr/commands/transcribe.py": (
        "tests/test_pipeline_runner.py",
        "tests/test_transcribe_profile.py",
    ),
    "qwen_asr/commands/translate.py": (
        "tests/test_pipeline_runner.py",
        "tests/test_translator_suspects.py",
    ),
    "qwen_asr/optimizer_bridge_stages.py": (
        "tests/test_optimizer_bridge_timing.py",
        "tests/test_pipeline_runner.py",
    ),
}

GROUPS: dict[str, dict[str, str]] = {
    "asr_align_runtime": {
        "title": "ASR, align, and runtime infrastructure",
        "purpose": "Model runtime, transcription, alignment, artifact state, and core stage behavior.",
        "rollback": "Restore edited runtime files from their topic backups; newly created modules remain review-only until explicit deletion approval.",
    },
    "split_readability": {
        "title": "Split and readability protection",
        "purpose": "Rule split boundaries, timing allocation, display duration, and protected short-response behavior.",
        "rollback": "Restore splitter and prompt files from their topic backups together with the matching split tests.",
    },
    "translation_mimo": {
        "title": "Translation and MiMo suspects-only flow",
        "purpose": "Translation guards plus MiMo candidate, request, checkpoint, application, audio, and output boundaries.",
        "rollback": "Restore translator/MiMo files and their paired tests from the corresponding extraction backups.",
    },
    "quality_realign": {
        "title": "Proofread realignment and quality gates",
        "purpose": "Content, ASS, final-quality, proofread-realign, normalize, and export protection.",
        "rollback": "Restore quality and realignment modules with their tests; do not bypass the pre-export quality gate.",
    },
    "mfa_diagnostics_benchmarks": {
        "title": "MFA, diagnostics, tools, and benchmarks",
        "purpose": "Experimental MFA boundaries, diagnostic tools, benchmark definitions, and reproducibility metadata.",
        "rollback": "Restore source/report files from backups; local environments, corpora, models, and run artifacts remain untouched.",
    },
    "cli_web_wiring": {
        "title": "CLI, WebUI, and pipeline wiring",
        "purpose": "Argument parsing, command dispatch, PipelineRunner orchestration, Web command construction, and status presentation.",
        "rollback": "Restore CLI/Web/pipeline files as one wiring set, then rerun CLI-help, WebUI, and PipelineRunner tests.",
    },
    "tests_docs_acceptance": {
        "title": "Tests, documentation, configuration, and acceptance",
        "purpose": "Regression coverage, developer tooling, documentation, inventory, and acceptance evidence.",
        "rollback": "Restore edited documents/configuration from backups; retain generated evidence until its owning change is reviewed.",
    },
    "unclassified": {
        "title": "Unclassified review paths",
        "purpose": "Paths that do not match an established source, test, documentation, or local-state boundary.",
        "rollback": "No action is authorized. Inspect origin and purpose before editing, ignoring, moving, archiving, or deleting.",
    },
}


def run_git(args: list[str], root: Path = ROOT) -> list[str]:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return [line for line in completed.stdout.splitlines() if line]


def collect_review_paths(root: Path = ROOT) -> tuple[list[str], dict[str, str]]:
    modified = run_git(["diff", "--name-only"], root)
    untracked = run_git(["ls-files", "--others", "--exclude-standard"], root)
    states = {path: "modified" for path in modified}
    states.update({path: "untracked" for path in untracked})
    return sorted(states), states


def collect_test_paths(root: Path = ROOT) -> set[str]:
    tracked = run_git(["ls-files", "tests/test_*.py"], root)
    untracked = run_git(["ls-files", "--others", "--exclude-standard", "tests/test_*.py"], root)
    return set(tracked) | set(untracked)


def is_new_production_module(path: str) -> bool:
    return (
        path.endswith(".py")
        and (path.startswith("qwen_asr/") or path.startswith("optimizer/"))
        and not path.endswith("/__init__.py")
        and "/__pycache__/" not in path
    )


def build_source_test_mapping(untracked_paths: list[str], test_paths: set[str]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for production_path in sorted(path for path in untracked_paths if is_new_production_module(path)):
        direct = f"tests/test_{Path(production_path).stem}.py"
        if direct in test_paths:
            mapped_tests = [direct]
            coverage_kind = "direct"
            reason = "A same-name focused test module exists."
        else:
            mapped_tests = [path for path in COMMAND_TEST_MAP.get(production_path, ()) if path in test_paths]
            coverage_kind = "integration" if mapped_tests else "unmapped"
            reason = (
                "Command handler behavior is covered through CLI, PipelineRunner, or stage-specific integration tests."
                if mapped_tests
                else "No focused or approved integration-test mapping was found."
            )
        records.append(
            {
                "production_path": production_path,
                "test_paths": mapped_tests,
                "coverage_kind": coverage_kind,
                "status": "MAPPED" if mapped_tests else "UNMAPPED",
                "reason": reason,
            }
        )
    return {
        "production_module_count": len(records),
        "mapped_count": sum(record["status"] == "MAPPED" for record in records),
        "unmapped_count": sum(record["status"] == "UNMAPPED" for record in records),
        "records": records,
    }


def group_for_path(path: str) -> str:
    lower = path.lower()
    name = Path(lower).name

    if lower.startswith(("docs/", "reports/", "scripts/", "samples/")) or name in {
        ".gitignore",
        "readme.md",
        "pyproject.toml",
        "requirements-dev.txt",
        "requirements.txt",
    }:
        return "tests_docs_acceptance"
    if lower.startswith(("benchmarks/", "tools/")):
        return "mfa_diagnostics_benchmarks"
    if lower.startswith("qwen_asr/web/") or lower in {
        "qwen_asr/cli.py",
        "qwen_asr/pipeline_runner.py",
        "qwen_asr/commands/__init__.py",
        "qwen_asr/commands/stages.py",
    }:
        return "cli_web_wiring"
    if lower.startswith("qwen_asr/commands/"):
        stem = Path(lower).stem
        if stem in {"prepare", "transcribe", "align"}:
            return "asr_align_runtime"
        if stem == "split":
            return "split_readability"
        if stem in {"translate", "mimo_proofread"}:
            return "translation_mimo"
        if stem in {"quality", "proofread_realign", "normalize", "export"}:
            return "quality_realign"
        return "cli_web_wiring"
    if lower.startswith("tests/"):
        if any(token in name for token in ("split", "readability", "token_boundary")):
            return "split_readability"
        if any(token in name for token in ("mimo", "translator", "translation")):
            return "translation_mimo"
        if any(token in name for token in ("quality", "proofread", "export", "content")):
            return "quality_realign"
        if any(token in name for token in ("mfa", "diagnose", "audit", "baseline", "tuning")):
            return "mfa_diagnostics_benchmarks"
        if any(token in name for token in ("cli", "webui", "pipeline")):
            return "cli_web_wiring"
        if any(token in name for token in ("align", "asr", "artifact", "model_runtime", "transcribe")):
            return "asr_align_runtime"
        return "tests_docs_acceptance"
    if lower.startswith("optimizer/"):
        return "translation_mimo" if "translat" in lower else "split_readability"
    if lower.startswith("qwen_asr/"):
        if any(token in lower for token in ("mfa", "diagnose", "audit", "baseline", "tuning_matrix", "ass_quality_diff")):
            return "mfa_diagnostics_benchmarks"
        if any(token in lower for token in ("mimo", "quality_suspects")):
            return "translation_mimo"
        if any(token in lower for token in ("quality", "proofread_realign", "ass_quality", "content_quality")):
            return "quality_realign"
        return "asr_align_runtime"
    return "unclassified"


def build_change_groups(review_paths: list[str], states: dict[str, str]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in GROUPS}
    for path in review_paths:
        grouped[group_for_path(path)].append({"path": path, "state": states[path]})
    groups = []
    for group_id, metadata in GROUPS.items():
        files = grouped[group_id]
        if not files:
            continue
        groups.append(
            {
                "id": group_id,
                **metadata,
                "file_count": len(files),
                "files": files,
            }
        )
    return {
        "review_path_count": len(review_paths),
        "group_count": len(groups),
        "unclassified_count": len(grouped["unclassified"]),
        "groups": groups,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_mapping_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Source And Test Mapping",
        "",
        f"Production modules: `{payload['production_module_count']}`",
        f"Mapped: `{payload['mapped_count']}`",
        f"Unmapped: `{payload['unmapped_count']}`",
        "",
        "| Production module | Coverage | Tests | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for record in payload["records"]:
        tests = "<br>".join(f"`{item}`" for item in record["test_paths"])
        lines.append(
            f"| `{record['production_path']}` | {record['coverage_kind']} | {tests} | "
            f"{record['status']} | {record['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_groups_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Review Change Groups",
        "",
        "This is a review and rollback plan only. No file has been staged, committed, pushed, moved, archived, or deleted.",
        "",
        f"Review paths: `{payload['review_path_count']}`",
        f"Groups: `{payload['group_count']}`",
        f"Unclassified paths: `{payload['unclassified_count']}`",
        "",
    ]
    for group in payload["groups"]:
        lines.extend(
            [
                f"## {group['title']}",
                "",
                f"- ID: `{group['id']}`",
                f"- Purpose: {group['purpose']}",
                f"- Rollback: {group['rollback']}",
                f"- Files: `{group['file_count']}`",
                "",
                "| State | Path |",
                "|---|---|",
            ]
        )
        lines.extend(f"| {item['state']} | `{item['path']}` |" for item in group["files"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_reports(root: Path = ROOT) -> tuple[dict[str, object], dict[str, object]]:
    review_paths, states = collect_review_paths(root)
    untracked = [path for path in review_paths if states[path] == "untracked"]
    mapping = build_source_test_mapping(untracked, collect_test_paths(root))
    groups = build_change_groups(review_paths, states)
    return mapping, groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate source/test and review-group reports for the current worktree.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--check", action="store_true", help="Build reports in memory without writing files.")
    args = parser.parse_args()

    mapping, groups = build_reports()
    summary = {
        "production_modules": mapping["production_module_count"],
        "unmapped_modules": mapping["unmapped_count"],
        "review_paths": groups["review_path_count"],
        "unclassified_paths": groups["unclassified_count"],
    }
    if args.check:
        print(json.dumps(summary, sort_keys=True))
        return 1 if mapping["unmapped_count"] else 0

    reports_dir = Path(args.reports_dir)
    write_json(reports_dir / "source_test_mapping.json", mapping)
    write_mapping_markdown(reports_dir / "source_test_mapping.md", mapping)
    write_json(reports_dir / "change_groups.json", groups)
    write_groups_markdown(reports_dir / "change_groups.md", groups)
    print(json.dumps(summary, sort_keys=True))
    return 1 if mapping["unmapped_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
