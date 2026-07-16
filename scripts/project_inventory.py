from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = ROOT / "reports"

LOCAL_STATE_DIRS = {
    ".model-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv-whisperx",
    ".venv-whisperx312",
    ".venv312",
    "backups",
    "dist",
    "tmp_codex_pytest",
    "tmp_manual_tests",
    "tmp_pytest",
    "workspaces",
}

KEEP_TOP_LEVEL = {
    "qwen_asr",
    "optimizer",
    "tests",
    "docs",
    "scripts",
    "samples",
    "benchmarks",
    "tools",
}


@dataclass(frozen=True)
class TreeStats:
    files: int
    bytes: int
    latest_mtime: float | None


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace")


def safe_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for item in root.rglob("*"):
        if item.is_file():
            yield item


def tree_stats(path: Path) -> TreeStats:
    files = 0
    total = 0
    latest: float | None = None
    if not path.exists():
        return TreeStats(0, 0, None)
    for file_path in iter_files(path):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        files += 1
        total += stat.st_size
        latest = stat.st_mtime if latest is None else max(latest, stat.st_mtime)
    return TreeStats(files, total, latest)


def file_record(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": safe_rel(path),
            "size_bytes": None,
            "modified_at": None,
            "status": "UNKNOWN",
            "reason": "Could not stat path.",
        }
    rel = safe_rel(path)
    status, reason = classify_path(rel, path, stat.st_size)
    return {
        "path": rel,
        "record_kind": "file" if path.is_file() else "directory",
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
    }


def classify_path(rel: str, path: Path, size: int | None = None) -> tuple[str, str]:
    parts = rel.split("/")
    top = parts[0]
    if top in {".git"}:
        return "KEEP", "Git metadata; excluded from cleanup reports."
    if top in LOCAL_STATE_DIRS:
        if top in {"backups", "workspaces"}:
            return "UNKNOWN", "Protected evidence or rollback state; requires task-level review before archive."
        return "KEEP", "Local runtime state; keep locally and ignore from Git."
    if rel == "$env" or top == "None":
        return "DELETE_CANDIDATE", "Suspicious generated path; deletion still requires explicit approval."
    if "__pycache__" in parts or rel.endswith(".pyc") or top in {".pytest_cache", ".ruff_cache"}:
        return "DELETE_CANDIDATE", "Rebuildable Python or tool cache; deletion requires approval."
    if rel.startswith("tools/mfa-env/") or rel.startswith("tools/mfa-root/") or rel.startswith("tools/micromamba/"):
        return "KEEP", "Local MFA or micromamba dependency state; ignored from Git."
    if rel.startswith("tools/diagnose_") and rel.endswith(".py"):
        return "KEEP", "Maintainable diagnostic script source."
    if "/runs/" in rel and rel.startswith("benchmarks/"):
        return "ARCHIVE_CANDIDATE", "Benchmark run artifact; keep manifest and archive only after approval."
    if top in KEEP_TOP_LEVEL:
        return "KEEP", "Source, tests, documentation, reproducible benchmark metadata, or sample input."
    if path.is_file() and top in {"main.py", "webapp.py", "start.bat", "pyproject.toml", "pytest.ini", "requirements.txt", "README.md", ".gitignore"}:
        return "KEEP", "Project entrypoint or configuration."
    return "UNKNOWN", "No cleanup rule matched; review manually."


def top_level_inventory() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if path.name == ".git":
            continue
        stats = tree_stats(path)
        modified = None if stats.latest_mtime is None else datetime.fromtimestamp(stats.latest_mtime, timezone.utc).isoformat()
        status, reason = classify_path(path.name, path, stats.bytes)
        records.append(
            {
                "path": path.name,
                "kind": "directory" if path.is_dir() else "file",
                "files": stats.files,
                "size_bytes": stats.bytes,
                "modified_at": modified,
                "status": status,
                "reason": reason,
            }
        )
    return records


def git_file_classification() -> dict[str, object]:
    tracked = [line for line in run_git(["ls-files"]).splitlines() if line]
    modified = [line for line in run_git(["diff", "--name-only"]).splitlines() if line]
    untracked = [line for line in run_git(["ls-files", "--others", "--exclude-standard"]).splitlines() if line]
    ignored = [line for line in run_git(["ls-files", "--others", "-i", "--exclude-standard"]).splitlines() if line]
    return {
        "tracked": [file_record(ROOT / item) for item in tracked],
        "modified": [file_record(ROOT / item) for item in modified],
        "untracked": [file_record(ROOT / item) for item in untracked],
        "ignored": [file_record(ROOT / item) for item in ignored],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_record(path: Path, status: str, reason: str) -> dict[str, object]:
    if path.is_file():
        record = file_record(path)
        record.update(
            {
                "record_kind": "file",
                "approval_required": True,
                "sha256": sha256(path),
                "sha256_kind": "file",
                "reason": reason,
            }
        )
        return record

    stats = tree_stats(path)
    return {
        "path": safe_rel(path),
        "record_kind": "directory_aggregate",
        "size_bytes": stats.bytes,
        "files": stats.files,
        "modified_at": None
        if stats.latest_mtime is None
        else datetime.fromtimestamp(stats.latest_mtime, timezone.utc).isoformat(),
        "status": status,
        "approval_required": True,
        "sha256": None,
        "sha256_kind": "not_applicable_directory_aggregate",
        "reason": reason,
    }


def archive_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    review_paths = set(run_git(["ls-files", "--others", "--exclude-standard"]).splitlines())
    review_paths.update(run_git(["ls-files", "--others", "-i", "--exclude-standard"]).splitlines())
    for rel in sorted(review_paths):
        path = ROOT / rel
        if not path.exists():
            continue
        status, reason = classify_path(rel, path)
        if status in {"ARCHIVE_CANDIDATE", "DELETE_CANDIDATE", "UNKNOWN"}:
            candidates.append(candidate_record(path, status, reason))
    for root_name in ["workspaces", "backups", "tmp_manual_tests"]:
        root = ROOT / root_name
        if root.exists():
            stats = tree_stats(root)
            candidates.append(
                {
                    "path": root_name,
                    "record_kind": "directory_aggregate",
                    "size_bytes": stats.bytes,
                    "files": stats.files,
                    "modified_at": None
                    if stats.latest_mtime is None
                    else datetime.fromtimestamp(stats.latest_mtime, timezone.utc).isoformat(),
                    "status": "UNKNOWN" if root_name != "tmp_manual_tests" else "ARCHIVE_CANDIDATE",
                    "approval_required": True,
                    "sha256": None,
                    "sha256_kind": "not_applicable_directory_aggregate",
                    "reason": "Protected large local state; directory-level archive/delete requires explicit approval.",
                }
            )
    return candidates


def build_p2_approval_status(
    candidates: list[dict[str, object]],
    generated_at: str,
    existing: dict[str, object] | None = None,
) -> dict[str, object]:
    existing = existing or {}
    approved_paths = [str(path) for path in existing.get("approved_paths", [])]
    status_counts: dict[str, int] = {}
    for item in candidates:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    missing_file_hashes = sum(
        1 for item in candidates if item.get("record_kind") == "file" and not item.get("sha256")
    )
    directory_aggregates = sum(1 for item in candidates if item.get("record_kind") == "directory_aggregate")
    return {
        "generated_at": generated_at,
        "status": "AWAITING_OPTIONAL_APPROVAL",
        "goal_blocking": False,
        "policy": {
            "audit_and_approval_package_required": True,
            "archive_or_delete_execution_required_for_goal": False,
            "path_level_approval_required": True,
            "bulk_directory_approval_inferred": False,
        },
        "source": "reports/archive_candidates.json",
        "candidate_count": len(candidates),
        "status_counts": status_counts,
        "candidate_size_bytes": sum(int(item.get("size_bytes") or 0) for item in candidates),
        "approval_required_count": sum(bool(item.get("approval_required")) for item in candidates),
        "approved_paths": approved_paths,
        "approved_count": len(approved_paths),
        "archived_count": int(existing.get("archived_count", 0) or 0),
        "deleted_count": int(existing.get("deleted_count", 0) or 0),
        "file_candidate_count": sum(1 for item in candidates if item.get("record_kind") == "file"),
        "directory_aggregate_count": directory_aggregates,
        "missing_file_sha256_count": missing_file_hashes,
        "decision": (
            "No archive or delete action is approved or executed. "
            "P2-D remains optional and does not block Goal completion."
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def write_inventory_md(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Project Inventory",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Git HEAD: `{payload['git']['head']}`",
        "",
        "## Top-Level Paths",
        "",
        "| Path | Kind | Files | Size | Status | Reason |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in payload["top_level"]:
        lines.append(
            f"| `{item['path']}` | {item['kind']} | {item['files']} | {format_size(item['size_bytes'])} | "
            f"{item['status']} | {item['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_git_md(path: Path, payload: dict[str, object]) -> None:
    lines = ["# Git File Classification", ""]
    for key in ["tracked", "modified", "untracked", "ignored"]:
        rows = payload[key]
        lines.extend([f"## {key.title()}", "", f"Count: `{len(rows)}`", "", "| Path | Size | Status | Reason |", "|---|---:|---|---|"])
        for item in rows:
            lines.append(f"| `{item['path']}` | {format_size(item['size_bytes'])} | {item['status']} | {item['reason']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_archive_md(path: Path, payload: list[dict[str, object]]) -> None:
    lines = [
        "# Archive And Delete Candidates",
        "",
        "No file or directory in this list has been deleted or moved. Every row requires explicit human approval before archive or deletion.",
        "",
        "Directory aggregate rows are inventory summaries, not files, and therefore do not claim a file SHA-256.",
        "",
        "| Path | Kind | Size | Status | Approval | SHA-256 | SHA semantics | Reason |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for item in payload:
        lines.append(
            f"| `{item['path']}` | {item.get('record_kind', '')} | {format_size(item.get('size_bytes'))} | "
            f"{item['status']} | {item.get('approval_required', True)} | {item.get('sha256') or ''} | "
            f"{item.get('sha256_kind', '')} | {item['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate project cleanup inventory reports.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Output directory for generated reports.")
    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    inventory = {
        "generated_at": generated_at,
        "git": {
            "head": run_git(["rev-parse", "HEAD"]).strip(),
            "status_short": run_git(["status", "--short"]).splitlines(),
            "diff_stat": run_git(["diff", "--stat"]).splitlines(),
        },
        "top_level": top_level_inventory(),
    }
    git_classification = git_file_classification()
    candidates = archive_candidates()
    approval_path = reports_dir / "p2_approval_status.json"
    existing_approval = None
    if approval_path.exists():
        try:
            existing_approval = json.loads(approval_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            existing_approval = None
    approval_status = build_p2_approval_status(candidates, generated_at, existing_approval)
    write_json(reports_dir / "project_inventory.json", inventory)
    write_json(reports_dir / "git_file_classification.json", git_classification)
    write_json(reports_dir / "archive_candidates.json", candidates)
    write_json(approval_path, approval_status)
    write_inventory_md(reports_dir / "project_inventory.md", inventory)
    write_git_md(reports_dir / "git_file_classification.md", git_classification)
    write_archive_md(reports_dir / "archive_candidates.md", candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
