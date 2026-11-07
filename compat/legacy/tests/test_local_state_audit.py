from __future__ import annotations

from pathlib import Path

from scripts.local_state_audit import (
    backup_topic,
    classify_workspace,
    references_for,
    tree_stats,
)


def test_tree_stats_counts_nested_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.txt").write_text("one", encoding="ascii")
    (tmp_path / "two.txt").write_text("two!", encoding="ascii")

    stats = tree_stats(tmp_path)

    assert stats.files == 2
    assert stats.bytes == 7
    assert stats.latest_mtime is not None


def test_workspace_classification_preserves_unknown_and_referenced_paths() -> None:
    status, _reason = classify_workspace("unreferenced-experiment", [])
    referenced_status, _reason = classify_workspace("historical-proof", ["docs/history.md"])
    baseline_status, _reason = classify_workspace("p6-final-regression-20260714-160541", [])

    assert status == "UNKNOWN"
    assert referenced_status == "KEY_EVIDENCE"
    assert baseline_status == "CURRENT_BASELINE"


def test_references_and_backup_topic_are_deterministic() -> None:
    documents = [
        ("docs/a.md", "workspaces/example/report.json"),
        ("docs/b.md", "no reference"),
    ]

    assert references_for("workspaces/example", documents) == ["docs/a.md"]
    assert backup_topic("p1-review-scope-20260715-195811") == "p1-review-scope"
