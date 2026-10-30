from __future__ import annotations

from qwen_asr.web.static_html import WORKBENCH_HTML, load_static_asset


def test_workbench_is_the_structured_function_shell() -> None:
    assert 'data-view="pipeline"' in WORKBENCH_HTML
    assert 'data-view="recovery"' in WORKBENCH_HTML
    assert 'data-view="review"' in WORKBENCH_HTML
    assert 'data-view="quality"' in WORKBENCH_HTML
    assert 'data-view="exports"' in WORKBENCH_HTML
    assert 'href="/legacy"' in WORKBENCH_HTML
    assert '<audio' in WORKBENCH_HTML
    assert 'id="reviewDraftState"' in WORKBENCH_HTML
    assert 'id="undoReviewButton"' in WORKBENCH_HTML
    assert "<th>操作</th>" in WORKBENCH_HTML
    assert 'class="hero"' not in WORKBENCH_HTML.lower()
    assert WORKBENCH_HTML.count('class="nav-label"') == 5
    assert 'aria-label="恢复队列"' in WORKBENCH_HTML


def test_workbench_assets_call_structured_apis() -> None:
    script, script_type = load_static_asset("workbench.js")
    stylesheet, style_type = load_static_asset("workbench.css")
    text = script.decode("utf-8")
    assert script_type.startswith("text/javascript")
    assert style_type.startswith("text/css")
    assert "/api/v1/workspaces" in text
    assert "/api/v1/workspace?" in text
    assert "/api/v1/workspace/recovery/action" in text
    assert "/api/v1/workspace/review" in text
    assert "/api/v1/workspace/review/edit" in text
    assert "/api/v1/workspace/review/undo" in text
    assert "saveReviewCue" in text
    assert "undoReviewEdit" in text
    assert "/api/v1/workspace/media" in text
    assert "/api/v1/workspace/export-file" in text
    assert "/api/v1/workspace/quality-evidence" in text
    assert "/api/v1/workspace/stage/start" in text
    assert "startWorkspaceStage" in text
    assert "savedStageSettings" in text
    assert "qualityAction" in text
    assert "openQualityReviewTarget" in text
    assert "/api/v1/job" in text
    assert "setInterval(refreshJob" in text
    assert "setInterval(refresh," not in text
    assert "SKIPPED_MUSIC_REGION" not in text
    css = stylesheet.decode("utf-8")
    assert ".nav-label,.nav-item .badge" in css
    assert len(stylesheet) > 1000
