from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_asr.web import server
from qwen_asr.web import workspace_api
from qwen_asr.web.server import _content_type_for_path, _subtitle_preview_content_type


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "workspaces"
    workdir = root / "sample"
    (workdir / "manifests").mkdir(parents=True)
    (workdir / "reports").mkdir()
    monkeypatch.setattr(workspace_api, "WORKSPACES_DIR", root)
    (workdir / "project.json").write_text(
        json.dumps({"source_name": "日本語 sample"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return workdir


def test_contract_freezes_align_states_and_secret_policy() -> None:
    contract = workspace_api.api_contract()

    assert contract["api_version"] == "v1"
    assert contract["data"]["align_states"] == ["completed_exact", "completed_coarse", "failed"]
    assert contract["data"]["music_region_state"] == "SKIPPED_MUSIC_REGION"
    assert contract["data"]["recovery"]["strategies"] == ["auto", "qwen", "mfa-local"]
    assert contract["data"]["recovery"]["completed_coarse_requires_verified_transcript"] is True
    assert "environment-only" in contract["data"]["security"]["secrets"]
    assert "POST /api/v1/workspace/review/edit" in contract["data"]["endpoints"]
    assert "POST /api/v1/workspace/review/undo" in contract["data"]["endpoints"]
    assert "POST /api/v1/workspace/stage/start" in contract["data"]["endpoints"]


def test_subtitle_preview_content_types_are_utf8() -> None:
    assert _content_type_for_path(Path("subtitle.srt")) == "application/x-subrip; charset=utf-8"
    assert _content_type_for_path(Path("subtitle.vtt")) == "text/vtt; charset=utf-8"
    assert _subtitle_preview_content_type(Path("subtitle.srt")) == "text/plain; charset=utf-8"


def test_workspace_scope_rejects_paths_outside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(workspace_api, "WORKSPACES_DIR", root)

    with pytest.raises(workspace_api.WorkspaceApiError) as exc_info:
        workspace_api.get_workspace_detail(str(outside))

    assert exc_info.value.code == "WORKDIR_OUT_OF_SCOPE"
    assert exc_info.value.status == 403


def test_corrupt_manifest_returns_structured_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    (workdir / "manifests" / "aligned_segments.json").write_text("{broken", encoding="utf-8")

    response = workspace_api.get_align_state(str(workdir))

    assert response["data"]["manifest_status"] == "corrupt"
    assert "JSONDecodeError" in response["data"]["manifest_error"]
    assert response["data"]["segments"] == []


def test_align_and_recovery_use_generic_manifest_and_music_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    aligned = [
        {
            "segment_id": "exact",
            "global_start_time": 1.0,
            "global_end_time": 2.0,
            "text": "長い台詞",
            "status": "completed",
            "alignment_unit": "token",
            "tokens": [{"text": "長", "start_time": 1.1, "end_time": 1.2}],
        },
        {
            "segment_id": "coarse",
            "global_start_time": 3.0,
            "global_end_time": 4.0,
            "text": "粗い",
            "status": "completed",
            "alignment_unit": "segment",
            "tokens": [],
        },
        {
            "segment_id": "short-failed",
            "global_start_time": 5.0,
            "global_end_time": 6.0,
            "text": "うん。",
            "status": "failed",
            "tokens": [],
            "error": "alignment token timing unreliable",
        },
        {
            "segment_id": "music-failed",
            "global_start_time": 10.0,
            "global_end_time": 11.0,
            "text": "song",
            "status": "failed",
            "tokens": [],
        },
    ]
    (workdir / "manifests" / "aligned_segments.json").write_text(
        json.dumps(aligned, ensure_ascii=False), encoding="utf-8"
    )
    (workdir / "reports" / "evidence.json").write_text(
        json.dumps({"intervals": {"op": {"start_ms": 9000, "end_ms": 12000}}}),
        encoding="utf-8",
    )

    align = workspace_api.get_align_state(str(workdir))["data"]
    recovery = workspace_api.get_recovery_queue(str(workdir))["data"]

    assert align["raw_counts"] == {"completed_exact": 1, "completed_coarse": 1, "failed": 2}
    assert align["dialogue_counts"] == {"completed_exact": 1, "completed_coarse": 1, "failed": 1}
    assert align["excluded_music_region_count"] == 1
    assert align["music_region_evidence_summary"]["subtitle_cues"] == {}
    assert recovery["total"] == 1
    assert recovery["short_response_count"] == 1
    assert recovery["items"][0]["segment_id"] == "short-failed"
    assert recovery["items"][0]["priority"] == "short_response"


def test_quality_fail_is_preserved_when_exports_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    (workdir / "exports").mkdir()
    (workdir / "exports" / "result.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
    (workdir / "reports" / "final_quality.json").write_text(
        json.dumps({"status": "FAIL", "summary": {"fail_count": 1}}), encoding="utf-8"
    )

    exports = workspace_api.get_exports(str(workdir))["data"]

    assert exports[0]["quality_status"] == "FAIL"
    assert exports[0]["delivery_state"] == "quality_gate_failed"


def test_export_file_path_must_come_from_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    (workdir / "exports").mkdir()
    allowed = workdir / "exports" / "result.srt"
    allowed.write_text("subtitle", encoding="utf-8")
    outside = tmp_path / "outside.srt"
    outside.write_text("outside", encoding="utf-8")

    assert workspace_api.get_export_file_path(str(workdir), str(allowed)) == allowed.resolve()
    with pytest.raises(workspace_api.WorkspaceApiError) as exc_info:
        workspace_api.get_export_file_path(str(workdir), str(outside))
    assert exc_info.value.code == "EXPORT_PATH_OUT_OF_SCOPE"


def test_quality_uses_live_music_excluded_alignment_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    (workdir / "manifests" / "aligned_segments.json").write_text(
        json.dumps(
            [
                {"segment_id": "dialogue", "status": "failed", "global_start_time": 1.0, "global_end_time": 2.0},
                {"segment_id": "op", "status": "failed", "global_start_time": 5.0, "global_end_time": 6.0},
            ]
        ),
        encoding="utf-8",
    )
    (workdir / "reports" / "music.json").write_text(
        json.dumps({"intervals": {"op": {"start_ms": 4500, "end_ms": 6500}}}), encoding="utf-8"
    )
    (workdir / "reports" / "final_quality.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "checks": [
                    {
                        "name": "alignment_health",
                        "status": "FAIL",
                        "failed_count": 2,
                        "failed_segment_ids": ["dialogue", "op"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    quality = workspace_api.get_quality_gate(str(workdir))["data"]
    alignment = next(item for item in quality["checks"] if item["name"] == "alignment_health")

    assert alignment["failed_count"] == 1
    assert alignment["skipped_music_region_count"] == 1
    assert alignment["target"]["view"] == "recovery"


def test_quality_targets_review_cues_and_controlled_evidence_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    report = workdir / "reports" / "content_quality.json"
    report.write_text(json.dumps({"status": "WARN"}), encoding="utf-8")
    (workdir / "reports" / "final_quality.json").write_text(
        json.dumps(
            {
                "status": "WARN",
                "checks": [
                    {"name": "content_quality", "status": "WARN", "report": str(report)},
                    {
                        "name": "subtitle_readability",
                        "status": "WARN",
                        "issues": [{"key": "27", "message": "long"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    quality = workspace_api.get_quality_gate(str(workdir))["data"]
    content = next(item for item in quality["checks"] if item["name"] == "content_quality")
    readability = next(item for item in quality["checks"] if item["name"] == "subtitle_readability")

    assert content["target"] == {"view": "evidence", "path": str(report)}
    assert readability["target"] == {"view": "review", "cue_ids": ["27"]}
    assert workspace_api.get_quality_evidence_path(str(workdir), str(report)) == report.resolve()

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(workspace_api.WorkspaceApiError) as exc_info:
        workspace_api.get_quality_evidence_path(str(workdir), str(outside))
    assert exc_info.value.code == "QUALITY_EVIDENCE_OUT_OF_SCOPE"


def test_workspace_listing_keeps_utf8_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)

    response = workspace_api.list_workspace_summaries()

    assert response["data"][0]["workdir"] == str(workdir.resolve())
    assert response["data"][0]["source_name"] == "日本語 sample"


def test_review_edit_and_undo_api_keep_formal_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    source = workdir / "manifests" / "normalized_segments.json"
    source.write_text(
        json.dumps(
            {
                "1": {
                    "start_time": 1000,
                    "end_time": 2000,
                    "original_subtitle": "原文",
                    "translated_subtitle": "翻译",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    source_before = source.read_bytes()

    edited = workspace_api.apply_review_edit(
        str(workdir),
        cue_id="1",
        original="修改原文",
        translation="修改翻译",
        start_ms=1000,
        end_ms=2000,
        expected_revision=0,
    )
    undone = workspace_api.apply_review_undo(
        str(workdir),
        expected_revision=edited["data"]["revision"],
    )

    assert edited["error"] is None
    assert edited["data"]["review"]["review_state"]["dirty"] is True
    assert undone["data"]["review"]["review_state"]["dirty"] is False
    assert source.read_bytes() == source_before


def test_workspace_stage_start_prepares_shared_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workdir = _workspace(tmp_path, monkeypatch)
    (workdir / "manifests" / "aligned_segments.json").write_text("[]", encoding="utf-8")

    payload = workspace_api.prepare_workspace_stage_start(str(workdir), stage="split", settings={})

    assert payload["stage"] == "split"
    assert payload["workdir"] == str(workdir)
    assert payload["resume"] is True


def test_versioned_stage_start_route_returns_structured_job(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object.__new__(server.Handler)
    handler.path = "/api/v1/workspace/stage/start"
    responses: list[tuple[dict, int]] = []
    monkeypatch.setattr(
        handler,
        "_read_json",
        lambda: {"workdir": "workspace", "stage": "split", "settings": {}},
    )
    monkeypatch.setattr(handler, "_send_json", lambda payload, status=200: responses.append((payload, status)))
    monkeypatch.setattr(
        server,
        "prepare_workspace_stage_start",
        lambda *_args, **_kwargs: {"stage": "split", "workdir": "workspace"},
    )
    monkeypatch.setattr(
        server,
        "start_job",
        lambda _payload: {"status": "running", "stage": "split", "workdir": "workspace"},
    )

    handler.do_POST()

    assert responses[0][1] == 202
    assert responses[0][0]["data"]["status"] == "running"
    assert responses[0][0]["error"] is None
