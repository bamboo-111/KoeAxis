from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_asr import text_proofread as tp
from qwen_asr.artifact_state import ArtifactState
from qwen_asr.models import WorkPaths
from qwen_asr.stages import stage_names_for_run


def _item(start: int, end: int, ja: str, zh: str, **extra: object) -> dict:
    return {"start_time": start, "end_time": end, "original_subtitle": ja, "translated_subtitle": zh, **extra}


MANIFEST = {
    "1": _item(0, 1000, "私たちは", "我们不仅是"),
    "2": _item(1000, 2000, "バンドだけでなく", "乐队"),
    "3": _item(2000, 3000, "雨だ雨だ", "下雨落下雨了", suspect_reason="typo?"),
    "4": _item(3000, 4000, "オーバーオールで", "工装裤"),
    "5": _item(4000, 5000, "ゴーグルで", "护目镜"),
}
# Segment B starts at 2.5 s, so line 3 (2.0-3.0 s) overlaps both segments but belongs to A.
SEGMENTS = [
    {"segment_id": "seg_A", "global_start_time": 0.0, "global_end_time": 3.0},
    {"segment_id": "seg_B", "global_start_time": 2.5, "global_end_time": 5.0},
]


def _targets(prompt: str) -> list[dict]:
    line = next(row for row in prompt.splitlines() if row.startswith("待校对字幕："))
    return json.loads(line.split("：", 1)[1])


class FakeClient:
    """Answers like the model: ``answers[id]`` is the suggestion for that line, everything else is ok."""

    def __init__(self, answers: dict[str, dict], *, fail_segments: tuple[str, ...] = ()) -> None:
        self.answers = answers
        self.fail_segments = fail_segments
        self.prompts: list[str] = []
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self.capacity = 2
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        with self._lock:
            self.prompts.append(prompt)
            self.calls.append(kwargs)
        targets = _targets(prompt)
        if any(row["id"] in self.fail_segments for row in targets):
            raise RuntimeError("bad request 400")
        rows = []
        for row in targets:
            rows.append(self.answers.get(row["id"], {"id": row["id"], "verdict": "ok"}))
        # extra answer for a neighbour line, keyed "<neighbour>@<first target>"
        for key, value in self.answers.items():
            if "@" in key and key.split("@")[1] == targets[0]["id"]:
                rows.append(value)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(rows, ensure_ascii=False)))],
            usage=None,
        )


def _workdir(tmp_path: Path, *, with_mimo: bool = True) -> WorkPaths:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.segments_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.segments_manifest.write_text(json.dumps(SEGMENTS), encoding="utf-8")
    target = paths.mimo_proofread_manifest if with_mimo else paths.translated_manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(MANIFEST, ensure_ascii=False), encoding="utf-8")
    return paths


def _fix(sid: str, new: str, error_type: str = "mistranslation", confidence: float = 0.9) -> dict:
    return {"id": sid, "verdict": "fix", "error_type": error_type, "suggested_translation": new,
            "reason": "r", "confidence": confidence}


def test_endpoints_from_env_reads_numbered_relays() -> None:
    env = {
        "GEMINI_API_KEY": "k0", "GEMINI_BASE_URL": "https://relay0.example/v1beta",
        "GEMINI_API_KEY_1": "k1", "GEMINI_BASE_URL_1": "http://127.0.0.1:8317",
        "GEMINI_MODEL_1": "gemini-3.8-flash-high", "GEMINI_CONCURRENCY_1": "5",
        "GEMINI_API_KEY_2": "k0", "GEMINI_BASE_URL_2": "https://relay0.example/v1beta",  # duplicate of slot 0
        "GEMINI_BASE_URL_3": "https://no-key.example",
    }
    endpoints = tp.endpoints_from_env(env, default_model="gemini-3.8-flash", default_concurrency=3)
    assert [(e.name, e.model, e.max_inflight) for e in endpoints] == [
        ("gemini", "gemini-3.8-flash", 3),
        ("gemini_1", "gemini-3.8-flash-high", 5),
    ]
    assert endpoints[1].base_url.endswith("/v1beta")


def test_every_line_is_proofread_by_exactly_one_request() -> None:
    requests = tp.build_requests(MANIFEST, SEGMENTS, [], tp.TextProofreadConfig())
    assert [(r.segment_id, r.target_ids, r.neighbor_ids) for r in requests] == [
        ("seg_A", ["1", "2", "3"], ["4"]),
        ("seg_B", ["4", "5"], ["3"]),
    ]
    targets = _targets(requests[0].prompt)
    assert targets[2]["hint"] == "typo?"
    assert "主持成员" not in requests[0].prompt
    header = tp.build_prompt_header("日语电台节目《刺刺广播》", "理名、夕莉、朱李")
    assert header == "你是日语电台节目《刺刺广播》的中文字幕校对编辑。主持成员：理名、夕莉、朱李。\n"


def test_select_edits_filters_and_resolves_conflicts() -> None:
    results = [
        {"segment_id": "seg_A", "target_ids": ["1", "2", "3"], "neighbor_ids": ["4"], "suggestions": [
            _fix("1", "我们", "duplicate"),
            _fix("2", "不仅是乐队", "duplicate"),
            _fix("3", "下雨了下雨了", confidence=0.7),          # below threshold
            _fix("4", "背带裤"),                               # neighbour, not duplicate
            {"id": "5", "verdict": "fix", "error_type": "kana", "suggested_translation": "x", "confidence": 1},
        ]},
        {"segment_id": "seg_B", "target_ids": ["4", "5"], "neighbor_ids": ["3"], "suggestions": [
            _fix("4", "背带裤", confidence=0.85),
            _fix("5", "ゴーグル"),                             # kana left
            _fix("3", "下雨了", "duplicate", 0.95),           # neighbour duplicate fix
            _fix("4", "穿背带裤", confidence=0.95),            # second answer for the same line wins
            {"id": "1", "verdict": "ok"},
        ]},
    ]
    accepted, rejected = tp.select_edits(MANIFEST, results)
    assert {sid: e["new"] for sid, e in accepted.items()} == {
        "1": "我们", "2": "不仅是乐队", "3": "下雨了", "4": "穿背带裤"}
    reasons = sorted((e["id"], e["rejected"]) for e in rejected)
    assert reasons == [
        ("3", "low_confidence"), ("4", "neighbor_non_duplicate"), ("4", "superseded"),
        ("5", "kana"), ("5", "not_in_request"),
    ]


def test_select_edits_rejects_copy_of_neighbour_and_style_only() -> None:
    results = [{"segment_id": "seg_B", "target_ids": ["4", "5"], "neighbor_ids": [], "suggestions": [
        _fix("4", "护目镜"),
        _fix("5", "戴护目镜", "style_only"),
    ]}]
    accepted, rejected = tp.select_edits(MANIFEST, results)
    assert accepted == {}
    assert sorted(e["rejected"] for e in rejected) == ["error_type", "same_as_neighbor"]


def test_run_writes_branch_report_and_resumes(tmp_path: Path) -> None:
    paths = _workdir(tmp_path)
    client = FakeClient({"3": _fix("3", "下雨了下雨了", confidence=0.95)})
    config = tp.TextProofreadConfig(effort="medium", model="m")
    report = tp.run_text_proofread(paths, config, client=client)

    assert report["status"] == "PASS"
    assert (report["request_count"], report["applied_count"], report["input"]) == (2, 1, "mimo")
    assert all(call["extra_body"] == {"reasoning_effort": "medium"} for call in client.calls)
    branch = json.loads(paths.text_proofread_manifest.read_text(encoding="utf-8"))
    assert branch["3"]["translated_subtitle"] == "下雨了下雨了"
    assert branch["3"]["text_proofread"]["previous_translation"] == "下雨落下雨了"
    for sid, item in MANIFEST.items():  # Japanese text and timing untouched
        assert (branch[sid]["original_subtitle"], branch[sid]["start_time"], branch[sid]["end_time"]) == (
            item["original_subtitle"], item["start_time"], item["end_time"])
    assert "下雨了下雨了" in paths.text_proofread_srt.read_text(encoding="utf-8")
    assert ArtifactState(paths).is_complete("text-proofread")

    second = FakeClient({})
    again = tp.run_text_proofread(paths, config, client=second)
    assert second.prompts == []
    assert (again["cached_count"], again["applied_count"]) == (2, 1)


def test_run_reports_partial_and_total_failure(tmp_path: Path) -> None:
    paths = _workdir(tmp_path, with_mimo=False)
    config = tp.TextProofreadConfig(max_attempts=1, retry_delay_s=0)
    partial = tp.run_text_proofread(paths, config, client=FakeClient({}, fail_segments=("4",)))
    assert (partial["status"], partial["input"], partial["failed_segments"]) == ("WARN", "translated", ["seg_B"])
    assert ArtifactState(paths).is_complete("text-proofread")

    paths.text_proofread_manifest.unlink()
    fresh = tp.TextProofreadConfig(max_attempts=1, retry_delay_s=0, resume=False)  # drop the cached seg_A
    failed = tp.run_text_proofread(paths, fresh, client=FakeClient({}, fail_segments=("1", "4")))
    assert failed["status"] == "FAIL"
    assert not paths.text_proofread_manifest.exists()
    assert not ArtifactState(paths).is_complete("text-proofread")


def test_run_without_relays_fails_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _workdir(tmp_path)
    for suffix in tp.ENV_SUFFIXES:
        monkeypatch.delenv(f"GEMINI_API_KEY{suffix}", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        tp.run_text_proofread(paths, tp.TextProofreadConfig())


def test_stage_order_and_config_from_args() -> None:
    names = stage_names_for_run(with_translate=True, with_mimo_proofread=True, with_text_proofread=True,
                                with_normalize=True)
    assert names[-6:] == ("mimo-proofread", "proofread-realign", "text-proofread", "quality-gate", "normalize",
                          "export")
    args = SimpleNamespace(text_proofread_effort="high", text_proofread_model="", text_proofread_concurrency=2,
                           text_proofread_program="P", text_proofread_hosts="H", glossary_xlsx=None,
                           resume=True, force=True)
    config = tp.config_from_args(args)
    assert (config.effort, config.model, config.concurrency, config.resume) == ("high", tp.DEFAULT_MODEL, 2, False)
    with pytest.raises(ValueError):
        tp.config_from_args(SimpleNamespace(text_proofread_effort="max"))


def test_cmd_skips_when_complete_and_clears_downstream_on_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _workdir(tmp_path)
    clients: list[FakeClient] = []
    real_run = tp.run_text_proofread

    def fake_run(work_paths, config, progress=None):
        clients.append(FakeClient({"3": _fix("3", "下雨了下雨了", confidence=0.95)}))
        return real_run(work_paths, config, client=clients[-1], progress=progress)

    monkeypatch.setattr(tp, "run_text_proofread", fake_run)
    args = SimpleNamespace(text_proofread_effort="medium", resume=True, force=False)
    assert tp.cmd_text_proofread(args, paths) == 0
    assert len(clients) == 1

    paths.normalized_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.normalized_manifest.write_text("{}", encoding="utf-8")
    assert tp.cmd_text_proofread(args, paths) == 0  # complete + same effort -> skipped
    assert len(clients) == 1 and paths.normalized_manifest.exists()

    forced = SimpleNamespace(text_proofread_effort="medium", resume=True, force=True)
    assert tp.cmd_text_proofread(forced, paths) == 0
    assert len(clients) == 2
    assert not paths.normalized_manifest.exists()  # stale normalize output removed
