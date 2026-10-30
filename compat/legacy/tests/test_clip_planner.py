from __future__ import annotations

import json

import pytest

from qwen_asr import clip_planner as cp

SRT = """1
00:00:01,000 --> 00:00:03,000
トゲナシトゲアリのトゲラジ
无刺有刺的刺刺广播

2
00:00:03,500 --> 00:00:30,000
こんにちは
大家好

3
00:00:30,000 --> 00:01:10,500
以上今週の声ログでした
以上就是本周的声日志

4
00:01:12,000 --> 00:01:20,000
お送りした曲は
播放的歌曲是
"""


def cues():
    return cp.parse_srt(SRT)


def test_parse_srt_reads_index_times_and_both_lines():
    parsed = cues()
    assert [c.index for c in parsed] == [1, 2, 3, 4]
    assert parsed[1].start == 3.5 and parsed[2].end == 70.5
    assert parsed[0].text == "トゲナシトゲアリのトゲラジ" and parsed[0].translation == "无刺有刺的刺刺广播"


def test_parse_srt_handles_bom_crlf_and_single_line():
    parsed = cp.parse_srt("\ufeff1\r\n00:00:00,000 --> 00:00:01,000\r\nはい\r\n")
    assert parsed == [cp.Cue(1, 0.0, 1.0, "はい", "")]


def test_cut_window_uses_measured_pre_and_post_roll():
    assert cp.cut_window(cues(), 1, 3) == (0.0, 71.5)
    assert cp.cut_window(cues(), 4, 4) == (71.0, 81.0)
    assert cp.cut_window(cues(), 1, 3, media_duration=71.0) == (0.0, 71.0)


def test_cut_window_pre_roll_stops_at_previous_cue():
    # previous cue ends 0.5s before: cut right where it ends
    assert cp.cut_window(cues(), 2, 3) == (3.0, 71.5)
    # previous cue runs right up to the first cue: keep only a short lead
    assert cp.cut_window(cues(), 3, 3) == (29.85, 71.5)


def test_cut_window_rejects_out_of_range():
    with pytest.raises(cp.ClipPlanError):
        cp.cut_window(cues(), 3, 2)
    with pytest.raises(cp.ClipPlanError):
        cp.cut_window(cues(), 1, 9)


def test_extract_json_object_from_fence_and_think():
    text = '<think>{"x":1}</think>说明\n```json\n{"candidates": [{"a": {"b": 1}}]}\n```'
    assert cp.extract_json_object(text) == {"candidates": [{"a": {"b": 1}}]}
    assert cp.extract_json_object('前言 {"k": 2} 后记') == {"k": 2}
    with pytest.raises(cp.ClipPlanError):
        cp.extract_json_object("no json")


def response():
    return {
        "core_judgment": "不错",
        "positioning": "日常趣事型",
        "candidates": [
            {"stars": 5, "start_cue": 1, "end_cue": 3, "content": "声ログ", "title_full": "完整", "title_short": "短/标题?",
             "reason": "好笑", "note": ""},
            {"stars": 9, "start_cue": 4, "end_cue": 2},  # invalid range
            {"stars": 2, "start_cue": 4, "end_cue": 4},  # too short
            "bad",
        ],
        "top3": [{"index": 0, "why": "最好"}, {"index": 1, "why": "丢弃的"}],
        "single_pick": {"index": 0, "why": "就它"},
    }


def test_build_plan_validates_and_renumbers():
    plan = cp.build_plan(response(), cues())
    assert [c["number"] for c in plan["candidates"]] == [1]
    first = plan["candidates"][0]
    assert (first["start"], first["end"], first["stars"]) == (0.0, 71.5, 5)
    assert first["title_short"] == "短 标题"
    assert plan["top3"] == [{"number": 1, "why": "最好"}]
    assert plan["single_pick"] == {"number": 1, "why": "就它"}
    assert len(plan["warnings"]) == 3
    json.dumps(plan, ensure_ascii=False)


def test_build_plan_without_valid_candidates_raises():
    with pytest.raises(cp.ClipPlanError):
        cp.build_plan({"candidates": [{"start_cue": 4, "end_cue": 4}]}, cues())
    with pytest.raises(cp.ClipPlanError):
        cp.build_plan({}, cues())


def test_render_markdown_lists_candidates_and_boundary_lines():
    md = cp.render_markdown(cp.build_plan(response(), cues()), cues(), title="第1期")
    assert "| 1 | ★★★★★ | 00:00.0–01:11.5 | 1.2 分 |" in md
    assert "开头「トゲナシトゲアリのトゲラジ」… 结尾「以上今週の声ログでした」" in md
    assert "## 如果只看一个片段" in md


def test_overlap_notes_flags_both_sides():
    a = cp.ClipCandidate(1, 5, 1, 2, 0.0, 60.0, "", "", "a", "")
    b = cp.ClipCandidate(2, 4, 2, 3, 50.0, 120.0, "", "", "b", "")
    c = cp.ClipCandidate(3, 3, 3, 4, 118.0, 200.0, "", "", "c", "")
    assert cp.overlap_notes([a, b, c]) == {1: [2], 2: [1]}


def test_apply_refinement_moves_only_inside_shown_windows():
    many = [cp.Cue(i, i * 10.0, i * 10.0 + 9.0, f"t{i}") for i in range(1, 101)]
    plan = cp.build_plan({"candidates": [{"stars": 4, "start_cue": 50, "end_cue": 60, "title_short": "x"},
                                         {"stars": 3, "start_cue": 70, "end_cue": 80, "title_short": "y"}]}, many)
    refined = cp.apply_refinement(plan, {"items": [
        {"number": 1, "start_cue": 45, "end_cue": 62, "why": "含开场"},
        {"number": 2, "start_cue": 5, "end_cue": 80, "why": "越界"},
    ]}, many)
    first, second = refined["candidates"]
    assert (first["start_cue"], first["end_cue"], first["start"], first["end"]) == (45, 62, 449.0, 630.0)
    assert (second["start_cue"], second["end_cue"]) == (70, 80)
    assert refined["boundary_changes"] == [{"number": 1, "from": [50, 60], "to": [45, 62], "why": "含开场"}]


def test_build_refine_messages_marks_current_boundaries():
    many = [cp.Cue(i, i * 10.0, i * 10.0 + 9.0, f"t{i}") for i in range(1, 101)]
    plan = cp.build_plan({"candidates": [{"stars": 4, "start_cue": 50, "end_cue": 60, "title_short": "x"}]}, many)
    user = cp.build_refine_messages(many, cp.candidates_from_plan(plan), prompt="P")[1]["content"]
    assert ">>START #50 " in user and ">>END #60 " in user
    assert "#20 " in user and "#19 " not in user and "#80 " in user and "#81 " not in user


def test_parse_and_apply_adjust():
    assert cp.parse_adjust("2:315-411") == (2, 315, 411)
    assert cp.parse_adjust(" 3：10 – 20 ") == (3, 10, 20)
    with pytest.raises(cp.ClipPlanError):
        cp.parse_adjust("2-315-411")
    base = cp.ClipCandidate(1, 5, 1, 3, 0.0, 71.5, "", "", "a", "")
    moved = cp.adjust_candidate(base, cues(), 2, 3)
    assert (moved.start_cue, moved.start, moved.end) == (2, 3.0, 71.5)


def test_slice_cues_shifts_and_clamps():
    sliced = cp.slice_cues(cues(), 2.5, 71.5)
    assert [(c.index, c.start, c.end, c.text) for c in sliced] == [
        (1, 0.0, 0.5, "トゲナシトゲアリのトゲラジ"), (2, 1.0, 27.5, "こんにちは"), (3, 27.5, 68.0, "以上今週の声ログでした")]
    srt = cp.render_srt(sliced)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:00,500\nトゲナシトゲアリのトゲラジ\n无刺有刺的刺刺广播\n")


def test_ffmpeg_cut_command_is_lossless_copy(tmp_path):
    cmd = cp.ffmpeg_cut_command(tmp_path / "a.mp3", tmp_path / "b.mp3", 2.5, 71.5)
    assert cmd[cmd.index("-ss") + 1] == "2.500" and cmd[cmd.index("-t") + 1] == "69.000"
    assert cmd[cmd.index("-c") + 1] == "copy" and cmd.index("-ss") < cmd.index("-i")


def test_clip_basename_is_filesystem_safe():
    c = cp.ClipCandidate(3, 5, 1, 2, 0.0, 60.0, "", "", 'a:b*c?"d', "")
    assert cp.clip_basename("45", c) == "#45-03 a b c d"


def test_recompute_times_follows_current_rule():
    plan = cp.build_plan({"candidates": [{"stars": 4, "start_cue": 2, "end_cue": 3, "title_short": "x"}]}, cues())
    stale = {**plan, "candidates": [{**plan["candidates"][0], "start": 0.0, "end": 1.0}]}
    fresh = cp.recompute_times(stale, cues())["candidates"][0]
    assert (fresh["start"], fresh["end"]) == (3.0, 71.5)


def test_slice_cues_drops_slivers_cut_by_clip_edge():
    sliced = cp.slice_cues(cues(), 2.85, 71.5)
    assert [c.text for c in sliced] == ["こんにちは", "以上今週の声ログでした"]


def test_verify_field_is_kept_and_rendered():
    resp = {"candidates": [{"stars": 3, "start_cue": 1, "end_cue": 3, "title_short": "x", "verify": "「語源」可能是误识别"}]}
    plan = cp.build_plan(resp, cues())
    assert plan["candidates"][0]["verify"] == "「語源」可能是误识别"
    assert "**需听音确认**：「語源」可能是误识别" in cp.render_markdown(plan, cues(), title="t")
    legacy = {**plan, "candidates": [{k: v for k, v in plan["candidates"][0].items() if k != "verify"}]}
    assert cp.candidates_from_plan(legacy)[0].verify == ""


def test_slice_cues_drops_sentence_cut_by_clip_end():
    sliced = cp.slice_cues(cues(), 2.5, 72.5)  # 0.5s of cue 4 falls into the fade-out tail
    assert [c.text for c in sliced][-1] == "以上今週の声ログでした"
    assert all(c.end <= 70.0 for c in sliced)


def test_format_clock_carries_rounded_seconds():
    assert cp.format_clock(119.96) == "02:00.0"
    assert cp.format_clock(3599.97) == "1:00:00.0"
    assert cp.format_clock(65.24) == "01:05.2"
