from pathlib import Path

import pytest

from qwen_asr.program_memory import (
    apply_to_cue,
    apply_to_manifest,
    apply_to_srt_text,
    load_program_memory,
    parse_program_memory,
)

TOGERADI = Path(__file__).resolve().parents[1] / "samples" / "program-memory-togeradi.json"


@pytest.fixture(scope="module")
def togeradi():
    return load_program_memory(TOGERADI)


@pytest.mark.parametrize(
    ("source", "translation", "want_source", "want_translation"),
    [
        ("interfm.jpトゲナシトゲアリのトゲラジ", "interfm.jp 无刺有刺的刺刺广播",
         "InterFMトゲナシトゲアリのトゲラジ", "InterFM 无刺有刺的刺刺广播"),
        ("インタFM トゲなしトゲありのトゲラジ", "INTERFM无刺有刺的刺刺广播",
         "InterFMトゲなしトゲありのトゲラジ", "InterFM无刺有刺的刺刺广播"),
        ("赤色の共済を11月3日", "红色的共济 将于11月3日", "赤色の響祭を11月3日", "《赤色响祭》 将于11月3日"),
        ("赤色の共催の予習のためにも", "为了赤色共同主办的演出的预习", "赤色の響祭の予習のためにも", "为了《赤色响祭》的预习"),
        ("会場はピアーアールイーナM M", "会场是皮亚竞技场M M", "会場はぴあアリーナMM", "会场是PIA ARENA MM"),
        ("アドレスはトゲラジ togeraji@interfm.jptogeradiinterfm.jp", "邮箱是 togeraji@interfm.jp",
         "アドレスはトゲラジ togeradi@interfm.jp", "邮箱是 togeradi@interfm.jp"),
        ("挑戦したいことなどを", "请发送到 刺刺广播 interfm.jp", "挑戦したいことなどを", "请发送到 togeradi@interfm.jp"),
        ("総撃の叫びのライブブルーレイ DVDが", "总击的叫喊 的现场蓝光 DVD",
         "奏檄の叫のライブブルーレイ DVDが", "《奏檄之叫》 的现场蓝光 DVD"),
        ("カナリリーのギターのリネちゃん", "Kanariri的吉他手小Rine", "Canna Lilyのギターのリネちゃん", "Canna Lily的吉他手小Rine"),
    ],
)
def test_togeradi_rules_fix_recurring_misrecognitions(togeradi, source, translation, want_source, want_translation):
    result = apply_to_cue(source, translation, togeradi)
    assert (result.source, result.translation) == (want_source, want_translation)


@pytest.mark.parametrize(
    ("source", "translation"),
    [
        # spoken e-mail address stays as spoken; only the station name before the band name is fixed
        ("トゲラジアットインタFMドットJPまで", "请发送到 togeradi@interfm.jp"),
        ("togeradi@interfm.jpSNSでの感想は", "togeradi@interfm.jp 在SNS上的感想"),
        # translation rule needs the concert name in the source line
        ("赤色のラムネ", "红色的竞赛"),
        ("《赤色响祭》", "《赤色响祭》"),
        # 衝撃の叫び is only the Budokan title when it names the live video
        ("衝撃の叫びが聞こえた", "听到了冲击的呐喊"),
    ],
)
def test_togeradi_rules_leave_correct_or_unrelated_text(togeradi, source, translation):
    result = apply_to_cue(source, translation, togeradi)
    assert (result.source, result.translation, result.hits) == (source, translation, [])


def test_already_fixed_translation_is_not_wrapped_twice(togeradi):
    result = apply_to_cue("赤色の響祭を", "《赤色响祭》将于", togeradi)
    assert result.translation == "《赤色响祭》将于"
    assert result.flags == []


def test_flags_report_cues_that_need_a_human(togeradi):
    assert [f["flag"] for f in apply_to_cue("PRE", "是PRE", togeradi).flags] == ["venue-fragment"]
    assert [f["flag"] for f in apply_to_cue("ピアールリーなM", "是Pia Arene MM", togeradi).flags] == ["venue-fragment"]
    assert [f["flag"] for f in apply_to_cue("M です", "就是这里", togeradi).flags] == ["venue-fragment"]
    missing = apply_to_cue("アドレスはトゲラジアットインタFMドットJP", "地址是", togeradi)
    assert [f["flag"] for f in missing.flags] == ["email-missing-in-translation"]


def test_apply_to_srt_text_keeps_crlf_timing_and_count(togeradi):
    srt = (
        "1\r\n00:00:01,000 --> 00:00:02,000\r\nインターフェムトゲなしトゲありのトゲラジ\r\nInterFM 无刺有刺的刺刺广播\r\n\r\n"
        "2\r\n00:00:02,000 --> 00:00:03,000\r\nはい\r\n好的\r\n"
    )
    fixed, report = apply_to_srt_text(srt, togeradi)
    assert fixed == srt.replace("インターフェム", "InterFM")
    assert report["changed_cue_count"] == 1
    assert report["rule_hit_counts"] == {"station-name-source": 1}


def test_apply_to_manifest_updates_items_in_place(togeradi):
    manifest = {
        "5": {"original_subtitle": "赤色の競賽を", "translated_subtitle": "红色的竞赛", "start_time": 1},
        "6": {"original_subtitle": "はい", "translated_subtitle": "好的"},
    }
    report = apply_to_manifest(manifest, togeradi)
    assert manifest["5"]["original_subtitle"] == "赤色の響祭を"
    assert manifest["5"]["translated_subtitle"] == "《赤色响祭》"
    assert manifest["6"] == {"original_subtitle": "はい", "translated_subtitle": "好的"}
    assert report["changed_cue_count"] == 1


def test_parse_rejects_bad_rules():
    with pytest.raises(ValueError, match="field"):
        parse_program_memory({"rules": [{"id": "x", "field": "both", "pattern": "a"}]})
    with pytest.raises(ValueError, match="invalid regex"):
        parse_program_memory({"rules": [{"id": "x", "field": "source", "pattern": "("}]})


def _fragment_workspace(tmp_path):
    """Proofread manifest split mid-title (as on episode 53) and its normalized whole-sentence cue."""
    from qwen_asr.models import WorkPaths
    from qwen_asr.storage import write_json_atomic

    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {"start_time": 0, "end_time": 900, "original_subtitle": "ライブ赤色の",
                  "translated_subtitle": "演唱会 红色的"},
            "2": {"start_time": 900, "end_time": 1500, "original_subtitle": "共済を",
                  "translated_subtitle": "共济」"},
            "3": {"start_time": 1600, "end_time": 2000, "original_subtitle": "PRE", "translated_subtitle": "是PRE"},
        },
    )
    write_json_atomic(
        paths.normalized_manifest,
        {
            "1": {"start_time": 0, "end_time": 1500, "original_subtitle": "ライブ赤色の共済を",
                  "translated_subtitle": "演唱会 红色的共济"},
            "2": {"start_time": 1600, "end_time": 2000, "original_subtitle": "PRE", "translated_subtitle": "是PRE"},
        },
    )
    return paths


def test_apply_to_work_paths_fixes_normalized_cues_and_srt(tmp_path):
    from qwen_asr.program_memory import apply_to_work_paths
    from qwen_asr.storage import read_json

    paths = _fragment_workspace(tmp_path)

    report = apply_to_work_paths(paths, TOGERADI)

    manifest = read_json(paths.normalized_manifest)
    assert manifest["1"]["original_subtitle"] == "ライブ赤色の響祭を"
    assert manifest["1"]["translated_subtitle"] == "演唱会 《赤色响祭》"
    assert "ライブ赤色の響祭を\n演唱会 《赤色响祭》" in paths.normalized_srt.read_text(encoding="utf-8")
    assert report["changed_cue_count"] == 1
    assert [item["id"] for item in report["flagged"]] == ["2"]
    assert report["source_reverts"] == {"1": {"before": "ライブ赤色の共済を", "after": "ライブ赤色の響祭を"}}

    # idempotent, and a re-run keeps the record made by the first run
    again = apply_to_work_paths(paths, TOGERADI)
    assert again["changed_cue_count"] == 0
    assert again["source_reverts"] == {"1": {"before": "ライブ赤色の共済を", "after": "ライブ赤色の響祭を"}}


def test_apply_to_work_paths_drops_stale_export(tmp_path):
    from qwen_asr.program_memory import apply_to_work_paths

    paths = _fragment_workspace(tmp_path)
    paths.subtitles_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.subtitles_srt.write_text("1\n00:00:00,000 --> 00:00:01,500\nライブ赤色の共済を\n", encoding="utf-8")

    apply_to_work_paths(paths, TOGERADI)

    assert not paths.subtitles_srt.exists()


def test_normalize_content_check_only_exempts_the_fixed_cue(tmp_path):
    # Episode 52: the same jingle line occurs in several cues; only the cue the rule fixed is exempt.
    from qwen_asr.final_quality_content import normalize_export_content_check
    from qwen_asr.program_memory import apply_to_work_paths
    from qwen_asr.storage import read_json, write_json_atomic

    paths = _fragment_workspace(tmp_path)
    upstream = read_json(paths.mimo_proofread_manifest)
    upstream["4"] = {"start_time": 3000, "end_time": 3500, "original_subtitle": "ライブ赤色の共済を",
                     "translated_subtitle": ""}
    write_json_atomic(paths.mimo_proofread_manifest, upstream)
    normalized = read_json(paths.normalized_manifest)
    normalized["3"] = {"start_time": 3000, "end_time": 3500, "original_subtitle": "ライブ赤色の共済を",
                       "translated_subtitle": ""}
    write_json_atomic(paths.normalized_manifest, normalized)
    apply_to_work_paths(paths, TOGERADI)  # fixes both cues 1 and 3
    assert normalize_export_content_check(paths)["status"] == "PASS"

    normalized = read_json(paths.normalized_manifest)
    normalized["3"]["original_subtitle"] = "ライブ赤色の共済を"  # cue 3 back to the old text: still consistent
    write_json_atomic(paths.normalized_manifest, normalized)
    assert normalize_export_content_check(paths)["status"] == "PASS"


def test_normalize_content_check_accepts_program_memory_fixes_only(tmp_path):
    from qwen_asr.final_quality_content import normalize_export_content_check
    from qwen_asr.program_memory import apply_to_work_paths
    from qwen_asr.storage import read_json, write_json_atomic

    paths = _fragment_workspace(tmp_path)
    apply_to_work_paths(paths, TOGERADI)

    assert normalize_export_content_check(paths)["status"] == "PASS"

    manifest = read_json(paths.normalized_manifest)
    manifest["2"]["original_subtitle"] = "PREです"  # an unrelated change must still fail
    write_json_atomic(paths.normalized_manifest, manifest)
    assert normalize_export_content_check(paths)["status"] == "FAIL"


def test_cmd_normalize_applies_program_memory(tmp_path, monkeypatch):
    import argparse

    from qwen_asr.commands import normalize as normalize_command
    from qwen_asr.commands import stages as stage_helpers
    from qwen_asr.storage import read_json

    paths = _fragment_workspace(tmp_path)
    paths.normalized_manifest.unlink()
    monkeypatch.setattr(stage_helpers, "_require_quality_gate_before_formal_output", lambda work_paths: 0)
    args = argparse.Namespace(
        force=False, resume=False, source="mimo", optimizer_root="optimizer",
        extend_ms=350, snap_gap_ms=200, min_blank_ms=300, program_memory_file=str(TOGERADI),
    )

    assert normalize_command.cmd_normalize(args, paths) == 0

    text = "".join(item["original_subtitle"] for item in read_json(paths.normalized_manifest).values())
    assert "響祭" in text and "共済" not in text
    assert read_json(paths.workdir / "reports" / "program_memory.json")["changed_cue_count"] >= 1


def test_cli_accepts_program_memory_file_on_run_and_normalize():
    from qwen_asr.cli import build_parser

    parser = build_parser()
    for argv in (["normalize"], ["run", "--media", "a.mp3"]):
        args = parser.parse_args([*argv, "--workdir", "w", "--program-memory-file", "rules.json"])
        assert args.program_memory_file == "rules.json"


def test_clearing_normalize_outputs_also_drops_the_stale_export(tmp_path):
    from qwen_asr.commands.stages import _clear_normalize_outputs

    paths = _fragment_workspace(tmp_path)
    paths.normalized_srt.write_text("x", encoding="utf-8")
    paths.subtitles_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.subtitles_srt.write_text("x", encoding="utf-8")

    _clear_normalize_outputs(paths)

    assert not paths.normalized_manifest.exists()
    assert not paths.normalized_srt.exists()
    assert not paths.subtitles_srt.exists()
    assert paths.mimo_proofread_manifest.exists()
