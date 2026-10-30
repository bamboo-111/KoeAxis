from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from optimizer.asr_data import ASRDataSeg
from qwen_asr.optimizer_bridge_stages import tidy_repeat_chains
from qwen_asr.repeat_chains import char_spans, merge_repeat_chains, move_degenerate_copies, rejoin_split_copies
from qwen_asr.speaker.assign import CueSpeaker, SpeakerAssigner, repeat_cuts
from qwen_asr.speaker.voiceprint import BLOCKED_SCORE, detect_members, local_regions, member_scores

MEMBERS = ["理名", "夕莉", "朱李"]
A = [0.7, 0.3, 0.2]  # clearly 理名
B = [0.3, 0.7, 0.2]  # clearly 夕莉
AMBIG = [0.5, 0.48, 0.2]


def _new_cue(text: str, start: int, end: int) -> ASRDataSeg:
    return ASRDataSeg(text, start, end)


def _cues(segments: list[ASRDataSeg]) -> list[tuple[str, int, int]]:
    return [(c.text, c.start_time, c.end_time) for c in segments]


# --- presence: low-scoring episodes and members heard only in one stretch ---------------------------------------


def _window_scores(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.full((n, 3), 0.3) + rng.normal(0, 0.01, (n, 3))


def test_detect_members_keeps_strong_member_with_low_p95() -> None:
    rng = np.random.default_rng(0)
    scores = _window_scores(1000, rng)
    scores[:500, 0] = 0.7
    scores[500:900, 1] = 0.7
    scores[900:, 2] = 0.48  # 10 % confident windows but p95 below 0.52 (public recording)
    present, stats = detect_members(scores, MEMBERS)
    assert present == MEMBERS
    assert stats["朱李"]["p95"] < 0.52


def test_local_regions_finds_member_dominating_one_stretch() -> None:
    rng = np.random.default_rng(1)
    hop = 0.2
    scores = _window_scores(3000, rng)  # 600 s
    scores[:, 0] = 0.7
    scores[1500:1640, 1] = 0.8  # 300-328 s: 夕莉 (4.7 % of the windows overall)
    scores[1500:1640, 0] = 0.3
    present, _ = detect_members(scores, MEMBERS, min_share=0.2)
    assert present == ["理名"]
    regions = local_regions(scores, MEMBERS, present, hop, win=0.75)
    assert list(regions) == ["夕莉"]
    [(start, end)] = regions["夕莉"]
    assert 296 <= start <= 300 and 328 <= end <= 332


def test_member_scores_block_local_member_outside_her_region() -> None:
    rng = np.random.default_rng(2)
    hop, win = 0.2, 0.75
    scores = _window_scores(3000, rng)
    scores[:, 0] = 0.7
    scores[1500:1640, 1] = 0.8
    scores[1500:1640, 0] = 0.3
    scores[2500:2505, 1] = 0.8  # a short stray match later on: must not count
    scores[2500:2505, 0] = 0.3
    n_frames = 60000
    used, frames, info = member_scores(scores, MEMBERS, n_frames, win=win, hop=hop)
    assert used == ["理名", "夕莉"]
    assert info["detected"] == ["理名"] and list(info["local"]) == ["夕莉"]
    assert (frames[: 29000, 1] == BLOCKED_SCORE).all()
    assert (frames[50050:50150, 1] == BLOCKED_SCORE).all()
    assert (frames[31000:33000, 1] > 0).all()
    # manual speakers: no local detection, scores everywhere
    used, frames, info = member_scores(scores, MEMBERS, n_frames, win=win, hop=hop, speakers=["理名", "朱李"])
    assert used == ["理名", "朱李"] and info["local"] == {}
    assert (frames > 0).all()


# --- repeated phrases inside one cue ------------------------------------------------------------------------------


def _tokens(*spec: tuple[str, list[float]]) -> tuple[list[str], np.ndarray]:
    return [s[0] for s in spec], np.array([s[1] for s in spec], np.float32)


def test_repeat_cuts_between_copies_said_by_different_members() -> None:
    texts, scores = _tokens(("ありがとう", B), ("ござい", B), ("ます", B), ("ありがとう", A), ("ござい", A), ("ます", A))
    assert repeat_cuts(texts, scores, 0.08) == [3]


def test_repeat_cuts_needs_both_copies_confident_and_different() -> None:
    texts, scores = _tokens(("ありがとう", B), ("ござい", B), ("ます", B), ("ありがとう", AMBIG), ("ござい", AMBIG),
                            ("ます", AMBIG))
    assert repeat_cuts(texts, scores, 0.08) == []
    texts, scores = _tokens(("やり", A), ("ます", A), ("やり", A), ("ます", A))
    assert repeat_cuts(texts, scores, 0.08) == []


def test_repeat_cuts_never_cut_inside_a_copy() -> None:
    # a shifted position ("ございますありがとう" twice) matches as well as the copy boundary
    texts, scores = _tokens(("ありがとう", B), ("ござい", B), ("ます", A), ("ありがとう", A), ("ござい", A), ("ます", A),
                            ("ありがとう", A), ("ござい", A), ("ます", A))
    assert repeat_cuts(texts, scores, 0.08) in ([], [3])


def test_split_cues_uses_repeat_cuts_and_keeps_text() -> None:
    texts = ["うます", "ぎて", "うます", "ぎ"]
    assigner = SpeakerAssigner(
        list(MEMBERS), texts, [(0, 300), (300, 600), (700, 1000), (1000, 1200)],
        np.array([B, B, A, A], np.float32), min_margin=0.5, repeat_margin=0.08,
    )
    cue = ASRDataSeg("うますぎて、うますぎ", 0, 1200)
    result, cuts = assigner.split_cues([cue], _new_cue)
    assert cuts == 1 and _cues(result) == [("うますぎて、", 0, 600), ("うますぎ", 700, 1200)]
    off = SpeakerAssigner(list(MEMBERS), texts, assigner.token_times, assigner.token_scores, 0.5, None)
    assert off.split_cues([cue], _new_cue) == ([cue], 0)


# --- repeated phrases across cues ---------------------------------------------------------------------------------


def test_merge_repeat_chains_joins_copies_unless_speaker_confidently_changes() -> None:
    cues = [ASRDataSeg("やります。", 0, 500), ASRDataSeg("やります。", 500, 1000), ASRDataSeg("やります。", 1000, 1500),
            ASRDataSeg("次の話。", 1600, 2600)]
    merged, joins = merge_repeat_chains(cues, _new_cue)
    assert joins == 2 and _cues(merged) == [("やります、やります、やります。", 0, 1500), ("次の話。", 1600, 2600)]

    labels = [CueSpeaker("理名", 0.3), CueSpeaker("夕莉", 0.3), CueSpeaker("夕莉", 0.02), CueSpeaker("", 0.0)]
    merged, joins = merge_repeat_chains(cues, _new_cue, labels=labels, min_margin=0.1)
    assert joins == 1 and [c.text for c in merged] == ["やります。", "やります、やります。", "次の話。"]


def test_merge_repeat_chains_leaves_short_backchannels_and_far_copies() -> None:
    cues = [ASRDataSeg("そう。", 0, 300), ASRDataSeg("そう。", 300, 600)]
    assert merge_repeat_chains(cues, _new_cue) == (cues, 0)
    far = [ASRDataSeg("やります。", 0, 500), ASRDataSeg("やります。", 3000, 3500)]
    assert merge_repeat_chains(far, _new_cue) == (far, 0)


def test_rejoin_split_copies_undoes_cut_inside_a_copy() -> None:
    cues = [ASRDataSeg("ありがとうございますありがとう", 0, 1600), ASRDataSeg("ございますありがとうございます。", 1600, 2960),
            ASRDataSeg("そう雨多いよ", 3600, 5000)]
    joined, count = rejoin_split_copies(cues, _new_cue)
    assert count == 1
    assert _cues(joined)[0] == ("ありがとうございますありがとうございますありがとうございます。", 0, 2960)
    # a cut on a copy boundary stays
    whole = [ASRDataSeg("やります", 0, 500), ASRDataSeg("やります", 500, 1000)]
    assert rejoin_split_copies(whole, _new_cue) == (whole, 0)


def _tok(text: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def test_move_degenerate_copy_back_and_keep_next_sentence_time() -> None:
    tokens = [_tok("やり", 0, 250), _tok("ます。", 250, 500), _tok("やり", 500, 750), _tok("ます。", 750, 1000),
              _tok("やり", 1010, 1170), _tok("ます。", 1170, 1250), _tok("メンバー", 1250, 1330), _tok("が", 1330, 1490),
              _tok("リスナー", 1490, 1810)]
    cues = [ASRDataSeg("やります。", 0, 500), ASRDataSeg("やります。", 500, 1000),
            ASRDataSeg("やります。メンバーがリスナー", 1010, 1810)]
    moved_cues, moved = move_degenerate_copies(cues, char_spans(tokens, cues), _new_cue)
    assert moved == 1
    assert _cues(moved_cues) == [("やります。", 0, 500), ("やります、やります。", 500, 1000), ("メンバーがリスナー", 1010, 1810)]


def test_move_keeps_a_real_copy_in_place() -> None:
    tokens = [_tok("ルーレット", 0, 560), _tok("ルーレット", 1120, 1440), _tok("かまる", 1440, 1840), _tok("ね", 1840, 1920)]
    cues = [ASRDataSeg("ルーレット", 0, 560), ASRDataSeg("ルーレットかまるね", 1120, 1920)]
    assert move_degenerate_copies(cues, char_spans(tokens, cues), _new_cue) == (cues, 0)


def test_tidy_repeat_chains_merges_squeezed_chain_and_preserves_content() -> None:
    tokens = [_tok("やり", 0, 250), _tok("ます。", 250, 500), _tok("やり", 500, 750), _tok("ます。", 750, 1000),
              _tok("やり", 1010, 1170), _tok("ます。", 1170, 1250), _tok("メンバー", 1250, 1330), _tok("が", 1330, 1490)]
    cues = [ASRDataSeg("やります。", 0, 500), ASRDataSeg("やります。", 500, 1000),
            ASRDataSeg("やります。メンバーが", 1010, 1490)]
    result = tidy_repeat_chains(cues, tokens, _new_cue)
    assert _cues(result) == [("やります、やります、やります。", 0, 1000), ("メンバーが", 1010, 1490)]


def test_single_cue_of_copies_gets_commas_between_copies() -> None:
    cues = [ASRDataSeg("ありがとうございます。ありがとうございます。", 0, 2000), ASRDataSeg("今日は晴れ。明日は雨。", 2100, 4000)]
    merged, joins = merge_repeat_chains(cues, _new_cue)
    assert joins == 0
    assert [c.text for c in merged] == ["ありがとうございます、ありがとうございます。", "今日は晴れ。明日は雨。"]
