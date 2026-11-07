"""Experimental wording assessment, not a production stage success gate.

Dependencies are explicit so the historical operations-module test hooks remain
observable. This module neither loads baselines nor invokes a provider.
"""


def wording_assessment(candidate, reference, source, *, text_marks, numbers):
    """Implemented checks only; proper nouns/bracket pairing stay unassessed."""
    left = text_marks(candidate)
    right = text_marks(reference)
    source_numbers = set(numbers(source))
    introduced = [number for number in left["numbers"] if number not in source_numbers]
    reasons = []
    texts_equal = left["sha256"] == right["sha256"]
    if not texts_equal and left["numbers"] != right["numbers"]:
        reasons.append("numbers")
    if introduced:
        reasons.append("new_number")
    if not texts_equal and left["negations"] != right["negations"]:
        reasons.append("negations")
    ratio = left["length"] / max(1, right["length"])
    if not texts_equal and (ratio < 0.80 or ratio > 1.25):
        reasons.append("length")
    if not texts_equal:
        for name in ("kana", "han"):
            if abs(left["language"][name] - right["language"][name]) > 0.15:
                reasons.append("language")
                break
    if texts_equal and not reasons:
        primary = "equal"
    elif reasons:
        primary = reasons[0]
    else:
        primary = "wording"
    return {
        "implemented_allowed": not reasons,
        "reason": primary,
        "reasons": reasons,
        "texts_equal": texts_equal,
        "proper_nouns": "same_text" if texts_equal else "NOT_ASSESSED",
        "bracket_pairing": "same_text" if texts_equal else "NOT_ASSESSED",
        "quote_char_mismatch": left["quotes"] != right["quotes"],
        "length_ratio": round(ratio, 4),
        "numbers_a": left["numbers"],
        "numbers_b": right["numbers"],
        "negations_a": left["negations"],
        "negations_b": right["negations"],
        "quote_chars_a": left["quotes"],
        "quote_chars_b": right["quotes"],
        "language_a": {key: round(value, 4) for key, value in left["language"].items()},
        "language_b": {key: round(value, 4) for key, value in right["language"].items()},
        "sha_a": left["sha256"],
        "sha_b": right["sha256"],
    }
