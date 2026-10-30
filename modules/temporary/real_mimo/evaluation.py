"""Experimental protocol-comparison wording checks; not an audio edit guard."""


def wording_assessment(candidate, reference, source, *, numbers, language_shares):
    reasons = []
    left = candidate or ""
    right = reference or ""
    if left == right:
        reasons.append("equal")
    cand_numbers = numbers(left)
    ref_numbers = numbers(right)
    source_numbers = set(numbers(source or ""))
    if cand_numbers != ref_numbers:
        reasons.append("numbers")
    if any(item not in source_numbers and item not in set(ref_numbers) for item in cand_numbers):
        reasons.append("new_number")
    ratio = (len(left) / len(right)) if right else None
    if ratio is not None and not (0.80 <= ratio <= 1.25) and left != right:
        reasons.append("length")
    shares_left = language_shares(left)
    shares_right = language_shares(right)
    if abs(shares_left["zh"] - shares_right["zh"]) > 0.15 or abs(shares_left["ja"] - shares_right["ja"]) > 0.15:
        reasons.append("language")
    hard = [reason for reason in reasons if reason in {"numbers", "new_number"}]
    return {
        "reasons": reasons,
        "hard": hard,
        "length_ratio": None if ratio is None else round(ratio, 4),
        "proper_nouns": "NOT_ASSESSED",
        "bracket_pairing": "NOT_ASSESSED",
        "positional_zip": False,
    }
