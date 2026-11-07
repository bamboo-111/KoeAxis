from __future__ import annotations

from optimizer.asr_cleanup import clean_subtitle_text as direct_clean_subtitle_text
from optimizer.text_metrics import count_words
from optimizer.text_utils import clean_asr_correction_text, clean_subtitle_text


def test_email_cleanup() -> None:
    assert clean_subtitle_text("test @ example . com") == "test@example.com"
    assert clean_subtitle_text("test at example . com") == "test@example.com"


def test_fixed_term_cleanup_smoke() -> None:
    source = "togeraji interfm j p"
    assert "togeraji@interfm.jp" in clean_asr_correction_text(source)


def test_text_utils_keeps_compat_exports() -> None:
    source = "test @ example . com"

    assert clean_subtitle_text(source) == direct_clean_subtitle_text(source)
    assert count_words("hello world") == 2
