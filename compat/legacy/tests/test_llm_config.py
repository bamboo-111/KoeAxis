from __future__ import annotations

from optimizer.llm_config import LLMConfig
from optimizer.translator import SubtitleTranslator
from qwen_asr.corrector import ASRCorrector


def test_llm_config_defaults() -> None:
    config = LLMConfig(model="m", base_url="http://localhost:8000/v1", api_key="k")

    assert config.disable_thinking is True
    assert config.timeout == 120.0
    assert config.llm_extra_body is None


def test_translator_and_corrector_keep_legacy_attrs_and_config() -> None:
    extra_body = {"thinking": {"type": "disabled"}}
    translator = SubtitleTranslator(
        thread_num=1,
        batch_num=2,
        model="model",
        base_url="http://localhost:8000/v1",
        api_key="key",
        target_language="zh",
        disable_thinking=False,
        llm_extra_body=extra_body,
        timeout=9,
    )
    corrector = ASRCorrector(
        model="model",
        base_url="http://localhost:8000/v1",
        api_key="key",
        thread_num=1,
        batch_num=2,
        disable_thinking=False,
        llm_extra_body=extra_body,
        timeout=9,
    )
    try:
        assert translator.model == translator.llm_config.model == "model"
        assert translator.disable_thinking is False
        assert translator.llm_config.llm_extra_body == extra_body
        assert corrector.base_url == corrector.llm_config.base_url
        assert corrector.llm_config.timeout == 9
    finally:
        translator.stop()
        corrector.stop()
