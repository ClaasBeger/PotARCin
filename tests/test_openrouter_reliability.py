from __future__ import annotations

import http.client

from framework.llm_api import (
    _openrouter_empty_content_error,
    _openrouter_extract_text,
    _openrouter_retry_count,
    _together_transient_error,
)
from scripts.run_all_dimensions_eval import _is_transient_error


def test_openrouter_extract_text_string_content():
    payload = {"choices": [{"message": {"content": "  hello  "}, "finish_reason": "stop"}]}
    assert _openrouter_extract_text(payload) == "hello"


def test_openrouter_empty_content_error_includes_diagnostics():
    payload = {
        "choices": [
            {
                "message": {"content": ""},
                "finish_reason": "length",
                "native_finish_reason": "MAX_TOKENS",
            }
        ],
        "usage": {"completion_tokens": 12, "prompt_tokens": 3},
    }
    err = _openrouter_empty_content_error(payload)
    msg = str(err)
    assert "did not contain any text content" in msg
    assert "finish_reason='length'" in msg
    assert "native_finish_reason='MAX_TOKENS'" in msg
    assert "completion_tokens" in msg


def test_together_transient_error_empty_content_and_incomplete_read():
    assert _together_transient_error(
        ValueError("OpenRouter Chat Completions response did not contain any text content.")
    )
    assert _together_transient_error(http.client.IncompleteRead(b"partial"))
    assert _together_transient_error(ValueError("IncompleteRead(3817 bytes read)"))


def test_dimension_transient_error_markers():
    assert _is_transient_error(
        "OpenRouter Chat Completions response did not contain any text content. finish_reason='length'"
    )
    assert _is_transient_error("IncompleteRead(7755 bytes read)")
    assert not _is_transient_error("Could not locate dict object in model output.")


def test_openrouter_retry_count_kimi_default(monkeypatch):
    monkeypatch.delenv("POTARCIN_OPENROUTER_MAX_RETRIES", raising=False)
    monkeypatch.delenv("POTARCIN_TOGETHER_MAX_RETRIES", raising=False)
    assert _openrouter_retry_count(model="moonshotai/Kimi-K2.5") == 3
    assert _openrouter_retry_count(model="minimax/minimax-m2.5:nitro") == 1


def test_openrouter_retry_count_env_override(monkeypatch):
    monkeypatch.setenv("POTARCIN_OPENROUTER_MAX_RETRIES", "5")
    assert _openrouter_retry_count(model="moonshotai/Kimi-K2.5") == 5
