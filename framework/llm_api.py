from __future__ import annotations

import concurrent.futures
import datetime as dt
import http.client
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal, Optional
from urllib import error, parse, request

Provider = Literal["openai", "anthropic", "google", "together", "openrouter"]

_VALID_PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "google", "together", "openrouter")


def _full_response_debug_enabled() -> bool:
    v = os.environ.get("POTARCIN_DEBUG_FULL_RESPONSE", "")
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _together_max_tokens() -> int:
    raw = os.environ.get("POTARCIN_TOGETHER_MAX_TOKENS", "131072").strip()
    try:
        val = int(raw)
    except Exception:
        val = 131072
    # Together/OpenAI-compatible APIs expect a positive integer.
    # Clamp to a sane upper bound instead of using undocumented sentinels like -1.
    if val <= 0:
        return 131072
    return min(val, 262144)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _is_kimi_together_model(model: str) -> bool:
    """Kimi (Moonshot on Together) — used for Kimi-only speed/safety toggles."""
    m = model.strip().lower()
    return m.startswith("moonshotai/") or "kimi" in m


def _kimi_fast_preset() -> bool:
    return os.environ.get("POTARCIN_KIMI_PRESET", "").strip().lower() == "fast"


def _together_effective_max_tokens(*, model: str) -> int:
    """
    Kimi-only tuning (other Together models use POTARCIN_TOGETHER_MAX_TOKENS only):
    - POTARCIN_KIMI_PRESET=fast: lower default max_tokens (faster, less reasoning spam).
    - POTARCIN_KIMI_MAX_TOKENS: explicit cap for Kimi (overrides generic TOGETHER default).
    """
    if _is_kimi_together_model(model) and _kimi_fast_preset():
        raw = os.environ.get("POTARCIN_KIMI_FAST_MAX_TOKENS", "32768").strip()
        try:
            v = int(raw)
        except Exception:
            v = 32768
        return min(max(1024, v), 262144)
    if _is_kimi_together_model(model):
        raw = os.environ.get("POTARCIN_KIMI_MAX_TOKENS", "").strip()
        if raw:
            try:
                v = int(raw)
            except Exception:
                v = _together_max_tokens()
            return min(max(1, v), 262144)
    return _together_max_tokens()


def _together_kimi_reasoning_param(*, model: str) -> dict[str, bool]:
    if not _is_kimi_together_model(model):
        return {"enabled": True}
    if _kimi_fast_preset():
        return {"enabled": _env_bool("POTARCIN_KIMI_FAST_REASONING", False)}
    return {"enabled": _env_bool("POTARCIN_KIMI_REASONING_ENABLED", True)}


def _together_kimi_reasoning_and_temperature(
    *, model: str, reasoning_effort: Optional[str]
) -> tuple[dict[str, bool], float]:
    """
    Kimi-only override:
    - reasoning_effort=none => disable thinking and use lower temperature for non-thinking mode.
    """
    if _is_kimi_together_model(model) and isinstance(reasoning_effort, str):
        if reasoning_effort.strip().lower() == "none":
            return {"enabled": False}, 0.6
    return _together_kimi_reasoning_param(model=model), 1.0


def _together_transient_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            TimeoutError,
            BrokenPipeError,
            ConnectionError,
            ConnectionAbortedError,
            ConnectionResetError,
            http.client.IncompleteRead,
        ),
    ):
        return True
    if isinstance(exc, error.URLError):
        return True
    msg = str(exc).lower()
    if "503" in msg or "502" in msg or "429" in msg or "504" in msg:
        return True
    for phrase in (
        "service unavailable",
        "rate limit",
        "overloaded",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "did not contain any text content",
        "incompleteread",
        "incomplete read",
    ):
        if phrase in msg:
            return True
    if "time" in msg and ("out" in msg or "timed" in msg):
        return True
    return False


def _openrouter_empty_content_error(payload: Any) -> ValueError:
    """Build a diagnostic ValueError when OpenRouter returns no extractable text."""
    finish_reason = None
    native_finish_reason = None
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                finish_reason = c0.get("finish_reason")
                native_finish_reason = c0.get("native_finish_reason")
        usage = payload.get("usage")
    else:
        usage = None
    parts = ["OpenRouter Chat Completions response did not contain any text content."]
    if finish_reason is not None:
        parts.append(f"finish_reason={finish_reason!r}")
    if native_finish_reason is not None:
        parts.append(f"native_finish_reason={native_finish_reason!r}")
    if usage is not None:
        try:
            parts.append(f"usage={json.dumps(usage, sort_keys=True)}")
        except Exception:
            parts.append(f"usage={usage!r}")
    return ValueError(" ".join(parts))


def _openrouter_retry_count(*, model: str) -> int:
    try:
        n = int(os.environ.get("POTARCIN_OPENROUTER_MAX_RETRIES", "0").strip() or 0)
    except Exception:
        n = 0
    if n > 0:
        return min(max(1, n), 8)
    # Kimi via OpenRouter often returns empty content / truncated streams; retry more.
    default = "3" if _is_kimi_together_model(model) else "1"
    try:
        n = int(os.environ.get("POTARCIN_TOGETHER_MAX_RETRIES", default).strip() or default)
    except Exception:
        n = int(default)
    return min(max(1, n), 8)


def _together_retry_count(*, model: str) -> int:
    # Keep retries conservative for Kimi to avoid nested retry explosions:
    # per-dimension retries already exist at a higher layer.
    default_retries = "1" if _is_kimi_together_model(model) else "2"
    try:
        n = int(os.environ.get("POTARCIN_TOGETHER_MAX_RETRIES", default_retries).strip())
    except Exception:
        n = int(default_retries)
    return min(max(1, n), 8)


def _together_sdk_wall_timeout_s(*, timeout_s: int, model: str) -> int:
    try:
        extra = int(os.environ.get("POTARCIN_TOGETHER_SDK_TIMEOUT_S", "0").strip() or 0)
    except Exception:
        extra = 0
    if extra > 0:
        return extra
    # Together/Kimi should closely track configured request timeout to avoid
    # silently stretching single calls into 15-minute waits.
    buffer_s = 45 if _is_kimi_together_model(model) else 30
    return max(60, timeout_s + buffer_s)


def _dump_full_response_debug(*, provider: str, model: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    print("=== FULL_API_RESPONSE_START ===")
    print(raw)
    print("=== FULL_API_RESPONSE_END ===")
    try:
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("evaluation") / "raw_api_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_model = "".join(ch for ch in model if ch.isalnum() or ch in {"-", "_"})
        out_path = out_dir / f"{ts}_{provider}_{safe_model}.json"
        out_path.write_text(raw + "\n", encoding="utf-8")
        print(f"Saved full API response: {out_path}")
    except Exception as e:
        print(f"Warning: failed to save full API response debug dump: {e}")
    return raw


def infer_provider(*, model: str, provider: Optional[str] = None) -> Provider:
    if provider is not None:
        p = provider.strip().lower()
        if p in _VALID_PROVIDERS:
            return p  # type: ignore[return-value]
        raise ValueError(
            f"Unsupported provider={provider!r}. Use one of: {', '.join(_VALID_PROVIDERS)}."
        )

    m = model.strip().lower()
    if m.startswith("minimax/") or m.startswith("openrouter/") or ":nitro" in m:
        return "openrouter"
    if (
        m.startswith("moonshotai/")
        or m.startswith("kimi")
        or m.startswith("minimaxai/")
        or "minimax" in m
    ):
        return "together"
    if m.startswith("claude") or m.startswith("anthropic/claude"):
        return "anthropic"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    return "openai"


def resolve_provider_config(
    *,
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[Provider, str, str]:
    resolved_provider = infer_provider(model=model, provider=provider)
    if resolved_provider == "openai":
        env_key_name = "OPENAI_API_KEY"
        env_base_name = "OPENAI_BASE_URL"
        default_base = "https://api.openai.com/v1"
    elif resolved_provider == "openrouter":
        env_key_name = "OPENROUTER_API_KEY"
        env_base_name = "OPENROUTER_BASE_URL"
        default_base = "https://openrouter.ai/api/v1"
    elif resolved_provider == "anthropic":
        env_key_name = "ANTHROPIC_API_KEY"
        env_base_name = "ANTHROPIC_BASE_URL"
        default_base = "https://api.anthropic.com/v1"
    elif resolved_provider == "together":
        env_key_name = "TOGETHER_API_KEY"
        env_base_name = "TOGETHER_BASE_URL"
        default_base = "https://api.together.xyz/v1"
    else:
        # Keep parity with ConceptARC-Rules: accept either GEMINI_API_KEY or GOOGLE_API_KEY.
        env_key_name = "GEMINI_API_KEY"
        env_base_name = "GOOGLE_BASE_URL"
        default_base = "https://generativelanguage.googleapis.com/v1beta"

    if resolved_provider == "google":
        effective_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    else:
        effective_key = api_key or os.environ.get(env_key_name)
    if not effective_key:
        raise ValueError(
            f"No API key provided for provider={resolved_provider}. "
            f"Set `api_key` or {env_key_name}."
        )
    effective_key = effective_key.strip()
    effective_base = (base_url or os.environ.get(env_base_name, default_base)).strip()
    return resolved_provider, effective_key, effective_base


def request_text_completion(
    *,
    prompt: str,
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    timeout_s: int = 180,
) -> tuple[str, Provider, str]:
    resolved_provider, effective_key, effective_base = resolve_provider_config(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )

    if resolved_provider == "openai":
        text = _openai_responses_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            tools_enabled=tools_enabled,
            timeout_s=timeout_s,
        )
    elif resolved_provider == "openrouter":
        try:
            to = int(os.environ.get("POTARCIN_OPENROUTER_TIMEOUT_S", "0").strip() or 0)
        except Exception:
            to = 0
        if to <= 0:
            # Optional compatibility fallback so existing timeout env still works.
            try:
                to = int(os.environ.get("POTARCIN_TOGETHER_TIMEOUT_S", "0").strip() or 0)
            except Exception:
                to = 0
        if to > 0:
            timeout_s = to
        elif _is_kimi_together_model(model):
            # Same practical floor as Together/Kimi; request body unchanged.
            timeout_s = max(timeout_s, 420)
        text = _openrouter_chat_completions_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            timeout_s=timeout_s,
        )
    elif resolved_provider == "anthropic":
        text = _anthropic_messages_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
        )
    elif resolved_provider == "together":
        try:
            to = int(os.environ.get("POTARCIN_TOGETHER_TIMEOUT_S", "0").strip() or 0)
        except Exception:
            to = 0
        if to > 0:
            timeout_s = to
        elif _is_kimi_together_model(model):
            # Slightly more permissive default for Kimi while staying practical.
            # Explicit env override (POTARCIN_TOGETHER_TIMEOUT_S) still wins.
            timeout_s = max(timeout_s, 420)
        text = _together_chat_completions_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
        )
    else:
        text = _google_generate_content_completion(
            prompt=prompt,
            model=model,
            api_key=effective_key,
            base_url=effective_base,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
        )
    return text.strip(), resolved_provider, effective_base


def _openrouter_extract_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    for c in choices:
        if not isinstance(c, dict):
            continue
        msg = c.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                if parts:
                    return "\n".join(parts)
        txt = c.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    return ""


def _openrouter_chat_completions_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: int = 180,
) -> str:
    # Match the Together MiniMax defaults used in this project.
    # Request body intentionally unchanged for MiniMax / prior Kimi comparability.
    max_tokens = _together_max_tokens()
    reasoning_cfg = {"enabled": True}
    temperature = 1.0
    top_p = 0.95
    n_attempts = _openrouter_retry_count(model=model)

    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": reasoning_cfg,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "PotARCin/1.0",
        },
        method="POST",
    )
    http_read_s = max(timeout_s, 120)
    last_err: Exception | None = None
    for attempt in range(1, n_attempts + 1):
        try:
            with request.urlopen(req, timeout=http_read_s) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            if _full_response_debug_enabled():
                return _dump_full_response_debug(
                    provider="openrouter_http",
                    model=model,
                    payload=parsed,
                )
            text = _openrouter_extract_text(parsed)
            if text:
                return text
            raise _openrouter_empty_content_error(parsed)
        except error.HTTPError as e:
            body_txt = e.read().decode("utf-8", errors="replace")
            err: Exception = ValueError(f"OpenRouter Chat Completions API error {e.code}: {body_txt}")
            last_err = err
        except Exception as e:
            last_err = e
        if attempt < n_attempts and last_err is not None and _together_transient_error(last_err):
            time.sleep(min(60.0, (2.0 ** (attempt - 1)) + random.random()))
            continue
        break
    if last_err is None:
        raise RuntimeError("OpenRouter: retry loop finished without a stored error (internal bug).")
    raise last_err


def _openai_responses_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    tools_enabled: bool = False,
    timeout_s: int = 180,
) -> str:
    url = base_url.rstrip("/") + "/responses"
    payload: dict[str, Any] = {"model": model, "input": prompt}
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if tools_enabled:
        payload["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"OpenAI Responses API error {e.code}: {body_txt}") from e
    parsed = json.loads(raw)
    out_text = parsed.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text
    parts: list[str] = []
    for item in parsed.get("output", []):
        for c in item.get("content", []):
            txt = c.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    return "\n".join(parts).strip()


def _anthropic_message_text_from_content(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        txt = getattr(block, "text", None)
        if isinstance(txt, str) and txt.strip():
            parts.append(txt)
            continue
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t)
    return "\n".join(parts).strip()


def _anthropic_messages_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    """
    Anthropic Messages API completion.

    Thinking mode (enabled + 120k by default): stream only. No messages.create(),
    no raw-HTTP fallback, and no wall-clock / client timeout — those were the
    credit-burn path on Run2 and must not cut off long 120k generations.
    """
    effort = _normalize_reasoning_effort(reasoning_effort)
    thinking_budget_tokens = _anthropic_thinking_budget_tokens(effort)
    # Anthropic requires: thinking.budget_tokens < max_tokens.
    max_tokens = 4096
    if thinking_budget_tokens is not None:
        max_tokens = max(max_tokens, thinking_budget_tokens + 1024)
    thinking_enabled = thinking_budget_tokens is not None

    def _build_req() -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effort is not None:
            req["output_config"] = {"effort": effort}
        if thinking_budget_tokens is not None:
            req["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
        return req

    def _make_client() -> Any:
        import anthropic  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url.rstrip("/") != "https://api.anthropic.com/v1":
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)

    # --- Thinking: stream on the calling thread; no timeouts; no fallbacks. ---
    if thinking_enabled:
        client = _make_client()
        req = _build_req()
        stream_fn = getattr(client.messages, "stream", None)
        if not callable(stream_fn):
            raise RuntimeError(
                "Anthropic thinking is enabled but messages.stream() is unavailable; "
                "refusing non-streaming fallback to avoid unbounded token burn."
            )
        with stream_fn(**req) as stream:
            parts: list[str] = []
            text_stream = getattr(stream, "text_stream", None)
            if text_stream is not None:
                for chunk in text_stream:
                    if isinstance(chunk, str) and chunk:
                        parts.append(chunk)
            text = "".join(parts).strip()
            if not text:
                final_msg = getattr(stream, "get_final_message", None)
                if callable(final_msg):
                    text = _anthropic_message_text_from_content(
                        getattr(final_msg(), "content", None)
                    )
            if text:
                return text
        raise ValueError(
            "Anthropic streaming response completed without any text content "
            "(thinking-only or empty). Refusing non-streaming fallback."
        )

    # --- Non-thinking: prior create() + ThreadPool + HTTP fallback behavior. ---
    anth_to = int(os.environ.get("POTARCIN_ANTHROPIC_TIMEOUT_S", "0").strip() or 0)
    if anth_to > 0:
        timeout_s = anth_to
    sdk_timeout_s = max(timeout_s, 900)
    max_attempts = 3

    def _sdk_call() -> str:
        client = _make_client()
        resp = client.messages.create(**_build_req())
        text = _anthropic_message_text_from_content(getattr(resp, "content", None))
        if text:
            return text
        raise ValueError("Anthropic SDK response did not contain any text content.")

    last_sdk_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_sdk_call)
                return fut.result(timeout=sdk_timeout_s)
        except concurrent.futures.TimeoutError:
            last_sdk_error = TimeoutError(
                f"Anthropic SDK call timed out after {sdk_timeout_s}s (attempt {attempt}/{max_attempts})."
            )
        except Exception as e:
            last_sdk_error = e
        if attempt < max_attempts:
            time.sleep(min(4, attempt))

    url = base_url.rstrip("/") + "/messages"
    payload = _build_req()
    body = json.dumps(payload).encode("utf-8")
    http_req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_req, timeout=sdk_timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Anthropic SDK failed ({last_sdk_error}); fallback HTTP API error {e.code}: {body_txt}"
        ) from e
    except Exception as e:
        raise ValueError(f"Anthropic SDK failed ({last_sdk_error}); fallback HTTP failed: {e}") from e

    parsed = json.loads(raw)
    text = _anthropic_message_text_from_content(parsed.get("content"))
    if text:
        return text
    raise ValueError(f"Anthropic SDK failed ({last_sdk_error}); fallback HTTP had no text content.")


def _google_generate_content_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    thinking_level = _map_google_thinking_level(reasoning_effort)
    sdk_timeout_s = max(timeout_s, 900)  # keep Gemini requests generous by default (15 min minimum)
    max_attempts = 3

    def _sdk_call() -> str:
        from google import genai  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url.rstrip("/") != "https://generativelanguage.googleapis.com/v1beta":
            kwargs["http_options"] = {"base_url": base_url.rstrip("/")}
        client = genai.Client(**kwargs)
        model_id = model if model.startswith("models/") else f"models/{model}"
        req_sdk: dict[str, Any] = {"model": model_id, "contents": [prompt]}
        if thinking_level is not None:
            req_sdk["config"] = {"thinking_config": {"thinking_level": thinking_level}}
        resp = client.models.generate_content(**req_sdk)
        txt = getattr(resp, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
        raise ValueError("Google SDK response did not contain any text content.")

    last_sdk_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_sdk_call)
                return fut.result(timeout=sdk_timeout_s)
        except concurrent.futures.TimeoutError:
            last_sdk_error = TimeoutError(
                f"Google SDK call timed out after {sdk_timeout_s}s (attempt {attempt}/{max_attempts})."
            )
        except Exception as e:
            last_sdk_error = e
        if attempt < max_attempts:
            time.sleep(min(4, attempt))

    # Fallback: raw HTTP.
    model_path = model if model.startswith("models/") else f"models/{model}"
    key_q = parse.quote(api_key, safe="")
    url = f"{base_url.rstrip('/')}/{model_path}:generateContent?key={key_q}"
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    if thinking_level is not None:
        payload["generationConfig"] = {"thinkingConfig": {"thinkingLevel": thinking_level}}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=sdk_timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Google SDK failed ({last_sdk_error}); fallback HTTP API error {e.code}: {body_txt}"
        ) from e
    except Exception as e:
        raise ValueError(f"Google SDK failed ({last_sdk_error}); fallback HTTP failed: {e}") from e

    parsed = json.loads(raw)
    parts: list[str] = []
    for cand in parsed.get("candidates", []):
        content = cand.get("content", {}) if isinstance(cand, dict) else {}
        for part in content.get("parts", []) if isinstance(content, dict) else []:
            txt = part.get("text") if isinstance(part, dict) else None
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
    text = "\n".join(parts).strip()
    if text:
        return text
    raise ValueError(f"Google SDK failed ({last_sdk_error}); fallback HTTP had no text content.")


def _together_chat_completions_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    max_tokens = _together_effective_max_tokens(model=model)
    reasoning_cfg, temperature = _together_kimi_reasoning_and_temperature(
        model=model, reasoning_effort=reasoning_effort
    )
    last_err: Exception | None = None
    n_attempts = _together_retry_count(model=model)
    for attempt in range(1, n_attempts + 1):
        try:
            return _together_chat_completions_completion_once(
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                reasoning_cfg=reasoning_cfg,
                temperature=temperature,
            )
        except Exception as e:
            last_err = e
            if attempt >= n_attempts or not _together_transient_error(e):
                break
            delay = min(60.0, (2.0 ** (attempt - 1)) + random.random())
            time.sleep(delay)
    if last_err is None:
        raise RuntimeError("Together: retry loop finished without a stored error (internal bug).")
    raise last_err


def _together_chat_completions_completion_once(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: int = 180,
    max_tokens: int,
    reasoning_cfg: dict[str, bool],
    temperature: float,
) -> str:
    sdk_error: Exception | None = None
    sdk_wall_s = _together_sdk_wall_timeout_s(timeout_s=timeout_s, model=model)

    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                        continue
                    nested = item.get("content")
                    if isinstance(nested, str) and nested.strip():
                        parts.append(nested.strip())
            if parts:
                return "\n".join(parts)
        if isinstance(content, dict):
            txt = content.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
            nested = content.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return ""

    def _extract_text_from_payload(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        # OpenAI-compatible surface
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                txt = _extract_text_from_content(choice.get("message"))
                if txt:
                    return txt
                txt2 = _extract_text_from_content(choice.get("text"))
                if txt2:
                    return txt2
        # Responses-like surface
        out_text = payload.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            return out_text.strip()
        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                txt = _extract_text_from_content(item.get("content"))
                if txt:
                    parts.append(txt)
            if parts:
                return "\n".join(parts)
        return ""

    # Prefer Together SDK: this matches user-verified behavior for Kimi.
    try:
        from together import Together  # type: ignore[import-not-found]

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url.rstrip("/") != "https://api.together.xyz/v1":
            client_kwargs["base_url"] = base_url.rstrip("/")
        client = Together(**client_kwargs)

        def _sdk_create() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning=reasoning_cfg,
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_sdk_create)
            try:
                resp = fut.result(timeout=sdk_wall_s)
            except concurrent.futures.TimeoutError as te:
                raise TimeoutError(
                    f"Together SDK call timed out after {sdk_wall_s}s (model={model!r})."
                ) from te
        choices = getattr(resp, "choices", None)
        if isinstance(choices, list):
            for choice in choices:
                msg = getattr(choice, "message", None)
                txt = _extract_text_from_content(getattr(msg, "content", None))
                if txt:
                    return txt
                # Some SDK responses expose text on the choice directly.
                txt2 = _extract_text_from_content(getattr(choice, "text", None))
                if txt2:
                    return txt2
        # Some SDK versions expose full payload via model_dump()/dict().
        dumped: dict[str, Any] | None = None
        if hasattr(resp, "model_dump"):
            try:
                dumped = resp.model_dump()  # type: ignore[assignment, attr-defined]
            except Exception:
                dumped = None
        if dumped is None and hasattr(resp, "dict"):
            try:
                dumped = resp.dict()  # type: ignore[assignment, attr-defined]
            except Exception:
                dumped = None
        if dumped is not None and _full_response_debug_enabled():
            # Special debug mode: return full payload unmodified (no extraction).
            return _dump_full_response_debug(
                provider="together_sdk",
                model=model,
                payload=dumped,
            )
        if dumped is not None:
            txt3 = _extract_text_from_payload(dumped)
            if txt3:
                return txt3
    except Exception as e:
        sdk_error = e

    http_read_s = max(timeout_s, 120)
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": reasoning_cfg,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "PotARCin/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=http_read_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"Together Chat Completions API error {e.code}: {body_txt}") from e

    parsed = json.loads(raw)
    if _full_response_debug_enabled():
        # Special debug mode: return full payload unmodified (no extraction).
        return _dump_full_response_debug(
            provider="together_http",
            model=model,
            payload=parsed,
        )
    txt = _extract_text_from_payload(parsed)
    if txt:
        return txt
    parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else []
    raise ValueError(
        "Together Chat Completions response did not contain any text content. "
        f"sdk_error={sdk_error!r}; response_keys={parsed_keys}"
    )


def _normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    if effort is None:
        return None
    e = effort.strip().lower()
    if not e:
        return None
    allowed = {"low", "medium", "high", "xhigh", "max"}
    return e if e in allowed else None


def _anthropic_thinking_budget_tokens(effort: Optional[str]) -> int | None:
    """
    Resolve Anthropic extended-thinking budget token configuration.

    Uses POTARCIN_ANTHROPIC_THINKING_BUDGET_TOKENS when set.
    Default with any valid reasoning effort: 120000 tokens.
    Set env var to 0 or negative to disable explicit thinking config.
    """
    normalized = _normalize_reasoning_effort(effort)
    if normalized is None:
        return None

    raw = os.environ.get("POTARCIN_ANTHROPIC_THINKING_BUDGET_TOKENS", "").strip()
    if raw:
        try:
            budget = int(raw)
        except Exception:
            budget = 120000
    else:
        budget = 120000

    if budget <= 0:
        return None
    return max(1024, budget)


def _map_google_thinking_level(effort: Optional[str]) -> Optional[str]:
    normalized = _normalize_reasoning_effort(effort)
    if normalized is None:
        return None
    if normalized == "xhigh" or normalized == "max":
        return "high"
    return normalized
