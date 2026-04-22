from __future__ import annotations

import concurrent.futures
import json
import os
import time
from typing import Any, Literal, Optional
from urllib import error, parse, request

Provider = Literal["openai", "anthropic", "google"]

_VALID_PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic", "google")


def infer_provider(*, model: str, provider: Optional[str] = None) -> Provider:
    if provider is not None:
        p = provider.strip().lower()
        if p in _VALID_PROVIDERS:
            return p  # type: ignore[return-value]
        raise ValueError(
            f"Unsupported provider={provider!r}. Use one of: {', '.join(_VALID_PROVIDERS)}."
        )

    m = model.strip().lower()
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
    elif resolved_provider == "anthropic":
        env_key_name = "ANTHROPIC_API_KEY"
        env_base_name = "ANTHROPIC_BASE_URL"
        default_base = "https://api.anthropic.com/v1"
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
    elif resolved_provider == "anthropic":
        text = _anthropic_messages_completion(
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


def _anthropic_messages_completion(
    *,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    effort = _normalize_reasoning_effort(reasoning_effort)
    sdk_timeout_s = max(timeout_s, 900)  # keep Claude requests generous by default (15 min minimum)
    max_attempts = 3

    # Match ConceptARC-Rules first: anthropic SDK client.
    def _sdk_call() -> str:
        import anthropic  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url.rstrip("/") != "https://api.anthropic.com/v1":
            kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**kwargs)
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effort is not None:
            req["output_config"] = {"effort": effort}
        resp = client.messages.create(**req)
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            txt = getattr(block, "text", None)
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
        text = "\n".join(parts).strip()
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

    # Fallback: raw HTTP.
    url = base_url.rstrip("/") + "/messages"
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort is not None:
        payload["output_config"] = {"effort": effort}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
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
        with request.urlopen(req, timeout=sdk_timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Anthropic SDK failed ({last_sdk_error}); fallback HTTP API error {e.code}: {body_txt}"
        ) from e
    except Exception as e:
        raise ValueError(f"Anthropic SDK failed ({last_sdk_error}); fallback HTTP failed: {e}") from e

    parsed = json.loads(raw)
    blocks = parsed.get("content")
    parts: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                txt = block.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
    text = "\n".join(parts).strip()
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


def _normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    if effort is None:
        return None
    e = effort.strip().lower()
    if not e:
        return None
    allowed = {"low", "medium", "high", "xhigh", "max"}
    return e if e in allowed else None


def _map_google_thinking_level(effort: Optional[str]) -> Optional[str]:
    normalized = _normalize_reasoning_effort(effort)
    if normalized is None:
        return None
    if normalized == "xhigh" or normalized == "max":
        return "high"
    return normalized
