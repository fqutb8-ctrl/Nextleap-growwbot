from __future__ import annotations

import os
import time

import httpx

import config
from rag.logging_utils import get_logger

log = get_logger("llm")

TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    pass


class LLMNotConfiguredError(LLMError):
    pass


def api_key() -> str:
    return (os.getenv(config.LLM_API_KEY_ENV) or "").strip()


def endpoint() -> str:
    base = (config.LLM_BASE_URL or "").rstrip("/")
    if not base:
        raise LLMNotConfiguredError(
            f"no base URL for provider {config.LLM_PROVIDER!r}; set LLM_BASE_URL"
        )
    return f"{base}/chat/completions"


def request_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def build_payload(system: str, user: str) -> dict:
    return {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "top_p": config.LLM_TOP_P,
        "max_tokens": config.LLM_MAX_TOKENS,
        "stream": False,
    }


def _extract(response: httpx.Response) -> str:
    if response.status_code in TRANSIENT_STATUS:
        raise httpx.HTTPStatusError(
            f"transient status {response.status_code}",
            request=response.request,
            response=response,
        )
    if response.status_code >= 400:
        raise LLMError(
            f"{config.LLM_PROVIDER} returned {response.status_code}: "
            f"{response.text[:300]}"
        )
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"malformed completion payload: {exc}") from exc
    if not isinstance(content, str):
        raise LLMError(f"completion content is {type(content).__name__}, expected str")
    return content.strip()


def generate(system: str, user: str) -> str:
    if not api_key() and config.LLM_PROVIDER not in ("ollama", "lmstudio"):
        raise LLMNotConfiguredError(
            f"environment variable {config.LLM_API_KEY_ENV} is empty; "
            f"provider {config.LLM_PROVIDER!r} needs an API key"
        )

    payload = build_payload(system, user)
    url = endpoint()
    headers = request_headers()
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            with httpx.Client(timeout=config.LLM_TIMEOUT_S) as client:
                response = client.post(url, json=payload, headers=headers)
            text = _extract(response)
            log.info(
                "generated",
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                chars=len(text),
                attempt=attempt + 1,
            )
            return text
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last_error = exc
            log.warn("llm_retry", attempt=attempt + 1, reason=str(exc)[:200])
            time.sleep(0.5)

    raise LLMError(
        f"generation failed after 2 attempts via {config.LLM_PROVIDER}: {last_error}"
    ) from last_error
