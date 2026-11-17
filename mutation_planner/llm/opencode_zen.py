"""OpenCode Zen HTTP API provider.

Zen serves each model through its native API format. The format and base URL for a
model are read from Zen's public model catalogue (``GET <base>/v1/models``), where each
entry has an ``api`` map such as ``{"messages": {"base_url": ".../anthropic/v1"}}``.
They can also be forced with ``--zen-api-format``/``--zen-base-url``.

Supported formats:

* ``chat-completions`` -> POST {base_url}/chat/completions   (OpenAI-compatible)
* ``responses``        -> POST {base_url}/responses          (OpenAI Responses)
* ``messages``         -> POST {base_url}/messages           (Anthropic Messages)
* ``generate-content`` -> POST {base_url}/models/{model}:generateContent (Gemini)

The API key is read from ``OPENCODE_ZEN_API_KEY`` (or ``OPENCODE_API_KEY``) and sent
as a bearer token, plus the header each native format expects.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .. import __version__
from .base import ProviderError, build_user_message, extract_json_object, redact

log = logging.getLogger(__name__)

DEFAULT_CATALOG_URL = "https://opencode.ai/inference/v1/models"
KEY_ENV_VARS = ("OPENCODE_ZEN_API_KEY", "OPENCODE_API_KEY")
FORMATS = ("chat-completions", "responses", "messages", "generate-content")
MAX_OUTPUT_TOKENS = 16_000

Transport = Callable[[str, str, dict[str, str], bytes | None, int], tuple[int, bytes]]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int) -> tuple[int, bytes]:
    headers = {"User-Agent": f"mutation-planner/{__version__}", **headers}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https endpoints
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class OpenCodeZenProvider:
    name = "opencode-zen"

    def __init__(
        self,
        model: str | None,
        timeout: int = 900,
        api_key: str | None = None,
        api_format: str | None = None,
        base_url: str | None = None,
        catalog_url: str = DEFAULT_CATALOG_URL,
        transport: Transport = urllib_transport,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not model:
            raise ProviderError("config", "opencode-zen requires --model (a Zen model id, e.g. from `curl https://opencode.ai/inference/v1/models`)")
        self.model = model.removeprefix("opencode/")
        self.timeout = timeout
        self.api_key = api_key or next((os.environ[v] for v in KEY_ENV_VARS if os.environ.get(v)), None)
        if api_format and api_format not in FORMATS:
            raise ProviderError("config", f"unknown Zen API format {api_format!r}; expected one of {', '.join(FORMATS)}")
        self.api_format = api_format
        self.base_url = base_url.rstrip("/") if base_url else None
        self.catalog_url = catalog_url
        self._transport = transport
        self.max_attempts = max_attempts
        self._sleep = sleep

    # -- endpoint resolution ------------------------------------------------

    def resolve_endpoint(self) -> tuple[str, str]:
        if self.api_format and self.base_url:
            return self.api_format, self.base_url
        status, body = self._transport("GET", self.catalog_url, {"Accept": "application/json"}, None, min(self.timeout, 60))
        if status != 200:
            raise ProviderError("http", f"could not read Zen model catalogue ({status}); pass --zen-api-format and --zen-base-url")
        try:
            models = {m["id"]: m for m in json.loads(body)["data"]}
        except (ValueError, KeyError, TypeError):
            raise ProviderError("bad_response", "unexpected Zen model catalogue format") from None
        entry = models.get(self.model)
        if entry is None:
            raise ProviderError("model", f"model {self.model!r} is not in the Zen catalogue")
        apis: dict[str, Any] = entry.get("api") or {}
        fmt = self.api_format or next((f for f in FORMATS if f in apis), None)
        if fmt is None or fmt not in apis and not self.base_url:
            raise ProviderError("config", f"no supported API format for {self.model!r} (catalogue lists {sorted(apis)})")
        base = self.base_url or apis[fmt]["base_url"].rstrip("/")
        return fmt, base

    # -- request building -----------------------------------------------------

    def build_request(self, fmt: str, base: str, system_prompt: str, user: str, schema: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if fmt == "chat-completions":
            return f"{base}/chat/completions", headers, {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
        if fmt == "responses":
            return f"{base}/responses", headers, {
                "model": self.model,
                "instructions": system_prompt,
                "input": user,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            }
        if fmt == "messages":
            headers |= {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
            return f"{base}/messages", headers, {
                "model": self.model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
        if fmt == "generate-content":
            headers |= {"x-goog-api-key": self.api_key or ""}
            return f"{base}/models/{self.model}:generateContent", headers, {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": MAX_OUTPUT_TOKENS},
            }
        raise ProviderError("config", f"unsupported format {fmt}")

    @staticmethod
    def extract_text(fmt: str, data: dict[str, Any]) -> str:
        try:
            if fmt == "chat-completions":
                return data["choices"][0]["message"]["content"]
            if fmt == "responses":
                if isinstance(data.get("output_text"), str):
                    return data["output_text"]
                return "".join(
                    c.get("text", "")
                    for item in data.get("output", []) if item.get("type") == "message"
                    for c in item.get("content", []) if c.get("type") in ("output_text", "text")
                )
            if fmt == "messages":
                return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            if fmt == "generate-content":
                return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            pass
        raise ProviderError("bad_response", f"unexpected {fmt} response shape: {redact(json.dumps(data))[:300]}")

    # -- main -----------------------------------------------------------------

    def generate_plan(self, system_prompt: str, context: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError("auth", f"no Zen API key: set one of {', '.join(KEY_ENV_VARS)}")
        fmt, base = self.resolve_endpoint()
        url, headers, payload = self.build_request(fmt, base, system_prompt, build_user_message(context, output_schema), output_schema)
        body = json.dumps(payload).encode()
        log.info("invoking OpenCode Zen (%s, model=%s, %d bytes)", fmt, self.model, len(body))
        for attempt in range(1, self.max_attempts + 1):
            try:
                status, raw = self._transport("POST", url, headers, body, self.timeout)
            except TimeoutError:
                raise ProviderError("timeout", f"Zen request timed out after {self.timeout}s") from None
            except OSError as exc:
                raise ProviderError("unavailable", f"network error contacting Zen: {exc}") from None
            if status == 200:
                try:
                    data = json.loads(raw)
                except ValueError:
                    raise ProviderError("bad_response", "Zen returned non-JSON body") from None
                return extract_json_object(self.extract_text(fmt, data))
            detail = redact(raw.decode(errors="replace"))
            if status in (401, 403):
                raise ProviderError("auth", f"Zen rejected the API key ({status}): {detail}")
            if status == 404:
                raise ProviderError("model", f"Zen endpoint/model not found ({status}): {detail}")
            if status in (429, 500, 502, 503, 504) and attempt < self.max_attempts:
                delay = min(30.0, 2.0 ** attempt) + random.random()
                log.warning("Zen returned %d; retrying in %.1fs (attempt %d/%d)", status, delay, attempt, self.max_attempts)
                self._sleep(delay)
                continue
            kind = "rate_limit" if status == 429 else "http"
            raise ProviderError(kind, f"Zen request failed ({status}): {detail}")
        raise ProviderError("http", "Zen request failed after retries")  # pragma: no cover
