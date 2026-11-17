"""OpenCode CLI provider (``opencode run --format json``).

Runs with a restricted inline configuration (OPENCODE_CONFIG_CONTENT) that denies every
permission except ``read``, and explicitly denies edits, shell, web access, sub-agents
and access outside the working directory. The working directory is an empty temporary
directory, so the remaining read access cannot reach the Bitcoin Core checkout. The
planning context is sent on stdin. Authentication uses OpenCode's existing
configuration (``opencode auth login``).

Note: denying ``read`` makes current OpenCode releases hang, so it stays allowed inside
the empty sandbox directory.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Callable

from .base import ProviderError, build_user_message, classify_failure, extract_json_object, redact

log = logging.getLogger(__name__)

RESTRICTED_CONFIG: dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {
        "*": "deny",
        "read": "allow",
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "task": "deny",
        "external_directory": "deny",
    },
    "share": "disabled",
    "autoupdate": False,
}


class OpenCodeProvider:
    name = "opencode"

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 900,
        executable: str = "opencode",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.model = model
        self.timeout = timeout
        self.executable = executable
        self._run = runner

    def build_command(self) -> list[str]:
        cmd = [self.executable, "run", "--pure", "--format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def generate_plan(self, system_prompt: str, context: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        # `opencode run` has no separate system-prompt flag; the prompt leads the message.
        message = (
            "<system_instructions>\n" + system_prompt + "\n</system_instructions>\n\n"
            + build_user_message(context, output_schema)
        )
        env = dict(os.environ)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(RESTRICTED_CONFIG)
        log.info("invoking OpenCode CLI (model=%s, %d chars of context)", self.model or "default", len(message))
        with tempfile.TemporaryDirectory(prefix="mutation-planner-opencode-") as cwd:
            try:
                proc = self._run(
                    self.build_command(), input=message, capture_output=True, text=True,
                    timeout=self.timeout, cwd=cwd, env=env,
                )
            except FileNotFoundError:
                raise ProviderError("unavailable", f"`{self.executable}` not found on PATH; install OpenCode (see README)") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("timeout", f"OpenCode did not finish within {self.timeout}s") from None
        return self.parse_output(proc.returncode, proc.stdout or "", proc.stderr or "")

    def parse_output(self, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
        texts: dict[str, list[str]] = {}
        order: list[str] = []
        errors: list[str] = []
        tool_calls: list[str] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            part = ev.get("part") or {}
            if etype == "text" and isinstance(part.get("text"), str):
                mid = part.get("messageID", "")
                if mid not in texts:
                    order.append(mid)
                texts.setdefault(mid, []).append(part["text"])
            elif etype == "error":
                err = ev.get("error") or {}
                errors.append(json.dumps(err)[:500] if not isinstance(err, str) else err)
            elif etype in ("tool_use", "tool"):
                tool_calls.append(str(part.get("tool") or part.get("name") or "?"))
        if tool_calls:
            log.warning("OpenCode model attempted tool calls (denied by config): %s", ", ".join(tool_calls))
        if errors or (returncode != 0 and not texts):
            detail = redact(" ".join(errors) + " " + stderr)
            raise ProviderError(classify_failure(detail), f"OpenCode failed (exit {returncode}): {detail.strip() or 'no output'}")
        if not texts:
            raise ProviderError("bad_response", "OpenCode produced no assistant text events")
        # The final assistant message holds the answer; earlier messages may be preamble.
        final = "".join(texts[order[-1]])
        return extract_json_object(final)
