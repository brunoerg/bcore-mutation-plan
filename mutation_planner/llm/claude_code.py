"""Claude Code CLI provider (``claude -p``), using the user's existing login.

The CLI is run as a pure reasoning component: no built-in tools (``--tools ""``), no MCP
servers, no user/project settings or hooks, no slash commands/skills, permission mode
``dontAsk``, in an empty temporary working directory. The planning context is sent on
stdin. Authentication is whatever the CLI is already configured with (subscription
login or ANTHROPIC_API_KEY); this module never reads credential storage.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from typing import Any, Callable

from .base import ProviderError, build_user_message, classify_failure, extract_json_object, redact

log = logging.getLogger(__name__)

REQUIRED_FLAGS = ("--print", "--output-format", "--tools", "--permission-mode", "--strict-mcp-config", "--system-prompt")
OPTIONAL_FLAGS = ("--json-schema", "--setting-sources", "--disable-slash-commands", "--no-session-persistence", "--disallowedTools", "--mcp-config")
DENY_TOOLS = "Bash,Edit,Write,MultiEdit,NotebookEdit,WebFetch,WebSearch,Task,Agent,Read,Glob,Grep"


class ClaudeCodeProvider:
    name = "claude-code"

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 900,
        executable: str = "claude",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.model = model
        self.timeout = timeout
        self.executable = executable
        self._run = runner
        self._flags: set[str] | None = None

    def _supported_flags(self) -> set[str]:
        if self._flags is None:
            exe = shutil.which(self.executable) if self._run is subprocess.run else self.executable
            if exe is None:
                raise ProviderError("unavailable", f"`{self.executable}` not found on PATH; install Claude Code (see README)")
            try:
                proc = self._run([self.executable, "--help"], capture_output=True, text=True, timeout=60)
            except FileNotFoundError:
                raise ProviderError("unavailable", f"`{self.executable}` not found on PATH") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("timeout", "`claude --help` timed out") from None
            help_text = (proc.stdout or "") + (proc.stderr or "")
            self._flags = {f for f in REQUIRED_FLAGS + OPTIONAL_FLAGS if f in help_text}
            missing = [f for f in REQUIRED_FLAGS if f not in self._flags]
            if missing:
                raise ProviderError(
                    "config",
                    "installed Claude Code CLI lacks options needed for safe tool-free invocation "
                    f"({', '.join(missing)}); upgrade Claude Code. Refusing to run with broader tool access.",
                )
        return self._flags

    def build_command(self, system_prompt: str, output_schema: dict[str, Any]) -> list[str]:
        flags = self._supported_flags()
        cmd = [
            self.executable, "--print",
            "--output-format", "json",
            "--tools", "",
            "--permission-mode", "dontAsk",
            "--strict-mcp-config",
        ]
        if "--mcp-config" in flags:
            cmd += ["--mcp-config", '{"mcpServers":{}}']
        if "--disallowedTools" in flags:
            cmd += ["--disallowedTools", DENY_TOOLS]
        if "--setting-sources" in flags:
            cmd += ["--setting-sources", ""]
        if "--disable-slash-commands" in flags:
            cmd += ["--disable-slash-commands"]
        if "--no-session-persistence" in flags:
            cmd += ["--no-session-persistence"]
        if "--json-schema" in flags:
            cmd += ["--json-schema", json.dumps(output_schema, separators=(",", ":"))]
        if self.model:
            cmd += ["--model", self.model]
        cmd += ["--system-prompt", system_prompt]
        return cmd

    def generate_plan(self, system_prompt: str, context: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        cmd = self.build_command(system_prompt, output_schema)
        message = build_user_message(context, output_schema)
        log.info("invoking Claude Code CLI (model=%s, %d chars of context)", self.model or "default", len(message))
        with tempfile.TemporaryDirectory(prefix="mutation-planner-claude-") as cwd:
            try:
                proc = self._run(cmd, input=message, capture_output=True, text=True, timeout=self.timeout, cwd=cwd)
            except FileNotFoundError:
                raise ProviderError("unavailable", f"`{self.executable}` not found on PATH") from None
            except subprocess.TimeoutExpired:
                raise ProviderError("timeout", f"Claude Code did not finish within {self.timeout}s") from None
        return self.parse_output(proc.returncode, proc.stdout or "", proc.stderr or "")

    @staticmethod
    def _envelope(stdout: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(stdout)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        for line in reversed(stdout.strip().splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "result":
                return obj
        return None

    def parse_output(self, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
        env = self._envelope(stdout)
        if env is None:
            detail = redact(stderr or stdout or f"exit status {returncode}")
            kind = classify_failure(detail) if returncode else "bad_response"
            raise ProviderError(kind, f"Claude Code returned no JSON result envelope (exit {returncode}): {detail}")
        if env.get("is_error") or returncode != 0 or env.get("subtype") not in (None, "success"):
            status = env.get("api_error_status")
            text = f"{env.get('result', '')} {env.get('api_error_type', '')} {env.get('subtype', '')} {stderr}"
            if status in (401, 403):
                kind = "auth"
            elif status == 404:
                kind = "model"
            elif status == 429:
                kind = "rate_limit"
            elif isinstance(status, int) and status >= 500:
                kind = "http"
            else:
                kind = classify_failure(text)
            raise ProviderError(kind, f"Claude Code reported an error: {redact(text.strip())}")
        structured = env.get("structured_output")
        if isinstance(structured, dict):
            return structured
        return extract_json_object(env.get("result", ""))
