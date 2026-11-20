"""Planning pipeline: one LLM call, deterministic validation and repair, optional single
retry, and application-side enrichment of the final artifact."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import llm_view
from .config import PlannerConfig
from .llm.base import LLMProvider, ProviderError
from .schemas import (
    LLMPlan,
    MutationContext,
    MutationPlan,
    PlanFile,
    PlanMetadata,
    PlanPR,
    PlanTests,
    PlanTestSelection,
    TestCategory,
    llm_output_schema,
)
from .validator import ValidationReport, parse_llm_plan, repair_plan, validate_plan

log = logging.getLogger(__name__)

PROMPT_FILE_NAME = "mutation_planner_system.md"
_VERSION_RE = re.compile(r"<!--\s*prompt-version:\s*([0-9A-Za-z.\-]+)\s*-->")


class PlanningError(RuntimeError):
    def __init__(
        self, message: str, report: ValidationReport | None = None, attempts: int = 0,
        attempt_errors: list[list[str]] | None = None,
    ):
        super().__init__(message)
        self.report = report
        self.attempts = attempts
        self.attempt_errors = attempt_errors or []


class IncompleteAnalysisError(PlanningError):
    pass


@dataclass
class Prompt:
    text: str
    version: str
    sha256: str
    path: str


def find_prompt_file(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for cand in (here.parent / "prompts" / PROMPT_FILE_NAME, here / "prompts" / PROMPT_FILE_NAME):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"cannot find prompts/{PROMPT_FILE_NAME}; pass --prompt-file")


def load_prompt(path: str | None = None) -> Prompt:
    p = find_prompt_file(path)
    text = p.read_text()
    m = _VERSION_RE.search(text)
    if not m:
        raise ValueError(f"{p} lacks a '<!-- prompt-version: X -->' header")
    return Prompt(text=text, version=m.group(1), sha256=hashlib.sha256(text.encode()).hexdigest(), path=str(p))


def static_prompt() -> Prompt:
    """Provenance for the ``static`` provider: its rules play the role of the prompt."""
    from .llm import static

    src = Path(static.__file__)
    text = src.read_text()
    return Prompt(
        text="", version=f"static-{static.STATIC_RULES_VERSION}",
        sha256=hashlib.sha256(text.encode()).hexdigest(), path=str(src),
    )


def context_sha256(ctx: MutationContext) -> str:
    return hashlib.sha256(ctx.model_dump_json().encode()).hexdigest()


@dataclass
class PlanResult:
    plan: MutationPlan
    attempts: int
    warnings: list[str] = field(default_factory=list)
    # Validation errors of every model response that failed, as returned (before repair).
    attempt_errors: list[list[str]] = field(default_factory=list)


NO_PRODUCTION_REASON = "No production code changed: the PR touches only {kinds}."


def run_planner(
    ctx: MutationContext,
    provider: LLMProvider,
    provider_name: str,
    config: PlannerConfig,
    prompt: Prompt,
) -> PlanResult:
    if not ctx.analysis_complete:
        raise IncompleteAnalysisError(
            "repository analysis is incomplete; refusing to ask the LLM for a plan: "
            + "; ".join(ctx.analysis_limitations)
        )
    max_tests = int(ctx.limits.get("max_tests_per_file", config.limits.max_tests_per_file))

    if not ctx.production_files:
        # Nothing to mutate, whatever a model would say: decide without an LLM call.
        kinds = sorted({c.classification.value for c in ctx.changed_files}) or ["nothing"]
        plan = LLMPlan.model_validate({
            "decision": {"mutate": False, "reason": NO_PRODUCTION_REASON.format(kinds=", ".join(kinds))},
            "files": [],
            "warnings": ["decided without an LLM call: no changed production file exists at the PR head"],
        })
        report = validate_plan(plan, ctx, max_tests)
        log.info("no production files changed; skipping the LLM call")
        final = enrich(plan, ctx, provider_name, getattr(provider, "model", None), prompt, 0, report)
        return PlanResult(plan=final, attempts=0, warnings=final.warnings)

    view = llm_view(ctx, config.include_inventory_index)
    schema = llm_output_schema()
    attempts = 0
    attempt_errors: list[list[str]] = []
    feedback: dict[str, Any] | None = None
    while True:
        attempts += 1
        payload = dict(view)
        if feedback:
            payload["previous_attempt"] = feedback
        log.info("LLM planning attempt %d via %s", attempts, provider_name)
        try:
            raw = provider.generate_plan(prompt.text, payload, schema)
        except ProviderError as exc:
            log.error("LLM request failed: %s", exc)
            raise
        plan, report = parse_llm_plan(raw)
        if plan is not None:
            report = validate_plan(plan, ctx, max_tests)
        if report.ok and plan is not None:
            log.info("validation passed (%d warnings)", len(report.warnings))
            break
        log.warning("validation failed with %d errors:\n  %s", len(report.errors), "\n  ".join(report.errors[:30]))
        attempt_errors.append(list(report.errors))
        if plan is not None:
            # Fix what can be fixed by removing or downgrading; a re-ask resends the whole context.
            repaired, notes = repair_plan(plan, ctx, max_tests)
            rreport = validate_plan(repaired, ctx, max_tests)
            if rreport.ok:
                log.warning("plan repaired deterministically (%d change(s)); no retry needed", len(notes))
                plan, report = repaired, ValidationReport(warnings=notes + rreport.warnings)
                break
        if attempts > config.max_llm_retries:
            raise PlanningError(
                f"LLM plan failed validation after {attempts} attempt(s)", report=report, attempts=attempts,
                attempt_errors=attempt_errors,
            )
        feedback = {
            "instruction": "Your previous response failed deterministic validation. Return a corrected, complete "
                           "plan. Use only IDs, paths and function names present in the context. Do not guess replacements.",
            "validation_errors": report.errors[:60],
            "previous_response": raw,
        }
    assert plan is not None
    final = enrich(plan, ctx, provider_name, getattr(provider, "model", None), prompt, attempts, report)
    return PlanResult(plan=final, attempts=attempts, warnings=final.warnings, attempt_errors=attempt_errors)


def enrich(
    plan: LLMPlan,
    ctx: MutationContext,
    provider_name: str,
    model: str | None,
    prompt: Prompt,
    attempts: int,
    report: ValidationReport,
) -> MutationPlan:
    """Add deterministic provenance and inventory data; never alters the LLM's selections."""
    inventory = ctx.test_inventory.by_id()
    changed = {c.path: c for c in ctx.changed_files}
    targets = {p.path: p for p in ctx.production_files}
    warnings: list[str] = list(plan.warnings)

    files: list[PlanFile] = []
    for f in plan.files:
        lines = list(f.changed_lines)
        if not lines:
            fn_names = set(f.functions)
            src = targets[f.path]
            picked = [fn for fn in src.functions if fn.name in fn_names or fn.short_name in fn_names]
            lines = [r for fn in picked for r in fn.changed_lines] or list(changed[f.path].changed_lines)
            if lines:
                warnings.append(f"{f.path}: changed_lines filled from deterministic diff analysis")
        tests = PlanTests(**{
            cat.value: [
                PlanTestSelection(**t.model_dump(), command=inventory[t.id].command)
                for t in getattr(f.tests, cat.value)
            ]
            for cat in TestCategory
        })
        files.append(PlanFile(
            path=f.path, priority=f.priority, reason=f.reason, functions=f.functions,
            changed_lines=lines, tests=tests, uncertainty=f.uncertainty,
        ))

    for w in report.warnings:
        if w not in warnings:
            warnings.append(w)
    for t in ctx.truncations:
        warnings.append(f"context truncated ({t.scope}): {t.detail}")
    warnings += [w for w in ctx.warnings if w not in warnings]

    return MutationPlan(
        pr=PlanPR(
            number=ctx.pr.number, base_sha=ctx.snapshot.base_sha,
            head_sha=ctx.snapshot.head_sha, merge_base_sha=ctx.snapshot.merge_base_sha,
        ),
        decision=plan.decision,
        files=files,
        warnings=warnings,
        metadata=PlanMetadata(
            provider=provider_name, model=model, prompt_version=prompt.version,
            prompt_sha256=prompt.sha256, context_sha256=context_sha256(ctx),
            tool_version=__version__, head_tree_sha=ctx.snapshot.head_tree_sha,
            validation_status="valid", llm_attempts=attempts,
        ),
    )


def run_metadata(
    ctx: MutationContext,
    provider_name: str | None,
    model: str | None,
    prompt: Prompt | None,
    status: str,
    errors: list[str] | None = None,
    attempts: int = 0,
    attempt_errors: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "base_sha": ctx.snapshot.base_sha,
        "head_sha": ctx.snapshot.head_sha,
        "merge_base_sha": ctx.snapshot.merge_base_sha,
        "head_tree_sha": ctx.snapshot.head_tree_sha,
        "provider": provider_name,
        "model": model,
        "prompt_version": prompt.version if prompt else None,
        "prompt_sha256": prompt.sha256 if prompt else None,
        "context_sha256": context_sha256(ctx),
        "tool_version": __version__,
        "llm_attempts": attempts,
        "validation_status": status,
        "validation_errors": errors or [],
        "attempt_errors": attempt_errors or [],
    }


def dump_json(obj: Any, path: str | Path) -> None:
    data = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
