"""Deterministic validation of LLM plans against the recorded repository context.

The validator never repairs a plan by guessing: it reports errors (which make the plan
invalid) and warnings (which are carried into the final artifact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .schemas import (
    PRODUCTION_CLASSES,
    WEAK_RELATIONSHIPS,
    LineRange,
    LLMPlan,
    MutationContext,
    MutationPlan,
    TestCategory,
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _supports(evidence_test: str, selected: str) -> bool:
    """Evidence about a suite supports its cases and vice versa (``unit:a`` / ``unit:a/b``)."""
    return (
        evidence_test == selected
        or evidence_test.startswith(selected + "/")
        or selected.startswith(evidence_test + "/")
    )


def parse_llm_plan(raw: Any) -> tuple[LLMPlan | None, ValidationReport]:
    report = ValidationReport()
    try:
        return LLMPlan.model_validate(raw), report
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            report.errors.append(f"schema: {loc}: {err['msg']}")
        return None, report


def validate_plan(plan: LLMPlan, ctx: MutationContext, max_tests_per_file: int) -> ValidationReport:
    report = ValidationReport()
    E, W = report.errors.append, report.warnings.append

    changed = {c.path: c for c in ctx.changed_files}
    targets = {p.path: p for p in ctx.production_files}
    inventory = ctx.test_inventory.by_id()
    evidence = {e.id: e for e in ctx.evidence}

    # Eligibility consistency (invariants 8, 9).
    if not plan.decision.mutate and plan.files:
        E("decision.mutate is false but files were selected")
    if plan.decision.mutate and not plan.files:
        E("decision.mutate is true but no files were selected")
    if not plan.decision.reason.strip():
        E("decision.reason is empty")

    seen_paths: set[str] = set()
    for i, f in enumerate(plan.files):
        where = f"files[{i}] ({f.path})"
        if f.path in seen_paths:
            E(f"{where}: file selected more than once")
        seen_paths.add(f.path)

        cf = changed.get(f.path)
        if cf is None:
            E(f"{where}: not a file changed by this PR")
            continue
        if cf.classification not in PRODUCTION_CLASSES:
            E(f"{where}: classified as {cf.classification.value}, not production code")
            continue
        if not cf.exists_at_head:
            E(f"{where}: file does not exist at the PR head (status {cf.status.value})")
            continue
        src = targets.get(f.path)
        if src is None:
            E(f"{where}: no analysed source context for this file")
            continue

        # Functions must be ones the analyzer found as changed.
        known = {fn.name for fn in src.functions} | {fn.short_name for fn in src.functions}
        for fn in f.functions:
            if fn not in known:
                E(f"{where}: function {fn!r} is not among the changed functions detected in this file "
                  f"({', '.join(sorted({x.name for x in src.functions})) or 'none detected'})")

        # Line ranges must lie in the file and touch changed code (invariant 6).
        allowed: list[LineRange] = list(cf.changed_lines)
        allowed += [LineRange(start=fn.start_line, end=fn.end_line) for fn in src.functions]
        allowed += [LineRange(start=a, end=a) for a in cf.deletion_anchors]
        n_lines = cf.head_line_count or 0
        for r in f.changed_lines:
            if r.end > n_lines:
                E(f"{where}: line range {r.start}-{r.end} is beyond end of file ({n_lines} lines)")
            elif not any(r.overlaps(a) for a in allowed):
                E(f"{where}: line range {r.start}-{r.end} does not overlap any changed line or changed function")

        total = 0
        no_tests = True
        for cat in TestCategory:
            sels = getattr(f.tests, cat.value)
            ids_seen: set[str] = set()
            for j, t in enumerate(sels):
                twhere = f"{where}.tests.{cat.value}[{j}] ({t.id})"
                total += 1
                no_tests = False
                if t.id in ids_seen:
                    E(f"{twhere}: duplicate test selection")
                ids_seen.add(t.id)
                entry = inventory.get(t.id)
                if entry is None:
                    E(f"{twhere}: test id not found in the discovered test inventory")
                    continue
                if entry.category != cat:
                    E(f"{twhere}: test is a {entry.category.value} test, listed under {cat.value}")
                if not t.reason.strip():
                    E(f"{twhere}: empty reason")
                rels = set()
                for eid in t.evidence:
                    ev = evidence.get(eid)
                    if ev is None:
                        E(f"{twhere}: evidence id {eid!r} does not exist")
                        continue
                    if not _supports(ev.test_id, t.id):
                        E(f"{twhere}: evidence {eid} is about {ev.test_id}, not this test")
                    if ev.source_file != f.path:
                        E(f"{twhere}: evidence {eid} relates to {ev.source_file}, not {f.path}")
                    rels.add(ev.relationship)
                if t.confidence in ("high", "medium"):
                    if not t.evidence:
                        E(f"{twhere}: {t.confidence} confidence requires at least one evidence reference")
                    elif rels and rels <= WEAK_RELATIONSHIPS:
                        E(f"{twhere}: filename similarity alone cannot justify {t.confidence} confidence")
                elif not t.evidence:
                    W(f"{twhere}: selected without collected evidence (low confidence, inferred by the model)")
            # Redundant suite + case selections.
            for tid in ids_seen:
                if any(other != tid and other.startswith(tid + "/") for other in ids_seen):
                    W(f"{where}: both suite {tid} and some of its cases are selected (redundant)")
        if total > max_tests_per_file:
            E(f"{where}: {total} tests selected, above the per-file budget of {max_tests_per_file}")
        if no_tests:
            W(f"{f.path}: no existing tests selected; mutants in this file may go undetected or be untested")
    return report


def validate_final_plan(raw: dict[str, Any], ctx: MutationContext) -> ValidationReport:
    """Re-validate a written ``mutation-plan.json`` against its context."""
    report = ValidationReport()
    try:
        plan = MutationPlan.model_validate(raw)
    except ValidationError as exc:
        report.errors += [f"schema: {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
        return report
    if plan.pr.head_sha != ctx.snapshot.head_sha or plan.pr.merge_base_sha != ctx.snapshot.merge_base_sha:
        report.errors.append("plan was produced for a different head/merge-base than this context")
    llm_like = LLMPlan.model_validate(
        {
            "decision": plan.decision.model_dump(),
            "files": [
                {
                    **f.model_dump(exclude={"tests"}),
                    "tests": {
                        k: [{kk: vv for kk, vv in t.items() if kk != "command"} for t in v]
                        for k, v in f.tests.model_dump().items()
                    },
                }
                for f in plan.files
            ],
            "warnings": [],
        }
    )
    sub = validate_plan(llm_like, ctx, int(ctx.limits.get("max_tests_per_file", 10)))
    report.errors += sub.errors
    report.warnings += sub.warnings
    return report
