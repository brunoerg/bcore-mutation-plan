"""Command-line interface: ``python -m mutation_planner``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .analysis import build_context, load_context
from .config import DEFAULT_CONTEXT_OUTPUT, DEFAULT_OUTPUT, DEFAULT_PROVIDER, DEFAULT_TIMEOUT, ContextLimits, PlannerConfig
from .git_analysis import GitError
from .llm import PROVIDERS, ProviderError, create_provider
from .planner import (
    IncompleteAnalysisError,
    PlanningError,
    dump_json,
    load_prompt,
    run_metadata,
    run_planner,
    static_prompt,
)
from .schemas import PRMetadata
from .validator import validate_final_plan

log = logging.getLogger("mutation_planner")

EXIT_OK, EXIT_PLAN_INVALID, EXIT_USAGE, EXIT_PROVIDER, EXIT_INCOMPLETE = 0, 1, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mutation_planner",
        description="Plan mutation testing for a Bitcoin Core PR: which changed production files to mutate "
                    "and which existing tests to run. Does not generate or run mutants.",
    )
    src = p.add_argument_group("input")
    src.add_argument("--repo", help="Path to a Bitcoin Core git checkout")
    src.add_argument("--base", help="Base commit (PR target branch tip)")
    src.add_argument("--head", help="Head commit (PR tip)")
    src.add_argument("--context-file", help="Plan from a saved mutation-context.json (no checkout needed)")
    src.add_argument("--pr-number", type=int)
    src.add_argument("--pr-title")
    src.add_argument("--pr-description")

    llm = p.add_argument_group("LLM")
    llm.add_argument("--provider", default=DEFAULT_PROVIDER, help=f"One of: {', '.join(PROVIDERS)} (default: {DEFAULT_PROVIDER}). 'static' is a deterministic, LLM-free baseline")
    llm.add_argument("--model", help="Model name/alias for the provider (e.g. sonnet, PROVIDER/MODEL, Zen model id)")
    llm.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="LLM call timeout in seconds")
    llm.add_argument("--max-retries", type=int, default=1, help="LLM retries after a validation failure (default 1)")
    llm.add_argument("--prompt-file", help="Override prompts/mutation_planner_system.md")
    llm.add_argument("--include-inventory-index", action="store_true",
                     help="Also send every unit suite, functional test and fuzz target ID, so the model can pick "
                          "tests that are not candidates (larger context)")
    llm.add_argument("--zen-api-format", choices=["chat-completions", "responses", "messages", "generate-content"])
    llm.add_argument("--zen-base-url", help="Override the Zen base URL for the selected format")

    out = p.add_argument_group("output")
    out.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Plan output path (default {DEFAULT_OUTPUT})")
    out.add_argument("--context-output", default=DEFAULT_CONTEXT_OUTPUT, help=f"Context output path (default {DEFAULT_CONTEXT_OUTPUT})")
    out.add_argument("--metadata-output", default="mutation-run-metadata.json", help="Run metadata path ('' to skip)")
    out.add_argument("--dry-run", action="store_true", help="Analyse and write the context only; no LLM call")
    out.add_argument("--validate-plan", metavar="PLAN", help="Validate an existing plan against --context-file and exit")

    lim = p.add_argument_group("limits")
    d = ContextLimits()
    lim.add_argument("--max-tests-per-file", type=int, default=d.max_tests_per_file)
    lim.add_argument("--max-candidates-per-file", type=int, default=d.max_candidates_per_file)
    lim.add_argument("--max-context-chars", type=int, default=d.max_total_context_chars)
    lim.add_argument("--max-call-depth", type=int, default=d.max_call_depth)

    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[mutation-planner] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.context_file and (args.repo or args.base or args.head):
        parser.error("--context-file cannot be combined with --repo/--base/--head")
    if args.context_file and args.dry_run:
        parser.error("--dry-run analyses a repository; it cannot be used with --context-file")
    if args.validate_plan and not args.context_file:
        parser.error("--validate-plan requires --context-file")
    if not args.context_file and not (args.repo and args.base and args.head):
        parser.error("either --context-file, or all of --repo, --base and --head, are required")

    config = PlannerConfig(
        limits=ContextLimits(
            max_tests_per_file=args.max_tests_per_file,
            max_candidates_per_file=args.max_candidates_per_file,
            max_total_context_chars=args.max_context_chars,
            max_call_depth=args.max_call_depth,
        ),
        max_llm_retries=args.max_retries,
        include_inventory_index=args.include_inventory_index,
        timeout=args.timeout,
    )

    # 1. Context: analyse the repository or load a saved context.
    try:
        if args.context_file:
            ctx = load_context(args.context_file)
            log.info("loaded context %s (head %s)", args.context_file, ctx.snapshot.head_sha[:12])
            if args.validate_plan:
                report = validate_final_plan(json.loads(Path(args.validate_plan).read_text()), ctx)
                for w in report.warnings:
                    log.warning("%s", w)
                if report.ok:
                    log.info("plan %s is valid for this context", args.validate_plan)
                    return EXIT_OK
                for e in report.errors:
                    log.error("%s", e)
                return EXIT_PLAN_INVALID
            if args.max_tests_per_file != ContextLimits().max_tests_per_file:
                ctx.limits["max_tests_per_file"] = args.max_tests_per_file
        else:
            pr = PRMetadata(number=args.pr_number, title=args.pr_title, description=args.pr_description)
            ctx = build_context(args.repo, args.base, args.head, pr, config)
            dump_json(ctx, args.context_output)
            log.info("wrote %s", args.context_output)
    except (GitError, OSError, ValueError) as exc:
        log.error("analysis failed: %s", exc)
        return EXIT_USAGE

    for limitation in ctx.analysis_limitations:
        log.warning("analysis limitation: %s", limitation)
    if args.dry_run:
        log.info("dry run: no LLM call made, no plan generated")
        return EXIT_OK if ctx.analysis_complete else EXIT_INCOMPLETE

    # 2. Plan with the selected provider.
    if args.provider == "static":
        # Deterministic baseline: no prompt, and a retry would return the same plan.
        prompt = static_prompt()
        config.max_llm_retries = 0
        if args.model:
            log.warning("--model is ignored by the static provider")
            args.model = None
    else:
        prompt = load_prompt(args.prompt_file)
    provider_cfg = {
        "model": args.model, "timeout": args.timeout,
        "zen_api_format": args.zen_api_format, "zen_base_url": args.zen_base_url,
    }
    meta_path = args.metadata_output or None
    try:
        provider = create_provider(args.provider, provider_cfg)
        log.info("provider: %s (model: %s)", args.provider,
                 "none" if args.provider == "static" else args.model or "provider default")
        result = run_planner(ctx, provider, args.provider, config, prompt)
    except IncompleteAnalysisError as exc:
        log.error("%s", exc)
        if meta_path:
            dump_json(run_metadata(ctx, args.provider, args.model, prompt, "incomplete_analysis", [str(exc)]), meta_path)
        return EXIT_INCOMPLETE
    except ProviderError as exc:
        log.error("LLM provider error: %s", exc)
        if meta_path:
            dump_json(run_metadata(ctx, args.provider, args.model, prompt, "provider_error", [str(exc)]), meta_path)
        return EXIT_PROVIDER
    except PlanningError as exc:
        log.error("%s", exc)
        errors = exc.report.errors if exc.report else []
        for e in errors:
            log.error("  %s", e)
        if meta_path:
            dump_json(run_metadata(ctx, args.provider, args.model, prompt, "invalid", errors, exc.attempts,
                                   exc.attempt_errors), meta_path)
        return EXIT_PLAN_INVALID

    dump_json(result.plan, args.output)
    if meta_path:
        dump_json(run_metadata(ctx, args.provider, args.model, prompt, "valid", attempts=result.attempts,
                               attempt_errors=result.attempt_errors), meta_path)
    n_tests = sum(len(t) for f in result.plan.files for t in (f.tests.unit, f.tests.functional, f.tests.fuzz))
    log.info(
        "wrote %s: mutate=%s, %d file(s), %d test selection(s), %d warning(s)",
        args.output, result.plan.decision.mutate, len(result.plan.files), n_tests, len(result.plan.warnings),
    )
    return EXIT_OK
