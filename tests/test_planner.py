"""End-to-end planning scenarios with mocked LLM providers (no network, no subscription)."""

import json

import pytest
from conftest import BASE_FILES, FakeProvider, replace

from mutation_planner.analysis import build_context, llm_view, load_context
from mutation_planner.cli import main
from mutation_planner.planner import (
    IncompleteAnalysisError,
    PlanningError,
    dump_json,
    load_prompt,
    run_planner,
)
from mutation_planner.schemas import ChangeStatus, FileClass, MutationPlan, PRMetadata

NO_MUTATE = {"decision": {"mutate": False, "reason": "No production code changed."}, "files": [], "warnings": []}


def plan_for(ctx, path, tests=None, functions=None, lines=None):
    """Build an LLM response selecting `path` with tests chosen from the context's candidates."""
    tests = tests or {}
    out = {"unit": [], "functional": [], "fuzz": []}
    for tid, conf in tests.items():
        cand = next(c for c in ctx.candidates if c.source_file == path and c.test_id == tid)
        out[cand.category.value].append({"id": tid, "reason": "evidence-backed", "confidence": conf, "evidence": cand.evidence[:2]})
    src = next(p for p in ctx.production_files if p.path == path)
    return {
        "decision": {"mutate": True, "reason": "behaviour changed"},
        "files": [{
            "path": path, "priority": "high", "reason": "logic changed",
            "functions": functions if functions is not None else [f.name for f in src.functions],
            "changed_lines": lines or [],
            "tests": out,
        }],
        "warnings": [],
    }


def run(ctx, responses, config, name="fake"):
    provider = FakeProvider(responses, name=name)
    return run_planner(ctx, provider, name, config, load_prompt()), provider


def test_scenario_a_documentation_only(repo, config):
    head = repo.change(files={"doc/release-notes.md": BASE_FILES["doc/release-notes.md"] + "- notes\n", "README.md": "# x\n"})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert ctx.production_files == [] and ctx.candidates == []
    assert {c.classification for c in ctx.changed_files} == {FileClass.documentation}
    result, provider = run(ctx, [NO_MUTATE], config)
    assert result.plan.decision.mutate is False and result.plan.files == []
    # nothing to decide: no LLM call is made
    assert provider.calls == [] and result.attempts == 0 and result.plan.metadata.llm_attempts == 0
    assert "documentation" in result.plan.decision.reason
    assert any("without an LLM call" in w for w in result.plan.warnings)


def test_scenario_a_llm_cannot_select_docs(repo, config):
    # A docs file next to a production change: selecting the docs file is invalid.
    head = repo.change(files={
        "doc/release-notes.md": BASE_FILES["doc/release-notes.md"] + "- notes\n",
        "src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0"),
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    bad = {"decision": {"mutate": True, "reason": "x"}, "files": [{
        "path": "doc/release-notes.md", "priority": "low", "reason": "x", "functions": [], "changed_lines": [],
        "tests": {"unit": [], "functional": [], "fuzz": []}}], "warnings": []}
    with pytest.raises(PlanningError) as exc:
        run(ctx, [bad, bad], config)
    assert exc.value.attempts == 2
    assert any("classified as documentation" in e for e in exc.value.report.errors)
    assert len(exc.value.attempt_errors) == 2


def test_scenario_b_test_only(repo, config):
    head = repo.change(files={
        "src/test/fee_tests.cpp": BASE_FILES["src/test/fee_tests.cpp"].replace("GetMinFee(0), 0", "GetMinFee(0), 0); BOOST_CHECK(true"),
        "test/functional/feature_fee.py": BASE_FILES["test/functional/feature_fee.py"] + "# more\n",
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert ctx.production_files == []
    assert {c.classification for c in ctx.changed_files} == {FileClass.test_source}
    result, provider = run(ctx, [NO_MUTATE], config)
    assert result.plan.files == [] and provider.calls == []


def test_scenario_c_production_bug_fix(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(number=7, title="policy: fix fee floor"), config)
    assert [p.path for p in ctx.production_files] == ["src/policy/fees.cpp"]
    assert [f.name for f in ctx.production_files[0].functions] == ["GetMinFee"]
    resp = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/min_fee_test": "high", "fuzz:fee_rate": "high"})
    result, provider = run(ctx, [resp], config)
    plan = result.plan
    assert plan.decision.mutate and plan.files[0].path == "src/policy/fees.cpp"
    assert plan.files[0].tests.unit[0].command == "build/bin/test_bitcoin --run_test=fee_tests/min_fee_test"
    assert plan.files[0].tests.fuzz[0].command == "FUZZ=fee_rate build_fuzz/bin/fuzz"
    # changed_lines were empty in the LLM output: filled deterministically and reported
    assert [(r.start, r.end) for r in plan.files[0].changed_lines] == [(9, 9)]
    assert any("filled from deterministic diff analysis" in w for w in plan.warnings)
    # provenance comes from the application
    assert plan.pr.number == 7 and plan.pr.head_sha == ctx.snapshot.head_sha
    assert plan.metadata.prompt_version == "1.1" and plan.metadata.validation_status == "valid"
    # the LLM saw the untrusted-data notice and the candidates, not the full inventory
    view = provider.calls[0]["context"]
    assert "untrusted" in view["note"]
    shown = {c["test_id"] for cs in view["candidate_tests"].values() for c in cs}
    assert "unit:fee_tests/min_fee_test" in shown
    assert "unit:amount_tests/space_test" not in shown
    assert "inventory_index" not in view


def test_scenario_d_mixed_changes_only_production_eligible(repo, config):
    head = repo.change(files={
        "src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "fee < 3", "fee <= 3"),
        "src/test/fee_tests.cpp": BASE_FILES["src/test/fee_tests.cpp"] + "// extra\n",
        "test/functional/feature_fee.py": BASE_FILES["test/functional/feature_fee.py"] + "# more\n",
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert [p.path for p in ctx.production_files] == ["src/policy/fees.cpp"]
    resp = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/dust_fee_test": "high"})
    result, _ = run(ctx, [resp], config)
    assert [f.path for f in result.plan.files] == ["src/policy/fees.cpp"]
    # A test file selected next to the production file is dropped deterministically, without a retry.
    selecting_test = plan_for(ctx, "src/policy/fees.cpp")
    selecting_test["files"].append({**selecting_test["files"][0], "path": "src/test/fee_tests.cpp", "functions": []})
    result, provider = run(ctx, [selecting_test], config)
    assert len(provider.calls) == 1 and [f.path for f in result.plan.files] == ["src/policy/fees.cpp"]
    assert any("auto-repaired: dropped src/test/fee_tests.cpp" in w for w in result.plan.warnings)
    assert any("test_source, not production" in e for e in result.attempt_errors[0])


def test_scenario_e_header_only_behavioral_change(repo, config):
    head = repo.change(files={"src/util/strutil.h": replace(BASE_FILES["src/util/strutil.h"], "c == ' ' ||", "c == ' ' || c == '\\n' ||")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    src = ctx.production_files[0]
    assert src.path == "src/util/strutil.h" and src.classification == FileClass.production_header
    assert [f.name for f in src.functions] == ["IsSpaceChar"]
    resp = plan_for(ctx, "src/util/strutil.h", {"unit:amount_tests/space_test": "high"})
    result, _ = run(ctx, [resp], config)
    assert result.plan.files[0].functions == ["IsSpaceChar"]


def test_scenario_f_no_relevant_tests(repo, config):
    head = repo.change(files={"src/new_feature.cpp": "int BrandNewFeatureLogic(int x)\n{\n    return x * 3;\n}\n"})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert not [c for c in ctx.candidates if c.source_file == "src/new_feature.cpp"]
    result, _ = run(ctx, [plan_for(ctx, "src/new_feature.cpp")], config)
    f = result.plan.files[0]
    assert f.tests.unit == f.tests.functional == f.tests.fuzz == []
    assert any("src/new_feature.cpp: no existing tests selected" in w for w in result.plan.warnings)


def test_retry_once_with_validation_errors_then_succeed(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    bad = plan_for(ctx, "src/policy/fees.cpp")
    bad["files"][0]["tests"]["unit"] = [{"id": "unit:fee_tests/ghost", "reason": "x", "confidence": "low", "evidence": []}]
    bad["decision"]["reason"] = " "  # not repairable: needs the model
    good = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/min_fee_test": "high"})
    result, provider = run(ctx, [bad, good], config)
    assert result.attempts == 2 and result.plan.metadata.llm_attempts == 2
    feedback = provider.calls[1]["context"]["previous_attempt"]
    assert any("unit:fee_tests/ghost" in e for e in feedback["validation_errors"])
    assert any("decision.reason is empty" in e for e in feedback["validation_errors"])
    assert result.attempt_errors == [feedback["validation_errors"]]
    # never silently replaced: the final plan contains only what the LLM returned on retry
    assert [t.id for t in result.plan.files[0].tests.unit] == ["unit:fee_tests/min_fee_test"]


def test_repairable_errors_need_no_retry(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    bad = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/min_fee_test": "high"})
    bad["files"][0]["tests"]["unit"].append({"id": "unit:fee_tests/ghost", "reason": "x", "confidence": "low", "evidence": []})
    result, provider = run(ctx, [bad], config)
    assert len(provider.calls) == 1 and result.attempts == 1
    assert [t.id for t in result.plan.files[0].tests.unit] == ["unit:fee_tests/min_fee_test"]
    assert any("auto-repaired" in w and "unit:fee_tests/ghost" in w for w in result.plan.warnings)
    assert any("unit:fee_tests/ghost" in e for e in result.attempt_errors[0])


def test_llm_view_is_compact_and_inventory_is_opt_in(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    view = llm_view(ctx)
    assert set(view) >= {"pr", "snapshot", "budget", "changed_files", "production_files", "candidate_tests"}
    assert not {"tests", "evidence", "inventory_index"} & set(view)
    cand = next(c for c in view["candidate_tests"]["src/policy/fees.cpp"] if c["test_id"] == "unit:fee_tests/min_fee_test")
    assert cand["defined_at"].startswith("src/test/fee_tests.cpp:")
    assert "category" not in cand and "build_condition" not in cand  # derivable / default build
    ids = {e["id"] for e in cand["evidence"]}
    full = next(c for c in ctx.candidates if c.test_id == "unit:fee_tests/min_fee_test" and c.source_file == "src/policy/fees.cpp")
    assert ids == set(full.evidence)
    for e in cand["evidence"]:
        assert e["relationship"] and ("line" in e or "at" in e)
        assert not {"description", "basis", "test_id", "source_file"} & set(e)
    full_view = llm_view(ctx, include_inventory=True)
    assert "unit:amount_tests" in full_view["inventory_index"]["unit_suites"]


def test_include_inventory_index_flag(repo, config, tmp_path, monkeypatch):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    ctx_path = tmp_path / "ctx.json"
    dump_json(ctx, ctx_path)
    resp = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/min_fee_test": "high"})
    for flag, expected in (([], False), (["--include-inventory-index"], True)):
        fake = FakeProvider([json.loads(json.dumps(resp))])
        monkeypatch.setattr("mutation_planner.cli.create_provider", lambda name, cfg, fake=fake: fake)
        assert main(["--context-file", str(ctx_path), "--output", str(tmp_path / "p.json"),
                     "--metadata-output", "", *flag]) == 0
        assert ("inventory_index" in fake.calls[0]["context"]) is expected


def test_scenario_j_equivalent_provider_results(repo, config):
    head = repo.change(files={"src/wallet/coinselection.cpp": replace(BASE_FILES["src/wallet/coinselection.cpp"], "total >= target", "total > target")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    resp = plan_for(ctx, "src/wallet/coinselection.cpp", {
        "unit:coinselection_tests/bnb_test": "high", "functional:wallet_fundrawtransaction.py": "low",
    })
    a, _ = run(ctx, [json.loads(json.dumps(resp))], config, name="claude-code")
    b, _ = run(ctx, [json.loads(json.dumps(resp))], config, name="opencode")
    da, db = a.plan.model_dump(mode="json"), b.plan.model_dump(mode="json")
    assert da["metadata"].pop("provider") == "claude-code" and db["metadata"].pop("provider") == "opencode"
    assert da == db


def test_scenario_k_renamed_and_deleted_files(repo, config):
    renamed_src = replace(BASE_FILES["src/util/strutil.h"], "c == ' ' ||", "c == ' ' || c == '\\r' ||")
    head = repo.change(
        files={"src/common/strutil.h": renamed_src},
        delete=["src/node/mempool_persist.cpp"],
        rename=[("src/util/strutil.h", "src/common/strutil.h")],
    )
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    changed = {c.path: c for c in ctx.changed_files}
    assert changed["src/common/strutil.h"].status == ChangeStatus.renamed
    assert changed["src/common/strutil.h"].old_path == "src/util/strutil.h"
    assert changed["src/node/mempool_persist.cpp"].status == ChangeStatus.deleted
    assert [p.path for p in ctx.production_files] == ["src/common/strutil.h"]
    assert any("mempool_persist.cpp" in w and "deleted" in w for w in ctx.warnings)
    result, _ = run(ctx, [plan_for(ctx, "src/common/strutil.h")], config)
    assert result.plan.files[0].path == "src/common/strutil.h"
    for path in ("src/node/mempool_persist.cpp", "src/util/strutil.h"):
        bad = plan_for(ctx, "src/common/strutil.h")
        bad["files"][0]["path"] = path
        bad["files"][0]["functions"] = []
        with pytest.raises(PlanningError) as exc:
            run(ctx, [bad, bad], config)
        assert any(("does not exist at the PR head" in e) or ("not a file changed" in e) for e in exc.value.report.errors)


def test_incomplete_analysis_refuses_llm_call(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    config.limits.max_total_context_chars = 500
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert not ctx.analysis_complete and ctx.analysis_limitations
    assert any(t.scope == "context-budget" for t in ctx.truncations)
    # eligible files are never silently dropped by the budget
    assert [p.path for p in ctx.production_files] == ["src/policy/fees.cpp"]
    with pytest.raises(IncompleteAnalysisError):
        run(ctx, [NO_MUTATE], config)


def test_scenario_m_context_replay_without_checkout(repo, config, tmp_path, monkeypatch):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx_path = tmp_path / "mutation-context.json"
    rc = main(["--repo", str(repo.path), "--base", repo.base, "--head", head, "--dry-run",
               "--context-output", str(ctx_path), "--metadata-output", ""])
    assert rc == 0 and ctx_path.exists()
    assert not (tmp_path / "mutation-plan.json").exists()

    # Remove the checkout entirely: replay must not need it.
    import shutil
    shutil.rmtree(repo.path)

    ctx = load_context(ctx_path)
    resp = plan_for(ctx, "src/policy/fees.cpp", {"unit:fee_tests/min_fee_test": "high"})
    fake = FakeProvider([resp], name="claude-code", model="sonnet")
    monkeypatch.setattr("mutation_planner.cli.create_provider", lambda name, cfg: fake)
    out = tmp_path / "mutation-plan.json"
    meta = tmp_path / "meta.json"
    rc = main(["--context-file", str(ctx_path), "--provider", "claude-code", "--model", "sonnet",
               "--output", str(out), "--metadata-output", str(meta)])
    assert rc == 0
    plan = MutationPlan.model_validate_json(out.read_text())
    assert plan.metadata.provider == "claude-code" and plan.metadata.model == "sonnet"
    assert json.loads(meta.read_text())["validation_status"] == "valid"
    # the replayed view is identical to what the original analysis would send
    assert fake.calls[0]["context"] == llm_view(ctx)
    # the written plan re-validates against the context
    assert main(["--context-file", str(ctx_path), "--validate-plan", str(out)]) == 0


def test_cli_reports_invalid_plan_and_writes_metadata(repo, config, tmp_path, monkeypatch):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    ctx_path = tmp_path / "ctx.json"
    dump_json(ctx, ctx_path)
    bad = {"decision": {"mutate": "maybe"}}
    monkeypatch.setattr("mutation_planner.cli.create_provider", lambda name, cfg: FakeProvider([bad, bad]))
    out, meta = tmp_path / "plan.json", tmp_path / "meta.json"
    rc = main(["--context-file", str(ctx_path), "--output", str(out), "--metadata-output", str(meta)])
    assert rc == 1 and not out.exists()
    m = json.loads(meta.read_text())
    assert m["validation_status"] == "invalid" and m["validation_errors"]


def test_cli_argument_errors(tmp_path):
    with pytest.raises(SystemExit):
        main(["--repo", "x"])
    with pytest.raises(SystemExit):
        main(["--context-file", "c.json", "--dry-run"])


def test_committed_context_fixture_replays_with_every_provider(tmp_path, monkeypatch, config):
    """Scenario M with a committed fixture: no Git repository exists at all."""
    from conftest import FIXTURES

    ctx_file = FIXTURES / "mutation-context.fixture.json"
    ctx = load_context(ctx_file)
    assert ctx.pr.number == 4242 and [p.path for p in ctx.production_files] == ["src/wallet/coinselection.cpp"]
    resp = plan_for(ctx, "src/wallet/coinselection.cpp", {
        "unit:coinselection_tests/bnb_test": "high", "functional:wallet_fundrawtransaction.py": "low",
    })
    outputs = {}
    for provider in ("claude-code", "opencode", "opencode-zen"):
        fake = FakeProvider([json.loads(json.dumps(resp))], name=provider, model="m")
        monkeypatch.setattr("mutation_planner.cli.create_provider", lambda name, cfg, fake=fake: fake)
        out = tmp_path / f"{provider}.json"
        assert main(["--context-file", str(ctx_file), "--provider", provider, "--model", "m",
                     "--output", str(out), "--metadata-output", ""]) == 0
        data = json.loads(out.read_text())
        assert data["metadata"].pop("provider") == provider
        outputs[provider] = data
        # PR description injection text reaches the LLM only inside the untrusted context
        assert "Ignore all previous instructions" in fake.calls[0]["context"]["pr"]["description"]
        assert "Ignore all previous instructions" not in fake.calls[0]["system_prompt"]
    assert outputs["claude-code"] == outputs["opencode"] == outputs["opencode-zen"]


def test_static_provider_baseline(repo, config, tmp_path):
    """--provider static: deterministic rules, same validator, no LLM."""
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "size <= 0", "size < 0")})
    ctx_path, out, meta = tmp_path / "ctx.json", tmp_path / "plan.json", tmp_path / "meta.json"
    rc = main(["--repo", str(repo.path), "--base", repo.base, "--head", head, "--provider", "static",
               "--model", "ignored", "--context-output", str(ctx_path), "--output", str(out),
               "--metadata-output", str(meta)])
    assert rc == 0
    plan = MutationPlan.model_validate_json(out.read_text())
    assert plan.decision.mutate is True
    f = plan.files[0]
    assert f.path == "src/policy/fees.cpp" and f.functions == ["GetMinFee"]
    assert f.priority == "high"  # src/policy/ is a critical area
    assert [(r.start, r.end) for r in f.changed_lines] == [(9, 9)]
    ids = {t.id: t for cat in (f.tests.unit, f.tests.functional, f.tests.fuzz) for t in cat}
    assert "unit:fee_tests/min_fee_test" in ids and ids["unit:fee_tests/min_fee_test"].confidence == "high"
    assert all(t.evidence for t in ids.values())
    assert plan.metadata.provider == "static" and plan.metadata.model is None
    assert plan.metadata.prompt_version.startswith("static-") and plan.metadata.llm_attempts == 1
    assert any("static baseline" in w for w in plan.warnings)
    assert json.loads(meta.read_text())["validation_status"] == "valid"
    # deterministic, and re-validates against its context
    out2 = tmp_path / "plan2.json"
    assert main(["--context-file", str(ctx_path), "--provider", "static", "--output", str(out2), "--metadata-output", ""]) == 0
    assert json.loads(out.read_text()) == json.loads(out2.read_text())
    assert main(["--context-file", str(ctx_path), "--validate-plan", str(out)]) == 0


def test_static_provider_no_production_changes(repo, config):
    from mutation_planner.llm.static import StaticProvider
    from mutation_planner.planner import static_prompt

    head = repo.change(files={"doc/release-notes.md": BASE_FILES["doc/release-notes.md"] + "- notes\n"})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    result = run_planner(ctx, StaticProvider(), "static", config, static_prompt())
    assert result.plan.decision.mutate is False and result.plan.files == []


def test_static_provider_on_committed_fixture(config):
    from conftest import FIXTURES

    from mutation_planner.llm.static import StaticProvider
    from mutation_planner.planner import static_prompt

    ctx = load_context(FIXTURES / "mutation-context.fixture.json")
    result = run_planner(ctx, StaticProvider(), "static", config, static_prompt())
    f = result.plan.files[0]
    assert f.path == "src/wallet/coinselection.cpp" and f.priority == "high"
    n = sum(len(x) for x in (f.tests.unit, f.tests.functional, f.tests.fuzz))
    assert 0 < n <= ctx.limits["max_tests_per_file"]
