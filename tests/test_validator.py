import copy

import pytest

from conftest import BASE_FILES, replace
from mutation_planner.analysis import build_context
from mutation_planner.schemas import LLMPlan, PRMetadata
from mutation_planner.validator import parse_llm_plan, repair_plan, validate_final_plan, validate_plan


@pytest.fixture
def fee_ctx(repo, config):
    head = repo.change(files={
        "src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "return 0;", "return 1;"),
        "src/node/mempool_persist.cpp": replace(BASE_FILES["src/node/mempool_persist.cpp"], '"mempool"', '"mempool.dat"'),
        "doc/release-notes.md": BASE_FILES["doc/release-notes.md"] + "- fee change\n",
    })
    return build_context(repo.path, repo.base, head, PRMetadata(number=1), config)


def ev_for(ctx, source, test_id):
    return next(c.evidence for c in ctx.candidates if c.source_file == source and c.test_id == test_id)


def good_plan(ctx) -> dict:
    return {
        "decision": {"mutate": True, "reason": "GetMinFee behaviour changed"},
        "files": [{
            "path": "src/policy/fees.cpp",
            "priority": "high",
            "reason": "fee floor logic changed",
            "functions": ["GetMinFee"],
            "changed_lines": [{"start": 10, "end": 10}],
            "tests": {
                "unit": [{"id": "unit:fee_tests/min_fee_test", "reason": "calls GetMinFee", "confidence": "high",
                          "evidence": ev_for(ctx, "src/policy/fees.cpp", "unit:fee_tests/min_fee_test")[:1]}],
                "functional": [],
                "fuzz": [{"id": "fuzz:fee_rate", "reason": "fuzzes GetMinFee", "confidence": "medium",
                          "evidence": ev_for(ctx, "src/policy/fees.cpp", "fuzz:fee_rate")[:1]}],
            },
        }],
        "warnings": [],
    }


def check(raw, ctx, max_tests=10):
    plan, report = parse_llm_plan(raw)
    if plan is None:
        return report
    return validate_plan(plan, ctx, max_tests)


def test_valid_plan_passes(fee_ctx):
    report = check(good_plan(fee_ctx), fee_ctx)
    assert report.ok, report.errors


def test_scenario_g_invalid_llm_output_schema(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["priority"] = "urgent"
    del raw["decision"]["reason"]
    raw["extra_field"] = 1
    report = check(raw, fee_ctx)
    assert not report.ok
    assert any("priority" in e for e in report.errors)
    assert any("decision.reason" in e for e in report.errors)
    assert any("extra_field" in e for e in report.errors)


def test_scenario_h_hallucinated_test(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["tests"]["unit"].append({"id": "unit:fee_tests/imaginary_case", "reason": "x", "confidence": "low", "evidence": []})
    raw["files"][0]["tests"]["functional"].append({"id": "functional:feature_made_up.py", "reason": "x", "confidence": "low", "evidence": []})
    report = check(raw, fee_ctx)
    assert sum("not found in the discovered test inventory" in e for e in report.errors) == 2


def test_scenario_i_name_similarity_cannot_justify_confidence(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"].append({
        "path": "src/node/mempool_persist.cpp", "priority": "low", "reason": "banner changed",
        "functions": ["DumpMempoolBanner"], "changed_lines": [],
        "tests": {"unit": [{"id": "unit:mempool_persist_tests", "reason": "same name", "confidence": "high",
                            "evidence": ev_for(fee_ctx, "src/node/mempool_persist.cpp", "unit:mempool_persist_tests")}],
                  "functional": [], "fuzz": []},
    })
    report = check(raw, fee_ctx)
    assert any("filename similarity alone" in e for e in report.errors)
    # the same selection at low confidence is acceptable
    raw["files"][1]["tests"]["unit"][0]["confidence"] = "low"
    assert check(raw, fee_ctx).ok


def test_scenario_l_invalid_evidence_references(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["tests"]["unit"][0]["evidence"] = ["ev-9999"]
    fuzz_ev = ev_for(fee_ctx, "src/policy/fees.cpp", "fuzz:fee_rate")[0]
    raw["files"][0]["tests"]["fuzz"].append(
        {"id": "fuzz:block_deserialize", "reason": "x", "confidence": "medium", "evidence": [fuzz_ev]}
    )
    report = check(raw, fee_ctx)
    assert any("ev-9999" in e and "does not exist" in e for e in report.errors)
    assert any("is about fuzz:fee_rate, not this test" in e for e in report.errors)


def test_medium_or_high_without_evidence_is_rejected(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["tests"]["unit"][0]["evidence"] = []
    assert any("requires at least one evidence" in e for e in check(raw, fee_ctx).errors)


def test_eligibility_consistency(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["decision"]["mutate"] = False
    assert any("mutate is false but files" in e for e in check(raw, fee_ctx).errors)
    raw = good_plan(fee_ctx)
    raw["files"] = []
    assert any("mutate is true but no files" in e for e in check(raw, fee_ctx).errors)


def test_non_target_files_rejected(fee_ctx):
    for path, needle in [
        ("doc/release-notes.md", "classified as documentation"),
        ("src/wallet/spend.cpp", "not a file changed by this PR"),
        ("src/test/fee_tests.cpp", "not a file changed by this PR"),
    ]:
        raw = good_plan(fee_ctx)
        raw["files"][0]["path"] = path
        assert any(needle in e for e in check(raw, fee_ctx).errors), path


def test_functions_lines_duplicates_categories_and_budget(fee_ctx):
    raw = good_plan(fee_ctx)
    f = raw["files"][0]
    f["functions"] = ["GetMinFee", "InventedFunction"]
    f["changed_lines"] = [{"start": 1, "end": 2}, {"start": 500, "end": 501}]
    f["tests"]["unit"].append(copy.deepcopy(f["tests"]["unit"][0]))
    f["tests"]["functional"].append({"id": "fuzz:fee_rate", "reason": "x", "confidence": "low", "evidence": []})
    report = check(raw, fee_ctx, max_tests=3)
    errs = "\n".join(report.errors)
    assert "'InventedFunction' is not among the changed functions" in errs
    assert "1-2 does not overlap" in errs
    assert "beyond end of file" in errs
    assert "duplicate test selection" in errs
    assert "is a fuzz test, listed under functional" in errs
    assert "above the per-file budget of 3" in errs


def test_scenario_f_no_tests_warns(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["tests"] = {"unit": [], "functional": [], "fuzz": []}
    report = check(raw, fee_ctx)
    assert report.ok and any("no existing tests selected" in w for w in report.warnings)


def test_validate_final_plan_detects_tampering(fee_ctx):
    from mutation_planner.planner import enrich, load_prompt
    from mutation_planner.validator import ValidationReport

    plan = enrich(LLMPlan.model_validate(good_plan(fee_ctx)), fee_ctx, "fake", "m", load_prompt(), 1, ValidationReport())
    data = plan.model_dump(mode="json")
    assert validate_final_plan(data, fee_ctx).ok
    data["files"][0]["tests"]["unit"][0]["id"] = "unit:nope"
    assert not validate_final_plan(data, fee_ctx).ok
    data = plan.model_dump(mode="json")
    data["pr"]["head_sha"] = "0" * 40
    assert any("different head" in e for e in validate_final_plan(data, fee_ctx).errors)


def test_repair_fixes_removable_errors(fee_ctx):
    raw = good_plan(fee_ctx)
    f = raw["files"][0]
    f["functions"] = ["GetMinFee", "InventedFunction"]
    f["changed_lines"] = [{"start": 10, "end": 10}, {"start": 1, "end": 2}, {"start": 500, "end": 501}]
    f["tests"]["unit"].append(copy.deepcopy(f["tests"]["unit"][0]))  # duplicate
    f["tests"]["unit"].append({"id": "unit:fee_tests/imaginary", "reason": "x", "confidence": "low", "evidence": []})
    f["tests"]["functional"].append(f["tests"]["fuzz"].pop())  # wrong category
    f["tests"]["unit"][0]["evidence"] = f["tests"]["unit"][0]["evidence"] + ["ev-9999"]
    raw["files"].append({**copy.deepcopy(f), "path": "doc/release-notes.md"})
    plan, report = parse_llm_plan(raw)
    assert not validate_plan(plan, fee_ctx, 10).ok

    fixed, notes = repair_plan(plan, fee_ctx, 10)
    report = validate_plan(fixed, fee_ctx, 10)
    assert report.ok, report.errors
    g = fixed.files[0]
    assert [x.path for x in fixed.files] == ["src/policy/fees.cpp"]
    assert g.functions == ["GetMinFee"]
    assert [(r.start, r.end) for r in g.changed_lines] == [(10, 10)]
    assert [t.id for t in g.tests.unit] == ["unit:fee_tests/min_fee_test"]
    assert "ev-9999" not in g.tests.unit[0].evidence and g.tests.unit[0].confidence == "high"
    assert [t.id for t in g.tests.fuzz] == ["fuzz:fee_rate"] and g.tests.functional == []
    assert all(n.startswith("auto-repaired:") for n in notes) and len(notes) >= 7


def test_repair_downgrades_unsupported_confidence_and_enforces_budget(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["files"][0]["tests"]["unit"][0]["evidence"] = ["ev-9999"]
    raw["files"][0]["tests"]["unit"].append({"id": "unit:fee_tests", "reason": "x", "confidence": "low", "evidence": []})
    plan, _ = parse_llm_plan(raw)
    fixed, notes = repair_plan(plan, fee_ctx, 2)
    assert validate_plan(fixed, fee_ctx, 2).ok
    u = {t.id: t for t in fixed.files[0].tests.unit}
    assert u["unit:fee_tests/min_fee_test"].confidence == "low"
    assert any("downgraded" in n for n in notes)
    # 3 selections, budget 2: the medium fuzz test is kept over the two (now) low ones
    total = sum(len(x) for x in (fixed.files[0].tests.unit, fixed.files[0].tests.functional, fixed.files[0].tests.fuzz))
    assert total == 2 and fixed.files[0].tests.fuzz and any("over the budget" in n for n in notes)


def test_repair_leaves_model_decisions_alone(fee_ctx):
    raw = good_plan(fee_ctx)
    raw["decision"]["reason"] = ""
    raw["files"][0]["path"] = "doc/release-notes.md"
    plan, _ = parse_llm_plan(raw)
    fixed, _ = repair_plan(plan, fee_ctx, 10)
    errs = validate_plan(fixed, fee_ctx, 10).errors
    assert "decision.reason is empty" in errs and "decision.mutate is true but no files were selected" in errs
