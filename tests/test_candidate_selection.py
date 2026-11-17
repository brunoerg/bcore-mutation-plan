from conftest import BASE_FILES, replace
from mutation_planner.analysis import build_context
from mutation_planner.candidate_selection import Hit, rank
from mutation_planner.config import ContextLimits
from mutation_planner.schemas import PRMetadata, Relationship, TestCategory, TestEntry


def cands(ctx, path):
    return {c.test_id: c for c in ctx.candidates if c.source_file == path}


def test_direct_symbol_reference_is_case_level(repo, config):
    head = repo.change(files={"src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "return 0;", "return 1;")})
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    c = cands(ctx, "src/policy/fees.cpp")
    assert Relationship.direct_symbol_reference in c["unit:fee_tests/min_fee_test"].relationships
    assert c["unit:fee_tests/min_fee_test"].strength == "direct"
    # dust_fee_test only mentions GetMinFee in a comment: no direct reference
    assert "unit:fee_tests/dust_fee_test" not in c or Relationship.direct_symbol_reference not in c["unit:fee_tests/dust_fee_test"].relationships
    assert Relationship.direct_symbol_reference in c["fuzz:fee_rate"].relationships
    ev = {e.id: e for e in ctx.evidence}
    first = ev[c["unit:fee_tests/min_fee_test"].evidence[0]]
    assert first.path == "src/test/fee_tests.cpp" and first.line == 8 and first.basis == "static"


def test_call_path_reaches_rpc_and_functional_tests(repo, config):
    head = repo.change(files={
        "src/wallet/coinselection.cpp": replace(BASE_FILES["src/wallet/coinselection.cpp"], "total >= target", "total > target"),
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    c = cands(ctx, "src/wallet/coinselection.cpp")
    assert c["unit:coinselection_tests/bnb_test"].strength == "direct"
    func = c["functional:wallet_fundrawtransaction.py"]
    assert Relationship.rpc_call_path in func.relationships and func.strength == "inferred"
    desc = next(e for e in ctx.evidence if e.id in func.evidence).description
    assert "fundrawtransaction -> ChooseSelectionResult -> SelectCoinsBnB" in desc
    # header include relationship for tests including wallet/coinselection.h
    assert Relationship.header_include in c["unit:coinselection_tests"].relationships


def test_changed_rpc_and_string_literal_are_direct(repo, config):
    head = repo.change(files={
        "src/wallet/rpc/spend.cpp": replace(BASE_FILES["src/wallet/rpc/spend.cpp"], '"Insufficient funds"', '"Insufficient funds"); (void)("Insufficient funds"'),
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    c = cands(ctx, "src/wallet/rpc/spend.cpp")
    rels = c["functional:wallet_fundrawtransaction.py"].relationships
    assert Relationship.rpc_method_call in rels and Relationship.string_literal_match in rels


def test_name_similarity_is_weak_only(repo, config):
    """Scenario I (analysis side): similar filenames produce only weak, heuristic evidence."""
    head = repo.change(files={
        "src/node/mempool_persist.cpp": replace(BASE_FILES["src/node/mempool_persist.cpp"], '"mempool"', '"mempool.dat"'),
    })
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    c = cands(ctx, "src/node/mempool_persist.cpp")
    sim = c["unit:mempool_persist_tests"]
    assert sim.relationships == [Relationship.name_similarity]
    assert sim.strength == "weak"
    assert all(e.basis == "heuristic" for e in ctx.evidence if e.id in sim.evidence)


def _entry(i, cat=TestCategory.unit):
    return TestEntry(id=f"unit:s{i}", category=cat, source_path="src/test/x.cpp", granularity="suite")


def test_truncation_keeps_direct_and_reserves_inferred_slots():
    inv = {f"unit:s{i}": _entry(i) for i in range(30)}
    hits = [Hit(f"unit:s{i}", Relationship.direct_symbol_reference, "p", i, "F", "d") for i in range(10)]
    hits += [Hit(f"unit:s{i}", Relationship.call_path, "p", i, "G", "d") for i in range(10, 20)]
    hits += [Hit(f"unit:s{i}", Relationship.name_similarity, "p", i, None, "d") for i in range(20, 30)]
    kept, trunc = rank(hits, inv, "src/a.cpp", ContextLimits(max_candidates_per_file=12))
    strengths = [c.strength for c, _ in kept]
    assert strengths.count("direct") == 9 and strengths.count("inferred") == 3 and "weak" not in strengths
    assert trunc is not None and "kept 12 of 30" in trunc.detail
