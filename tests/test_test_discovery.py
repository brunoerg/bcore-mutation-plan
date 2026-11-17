from mutation_planner.config import RepoConventions
from mutation_planner.git_analysis import GitRepo
from mutation_planner.test_discovery import _code_lines, cmake_commands, discover_tests, parse_cmake


def inventory(repo):
    r = GitRepo(repo.path)
    return discover_tests(r, repo.base, r.list_files(repo.base), RepoConventions())


def test_unit_suites_and_cases(repo):
    inv, _ = inventory(repo)
    ids = {t.id: t for t in inv.unit}
    assert {"unit:fee_tests", "unit:fee_tests/min_fee_test", "unit:fee_tests/dust_fee_test",
            "unit:amount_tests", "unit:amount_tests/space_test",
            "unit:coinselection_tests", "unit:coinselection_tests/bnb_test"} <= set(ids)
    case = ids["unit:fee_tests/min_fee_test"]
    assert case.granularity == "case" and case.suite == "fee_tests" and case.case == "min_fee_test"
    assert case.executable == "test_bitcoin"
    assert case.selector == "--run_test=fee_tests/min_fee_test"
    assert case.registered_in_build is True
    assert case.build_condition == "BUILD_TESTS"
    assert ids["unit:fee_tests"].granularity == "suite"
    # wallet tests are added from src/wallet/test via add_subdirectory under if(ENABLE_WALLET)
    assert ids["unit:coinselection_tests/bnb_test"].build_condition == "BUILD_TESTS AND ENABLE_WALLET"
    # the "/*" inside a string literal must not hide the rest of amount_tests.cpp
    assert ids["unit:amount_tests"].end_line is not None


def test_no_invented_cases_from_comments(repo):
    inv, _ = inventory(repo)
    assert not any("GetMinFee" in t.id for t in inv.unit)


def test_functional_scripts_follow_test_runner(repo):
    inv, _ = inventory(repo)
    ids = {t.id: t for t in inv.functional}
    assert set(ids) == {
        "functional:wallet_fundrawtransaction.py", "functional:feature_fee.py", "functional:wallet_multi.py",
        "functional:tool_check.py", "functional:feature_pruning.py",
    }
    assert ids["functional:wallet_multi.py"].variants == ["--usecli"]
    assert ids["functional:tool_check.py"].variants == ["--flag"]  # resolved from a module constant
    assert ids["functional:feature_pruning.py"].extended is True
    assert ids["functional:wallet_fundrawtransaction.py"].extended is False
    assert ids["functional:wallet_fundrawtransaction.py"].scenarios == ["run_test", "test_insufficient"]
    assert ids["functional:wallet_fundrawtransaction.py"].command == "build/test/functional/test_runner.py wallet_fundrawtransaction.py"
    assert any("mempool_helper.py" in n for n in inv.notes)  # helper, not a test


def test_fuzz_targets_including_wrapper_macros(repo):
    inv, _ = inventory(repo)
    ids = {t.id: t for t in inv.fuzz}
    assert set(ids) == {"fuzz:fee_rate", "fuzz:fee_hidden", "fuzz:block_deserialize"}
    assert ids["fuzz:fee_hidden"].hidden is True and ids["fuzz:fee_hidden"].init_function == "initialize_fee"
    assert ids["fuzz:fee_rate"].selector == "FUZZ=fee_rate"
    assert ids["fuzz:fee_rate"].build_condition == "BUILD_FUZZ_BINARY"
    assert ids["fuzz:block_deserialize"].source_path == "src/test/fuzz/deserialize.cpp"


def test_cmake_parser_handles_quoted_parens_and_else():
    text = (
        'string(REGEX MATCHALL "(A|B)\\\\(([a-z]+)" out "${x}")\n'
        "if(FOO)\n  add_executable(t a.cpp)\nelse()\n  target_sources(t PRIVATE b.cpp)\nendif()\n"
    )
    assert [c[0] for c in cmake_commands(text)] == ["string", "if", "add_executable", "else", "target_sources", "endif"]
    info = parse_cmake({"src/x/CMakeLists.txt": text})
    assert info.source_condition["src/x/a.cpp"] == "FOO"
    assert info.source_condition["src/x/b.cpp"] == "NOT (FOO)"


def test_code_lines_respects_strings():
    lines = _code_lines('a = "x/*y"; // c\nb /* c */ = 1;\n/* multi\nline */ z;')
    assert lines == ['a = "x/*y"; ', "b  = 1;", "", " z;"]
