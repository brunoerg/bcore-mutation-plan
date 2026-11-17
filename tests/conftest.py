"""Fixtures: a miniature repository following Bitcoin Core's layout and conventions."""

from __future__ import annotations

import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mutation_planner.config import PlannerConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


BASE_FILES: dict[str, str] = {
    "README.md": "# Bitcoin Core (fixture)\n",
    "doc/release-notes.md": "Release notes\n=============\n",
    "CMakeLists.txt": _d("""
        project(BitcoinCore)
        add_subdirectory(src)
    """),
    "src/CMakeLists.txt": _d("""
        add_library(bitcoin_node policy/fees.cpp wallet/coinselection.cpp wallet/spend.cpp)
        if(BUILD_TESTS)
          add_subdirectory(test)
        endif()
        if(BUILD_FUZZ_BINARY)
          add_subdirectory(test/fuzz)
        endif()
    """),
    "src/test/CMakeLists.txt": _d("""
        add_executable(test_bitcoin
          main.cpp
          fee_tests.cpp
          amount_tests.cpp
          mempool_persist_tests.cpp
        )
        function(add_boost_test source_file)
          string(REGEX MATCHALL "(BOOST_FIXTURE_TEST_SUITE|BOOST_AUTO_TEST_SUITE)\\\\(([A-Za-z0-9_]+)" out "${source_file}")
        endfunction()
        if(ENABLE_WALLET)
          add_subdirectory(${PROJECT_SOURCE_DIR}/src/wallet/test wallet)
        endif()
    """),
    "src/wallet/test/CMakeLists.txt": _d("""
        target_sources(test_bitcoin
          PRIVATE
            coinselection_tests.cpp
        )
    """),
    "src/test/fuzz/CMakeLists.txt": _d("""
        add_executable(fuzz
          fee.cpp
          deserialize.cpp
        )
    """),
    "src/policy/fees.h": _d("""
        #ifndef BITCOIN_POLICY_FEES_H
        #define BITCOIN_POLICY_FEES_H
        #include <cstdint>
        using CAmount = int64_t;
        CAmount GetMinFee(CAmount size);
        bool IsDustFee(CAmount fee);
        #endif
    """),
    "src/policy/fees.cpp": _d("""
        #include <policy/fees.h>

        namespace {
        constexpr CAmount FEE_PER_BYTE{1};
        } // namespace

        CAmount GetMinFee(CAmount size)
        {
            if (size <= 0) return 0;
            return size * FEE_PER_BYTE;
        }

        bool IsDustFee(CAmount fee)
        {
            return fee < 3;
        }
    """),
    "src/wallet/coinselection.h": _d("""
        #ifndef BITCOIN_WALLET_COINSELECTION_H
        #define BITCOIN_WALLET_COINSELECTION_H
        #include <vector>
        namespace wallet {
        struct OutputGroup { long value; };
        class SelectionResult
        {
        public:
            explicit SelectionResult(long target) : m_target(target) {}
            long GetTarget() const { return m_target; }
        private:
            long m_target;
        };
        bool SelectCoinsBnB(std::vector<OutputGroup>& pool, long target);
        } // namespace wallet
        #endif
    """),
    "src/wallet/coinselection.cpp": _d("""
        #include <wallet/coinselection.h>

        namespace wallet {
        bool SelectCoinsBnB(std::vector<OutputGroup>& pool, long target)
        {
            long total = 0;
            for (const auto& g : pool) {
                total += g.value;
                if (total >= target) return true;
            }
            return false;
        }
        } // namespace wallet
    """),
    "src/wallet/spend.cpp": _d("""
        #include <wallet/coinselection.h>

        namespace wallet {
        bool ChooseSelectionResult(std::vector<OutputGroup>& pool, long target) EXCLUSIVE_LOCKS_REQUIRED(cs_wallet)
        {
            return SelectCoinsBnB(pool, target);
        }
        } // namespace wallet
    """),
    "src/wallet/rpc/spend.cpp": _d("""
        #include <rpc/util.h>

        namespace wallet {
        static RPCMethod fundrawtransaction()
        {
            return RPCMethod{
                "fundrawtransaction",
                "Fund a transaction.",
                {},
                [&](const RPCMethod& self, const JSONRPCRequest& request) -> UniValue
                {
                    std::vector<OutputGroup> pool;
                    if (!ChooseSelectionResult(pool, 10)) throw JSONRPCError(RPC_WALLET_INSUFFICIENT_FUNDS, "Insufficient funds");
                    return UniValue{};
                },
            };
        }
        } // namespace wallet
    """),
    "src/node/mempool_persist.cpp": _d("""
        #include <string>

        std::string DumpMempoolBanner()
        {
            return "mempool";
        }
    """),
    "src/util/strutil.h": _d("""
        #ifndef BITCOIN_UTIL_STRUTIL_H
        #define BITCOIN_UTIL_STRUTIL_H
        // Character helpers.
        inline bool IsSpaceChar(char c)
        {
            return c == ' ' || c == '\\t';
        }
        #endif
    """),
    "src/leveldb/db.cc": "int leveldb_main() { return 0; }\n",
    "src/test/main.cpp": "#define BOOST_TEST_MODULE Bitcoin Core Test Suite\n",
    "src/test/fee_tests.cpp": _d("""
        #include <policy/fees.h>
        #include <boost/test/unit_test.hpp>

        BOOST_FIXTURE_TEST_SUITE(fee_tests, BasicTestingSetup)

        BOOST_AUTO_TEST_CASE(min_fee_test)
        {
            BOOST_CHECK_EQUAL(GetMinFee(10), 10);
            BOOST_CHECK_EQUAL(GetMinFee(0), 0);
        }

        BOOST_AUTO_TEST_CASE(dust_fee_test)
        {
            BOOST_CHECK(IsDustFee(1));
            // GetMinFee is only mentioned in this comment
        }

        BOOST_AUTO_TEST_SUITE_END()
    """),
    "src/test/amount_tests.cpp": _d("""
        #include <boost/test/unit_test.hpp>
        #include <util/strutil.h>

        BOOST_AUTO_TEST_SUITE(amount_tests)
        BOOST_AUTO_TEST_CASE(space_test)
        {
            BOOST_CHECK(IsSpaceChar(' '));
            const char* desc = "wpkh(xpub/0/*)"; // the /* inside a string is not a comment
        }
        BOOST_AUTO_TEST_SUITE_END()
    """),
    "src/test/mempool_persist_tests.cpp": _d("""
        #include <boost/test/unit_test.hpp>

        BOOST_AUTO_TEST_SUITE(mempool_persist_tests)
        BOOST_AUTO_TEST_CASE(roundtrip)
        {
            BOOST_CHECK(true);
        }
        BOOST_AUTO_TEST_SUITE_END()
    """),
    "src/wallet/test/coinselection_tests.cpp": _d("""
        #include <wallet/coinselection.h>
        #include <boost/test/unit_test.hpp>

        namespace wallet {
        BOOST_FIXTURE_TEST_SUITE(coinselection_tests, BasicTestingSetup)

        BOOST_AUTO_TEST_CASE(bnb_test)
        {
            std::vector<OutputGroup> pool{{5}, {6}};
            BOOST_CHECK(SelectCoinsBnB(pool, 10));
        }

        BOOST_AUTO_TEST_CASE(result_test)
        {
            SelectionResult r{3};
            BOOST_CHECK_EQUAL(r.GetTarget(), 3);
        }

        BOOST_AUTO_TEST_SUITE_END()
        } // namespace wallet
    """),
    "src/test/fuzz/fuzz.h": _d("""
        #define FUZZ_TARGET(...) DETAIL_FUZZ(__VA_ARGS__)
    """),
    "src/test/fuzz/fee.cpp": _d("""
        #include <policy/fees.h>
        #include <test/fuzz/fuzz.h>

        FUZZ_TARGET(fee_rate)
        {
            (void)GetMinFee(buffer.size());
        }

        FUZZ_TARGET(fee_hidden, .init = initialize_fee, .hidden = true)
        {
        }
    """),
    "src/test/fuzz/deserialize.cpp": _d("""
        #include <test/fuzz/fuzz.h>

        #define FUZZ_TARGET_DESERIALIZE(name, code) \\
            FUZZ_TARGET(name, .init = initialize_deserialize) \\
            {                                                 \\
                code                                          \\
            }

        FUZZ_TARGET_DESERIALIZE(block_deserialize, {
            int x = 0;
        })
    """),
    "test/lint/lint_ignore_dirs.py": 'SHARED_EXCLUDED_SUBTREES = ["src/leveldb/"]\n',
    "test/functional/test_runner.py": _d("""
        #!/usr/bin/env python3
        import os
        TOOL_CHECK = "tool_check.py --flag"
        EXTENDED_SCRIPTS = [
            'feature_pruning.py',
        ]
        BASE_SCRIPTS = [
            TOOL_CHECK,
            'wallet_fundrawtransaction.py',
            'feature_fee.py',
            'wallet_multi.py',
            'wallet_multi.py --usecli',
        ]
        NON_SCRIPTS = [
            "combine_logs.py",
            "test_runner.py",
        ]
        raise SystemExit("must never be executed by the planner")
    """),
    "test/functional/wallet_fundrawtransaction.py": _d("""
        class FundTest:
            def run_test(self):
                self.nodes[0].fundrawtransaction("00")
                self.test_insufficient()

            def test_insufficient(self):
                assert_raises_rpc_error(-4, "Insufficient funds", self.nodes[0].fundrawtransaction, "00")
    """),
    "test/functional/feature_fee.py": "class F:\n    def run_test(self):\n        self.nodes[0].getmempoolinfo()\n",
    "test/functional/wallet_multi.py": "class W:\n    def run_test(self):\n        pass\n",
    "test/functional/tool_check.py": "class T:\n    def run_test(self):\n        pass\n",
    "test/functional/feature_pruning.py": "class P:\n    def run_test(self):\n        pass\n",
    "test/functional/combine_logs.py": "print('helper')\n",
    "test/functional/mempool_helper.py": "# not registered in test_runner.py\n",
    "test/functional/test_framework/util.py": "def assert_raises_rpc_error(*a):\n    pass\n",
}


def git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(repo), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    }
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def write_files(repo: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)


@dataclass
class FixtureRepo:
    path: Path
    base: str

    def change(
        self,
        files: dict[str, str | bytes] | None = None,
        delete: list[str] | None = None,
        rename: list[tuple[str, str]] | None = None,
        branch: str = "pr",
    ) -> str:
        """Create a PR branch from base with the given changes; return its head SHA."""
        git(self.path, "checkout", "-q", "-B", branch, self.base)
        for old, new in rename or []:
            (self.path / new).parent.mkdir(parents=True, exist_ok=True)
            git(self.path, "mv", old, new)
        write_files(self.path, files or {})
        for d in delete or []:
            git(self.path, "rm", "-q", d)
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", "pr change")
        head = git(self.path, "rev-parse", "HEAD")
        git(self.path, "checkout", "-q", "main")
        return head

    def read(self, rel: str) -> str:
        return BASE_FILES[rel]


@pytest.fixture
def repo(tmp_path: Path) -> FixtureRepo:
    path = tmp_path / "bitcoin"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    write_files(path, BASE_FILES)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return FixtureRepo(path=path, base=git(path, "rev-parse", "HEAD"))


@pytest.fixture
def config() -> PlannerConfig:
    return PlannerConfig()


def replace(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new)


class FakeProvider:
    """Returns canned responses (a list consumed per call) and records calls."""

    def __init__(self, responses: list[Any], name: str = "fake", model: str | None = "fake-model"):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.name = name
        self.model = model

    def generate_plan(self, system_prompt: str, context: dict[str, Any], output_schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "context": context, "schema": output_schema})
        resp = self.responses.pop(0)
        if callable(resp):
            return resp(context)
        if isinstance(resp, Exception):
            raise resp
        return resp
