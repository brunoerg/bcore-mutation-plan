"""Planner configuration: limits, repository conventions and defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_PROVIDER = "claude-code"
DEFAULT_OUTPUT = "mutation-plan.json"
DEFAULT_CONTEXT_OUTPUT = "mutation-context.json"
DEFAULT_TIMEOUT = 900


@dataclass
class ContextLimits:
    """Budgets that keep the LLM context bounded. All are configurable from the CLI."""

    max_tests_per_file: int = 10
    max_candidates_per_file: int = 40
    max_evidence_per_candidate: int = 3
    max_function_snippet_lines: int = 160
    snippet_context_lines: int = 12
    max_patch_chars_per_file: int = 24_000
    max_unattributed_snippet_lines: int = 60
    max_callers_per_symbol: int = 12
    max_call_depth: int = 8
    max_call_frontier: int = 24
    max_module_area_candidates: int = 5
    max_total_context_chars: int = 600_000
    min_symbol_length: int = 4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepoConventions:
    """Bitcoin Core layout conventions.

    These are fallbacks: where the repository itself records the information (CMake
    target sources, test_runner.py script lists, lint_ignore_dirs.py subtrees) the
    repository wins.
    """

    source_root: str = "src/"
    functional_dir: str = "test/functional/"
    functional_runner: str = "test/functional/test_runner.py"
    subtree_lint_file: str = "test/lint/lint_ignore_dirs.py"
    fallback_subtrees: tuple[str, ...] = (
        "src/leveldb/",
        "src/crc32c/",
        "src/secp256k1/",
        "src/minisketch/",
        "src/ipc/libmultiprocess/",
        "src/crypto/ctaes/",
    )
    # Files produced by generator scripts that carry no "generated" banner.
    known_generated: tuple[str, ...] = (
        "src/chainparamsseeds.h",
        "src/qt/bitcoinstrings.cpp",
        "src/qt/locale/",
    )
    unit_test_binary: str = "test_bitcoin"
    fuzz_binary: str = "fuzz"
    # Functional test prefixes that broadly exercise a production area. Used only for
    # low-strength "module_area" candidates.
    area_functional_prefixes: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "src/wallet/": ("wallet_",),
            "src/net": ("p2p_",),
            "src/net_processing": ("p2p_",),
            "src/node/txdownloadman": ("p2p_",),
            "src/txmempool": ("mempool_",),
            "src/policy/": ("mempool_", "feature_fee"),
            "src/validation": ("feature_", "mempool_"),
            "src/consensus/": ("feature_",),
            "src/script/": ("feature_", "rpc_"),
            "src/rpc/": ("rpc_",),
            "src/index/": ("feature_", "rpc_"),
            "src/node/miner": ("mining_",),
            "src/httpserver": ("interface_http", "interface_rest"),
            "src/rest": ("interface_rest",),
            "src/zmq/": ("interface_zmq",),
        }
    )


@dataclass
class PlannerConfig:
    limits: ContextLimits = field(default_factory=ContextLimits)
    conventions: RepoConventions = field(default_factory=RepoConventions)
    max_llm_retries: int = 1
    timeout: int = DEFAULT_TIMEOUT
