"""Deterministic discovery of candidate tests for each changed production file.

Every relationship carries evidence (file, line, description). Static references are
facts about test *source text*; they are never presented as dynamic coverage.
Relationships through callers or production areas are marked as inferred, and
filename similarity is recorded as weak, heuristic evidence only.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import ContextLimits, RepoConventions
from .schemas import (
    DIRECT_RELATIONSHIPS,
    INFERRED_RELATIONSHIPS,
    Candidate,
    ChangedFile,
    Evidence,
    Relationship,
    SourceFileContext,
    TestCategory,
    TestEntry,
    TestInventory,
    TruncationRecord,
)
from .source_analysis import extract_functions, is_specific_symbol
from .test_discovery import TestSources, _code_lines

log = logging.getLogger(__name__)

WEIGHTS: dict[Relationship, float] = {
    Relationship.direct_symbol_reference: 1.0,
    Relationship.rpc_method_call: 0.95,
    Relationship.string_literal_match: 0.85,
    Relationship.class_reference: 0.7,
    Relationship.header_include: 0.55,
    Relationship.rpc_call_path: 0.5,
    Relationship.call_path: 0.45,
    Relationship.module_area: 0.15,
    Relationship.name_similarity: 0.1,
    Relationship.coverage: 1.0,
}
BASIS = {
    **{r: "static" for r in DIRECT_RELATIONSHIPS},
    **{r: "inferred" for r in INFERRED_RELATIONSHIPS},
    Relationship.name_similarity: "heuristic",
    Relationship.coverage: "coverage",
}


@dataclass
class Hit:
    test_id: str
    relationship: Relationship
    path: str
    line: int | None
    symbol: str | None
    description: str


class TestLocator:
    """Maps (test file, line) to the most specific inventory entry."""

    __test__ = False

    def __init__(self, inventory: TestInventory, sources: TestSources):
        self.by_file: dict[str, list[TestEntry]] = defaultdict(list)
        for t in inventory.all():
            self.by_file[t.source_path].append(t)
        self.sources = sources
        self._code: dict[str, list[str]] = {}

    def code_lines(self, path: str) -> list[str]:
        if path not in self._code:
            text = self.sources.files.get(path, "")
            if path.endswith(".py"):
                self._code[path] = ["" if ln.lstrip().startswith("#") else ln for ln in text.splitlines()]
            else:
                self._code[path] = _code_lines(text)
        return self._code[path]

    def attribute(self, path: str, line: int | None) -> list[str]:
        entries = self.by_file.get(path, [])
        if not entries:
            return []
        if entries[0].category == TestCategory.functional:
            return [entries[0].id]

        def inside(t: TestEntry) -> bool:
            return line is not None and t.line is not None and t.end_line is not None and t.line <= line <= t.end_line

        for gran in ("case", "target"):
            hit = [t for t in entries if t.granularity == gran and inside(t)]
            if hit:
                return [hit[0].id]
        suites = [t for t in entries if t.granularity == "suite" and inside(t)]
        if suites:
            return [max(suites, key=lambda t: t.line or 0).id]
        # Outside any case/target (file-level helper, include): whole file.
        tops = [t for t in entries if t.granularity in ("suite", "target")]
        return [t.id for t in tops if "/" not in (t.suite or "")] or [t.id for t in entries]

    def file_level(self, path: str) -> list[str]:
        return self.attribute(path, None)


def _word_re(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + r")(?![A-Za-z0-9_])")


def _include_path(path: str) -> str:
    return path[4:] if path.startswith("src/") else path


def _bare_stem(path: str) -> str:
    stem = PurePosixPath(path).stem.lower()
    for suffix in ("_tests", "_test", "_impl"):
        stem = stem.removesuffix(suffix)
    return stem


def similar_names(a: str, b: str) -> bool:
    """Whole-stem similarity (e.g. coinselection.cpp ~ coinselection_tests.cpp)."""
    sa, sb = _bare_stem(a), _bare_stem(b)
    return sa == sb or (len(sa) >= 6 and sa in sb) or (len(sb) >= 6 and sb in sa)


def _stem_tokens(path: str) -> set[str]:
    stem = PurePosixPath(path).stem.lower()
    for suffix in ("_tests", "_test", "_impl"):
        stem = stem.removesuffix(suffix)
    return {stem} | {t for t in re.split(r"[_\-.]", stem) if len(t) >= 5}


class CandidateFinder:
    def __init__(
        self,
        inventory: TestInventory,
        sources: TestSources,
        production_texts: dict[str, str],
        head_files: set[str],
        conventions: RepoConventions,
        limits: ContextLimits,
    ):
        self.inv = inventory
        self.sources = sources
        self.locator = TestLocator(inventory, sources)
        self.production_texts = production_texts
        self.head_files = head_files
        self.conv = conventions
        self.limits = limits
        self._func_cache: dict[str, list] = {}
        self.cpp_test_files = sorted(p for p in sources.files if not p.endswith(".py") and p in self.locator.by_file)
        self.cpp_test_files_all = sorted(p for p in sources.files if not p.endswith(".py"))
        self.py_test_files = sorted(p for p in sources.files if p.endswith(".py") and p in self.locator.by_file)

    # -- search primitives -------------------------------------------------

    def _search(self, files: list[str], pattern: re.Pattern, quick: list[str]) -> list[tuple[str, int, str]]:
        out = []
        for path in files:
            text = self.sources.files.get(path, "")
            if not any(q in text for q in quick):
                continue
            for n, line in enumerate(self.locator.code_lines(path), 1):
                if m := pattern.search(line):
                    out.append((path, n, m.group(1) if m.groups() else m.group(0)))
        return out

    def _hits_for_symbols(self, symbols: dict[str, str], rel: Relationship, files: list[str], src: str, what: str) -> list[Hit]:
        """symbols: symbol -> explanation of why it matters."""
        if not symbols:
            return []
        hits = []
        for path, line, sym in self._search(files, _word_re(list(symbols)), list(symbols)):
            for tid in self.locator.attribute(path, line):
                hits.append(Hit(tid, rel, path, line, sym, f"{path}:{line} references {what} `{sym}` ({symbols[sym]})"))
        return hits

    def _rpc_hits(self, rpcs: dict[str, str], rel: Relationship) -> list[Hit]:
        if not rpcs:
            return []
        names = "|".join(re.escape(r) for r in rpcs)
        py_re = re.compile(rf"(?:\.({names})\s*\(|[\"']({names})[\"'\s])")
        cpp_re = re.compile(rf"\"({names})[\"\s]")
        hits = []
        for files, pat in ((self.py_test_files, py_re), (self.cpp_test_files, cpp_re)):
            for path in files:
                text = self.sources.files.get(path, "")
                if not any(r in text for r in rpcs):
                    continue
                for n, line in enumerate(self.locator.code_lines(path), 1):
                    m = pat.search(line)
                    if not m:
                        continue
                    rpc = next(g for g in m.groups() if g)
                    for tid in self.locator.attribute(path, n):
                        hits.append(Hit(tid, rel, path, n, rpc, f"{path}:{n} calls RPC `{rpc}` ({rpcs[rpc]})"))
        return hits

    def _literal_hits(self, literals: list[str], src: str) -> list[Hit]:
        hits = []
        files = self.cpp_test_files + self.py_test_files
        for lit in literals:
            for path in files:
                if lit not in self.sources.files.get(path, ""):
                    continue
                for n, line in enumerate(self.locator.code_lines(path), 1):
                    if lit in line:
                        for tid in self.locator.attribute(path, n):
                            hits.append(Hit(
                                tid, Relationship.string_literal_match, path, n, lit,
                                f"{path}:{n} contains string `{lit[:80]}` that appears on a changed line of {src}",
                            ))
        return hits

    def _include_hits(self, headers: list[str], src: str) -> list[Hit]:
        hits = []
        for hdr in headers:
            inc = re.compile(rf'^\s*#\s*include\s*[<"]{re.escape(hdr)}[>"]')
            for path in self.cpp_test_files_all:
                text = self.sources.files.get(path, "")
                if hdr not in text:
                    continue
                for n, line in enumerate(text.splitlines(), 1):
                    if inc.match(line):
                        targets = self.locator.file_level(path)
                        if not targets:  # included by a test utility, not a test file
                            continue
                        for tid in targets:
                            hits.append(Hit(tid, Relationship.header_include, path, n, hdr, f"{path}:{n} includes <{hdr}> ({src} or its header)"))
        return hits

    # -- call paths ----------------------------------------------------------

    def _functions_of(self, path: str):
        if path not in self._func_cache:
            text = self.production_texts.get(path, "")
            self._func_cache[path] = extract_functions(text.encode())[0] if text else []
        return self._func_cache[path]

    def callers_of(self, symbol: str, exclude: set[tuple[str, str]]) -> list[tuple[str, str, int, str | None]]:
        """Production functions whose bodies call ``symbol``: (caller, file, line, rpc)."""
        from .source_analysis import rpc_name_in

        call = re.compile(rf"(?<![A-Za-z0-9_~]){re.escape(symbol)}\s*(?:<[^;()]*>)?\s*\(")
        callers: list[tuple[str, str, int, str | None]] = []
        seen: set[tuple[str, str]] = set()
        for path in sorted(self.production_texts):
            text = self.production_texts[path]
            if symbol not in text or not call.search(text):
                continue
            for n, line in enumerate(_code_lines(text), 1):
                if not call.search(line):
                    continue
                funcs = [f for f in self._functions_of(path) if f.start_line <= n <= f.end_line]
                if not funcs:
                    continue
                f = min(funcs, key=lambda f: f.end_line - f.start_line)
                short = f.name.split("::")[-1]
                key = (path, f.name)
                if short == symbol or key in seen or key in exclude or n == f.start_line:
                    continue
                seen.add(key)
                callers.append((f.name, path, n, rpc_name_in(f.body_text)))
                if len(callers) >= self.limits.max_callers_per_symbol:
                    return callers
        return callers

    # -- main entry ------------------------------------------------------------

    def find(self, ctx: SourceFileContext, changed: ChangedFile) -> list[Hit]:
        path = ctx.path
        min_len = self.limits.min_symbol_length
        hits: list[Hit] = []

        func_syms: dict[str, str] = {}
        for f in ctx.functions:
            if f.kind in ("constructor", "destructor"):
                continue
            if is_specific_symbol(f.short_name, min_len):
                func_syms.setdefault(f.short_name, f"changed function {f.name} in {path}")
        hits += self._hits_for_symbols(func_syms, Relationship.direct_symbol_reference, self.cpp_test_files, path, "changed function")

        class_syms = {c: f"class/struct changed in {path}" for c in ctx.affected_classes if is_specific_symbol(c, min_len)}
        for f in ctx.functions:
            if f.kind in ("constructor", "destructor") and f.enclosing_scope:
                class_syms.setdefault(f.enclosing_scope[-1], f"constructor/destructor changed in {path}")
            elif "::" in f.name:
                owner = f.name.split("::")[-2]
                if is_specific_symbol(owner, min_len) and owner[:1].isupper():
                    class_syms.setdefault(owner, f"owner of changed method {f.name}")
        class_syms = {k: v for k, v in class_syms.items() if k not in func_syms}
        hits += self._hits_for_symbols(class_syms, Relationship.class_reference, self.cpp_test_files, path, "affected type")

        rpcs = {f.rpc_name: f"RPC defined by changed function {f.name} in {path}" for f in ctx.functions if f.rpc_name}
        hits += self._rpc_hits(rpcs, Relationship.rpc_method_call)

        hits += self._literal_hits(ctx.changed_string_literals, path)

        headers = []
        if path.endswith((".h", ".hpp")):
            headers.append(_include_path(path))
        else:
            hdr = str(PurePosixPath(path).with_suffix(".h"))
            if hdr in self.head_files:
                headers.append(_include_path(hdr))
        hits += self._include_hits(headers, path)

        # Walk up the static call graph (bounded BFS) to reach tests and RPC handlers
        # that exercise the changed functions indirectly.
        exclude = {(path, f.name) for f in ctx.functions}
        caller_syms: dict[str, str] = {}
        caller_rpcs: dict[str, str] = {}
        frontier = {sym: sym for sym in func_syms}
        visited = set(func_syms)
        for depth in range(1, self.limits.max_call_depth + 1):
            nxt: dict[str, str] = {}
            for sym, chain in frontier.items():
                for caller, cpath, cline, rpc in self.callers_of(sym, exclude):
                    short = caller.split("::")[-1]
                    route = f"{caller} -> {chain}"
                    why = f"static call chain {route} (first hop at {cpath}:{cline}, depth {depth}); inferred"
                    if rpc:
                        caller_rpcs.setdefault(rpc, why)
                    if is_specific_symbol(short, min_len) and short not in visited:
                        visited.add(short)
                        caller_syms[short] = why
                        nxt.setdefault(short, route)
            frontier = dict(list(nxt.items())[: self.limits.max_call_frontier])
            if not frontier:
                break
        hits += self._hits_for_symbols(caller_syms, Relationship.call_path, self.cpp_test_files, path, "caller")
        rpc_hits = self._rpc_hits({k: v for k, v in caller_rpcs.items() if k not in rpcs}, Relationship.rpc_call_path)
        hits += rpc_hits

        # Production area -> functional test families (low strength, inferred), ranked by
        # how often the script mentions the file's name tokens or reachable RPCs.
        stem = _stem_tokens(path)
        keywords = list(stem) + list(caller_rpcs) + list(rpcs)
        already = {h.test_id for h in hits}
        area: list[TestEntry] = []
        for prefix, families in self.conv.area_functional_prefixes.items():
            if path.startswith(prefix):
                area += [t for t in self.inv.functional if PurePosixPath(t.source_path).name.startswith(families)]

        def relevance(t: TestEntry) -> int:
            text = self.sources.files.get(t.source_path, "")
            return sum(text.count(k) for k in keywords)

        ranked_area = sorted({t.id: t for t in area if t.id not in already}.values(), key=lambda t: (-relevance(t), t.id))
        for t in ranked_area[: self.limits.max_module_area_candidates]:
            if relevance(t) == 0:
                break
            hits.append(Hit(t.id, Relationship.module_area, t.source_path, None, None,
                            f"{t.source_path} belongs to a functional-test family for {path}'s area and mentions related names {relevance(t)} times (inferred, not verified)"))

        # Filename similarity: weak by design.
        for tpath, ids in self.sources.tests_in_file.items():
            if similar_names(path, tpath):
                for tid in self.locator.file_level(tpath) or ids[:1]:
                    hits.append(Hit(tid, Relationship.name_similarity, tpath, None, None,
                                    f"{tpath} has a name similar to {path} (heuristic only; not evidence of execution)"))
        return hits


def rank(hits: list[Hit], inventory: dict[str, TestEntry], source_file: str, limits: ContextLimits) -> tuple[list[tuple[Candidate, list[Hit]]], TruncationRecord | None]:
    grouped: dict[str, list[Hit]] = defaultdict(list)
    for h in hits:
        if h.test_id in inventory:
            grouped[h.test_id].append(h)
    scored = []
    for tid, hs in grouped.items():
        rels = sorted({h.relationship for h in hs}, key=lambda r: -WEIGHTS[r])
        score = WEIGHTS[rels[0]] + 0.05 * (len(rels) - 1) + 0.01 * min(len(hs), 5)
        strength = "direct" if rels[0] in DIRECT_RELATIONSHIPS else "inferred" if rels[0] in INFERRED_RELATIONSHIPS else "weak"
        hs_sorted = sorted(hs, key=lambda h: (-WEIGHTS[h.relationship], h.path, h.line or 0))
        # keep one evidence item per relationship first, then fill
        picked: list[Hit] = []
        for r in rels:
            picked.append(next(h for h in hs_sorted if h.relationship == r))
        for h in hs_sorted:
            if h not in picked:
                picked.append(h)
        picked = picked[: limits.max_evidence_per_candidate]
        cand = Candidate(
            source_file=source_file, test_id=tid, category=inventory[tid].category,
            relationships=rels, strength=strength, score=round(min(score, 1.5), 3), evidence=[],
        )
        scored.append((cand, picked))
    scored.sort(key=lambda cp: (-cp[0].score, cp[0].test_id))

    cap = limits.max_candidates_per_file
    if len(scored) <= cap:
        return scored, None
    # Keep the strongest overall, but reserve room so indirect candidates are not crowded out.
    direct = [s for s in scored if s[0].strength == "direct"]
    inferred = [s for s in scored if s[0].strength == "inferred"]
    weak = [s for s in scored if s[0].strength == "weak"]
    reserve = min(len(inferred), max(cap // 4, 1))
    kept = direct[: cap - reserve]
    kept += inferred[: cap - len(kept)]
    kept += weak[: max(cap - len(kept), 0)]
    kept.sort(key=lambda cp: (-cp[0].score, cp[0].test_id))
    dropped = len(scored) - len(kept)
    dropped_direct = len(direct) - sum(1 for k in kept if k[0].strength == "direct")
    rec = TruncationRecord(
        scope=f"candidates:{source_file}",
        detail=f"kept {len(kept)} of {len(scored)} candidate tests (dropped {dropped}, of which {dropped_direct} direct); raise --max-candidates-per-file to include more",
    )
    return kept, rec


def find_candidates(
    targets: list[SourceFileContext],
    changed: dict[str, ChangedFile],
    finder: CandidateFinder,
    limits: ContextLimits,
) -> tuple[list[Candidate], list[Evidence], list[TruncationRecord]]:
    inv = finder.inv.by_id()
    candidates: list[Candidate] = []
    evidence: list[Evidence] = []
    truncations: list[TruncationRecord] = []
    counter = 0
    for ctx in sorted(targets, key=lambda c: c.path):
        hits = finder.find(ctx, changed[ctx.path])
        ranked, trunc = rank(hits, inv, ctx.path, limits)
        if trunc:
            truncations.append(trunc)
        for cand, picked in ranked:
            for h in picked:
                counter += 1
                eid = f"ev-{counter:04d}"
                evidence.append(Evidence(
                    id=eid, source_file=ctx.path, test_id=h.test_id, relationship=h.relationship,
                    basis=BASIS[h.relationship], path=h.path, line=h.line, symbol=h.symbol, description=h.description,
                ))
                cand.evidence.append(eid)
            candidates.append(cand)
        log.info("candidates for %s: %d (from %d raw relationships)", ctx.path, len(ranked), len(hits))
    return candidates, evidence, truncations
