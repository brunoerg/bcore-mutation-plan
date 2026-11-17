"""Discover Bitcoin Core's unit, functional and fuzz tests at the PR head revision.

Discovery follows the repository's own definitions:

* Unit tests: Boost.Test suite/case macros in sources that CMake adds to a test
  executable (``add_executable``/``target_sources``), mirroring ``add_boost_test`` in
  ``src/test/CMakeLists.txt``.
* Functional tests: the ``BASE_SCRIPTS``/``EXTENDED_SCRIPTS`` lists of
  ``test/functional/test_runner.py``, read with ``ast`` (never executed).
* Fuzz targets: ``FUZZ_TARGET(name, ...)`` registrations and macros that wrap it
  (e.g. ``FUZZ_TARGET_DESERIALIZE``), plus the CMake ``fuzz`` target sources.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .config import RepoConventions
from .schemas import TestCategory, TestEntry, TestInventory

log = logging.getLogger(__name__)


@dataclass
class TestSources:
    """Raw text of discovered test files (not serialized; used for candidate search)."""

    __test__ = False
    files: dict[str, str] = field(default_factory=dict)
    # path -> list of test ids defined in that file
    tests_in_file: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CMake parsing (very small subset: target source lists and if() conditions)
# ---------------------------------------------------------------------------

_CMAKE_CMD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_TARGET_CMDS = {"add_executable", "target_sources", "add_library"}


@dataclass
class CMakeInfo:
    source_target: dict[str, str] = field(default_factory=dict)  # repo path -> target
    source_condition: dict[str, str | None] = field(default_factory=dict)


def _strip_root_vars(token: str) -> str:
    for var in ("${PROJECT_SOURCE_DIR}/", "${CMAKE_SOURCE_DIR}/"):
        if token.startswith(var):
            return token[len(var):]
    if token.startswith("${CMAKE_CURRENT_SOURCE_DIR}/"):
        return "./" + token[len("${CMAKE_CURRENT_SOURCE_DIR}/"):]
    return token


def _strip_cmake_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def cmake_commands(text: str) -> list[tuple[str, list[str]]]:
    """Split CMake source into (command, args) pairs, handling multi-line invocations."""
    clean = _strip_cmake_comments(text)
    cmds: list[tuple[str, list[str]]] = []
    pos = 0
    while m := _CMAKE_CMD_RE.search(clean, pos):
        depth, i = 1, m.end()
        in_str = False
        while i < len(clean) and depth:
            c = clean[i]
            if in_str:
                if c == "\\":
                    i += 1
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            else:
                depth += {"(": 1, ")": -1}.get(c, 0)
            i += 1
        cmds.append((m.group(1).lower(), clean[m.end() : i - 1].split()))
        pos = i
    return cmds


def _walk_conditions(cmds: list[tuple[str, list[str]]]):
    """Yield (command, args, active conditions) tracking if/elseif/else/endif."""
    stack: list[list[str]] = []  # each frame: list of branch conditions seen so far
    for name, args in cmds:
        if name == "if":
            stack.append([" ".join(args)])
            continue
        if name in ("elseif", "else") and stack:
            frame = stack[-1]
            prev = " OR ".join(frame)
            frame.append(f"NOT ({prev})" + (f" AND {' '.join(args)}" if name == "elseif" else ""))
            continue
        if name == "endif" and stack:
            stack.pop()
            continue
        yield name, args, [f[-1] for f in stack]


def parse_cmake(files: dict[str, str]) -> CMakeInfo:
    info = CMakeInfo()
    parsed = {path: cmake_commands(text) for path, text in files.items()}
    # child directory -> (directory of the CMakeLists that added it, conditions there)
    dir_parent: dict[str, tuple[str, list[str]]] = {}
    for path, cmds in parsed.items():
        base = posixpath.dirname(path)
        for name, args, conds in _walk_conditions(cmds):
            if name == "add_subdirectory" and args:
                target = _strip_root_vars(args[0])
                sub = target if target.startswith("src/") else posixpath.normpath(posixpath.join(base, target))
                dir_parent[sub] = (base, conds)

    def inherited(d: str, depth: int = 0) -> list[str]:
        if depth > 32 or not d or d == ".":
            return []
        if d in dir_parent:
            parent, conds = dir_parent[d]
            return inherited(parent, depth + 1) + conds
        return inherited(posixpath.dirname(d), depth + 1)

    for path, cmds in parsed.items():
        base = posixpath.dirname(path)
        for name, args, conds in _walk_conditions(cmds):
            if name not in _TARGET_CMDS or not args:
                continue
            target = args[0]
            cond = list(dict.fromkeys(inherited(base) + conds))
            for tok in args[1:]:
                tok = _strip_root_vars(tok)
                if tok.endswith((".cpp", ".cc", ".c")) and "$" not in tok:
                    src = tok if tok.startswith("src/") else posixpath.normpath(posixpath.join(base, tok))
                    info.source_target.setdefault(src, target)
                    info.source_condition.setdefault(src, " AND ".join(cond) or None)
    return info


# ---------------------------------------------------------------------------
# Unit tests (Boost.Test)
# ---------------------------------------------------------------------------

_SUITE_RE = re.compile(r"^\s*BOOST_(?:AUTO|FIXTURE)_TEST_SUITE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)")
_SUITE_END_RE = re.compile(r"^\s*BOOST_AUTO_TEST_SUITE_END\s*\(")
_CASE_RE = re.compile(
    r"^\s*BOOST_(?:AUTO_TEST_CASE|FIXTURE_TEST_CASE|DATA_TEST_CASE|AUTO_TEST_CASE_TEMPLATE|FIXTURE_TEST_CASE_TEMPLATE)"
    r"\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)"
)


def _code_lines(text: str) -> list[str]:
    """Return lines with comments blanked (line numbering preserved).

    String and character literals are respected, so ``"xpub/0/*"`` or ``"http://"``
    inside a string do not start a comment.
    """
    out: list[str] = []
    in_block = False
    for line in text.splitlines():
        res: list[str] = []
        i, n = 0, len(line)
        quote: str | None = None
        while i < n:
            c = line[i]
            if in_block:
                j = line.find("*/", i)
                if j < 0:
                    i = n
                else:
                    in_block = False
                    i = j + 2
                continue
            if quote:
                res.append(c)
                if c == "\\" and i + 1 < n:
                    res.append(line[i + 1])
                    i += 2
                    continue
                if c == quote:
                    quote = None
                i += 1
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_block = True
                i += 2
                continue
            if c == '"' or (c == "'" and not (i > 0 and line[i - 1].isalnum())):
                quote = c
            res.append(c)
            i += 1
        out.append("".join(res))
    return out


def discover_unit_tests(
    files: dict[str, str], cmake: CMakeInfo, conventions: RepoConventions
) -> tuple[list[TestEntry], list[str]]:
    entries: dict[str, TestEntry] = {}
    notes: list[str] = []
    for path in sorted(files):
        text = files[path]
        if "BOOST_" not in text:
            continue
        lines = _code_lines(text)
        target = cmake.source_target.get(path)
        registered = target is not None
        executable = target or conventions.unit_test_binary
        cond = cmake.source_condition.get(path)
        suite_stack: list[tuple[str, int]] = []
        cases: list[tuple[str, str, int]] = []  # (suite path, case, line)
        suite_spans: list[tuple[str, int, int]] = []
        for n, line in enumerate(lines, 1):
            if m := _SUITE_RE.match(line):
                suite_stack.append((m.group(1), n))
            elif _SUITE_END_RE.match(line):
                if suite_stack:
                    full = "/".join(s for s, _ in suite_stack)
                    suite_spans.append((full, suite_stack[-1][1], n))
                    suite_stack.pop()
            elif m := _CASE_RE.match(line):
                cases.append(("/".join(s for s, _ in suite_stack), m.group(1), n))
        if not suite_spans and not cases:
            continue
        if suite_stack:
            notes.append(f"{path}: unbalanced BOOST test suite macros")
        for full, start, end in suite_spans:
            tid = f"unit:{full}"
            if tid in entries:
                notes.append(f"suite {full} defined in multiple files; first definition kept for {tid}")
                continue
            entries[tid] = TestEntry(
                id=tid, category=TestCategory.unit, source_path=path, line=start, end_line=end,
                suite=full, granularity="suite", executable=executable,
                selector=f"--run_test={full}",
                command=f"build/bin/{executable} --run_test={full}",
                build_condition=cond, registered_in_build=registered,
            )
        for idx, (suite, case, line) in enumerate(cases):
            nxt = [c[2] for c in cases[idx + 1 :]] + [e for s, b, e in suite_spans if s == suite and e > line]
            end = min(nxt) - 1 if nxt else len(lines)
            full = f"{suite}/{case}" if suite else case
            tid = f"unit:{full}"
            entries[tid] = TestEntry(
                id=tid, category=TestCategory.unit, source_path=path, line=line, end_line=end,
                suite=suite or None, case=case, granularity="case", executable=executable,
                selector=f"--run_test={full}",
                command=f"build/bin/{executable} --run_test={full}",
                build_condition=cond, registered_in_build=registered,
            )
        if not registered:
            notes.append(f"{path}: Boost tests found but file is not in any CMake test target")
    return list(entries.values()), notes


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

_SCENARIO_RE = re.compile(r"^\s*def\s+((?:test|run)_[A-Za-z0-9_]*)\s*\(\s*self", re.M)


def _literal_list(tree: ast.Module, name: str, consts: dict[str, str]) -> list[str] | None:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(node.value, ast.List):
            out = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append(elt.value)
                elif isinstance(elt, ast.Name) and elt.id in consts:
                    out.append(consts[elt.id])
            return out
    return None


def parse_test_runner(text: str) -> tuple[list[str], list[str], list[str]] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    consts = {
        t.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    base = _literal_list(tree, "BASE_SCRIPTS", consts)
    extended = _literal_list(tree, "EXTENDED_SCRIPTS", consts)
    non = _literal_list(tree, "NON_SCRIPTS", consts) or []
    if base is None:
        return None
    return base, extended or [], non


def discover_functional_tests(
    head_files: list[str], files: dict[str, str], runner_text: str | None, conventions: RepoConventions
) -> tuple[list[TestEntry], list[str]]:
    notes: list[str] = []
    fdir = conventions.functional_dir
    present = {p[len(fdir):] for p in head_files if p.startswith(fdir) and "/" not in p[len(fdir):] and p.endswith(".py")}
    parsed = parse_test_runner(runner_text) if runner_text else None
    scripts: dict[str, dict] = {}
    if parsed is None:
        notes.append("test_runner.py script lists unavailable; falling back to filename conventions")
        for name in sorted(present):
            if re.match(r"^(feature|interface|mempool|mining|p2p|rpc|tool|wallet|example)_.*\.py$", name):
                scripts[name] = {"variants": [""], "extended": False}
        non: list[str] = []
    else:
        base, extended, non = parsed
        for group, is_ext in ((base, False), (extended, True)):
            for spec in group:
                name, _, args = spec.partition(" ")
                d = scripts.setdefault(name, {"variants": [], "extended": is_ext})
                d["variants"].append(args.strip())
                d["extended"] = d["extended"] and is_ext
    entries = []
    for name in sorted(scripts):
        path = fdir + name
        if name not in present:
            notes.append(f"functional test {name} listed in test_runner.py but missing at head")
            continue
        text = files.get(path, "")
        variants = sorted({v for v in scripts[name]["variants"] if v})
        entries.append(
            TestEntry(
                id=f"functional:{name}", category=TestCategory.functional, source_path=path,
                granularity="script", executable="test/functional/test_runner.py",
                selector=name, command=f"build/test/functional/test_runner.py {name}",
                extended=scripts[name]["extended"], variants=variants,
                scenarios=sorted(set(_SCENARIO_RE.findall(text)))[:30],
            )
        )
    helpers = sorted(present - set(scripts) - set(non))
    if helpers:
        notes.append(f"{len(helpers)} python files in {fdir} are not registered test scripts: {', '.join(helpers[:10])}")
    return entries, notes


# ---------------------------------------------------------------------------
# Fuzz targets
# ---------------------------------------------------------------------------

_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)[^)]*\)((?:[^\n]*\\\n)*[^\n]*)", re.M)


def fuzz_wrapper_macros(files: dict[str, str]) -> set[str]:
    macros = {"FUZZ_TARGET"}
    for text in files.values():
        for m in _DEFINE_RE.finditer(text):
            name, first_param, body = m.groups()
            if name in ("FUZZ_TARGET", "DETAIL_FUZZ"):
                continue
            if re.search(rf"\bFUZZ_TARGET\s*\(\s*{first_param}\b", body):
                macros.add(name)
    return macros


def discover_fuzz_targets(
    files: dict[str, str], cmake: CMakeInfo, conventions: RepoConventions
) -> tuple[list[TestEntry], list[str]]:
    notes: list[str] = []
    candidates = {p: t for p, t in files.items() if "FUZZ_TARGET" in t}
    macros = fuzz_wrapper_macros(candidates)
    target_re = re.compile(rf"^\s*({'|'.join(sorted(macros))})\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(.*)$")
    entries: dict[str, TestEntry] = {}
    for path in sorted(candidates):
        lines = _code_lines(candidates[path])
        found: list[tuple[str, int, str, str]] = []
        in_define = False
        for n, line in enumerate(lines, 1):
            # Skip preprocessor directives and their backslash continuation lines (macro bodies).
            if in_define or line.lstrip().startswith("#"):
                in_define = line.rstrip().endswith("\\")
                continue
            if m := target_re.match(line):
                found.append((m.group(2), n, m.group(3), m.group(1)))
        target = cmake.source_target.get(path)
        for i, (name, line, rest, _macro) in enumerate(found):
            end = found[i + 1][1] - 1 if i + 1 < len(found) else len(lines)
            init = re.search(r"\.init\s*=\s*([A-Za-z_][A-Za-z0-9_:]*)", rest)
            tid = f"fuzz:{name}"
            if tid in entries:
                notes.append(f"fuzz target {name} registered more than once ({path})")
                continue
            entries[tid] = TestEntry(
                id=tid, category=TestCategory.fuzz, source_path=path, line=line, end_line=end,
                granularity="target", executable=target or conventions.fuzz_binary,
                selector=f"FUZZ={name}", command=f"FUZZ={name} build_fuzz/bin/{target or conventions.fuzz_binary}",
                build_condition=cmake.source_condition.get(path),
                registered_in_build=target is not None,
                hidden=bool(re.search(r"\.hidden\s*=\s*true", rest)),
                init_function=init.group(1) if init else None,
            )
        if found and target is None:
            notes.append(f"{path}: fuzz targets found but file is not in a CMake fuzz target")
    return list(entries.values()), notes


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def is_test_cpp(path: str) -> bool:
    p = PurePosixPath(path)
    return path.startswith("src/") and p.suffix in (".cpp", ".h") and any(part in ("test", "tests") for part in p.parts[:-1])


def discover_tests(repo, head_sha: str, head_files: list[str], conventions: RepoConventions) -> tuple[TestInventory, TestSources]:
    cmake_paths = [p for p in head_files if p.startswith("src/") and p.endswith("CMakeLists.txt")]
    if "CMakeLists.txt" in head_files:
        cmake_paths.append("CMakeLists.txt")
    test_cpp = [p for p in head_files if is_test_cpp(p)]
    fdir = conventions.functional_dir
    functional_py = [p for p in head_files if p.startswith(fdir) and p.endswith(".py") and "/" not in p[len(fdir):]]
    raw = repo.read_files(head_sha, cmake_paths + test_cpp + functional_py + [conventions.functional_runner])
    text = {p: (b.decode("utf-8", errors="replace") if b is not None else None) for p, b in raw.items()}

    cmake = parse_cmake({p: t for p, t in text.items() if p in cmake_paths and t})
    cpp_files = {p: text[p] for p in test_cpp if text.get(p)}
    unit, unit_notes = discover_unit_tests({p: t for p, t in cpp_files.items() if p.endswith(".cpp")}, cmake, conventions)
    fuzz, fuzz_notes = discover_fuzz_targets(cpp_files, cmake, conventions)
    py_files = {p: text[p] for p in functional_py if text.get(p)}
    functional, func_notes = discover_functional_tests(head_files, py_files, text.get(conventions.functional_runner), conventions)

    inv = TestInventory(unit=unit, functional=functional, fuzz=fuzz, notes=unit_notes + func_notes + fuzz_notes)
    sources = TestSources(files={**cpp_files, **py_files})
    for t in inv.all():
        sources.tests_in_file.setdefault(t.source_path, []).append(t.id)
    log.info(
        "test inventory: %d unit (%d suites), %d functional, %d fuzz",
        len(unit), sum(1 for u in unit if u.granularity == "suite"), len(functional), len(fuzz),
    )
    return inv, sources
