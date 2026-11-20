"""Git access and PR diff analysis.

All repository reads go through Git object storage at an explicit commit. The working
tree is never consulted, so the analysis always reflects the requested head commit.

Security: every Git invocation uses an argument list (never ``shell=True``), disables
external diff/textconv drivers, hooks and fsmonitor, and never runs repository code.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import ChangeStatus, LineRange

log = logging.getLogger(__name__)

_SAFE_GIT_CONFIG = [
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.quotepath=off",
    "-c", "diff.external=",
    "-c", "protocol.allow=never",
]
_SHA_RE = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
# Revisions accepted from the command line: SHAs, branch/tag names, HEAD~N, origin/x.
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^@{}-]{0,254}$")


class GitError(RuntimeError):
    pass


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["LC_ALL"] = "C"
    return env


def run_git(repo: Path, args: list[str], *, input: bytes | None = None, check: bool = True) -> bytes:
    cmd = ["git", *_SAFE_GIT_CONFIG, "-C", str(repo), *args]
    try:
        proc = subprocess.run(cmd, input=input, capture_output=True, env=_git_env(), check=False)
    except FileNotFoundError as exc:  # pragma: no cover - environment problem
        raise GitError("git executable not found on PATH") from exc
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


class GitRepo:
    """Read-only view of a repository's object database."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_dir():
            raise GitError(f"repository path does not exist: {self.path}")
        top = run_git(self.path, ["rev-parse", "--show-toplevel"]).decode().strip()
        self.path = Path(top)

    def resolve_commit(self, ref: str) -> str:
        if not _REF_RE.match(ref) or ref.startswith("-"):
            raise GitError(f"refusing suspicious revision argument: {ref!r}")
        out = run_git(self.path, ["rev-parse", "--verify", "--quiet", "--end-of-options", f"{ref}^{{commit}}"], check=False)
        sha = out.decode().strip()
        if not _SHA_RE.match(sha):
            raise GitError(f"cannot resolve {ref!r} to a commit in {self.path}")
        return sha

    def merge_base(self, a: str, b: str) -> str:
        out = run_git(self.path, ["merge-base", a, b], check=False).decode().strip()
        if not _SHA_RE.match(out):
            raise GitError(f"no merge base between {a} and {b}")
        return out

    def commit_count(self, start: str, end: str) -> int:
        """Number of commits reachable from ``end`` but not from ``start`` (SHAs only)."""
        out = run_git(self.path, ["rev-list", "--count", f"{start}..{end}"]).decode().strip()
        return int(out or 0)

    def merge_commits(self, start: str, end: str, limit: int = 5) -> tuple[int, list[tuple[str, str]]]:
        """(count, first ``limit`` (sha, subject) pairs) of merge commits in ``start..end``."""
        count = int(run_git(self.path, ["rev-list", "--count", "--merges", f"{start}..{end}"]).decode().strip() or 0)
        out = run_git(self.path, ["log", "--merges", f"-n{limit}", "--format=%H%x00%s", f"{start}..{end}"])
        pairs = []
        for line in out.decode("utf-8", errors="replace").splitlines():
            sha, _, subject = line.partition("\0")
            pairs.append((sha, subject))
        return count, pairs

    def tree_sha(self, commit: str) -> str:
        return run_git(self.path, ["rev-parse", f"{commit}^{{tree}}"]).decode().strip()

    def list_files(self, commit: str) -> list[str]:
        out = run_git(self.path, ["ls-tree", "-r", "-z", "--name-only", "--full-tree", commit])
        return [p for p in out.decode("utf-8", errors="replace").split("\0") if p]

    def read_files(self, commit: str, paths: list[str]) -> dict[str, bytes | None]:
        """Read many files at ``commit`` with a single ``git cat-file --batch``."""
        if not paths:
            return {}
        request = "".join(f"{commit}:{p}\n" for p in paths).encode()
        out = run_git(self.path, ["cat-file", "--batch"], input=request)
        result: dict[str, bytes | None] = {}
        pos = 0
        for p in paths:
            nl = out.index(b"\n", pos)
            header = out[pos:nl].decode(errors="replace").split()
            pos = nl + 1
            if len(header) == 3 and header[1] == "blob":
                size = int(header[2])
                result[p] = out[pos : pos + size]
                pos += size + 1
            else:  # "missing", or a tree/submodule
                result[p] = None
        return result

    def read_text(self, commit: str, path: str) -> str | None:
        data = self.read_files(commit, [path]).get(path)
        return None if data is None else data.decode("utf-8", errors="replace")


@dataclass
class RawChange:
    path: str
    old_path: str | None
    status: ChangeStatus
    head_blob_sha: str | None
    binary: bool = False
    submodule: bool = False
    additions: int = 0
    deletions: int = 0
    patch: str = ""
    changed_lines: list[LineRange] = field(default_factory=list)
    deletion_anchors: list[int] = field(default_factory=list)


_STATUS = {
    "A": ChangeStatus.added,
    "M": ChangeStatus.modified,
    "D": ChangeStatus.deleted,
    "R": ChangeStatus.renamed,
    "C": ChangeStatus.copied,
    "T": ChangeStatus.type_changed,
}
_NULL_SHA = "0" * 40
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_raw_diff(data: bytes) -> list[RawChange]:
    """Parse ``git diff --raw -z -M --no-abbrev`` output."""
    tokens = data.decode("utf-8", errors="replace").split("\0")
    changes: list[RawChange] = []
    i = 0
    while i < len(tokens):
        meta = tokens[i]
        if not meta:
            i += 1
            continue
        if not meta.startswith(":"):
            raise GitError(f"unexpected raw diff token: {meta!r}")
        _old_mode, new_mode, _old_sha, new_sha, status = meta[1:].split(" ")
        code = status[0]
        if code in ("R", "C"):
            old_path, path = tokens[i + 1], tokens[i + 2]
            i += 3
        else:
            old_path, path = None, tokens[i + 1]
            i += 2
        st = _STATUS.get(code)
        if st is None:
            log.warning("skipping unsupported diff status %s for %s", status, path)
            continue
        changes.append(
            RawChange(
                path=path,
                old_path=old_path,
                status=st,
                head_blob_sha=None if st == ChangeStatus.deleted or new_sha == _NULL_SHA else new_sha,
                submodule=new_mode == "160000",
            )
        )
    return changes


def parse_numstat(data: bytes) -> dict[str, tuple[int | None, int | None]]:
    """Parse ``git diff --numstat -z -M``; binary files report ``None`` counts."""
    tokens = data.decode("utf-8", errors="replace").split("\0")
    result: dict[str, tuple[int | None, int | None]] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok:
            i += 1
            continue
        added, deleted, rest = tok.split("\t", 2)
        if rest == "":  # rename/copy: next two tokens are old and new path
            path = tokens[i + 2]
            i += 3
        else:
            path = rest
            i += 1
        result[path] = (None if added == "-" else int(added), None if deleted == "-" else int(deleted))
    return result


def parse_patch_lines(patch: str) -> tuple[list[LineRange], list[int]]:
    """Compute head-side changed line ranges and pure-deletion anchors from a unified diff."""
    added: list[int] = []
    anchors: list[int] = []
    new_line = 0
    in_hunk = False
    pending_delete = False
    for line in patch.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            if pending_delete:
                anchors.append(max(new_line - 1, 1))
            new_line = int(m.group(3))
            in_hunk = True
            pending_delete = False
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        tag = line[:1]
        if tag == "+":
            added.append(new_line)
            new_line += 1
            pending_delete = False
        elif tag == "-":
            pending_delete = True
        elif tag == " ":
            if pending_delete:
                anchors.append(max(new_line - 1, 1))
                pending_delete = False
            new_line += 1
    if pending_delete:
        anchors.append(max(new_line - 1, 1))
    return _to_ranges(added), sorted(set(anchors))


def _to_ranges(lines: list[int]) -> list[LineRange]:
    ranges: list[LineRange] = []
    for ln in sorted(set(lines)):
        if ranges and ln == ranges[-1].end + 1:
            ranges[-1] = LineRange(start=ranges[-1].start, end=ln)
        else:
            ranges.append(LineRange(start=ln, end=ln))
    return ranges


_DIFF_OPTS = ["--no-color", "--no-ext-diff", "--no-textconv", "-M", "--src-prefix=a/", "--dst-prefix=b/"]


@dataclass
class DiffResult:
    base_sha: str
    head_sha: str
    merge_base_sha: str
    head_tree_sha: str
    changes: list[RawChange]
    commit_count: int = 0
    merge_count: int = 0
    merge_samples: list[tuple[str, str]] = field(default_factory=list)


def analyze_diff(repo: GitRepo, base: str, head: str) -> DiffResult:
    base_sha = repo.resolve_commit(base)
    head_sha = repo.resolve_commit(head)
    merge_base_sha = repo.merge_base(base_sha, head_sha)
    log.info("base=%s head=%s merge-base=%s", base_sha[:12], head_sha[:12], merge_base_sha[:12])

    raw = run_git(repo.path, ["diff", "--raw", "-z", "--no-abbrev", *_DIFF_OPTS, merge_base_sha, head_sha])
    changes = parse_raw_diff(raw)
    numstat = parse_numstat(run_git(repo.path, ["diff", "--numstat", "-z", *_DIFF_OPTS, merge_base_sha, head_sha]))

    for ch in changes:
        added, deleted = numstat.get(ch.path, (0, 0))
        ch.binary = added is None or ch.submodule
        ch.additions = added or 0
        ch.deletions = deleted or 0
        if ch.binary:
            continue
        paths = [p for p in (ch.old_path, ch.path) if p]
        patch = run_git(
            repo.path,
            ["diff", "-U3", *_DIFF_OPTS, merge_base_sha, head_sha, "--", *paths],
        ).decode("utf-8", errors="replace")
        ch.patch = patch
        if ch.status != ChangeStatus.deleted:
            ch.changed_lines, ch.deletion_anchors = parse_patch_lines(patch)
    n_commits = repo.commit_count(merge_base_sha, head_sha)
    n_merges, merges = repo.merge_commits(merge_base_sha, head_sha)
    log.info("diff: %d changed files in %d commit(s) since the merge base", len(changes), n_commits)
    return DiffResult(
        base_sha, head_sha, merge_base_sha, repo.tree_sha(head_sha), changes,
        commit_count=n_commits, merge_count=n_merges, merge_samples=merges,
    )
