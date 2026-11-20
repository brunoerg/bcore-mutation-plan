import pytest

from conftest import BASE_FILES, git, replace, write_files
from mutation_planner.git_analysis import GitError, GitRepo, analyze_diff, parse_numstat, parse_patch_lines, parse_raw_diff
from mutation_planner.schemas import ChangeStatus


def by_path(diff):
    return {c.path: c for c in diff.changes}


def test_statuses_rename_delete_add_binary(repo):
    head = repo.change(
        files={
            "src/policy/fees.cpp": replace(BASE_FILES["src/policy/fees.cpp"], "return fee < 3;", "return fee < 5;"),
            "src/new_module.cpp": "int NewThing() { return 1; }\n",
            "share/pixmaps/icon.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00binary\x00data",
        },
        delete=["src/node/mempool_persist.cpp"],
        rename=[("src/util/strutil.h", "src/common/strutil.h")],
    )
    diff = analyze_diff(GitRepo(repo.path), repo.base, head)
    ch = by_path(diff)
    assert ch["src/policy/fees.cpp"].status == ChangeStatus.modified
    assert ch["src/new_module.cpp"].status == ChangeStatus.added
    assert ch["src/node/mempool_persist.cpp"].status == ChangeStatus.deleted
    assert ch["src/node/mempool_persist.cpp"].head_blob_sha is None
    renamed = ch["src/common/strutil.h"]
    assert renamed.status == ChangeStatus.renamed and renamed.old_path == "src/util/strutil.h"
    assert ch["share/pixmaps/icon.png"].binary
    fees = ch["src/policy/fees.cpp"]
    assert [(r.start, r.end) for r in fees.changed_lines] == [(15, 15)]
    assert fees.additions == 1 and fees.deletions == 1


def test_merge_base_is_used_not_base_tip(repo):
    head = repo.change(files={"src/policy/fees.cpp": BASE_FILES["src/policy/fees.cpp"] + "// pr change\n"})
    # The base branch moves on after the PR branched off.
    write_files(repo.path, {"src/wallet/spend.cpp": BASE_FILES["src/wallet/spend.cpp"] + "// main moved\n"})
    git(repo.path, "commit", "-qam", "main moves")
    new_base = git(repo.path, "rev-parse", "HEAD")
    diff = analyze_diff(GitRepo(repo.path), new_base, head)
    assert diff.base_sha == new_base
    assert diff.merge_base_sha == repo.base
    assert set(by_path(diff)) == {"src/policy/fees.cpp"}


def test_working_tree_is_never_used(repo):
    head = repo.change(files={"src/policy/fees.cpp": BASE_FILES["src/policy/fees.cpp"] + "// committed\n"})
    # Dirty, uncommitted working-tree edits must not leak into the analysis.
    write_files(repo.path, {"src/policy/fees.cpp": "garbage\n", "src/untracked.cpp": "int x;\n"})
    diff = analyze_diff(GitRepo(repo.path), repo.base, head)
    assert set(by_path(diff)) == {"src/policy/fees.cpp"}
    text = GitRepo(repo.path).read_text(head, "src/policy/fees.cpp")
    assert text.endswith("// committed\n")


def test_rejects_suspicious_revisions(repo):
    r = GitRepo(repo.path)
    with pytest.raises(GitError):
        r.resolve_commit("--output=/tmp/x")
    with pytest.raises(GitError):
        r.resolve_commit("does-not-exist")


def test_parse_patch_lines_additions_and_pure_deletions():
    patch = (
        "@@ -10,4 +10,3 @@\n"
        " ctx\n"
        "-removed only\n"
        " ctx2\n"
        "-old\n"
        "+new\n"
        "@@ -30,0 +29,2 @@\n"
        "+added1\n"
        "+added2\n"
    )
    ranges, anchors = parse_patch_lines(patch)
    assert [(r.start, r.end) for r in ranges] == [(12, 12), (29, 30)]
    assert anchors == [10]


def test_parse_raw_and_numstat_with_rename():
    raw = b":100644 100644 aaaa bbbb R087\0old/a.cpp\0new/a.cpp\0:100644 000000 cccc " + b"0" * 40 + b" D\0gone.cpp\0"
    changes = parse_raw_diff(raw.replace(b"aaaa", b"a" * 40).replace(b"bbbb", b"b" * 40).replace(b"cccc", b"c" * 40))
    assert changes[0].status == ChangeStatus.renamed and changes[0].old_path == "old/a.cpp"
    assert changes[1].status == ChangeStatus.deleted and changes[1].head_blob_sha is None
    ns = parse_numstat(b"3\t1\t\0old/a.cpp\0new/a.cpp\0-\t-\timg.png\0")
    assert ns == {"new/a.cpp": (3, 1), "img.png": (None, None)}


def _stale_base_repo(repo):
    """Upstream merges a PR into main, the PR branch is rebased onto the new main, and
    --base is the old main: the upstream work ends up in merge-base..head."""
    old_main = repo.base
    git(repo.path, "checkout", "-q", "-b", "upstream-work", old_main)
    write_files(repo.path, {"src/wallet/spend.cpp": BASE_FILES["src/wallet/spend.cpp"] + "// upstream\n"})
    git(repo.path, "commit", "-qam", "upstream work")
    git(repo.path, "checkout", "-q", "main")
    git(repo.path, "merge", "-q", "--no-ff", "-m", "Merge bitcoin/bitcoin#1: upstream work", "upstream-work")
    git(repo.path, "checkout", "-q", "-B", "pr", "main")
    write_files(repo.path, {"src/policy/fees.cpp": BASE_FILES["src/policy/fees.cpp"] + "// pr\n"})
    git(repo.path, "commit", "-qam", "pr change")
    head = git(repo.path, "rev-parse", "HEAD")
    git(repo.path, "checkout", "-q", "main")
    return old_main, head


def test_stale_base_is_reported(repo, config):
    from mutation_planner.analysis import build_context
    from mutation_planner.schemas import PRMetadata

    old_main, head = _stale_base_repo(repo)
    diff = analyze_diff(GitRepo(repo.path), old_main, head)
    assert diff.merge_count == 1 and diff.commit_count == 3
    assert diff.merge_samples[0][1] == "Merge bitcoin/bitcoin#1: upstream work"
    # the upstream file shows up as a "PR change": exactly the symptom being detected
    assert "src/wallet/spend.cpp" in by_path(diff)
    ctx = build_context(repo.path, old_main, head, PRMetadata(), config)
    assert any("looks stale" in w and "1 merge commit" in w for w in ctx.warnings)

    # With the fresh base, the range is the PR's single commit and nothing is reported.
    fresh = git(repo.path, "rev-parse", "main")
    ctx = build_context(repo.path, fresh, head, PRMetadata(), config)
    assert not any("looks stale" in w for w in ctx.warnings)
    assert {c.path for c in ctx.changed_files} == {"src/policy/fees.cpp"}


def test_long_history_is_reported(repo, config):
    from mutation_planner.analysis import build_context
    from mutation_planner.schemas import PRMetadata

    git(repo.path, "checkout", "-q", "-B", "pr", repo.base)
    for i in range(3):
        write_files(repo.path, {"src/policy/fees.cpp": BASE_FILES["src/policy/fees.cpp"] + f"// {i}\n"})
        git(repo.path, "commit", "-qam", f"c{i}")
    head = git(repo.path, "rev-parse", "HEAD")
    git(repo.path, "checkout", "-q", "main")
    assert not any("stale" in w for w in build_context(repo.path, repo.base, head, PRMetadata(), config).warnings)
    config.conventions.stale_base_max_commits = 2
    ctx = build_context(repo.path, repo.base, head, PRMetadata(), config)
    assert any("3 commits (more than 2)" in w for w in ctx.warnings)
