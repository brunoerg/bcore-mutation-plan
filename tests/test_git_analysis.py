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
