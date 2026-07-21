"""Phase 2: ops/retrieval — read-only, path-safe, bounded executors."""

from pathlib import Path

from orchestrator.ops import retrieval


def test_resolve_read_path_relative_ok(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    p = retrieval.resolve_read_path(tmp_path, wt, [], "src/a.ts")
    assert p == wt / "src/a.ts"


def test_resolve_read_path_refuses_escape(tmp_path):
    wt = tmp_path / "wt"
    assert retrieval.resolve_read_path(tmp_path, wt, [], "../secret") is None
    assert retrieval.resolve_read_path(tmp_path, wt, [], "a/../../secret") is None
    assert retrieval.resolve_read_path(tmp_path, wt, [], "/etc/passwd") is None


def test_resolve_read_path_untracked_prefix_uses_repo(tmp_path):
    repo = tmp_path / "repo"
    wt = tmp_path / "wt"
    p = retrieval.resolve_read_path(repo, wt, ["docs/"], "docs/DESIGN.md")
    assert p == repo / "docs/DESIGN.md"      # docs come from primary checkout
    p2 = retrieval.resolve_read_path(repo, wt, ["docs/"], "src/x.ts")
    assert p2 == wt / "src/x.ts"             # code comes from the worktree


def test_read_file_states(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    text, status = retrieval.read_file(f, 1000)
    assert status == "ok" and text == "hello"
    assert retrieval.read_file(tmp_path / "nope", 1000) == (None, "missing")
    text, status = retrieval.read_file(f, 2)   # 5 bytes > cap 2
    assert status == "too_large" and text is None


def test_ls_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x")
    names = retrieval.ls_dir(tmp_path)
    assert "a.txt" in names and "sub/" in names
    assert retrieval.ls_dir(tmp_path / "a.txt") is None   # not a dir


def test_grep_finds_and_caps(tmp_path):
    (tmp_path / "a.txt").write_text("foo bar\nfoo baz\n")
    (tmp_path / "b.txt").write_text("nothing here\n")
    matches, status = retrieval.grep(tmp_path, "foo", max_matches=40, snippet_lines=4)
    assert status == "ok"
    assert any("a.txt" in m and "foo" in m for m in matches)
    # cap enforced
    capped, _ = retrieval.grep(tmp_path, "foo", max_matches=1, snippet_lines=4)
    assert len(capped) == 1


def test_grep_pattern_length_capped(tmp_path):
    long = "a" * (retrieval.MAX_PATTERN_LEN + 1)
    matches, status = retrieval.grep(tmp_path, long, max_matches=40, snippet_lines=4)
    assert status == "pattern_too_long" and matches == []


def test_grep_no_matches_is_empty(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    matches, status = retrieval.grep(tmp_path, "zzz", max_matches=40, snippet_lines=4)
    assert status == "empty" and matches == []
