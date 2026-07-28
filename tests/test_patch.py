import pytest

from orchestrator.core.errors import PatchError
from orchestrator.ops.patch import apply_response, parse_worker_response


def test_parse_full_file():
    text = 'plan line\n<file path="src/a.ts">\nconst x = 1;\n</file>'
    parsed = parse_worker_response(text)
    assert parsed.files == {"src/a.ts": "const x = 1;\n"}
    assert parsed.plan == "plan line"


def test_parse_search_replace():
    text = (
        '<edit path="src/b.ts">\n'
        "<<<<<<< SEARCH\nconst old = 1;\n=======\nconst new_ = 2;\n>>>>>>> REPLACE\n"
        "</edit>"
    )
    parsed = parse_worker_response(text)
    assert parsed.edits["src/b.ts"] == [("const old = 1;", "const new_ = 2;")]


def test_parse_need_files():
    parsed = parse_worker_response("<need_files>\nsrc/x.ts\nsrc/y.ts\n</need_files>")
    assert parsed.need_files == ["src/x.ts", "src/y.ts"]
    assert parsed.is_empty is False
    assert parsed.touched_paths == set()


def test_apply_full_and_edit(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/b.ts").write_text("const old = 1;\nexport {};\n")
    parsed = parse_worker_response(
        '<file path="src/a.ts">\nnew file\n</file>\n'
        '<edit path="src/b.ts">\n'
        "<<<<<<< SEARCH\nconst old = 1;\n=======\nconst brand = 2;\n>>>>>>> REPLACE\n"
        "</edit>"
    )
    written = apply_response(tmp_path, parsed, ["src/a.ts", "src/b.ts"])
    assert sorted(written) == ["src/a.ts", "src/b.ts"]
    assert (tmp_path / "src/b.ts").read_text() == "const brand = 2;\nexport {};\n"


def test_apply_rejects_path_outside_allowlist(tmp_path):
    parsed = parse_worker_response('<file path="src/evil.ts">\nx\n</file>')
    with pytest.raises(PatchError, match="allowlist"):
        apply_response(tmp_path, parsed, ["src/ok.ts"])


def test_apply_rejects_traversal(tmp_path):
    parsed = parse_worker_response('<file path="../evil.ts">\nx\n</file>')
    with pytest.raises(PatchError, match="Unsafe"):
        apply_response(tmp_path, parsed, None)


def test_apply_rejects_ambiguous_search(tmp_path):
    (tmp_path / "c.ts").write_text("dup\ndup\n")
    parsed = parse_worker_response(
        '<edit path="c.ts">\n<<<<<<< SEARCH\ndup\n=======\nx\n>>>>>>> REPLACE\n</edit>')
    with pytest.raises(PatchError, match="not unique"):
        apply_response(tmp_path, parsed, None)


def test_apply_rejects_missing_search(tmp_path):
    (tmp_path / "d.ts").write_text("hello\n")
    parsed = parse_worker_response(
        '<edit path="d.ts">\n<<<<<<< SEARCH\nabsent\n=======\nx\n>>>>>>> REPLACE\n</edit>')
    with pytest.raises(PatchError, match="not found"):
        apply_response(tmp_path, parsed, None)


def test_apply_rejects_symlink_escape(tmp_path):
    """A symlinked directory inside the worktree defeats the lexical relative /
    '..' checks: `link/pwned.txt` is a perfectly clean relative path that
    resolves outside the tree."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    parsed = parse_worker_response('<file path="link/pwned.txt">\nx\n</file>')
    with pytest.raises(PatchError, match="escapes the worktree"):
        apply_response(root, parsed, ["link/pwned.txt"])   # even when allowlisted
    assert not (outside / "pwned.txt").exists()            # nothing written


def test_symlink_escape_is_rejected_before_any_write(tmp_path):
    """The escaping path must not take a legitimate sibling write with it into a
    half-applied state — nothing is written when any path is rejected."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "wt"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    parsed = parse_worker_response(
        '<file path="ok.txt">\nfine\n</file>\n'
        '<file path="link/pwned.txt">\nx\n</file>')
    with pytest.raises(PatchError):
        apply_response(root, parsed, ["ok.txt", "link/pwned.txt"])
    assert not (outside / "pwned.txt").exists()
    assert not (root / "ok.txt").exists()


def test_symlinked_file_inside_root_still_allowed(tmp_path):
    """A symlink that stays inside the worktree is not an escape — only the
    resolved location matters, so ordinary in-tree links keep working."""
    root = tmp_path / "wt"
    (root / "real").mkdir(parents=True)
    (root / "alias").symlink_to(root / "real", target_is_directory=True)
    parsed = parse_worker_response('<file path="alias/a.txt">\nhi\n</file>')
    written = apply_response(root, parsed, ["alias/a.txt"])
    assert written == ["real/a.txt"]                       # reported resolved
    assert (root / "real/a.txt").read_text() == "hi\n"
