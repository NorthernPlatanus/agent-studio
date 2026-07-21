"""Phase 4: structural project-map — deterministic index, gitignore-respecting
file list, and digest-regenerates-only-on-change."""

import subprocess
from pathlib import Path

from orchestrator.core.config import Config
from orchestrator.ops import projectmap


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.ts").write_text(
        "export function foo() {}\nexport class Bar {}\nconst hidden = 1;\n")
    (tmp_path / "src" / "b.py").write_text("def baz():\n    pass\nclass Qux:\n    pass\n")
    (tmp_path / "README.md").write_text("# readme\n")
    return tmp_path


def test_symbol_index_deterministic(tmp_path):
    root = _fixture(tmp_path)
    files = projectmap.list_files(root)
    idx1 = projectmap.symbol_index(root, files)
    idx2 = projectmap.symbol_index(root, files)
    assert idx1 == idx2
    assert idx1["src/a.ts"] == ["Bar", "foo"]
    assert idx1["src/b.py"] == ["Qux", "baz"]
    assert "README.md" not in idx1        # non-source skipped


def test_list_files_respects_gitignore(tmp_path):
    root = _fixture(tmp_path)
    (root / ".gitignore").write_text("ignored/\n")
    (root / "ignored").mkdir()
    (root / "ignored" / "secret.ts").write_text("export function nope() {}\n")
    subprocess.run(["git", "init", "-q"], cwd=root)
    subprocess.run(["git", "add", "-A"], cwd=root)
    files = projectmap.list_files(root)
    assert "src/a.ts" in files
    assert not any("ignored/" in f for f in files)   # gitignored path excluded


def test_render_and_hash_change(tmp_path):
    root = _fixture(tmp_path)
    files = projectmap.list_files(root)
    idx = projectmap.symbol_index(root, files)
    md = projectmap.render_map(files, idx)
    assert "## Tree" in md and "src/a.ts" in md and "foo" in md

    h1 = projectmap.structure_hash(idx)
    # adding a symbol changes the hash; reformatting a comment does not
    (root / "src" / "a.ts").write_text(
        "export function foo() {}\nexport class Bar {}\nexport function NEW() {}\n")
    idx2 = projectmap.symbol_index(root, projectmap.list_files(root))
    assert projectmap.structure_hash(idx2) != h1


def _cfg(tmp_path):
    return Config({"projectmap": {"enabled": True, "path": "projectmap.md",
                                  "digest": True}}, "proj", tmp_path)


def test_regenerate_writes_map_and_digest_gating(tmp_path):
    src = _fixture(tmp_path / "repo")
    cfg = _cfg(tmp_path)
    out = tmp_path / "projects" / "proj" / "projectmap.md"

    calls = []
    def digest_fn(files, symbols):
        calls.append(1)
        return "DIGEST TEXT"

    h1 = projectmap.regenerate(cfg, src, out, digest_fn=digest_fn)
    assert out.exists() and "DIGEST TEXT" in out.read_text()
    assert len(calls) == 1                       # digest ran (hash was new)

    # unchanged structure -> digest skipped on the next run
    projectmap.regenerate(cfg, src, out, digest_fn=digest_fn)
    assert len(calls) == 1                        # not called again

    # structural change -> digest runs again
    (src / "src" / "c.ts").write_text("export function added() {}\n")
    projectmap.regenerate(cfg, src, out, digest_fn=digest_fn)
    assert len(calls) == 2


def test_map_path(tmp_path):
    cfg = _cfg(tmp_path)
    assert projectmap.map_path(cfg) == tmp_path / "projects" / "proj" / "projectmap.md"
