from types import SimpleNamespace

from orchestrator.nodes.worker import _read_task_file


def make_ctx(repo):
    cfg = SimpleNamespace(
        project=SimpleNamespace(untracked_doc_prefixes=[]),
        repo_path=lambda: repo,
    )
    return SimpleNamespace(cfg=cfg)


def test_read_task_file_reads_relative(tmp_path):
    (tmp_path / "ok.txt").write_text("fine")
    assert _read_task_file(make_ctx(tmp_path), tmp_path, "ok.txt") == "fine"


def test_read_task_file_rejects_escape(tmp_path):
    """need_files comes from LLM output — requests must never leave the tree."""
    ctx = make_ctx(tmp_path)
    assert _read_task_file(ctx, tmp_path, "../secret.txt") is None
    assert _read_task_file(ctx, tmp_path, "a/../../secret.txt") is None
    assert _read_task_file(ctx, tmp_path, "/etc/passwd") is None


def test_read_task_file_missing_is_none(tmp_path):
    assert _read_task_file(make_ctx(tmp_path), tmp_path, "absent.txt") is None
