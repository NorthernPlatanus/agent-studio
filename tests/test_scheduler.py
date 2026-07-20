from orchestrator.engine.scheduler import deps_satisfied, next_batch


def T(id, status="ready", deps=None, writes=None, milestone="M1"):
    return {"id": id, "status": status, "deps": deps or [],
            "files_write": writes if writes is not None else [f"src/{id}.ts"],
            "milestone": milestone}


def test_deps_block():
    a, b = T("T-1", status="running"), T("T-2", deps=["T-1"])
    assert not deps_satisfied(b, {"T-1": a})
    a["status"] = "done"
    assert deps_satisfied(b, {"T-1": a})


def test_unknown_dep_does_not_block():
    b = T("T-2", deps=["EXTERNAL-9"])
    assert deps_satisfied(b, {})


def test_batch_respects_write_conflicts():
    tasks = [T("T-1", writes=["src/a.ts"]),
             T("T-2", writes=["src/a.ts", "src/b.ts"]),
             T("T-3", writes=["src/c.ts"])]
    batch = next_batch(tasks, max_parallel=3)
    assert [t["id"] for t in batch] == ["T-1", "T-3"]


def test_batch_width_cap():
    tasks = [T(f"T-{i}") for i in range(5)]
    assert len(next_batch(tasks, max_parallel=2)) == 2


def test_only_ready_scheduled():
    tasks = [T("T-1", status="done"), T("T-2", status="needs_plan"),
             T("T-3", status="human_only"), T("T-4")]
    assert [t["id"] for t in next_batch(tasks, 4)] == ["T-4"]


def test_no_writes_runs_alone():
    tasks = [T("T-1", writes=[]), T("T-2")]
    batch = next_batch(tasks, 2)
    assert [t["id"] for t in batch] == ["T-1"]


def test_dep_chain_orders_waves():
    tasks = [T("T-1"), T("T-2", deps=["T-1"])]
    assert [t["id"] for t in next_batch(tasks, 2)] == ["T-1"]
    tasks[0]["status"] = "done"
    assert [t["id"] for t in next_batch(tasks, 2)] == ["T-2"]
