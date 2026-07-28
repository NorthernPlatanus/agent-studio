"""Item 8: the candidates reducer must not store one full transcript per attempt.

`operator.add` kept every attempt's complete chat — including the frozen prefix
with every files_read file inlined — so the checkpoint grew ~quadratically in
attempts and `resume` had to deserialize all of it.
"""

import json

from orchestrator.core.state import add_candidates, latest_candidates

BIG = "x" * 20000        # stands in for an inlined files_read block


def C(cid, attempt, status="gate_failed", messages=None):
    return {"cand_id": cid, "attempt": attempt, "status": status,
            "gate_log": f"log {attempt}", "diff": "", "error": "",
            "messages": messages if messages is not None
            else [{"role": "user", "content": BIG}]}


def test_only_the_newest_attempt_keeps_its_transcript():
    state = add_candidates([], [C("w", 1)])
    state = add_candidates(state, [C("w", 2)])
    state = add_candidates(state, [C("w", 3)])
    assert [c["attempt"] for c in state] == [1, 2, 3]      # audit trail intact
    assert [len(c["messages"]) for c in state] == [0, 0, 1]


def test_superseded_attempts_keep_their_audit_fields():
    state = add_candidates([C("w", 1)], [C("w", 2)])
    assert state[0]["status"] == "gate_failed"
    assert state[0]["gate_log"] == "log 1"                 # still explains attempt 1


def test_each_candidate_keeps_its_own_newest():
    state = add_candidates([], [C("a", 1), C("b", 1)])
    state = add_candidates(state, [C("a", 2)])
    by_key = {(c["cand_id"], c["attempt"]): c for c in state}
    assert len(by_key[("a", 2)]["messages"]) == 1
    assert len(by_key[("b", 1)]["messages"]) == 1          # b never re-ran
    assert len(by_key[("a", 1)]["messages"]) == 0


def test_same_attempt_supersession_matches_latest_candidates():
    """A visual_failed copy is appended at the SAME attempt; latest_candidates
    picks the later one, so the reducer must leave the transcript there."""
    state = add_candidates([], [C("w", 1, "gate_passed")])
    state = add_candidates(state, [C("w", 1, "visual_failed")])
    assert latest_candidates({"candidates": state})["w"]["status"] == "visual_failed"
    assert [(c["status"], len(c["messages"])) for c in state] == \
        [("gate_passed", 0), ("visual_failed", 1)]


def test_warm_chain_still_reachable_after_reduction():
    """fan_out warms a retry from latest_candidates — that path must still find
    the messages, or every retry would restart cold."""
    state = add_candidates([C("w", 1)], [C("w", 2)])
    warm = latest_candidates({"candidates": state})["w"]["messages"]
    assert warm and warm[0]["content"] == BIG


def test_checkpoint_size_stays_flat_across_attempts():
    state = add_candidates([], [C("w", 1)])
    one = len(json.dumps(state))
    for attempt in range(2, 5):
        state = add_candidates(state, [C("w", attempt)])
    four = len(json.dumps(state))
    # Four attempts must not cost ~4x one attempt: only the newest transcript is
    # stored, so growth is the small per-attempt audit record, not the prefix.
    assert four < one * 1.5


def test_reducer_is_idempotent():
    state = add_candidates([C("w", 1)], [C("w", 2)])
    assert add_candidates(state, []) == state


def test_handles_empty_and_none():
    assert add_candidates(None, None) == []
    assert add_candidates(None, [C("w", 1)])[0]["cand_id"] == "w"


async def test_reducer_behaves_inside_a_real_langgraph_run():
    """The pure-function tests above don't prove LangGraph applies the reducer as
    expected across parallel Send branches — this drives the actual shape: two
    candidates fanned out per attempt, three attempts."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send
    from orchestrator.core.state import TaskState

    b = StateGraph(TaskState)

    async def dispatch(s):
        return {"attempt": s.get("attempt", 0) + 1}

    def fan(s):
        return [Send("work", {"cand_id": c, "attempt": s["attempt"]})
                for c in ("a", "b")]

    async def work(p):
        return {"candidates": [{"cand_id": p["cand_id"], "attempt": p["attempt"],
                                "status": "gate_failed",
                                "messages": [{"role": "user", "content": BIG}]}]}

    b.add_node("dispatch", dispatch)
    b.add_node("work", work)
    b.add_node("collect", lambda s: {})
    b.add_edge(START, "dispatch")
    b.add_conditional_edges("dispatch", fan, ["work"])
    b.add_edge("work", "collect")
    b.add_conditional_edges("collect",
                            lambda s: "dispatch" if s["attempt"] < 3 else END,
                            ["dispatch", END])

    out = await b.compile().ainvoke({"attempt": 0})
    cands = out["candidates"]
    assert len(cands) == 6                               # every attempt recorded
    assert sum(len(c["messages"]) for c in cands) == 2   # one transcript each
    latest = latest_candidates(out)
    assert {cid: c["attempt"] for cid, c in latest.items()} == {"a": 3, "b": 3}
    assert latest["a"]["messages"], "the warm chain must survive the reduction"
