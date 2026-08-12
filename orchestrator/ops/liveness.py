"""Is a run that *says* `running` actually running?

`runner` writes `done` on the normal path and `paused` on the budget/limit path,
but nothing writes a terminal status when the process dies without unwinding —
SIGKILL, a hard Ctrl-C, a laptop sleeping through a run, an OOM. The row keeps
its creation-time default of `running` forever, and every consumer that trusts
that column (`latest_run`, the panel's run pill, `status`) reports a run that has
been dead for days as live.

There is no pid to check: runs are started from a terminal as often as from the
panel, and the pid would be stale across a reboot anyway. What *is* recorded is
the run's own footprint — every LLM call writes a `usage` row and every stage
transition writes an `events` row — so liveness is measured as "has this run
touched the store recently".

The threshold is generous on purpose. A planner call against a real repo takes
minutes (measured: 385-425k tokens of agentic Read/Grep), and a run that is
merely slow must never be declared dead, because the remedy (`reconcile`) writes
a terminal status the real process would then contradict. Fifteen minutes is
several times the slowest single call observed and still catches the case this
exists for, where the gap is measured in days.
"""

from __future__ import annotations

import time

#: A `running` run silent for longer than this is presumed abandoned.
STALE_AFTER_S = 15 * 60

#: Statuses that claim the run is still going.
LIVE_STATUSES = ("running",)


def is_stale(status: str, last_activity_at: float | None, *,
             now: float | None = None, after_s: float = STALE_AFTER_S) -> bool:
    """True when `status` claims the run is live but nothing has happened since.

    `paused` is never stale: it is a terminal-for-now status a human or a resume
    is expected to act on, and the whole point of the pause path is that it was
    written deliberately.
    """
    if status not in LIVE_STATUSES:
        return False
    if last_activity_at is None:
        return True
    return (now if now is not None else time.time()) - last_activity_at > after_s
