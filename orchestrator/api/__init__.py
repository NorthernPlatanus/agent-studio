"""HTTP control panel layer over the orchestrator.

A read layer plus (from phase 3) a subprocess supervisor — never a second
implementation of the pipeline. The graph runs only through the CLI, so the
panel can never diverge from what a terminal run does.
"""

from __future__ import annotations
