"""Preset resolution — one named backend binding, flattened over an entry.

A preset is a COMPLETE, named backend binding (`presets:` in config/default.yaml):
provider + model + effort + the price shape for that model. `worker_models.<key>`
and `roles.<role>` may point at one with `preset: <name>` instead of repeating
those five keys, which is what makes "move this worker from a metered API to a
CLI riding a subscription" a one-line edit rather than a five-line one.

Resolution rule, and the only one: **the preset fills in, the entry wins.** An
entry that names no preset resolves to itself, unchanged — that is the whole
back-compat story, and every profile written before presets existed keeps its
exact previous behavior.

This lives in `core/` rather than in `core/context.py` because two unrelated
callers need it and one of them is below context in the import graph:
`RunContext` (dispatch: which provider/model/effort a role or candidate gets)
and `ops/pricing.py` (accounting: what a model costs). `context` imports
`ops.budget`, so a resolver inside `context` would make `ops.pricing -> context`
a cycle. A leaf module with no orchestrator imports at all avoids the question.

Everything here tolerates BOTH a `Section` (the normal `Config` view) and a
plain dict, because provider unit tests build their `cfg` as a bare `Section`
over a literal dict and the price table has to work there too.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("orchestrator.presets")


def as_dict(value: Any) -> dict:
    """A plain dict view of a Section / dict / anything else.

    `Section.as_dict()` hands back the underlying dict rather than a copy, so
    nested values (a preset's `long_context:` block) stay plain dicts too — which
    is what `ops.pricing` wants to read without knowing about Section at all.

    A scalar maps to `{}` rather than raising, because these sections are not
    uniformly dicts of entries: `roles:` holds `smart_provider: claude_cli`
    (a string) alongside the role bindings, and every caller here wants "not an
    entry, skip it" rather than a TypeError from the middle of a price table.
    """
    if isinstance(value, dict):
        return value
    getter = getattr(value, "as_dict", None)
    if callable(getter):
        inner = getter()
        return inner if isinstance(inner, dict) else {}
    return {}


def section(cfg: Any, name: str) -> dict:
    """Top-level config section as a plain dict ({} when absent)."""
    getter = getattr(cfg, "get", None)
    return as_dict(getter(name) if callable(getter) else None)


def resolve_entry(cfg: Any, entry: Any) -> dict:
    """Flatten `entry` over the preset it names. Returns a NEW plain dict.

    Precedence is preset-then-entry, key by key: a preset supplies
    provider/model/effort/prices, and any of those written explicitly on the
    entry still wins. That ordering is deliberate — the preset is the shared
    default and the entry is the local exception, so overriding one field of a
    preset must not force you to restate the other four.

    An unresolvable preset name is a WARNING here and an ERROR in
    `core.validate`: dispatch must never crash on a config typo three tasks into
    a run, and validation is where that typo gets caught before the first call.
    """
    data = as_dict(entry)
    name = data.get("preset")
    if not name:
        return dict(data)
    preset = section(cfg, "presets").get(name)
    if preset is None:
        log.warning("preset %r is not defined in presets: — using the entry as "
                    "written (run `validate_config` to catch this earlier)", name)
        return dict(data)
    merged = dict(as_dict(preset))
    # The entry wins, but only where it actually SAYS something: a key present
    # with a null value (`effort: null`, the shape every shipped role entry uses)
    # means "unset", not "override the preset with nothing".
    for key, value in data.items():
        if value is not None or key not in merged:
            merged[key] = value
    return merged
