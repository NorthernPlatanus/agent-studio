"""`orchestrator serve` argument surface.

No server is started here: the point is that the defaults are loopback + 8787 and
that the flag names match what `START-HERE.md` documents, since the runbook and
`studio-verify`'s smoke pass both type them literally.
"""

from __future__ import annotations

from orchestrator.cli import build_parser


def test_serve_defaults_to_loopback_and_8787():
    args = build_parser().parse_args(["serve"])
    assert args.command == "serve"
    assert (args.host, args.port, args.reload) == ("127.0.0.1", 8787, False)


def test_serve_accepts_host_port_reload_and_the_common_flags():
    args = build_parser().parse_args(
        ["serve", "--host", "127.0.0.1", "--port", "8788", "--reload",
         "--project", "example", "-v"])
    assert (args.host, args.port, args.reload) == ("127.0.0.1", 8788, True)
    assert args.project == "example" and args.verbose
