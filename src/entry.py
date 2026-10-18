"""Lightweight entry point for the local viewer and optional backend CLI."""

from __future__ import annotations

import sys


def _report(args: list[str]) -> int:
    """``graphweave report <trace.json>``: write the Markdown report beside a trace."""
    if len(args) != 1:
        print("usage: graphweave report <trace.json>", file=sys.stderr)
        return 2
    from .tracing.report import report_file

    try:
        print(f"Wrote {report_file(args[0])}")
    except FileNotFoundError:
        print(f"no such trace file: {args[0]}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"could not read {args[0]}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "install":
        from .skill.install import main as install_main

        return install_main(args[1:])
    if args and args[0] == "report":
        return _report(args[1:])
    if args and args[0] == "github-trace":
        from .local_github import main as github_main

        return github_main(args[1:])
    if args and args[0] in {"ingest", "view", "trace"}:
        from .cli import main as backend_main

        return backend_main(args)
    from .tracing.web import main as viewer_main

    return viewer_main(args)
