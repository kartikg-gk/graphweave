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


def _import_otel(args: list[str]) -> int:
    """``graphweave import-otel <spans.json> [--out DIR]``: OpenTelemetry spans to traces."""
    import argparse
    import re
    from pathlib import Path

    from .tracing import save, score_overlaps
    from .tracing.otel import traces_from_file
    from .tracing.report import write_report

    parser = argparse.ArgumentParser(prog="graphweave import-otel",
                                     description="Turn exported OpenTelemetry spans into graphweave traces.")
    parser.add_argument("spans", type=Path, help="OTLP JSON or SDK span JSON, one document or one per line")
    parser.add_argument("--out", type=Path, default=Path("graphweave_out"), help="folder for the traces")
    options = parser.parse_args(args)
    try:
        traces = traces_from_file(options.spans)
    except FileNotFoundError:
        print(f"no such file: {options.spans}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"could not read {options.spans}: {exc}", file=sys.stderr)
        return 1
    if not traces:
        print("No traces with a retrieval step were found in that file.", file=sys.stderr)
        return 1
    options.out.mkdir(parents=True, exist_ok=True)
    for index, trace in enumerate(traces, start=1):
        slug = "-".join(re.findall(r"[a-z0-9]+", trace.query.lower())[:6]) or "trace"
        path = save(score_overlaps(trace), options.out / f"otel-{index:02d}-{slug}.json")
        write_report(trace, path)
        print(f"Wrote {path} ({len(trace.items)} retrieved)")
    print(f"View with: graphweave {options.out}")
    return 0


def _verify(args: list[str]) -> int:
    """``graphweave verify <trace.json>``: leave-one-out check of the used/unused verdict."""
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="graphweave verify",
        description="Re-ask the question with each retrieved item removed, and compare what the "
                    "answer needed with the word-overlap verdict. Makes (items + 2) model calls.")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", default=os.environ.get("GRAPHWEAVE_VERIFY_MODEL", "google/gemini-2.5-flash-lite"))
    parser.add_argument("--base-url", default=os.environ.get("GRAPHWEAVE_VERIFY_BASE_URL"))
    options = parser.parse_args(args)

    key = (os.environ.get("GRAPHWEAVE_VERIFY_KEY") or os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    if not key:
        print("graphweave verify needs a model key: set OPENROUTER_API_KEY, OPENAI_API_KEY "
              "or GRAPHWEAVE_VERIFY_KEY.", file=sys.stderr)
        return 2
    base_url = options.base_url
    if base_url is None and key != os.environ.get("OPENAI_API_KEY"):
        # A graphweave or OpenRouter key: OpenRouter, unless --base-url says otherwise.
        base_url = "https://openrouter.ai/api/v1"
    try:
        from .verify import openai_ask, render, verify
    except ImportError:
        print('graphweave verify needs the openai package: pip install "graphweave[verify]"', file=sys.stderr)
        return 1
    from .tracing.store import load

    try:
        trace = load(options.trace)
    except FileNotFoundError:
        print(f"no such trace file: {options.trace}", file=sys.stderr)
        return 1
    if not trace.items:
        print("That trace has no retrieved items to check.", file=sys.stderr)
        return 1
    try:
        result = verify(trace, openai_ask(options.model, api_key=key, base_url=base_url))
    except Exception as exc:  # noqa: BLE001 - model or network failure, reported plainly
        print(f"graphweave verify: the model call failed: {exc}", file=sys.stderr)
        return 1
    destination = options.trace.with_name(f"{options.trace.stem}.verify.json")
    destination.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    print(render(result))
    print()
    print(f"Wrote {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "install":
        from .skill.install import main as install_main

        return install_main(args[1:])
    if args and args[0] == "report":
        return _report(args[1:])
    if args and args[0] == "import-otel":
        return _import_otel(args[1:])
    if args and args[0] == "verify":
        return _verify(args[1:])
    if args and args[0] == "github-trace":
        from .local_github import main as github_main

        return github_main(args[1:])
    if args and args[0] in {"ingest", "view", "trace"}:
        from .cli import main as backend_main

        return backend_main(args)
    from .tracing.web import main as viewer_main

    return viewer_main(args)
