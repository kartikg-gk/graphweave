"""``graphweave-mcp``: the local tools as an MCP server over stdio.

An assistant that speaks MCP can call graphweave directly instead of running
shell commands. Three tools:

* ``trace_repo``    — ask a question about a public repository; returns the report.
* ``explain_trace`` — the report for a trace file already on disk.
* ``list_traces``   — the traces in a folder, with the question each one asked.

Everything runs on this machine; the only network calls go to GitHub's API.
Install with ``pip install "graphweave[mcp]"`` and register the command with
the assistant, for example ``claude mcp add graphweave -- graphweave-mcp``.

Works with both major versions of the ``mcp`` package: 1.x registers handlers
with decorators, 2.x passes them to the ``Server`` constructor.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from .local_github import DEFAULT_CACHE, REPO_PATTERN, run_trace
from .tracing.report import render_markdown, report_file
from .tracing.store import load

SERVER_NAME = "graphweave"
DEFAULT_FOLDER = Path("graphweave_out")


class ToolInputError(ValueError):
    """Arguments the tool cannot act on; reported to the assistant, not raised."""


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "-".join(words) or "question"


def _count(arguments: dict, name: str, default: int, ceiling: int = 100) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or not 0 <= value <= ceiling:
        raise ToolInputError(f"{name} must be a whole number from 0 to {ceiling}")
    return value


def trace_repo(arguments: dict) -> str:
    repository = str(arguments.get("repo", "")).strip()
    question = str(arguments.get("question", "")).strip()
    if not REPO_PATTERN.fullmatch(repository) or any(part in {".", ".."} for part in repository.split("/")):
        raise ToolInputError("repo must be owner/name, for example psf/requests")
    if not question:
        raise ToolInputError("question must contain text")
    out = DEFAULT_FOLDER / f"{repository.replace('/', '_')}-{_slug(question)}.json"
    _, destination, report = run_trace(
        repository, question,
        prs=_count(arguments, "prs", 15), issues=_count(arguments, "issues", 15),
        commits=_count(arguments, "commits", 20),
        reviews=bool(arguments.get("reviews", False)), files=bool(arguments.get("files", False)),
        out=out, cache_dir=DEFAULT_CACHE,
    )
    text = report.read_text(encoding="utf-8")
    return f"{text}\n---\nTrace: {destination}\nView the map: graphweave {destination}\n"


def explain_trace(arguments: dict) -> str:
    path = Path(str(arguments.get("path", "")))
    if not path.is_file():
        raise ToolInputError(f"no trace file at {path}")
    report = report_file(path)
    return report.read_text(encoding="utf-8")


def list_traces(arguments: dict) -> str:
    folder = Path(str(arguments.get("folder") or DEFAULT_FOLDER))
    if not folder.is_dir():
        return f"No traces yet: {folder} does not exist."
    lines = []
    for path in sorted(folder.glob("*.json")):
        try:
            lines.append(f"- {path}: {load(path).query}")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return "\n".join(lines) if lines else f"No traces in {folder}."


TOOLS: dict[str, tuple[Any, str, dict]] = {
    "trace_repo": (
        trace_repo,
        "Answer a question about a public GitHub repository's recent pull requests, issues "
        "and commits. Returns a Markdown report: the answer, the retrieved items it used and "
        "did not use (with links), and every link with whether it came from GitHub's records "
        "or from text. Name people from AUTHORED/REVIEWED/REPORTED links.",
        {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name, e.g. psf/requests"},
                "question": {"type": "string"},
                "reviews": {"type": "boolean", "description": "also fetch PR reviews (who reviewed)"},
                "files": {"type": "boolean", "description": "also fetch files changed by each PR"},
                "prs": {"type": "integer", "description": "recent PRs to read (default 15)"},
                "issues": {"type": "integer", "description": "recent issues to read (default 15)"},
                "commits": {"type": "integer", "description": "recent commits to read (default 20)"},
            },
            "required": ["repo", "question"],
        },
    ),
    "explain_trace": (
        explain_trace,
        "The Markdown report for an existing graphweave trace file: what was retrieved, "
        "what the answer used, and why each item ranked.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
    "list_traces": (
        list_traces,
        "List saved traces in a folder (default graphweave_out) with the question each asked.",
        {"type": "object", "properties": {"folder": {"type": "string"}}},
    ),
}


def call(name: str, arguments: dict) -> tuple[str, bool]:
    """Run a tool; return (text, is_error). Failures are reported, not raised."""
    if name not in TOOLS:
        return f"Unknown tool: {name}", True
    try:
        return TOOLS[name][0](arguments), False
    except ToolInputError as exc:
        return str(exc), True
    except Exception as exc:  # noqa: BLE001 - one failed call must not end the server
        return f"{name} failed: {exc}", True


def build_server():
    from mcp import types
    from mcp.server.lowlevel import Server

    def tool_list() -> list:
        return [types.Tool(name=name, description=description, inputSchema=schema)
                for name, (_, description, schema) in TOOLS.items()]

    async def run_tool(name: str, arguments: dict) -> tuple[list, bool]:
        # Fetching from GitHub blocks; keep it off the event loop.
        text, failed = await asyncio.to_thread(call, name, dict(arguments or {}))
        return [types.TextContent(type="text", text=text)], failed

    if hasattr(Server, "list_tools"):  # mcp 1.x
        server = Server(SERVER_NAME)

        async def list_tools() -> list:
            return tool_list()

        async def call_tool(name: str, arguments: dict) -> list:
            content, failed = await run_tool(name, arguments)
            if failed:
                # 1.x reports a raised error to the client as an error result.
                raise RuntimeError(content[0].text)
            return content

        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    async def on_list_tools(ctx, params) -> types.ListToolsResult:  # mcp 2.x
        return types.ListToolsResult(tools=tool_list())

    async def on_call_tool(ctx, params) -> types.CallToolResult:
        content, failed = await run_tool(params.name, dict(params.arguments or {}))
        return types.CallToolResult(content=content, isError=failed)

    return Server(SERVER_NAME, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def main(argv: list[str] | None = None) -> int:
    try:
        from mcp.server.stdio import stdio_server
    except ImportError:
        import sys

        print('graphweave-mcp needs the mcp package: pip install "graphweave[mcp]"', file=sys.stderr)
        return 1
    server = build_server()

    async def serve() -> None:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
