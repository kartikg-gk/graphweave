"""The MCP server, driven by a real MCP client over stdio. Skipped without ``mcp``."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag.mcp_server import call
from graphrag.tracing import capture, save

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - optional extra
    ClientSession = None

ANSWER = "Sessions retry failed requests with exponential backoff up to three times."
ITEMS = [
    {"id": "doc-1", "label": "Retry policy", "content": ANSWER},
    {"id": "doc-2", "label": "Proxy config", "content": "Proxies come from HTTPS_PROXY."},
]


class ToolTests(unittest.TestCase):
    def test_bad_input_is_reported_not_raised(self):
        text, failed = call("trace_repo", {"repo": "not a repo", "question": "x"})
        self.assertTrue(failed)
        self.assertIn("owner/name", text)
        self.assertEqual(call("nope", {})[1], True)

    def test_explain_and_list_on_disk(self):
        with TemporaryDirectory() as folder:
            path = save(capture("How do sessions retry?", ITEMS, answer=ANSWER), Path(folder) / "run.json")
            text, failed = call("explain_trace", {"path": str(path)})
            self.assertFalse(failed)
            self.assertIn("used 1", text)
            listing, _ = call("list_traces", {"folder": folder})
            self.assertIn("How do sessions retry?", listing)


@unittest.skipIf(ClientSession is None, "mcp is not installed")
class StdioTests(unittest.TestCase):
    def test_a_client_lists_and_calls_tools(self):
        async def session() -> tuple[list[str], str, bool]:
            with TemporaryDirectory() as folder:
                path = save(capture("How do sessions retry?", ITEMS, answer=ANSWER), Path(folder) / "run.json")
                params = StdioServerParameters(command=sys.executable, args=["-m", "graphrag.mcp_server"],
                                               env=dict(os.environ))
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        names = [tool.name for tool in (await client.list_tools()).tools]
                        result = await client.call_tool("explain_trace", {"path": str(path)})
                        bad = await client.call_tool("explain_trace", {"path": "missing.json"})
                        return names, result.content[0].text, getattr(bad, "is_error", getattr(bad, "isError", None))

        names, text, bad_is_error = asyncio.run(session())
        self.assertEqual(sorted(names), ["explain_trace", "list_traces", "trace_repo"])
        self.assertIn("**Retrieved 2, used 1.**", text)
        self.assertTrue(bad_is_error)


if __name__ == "__main__":
    unittest.main()
