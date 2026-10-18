"""OpenTelemetry import, on spans recorded from a real instrumented LlamaIndex query.

The fixtures were exported from ``openinference-instrumentation-llama-index``
with the OpenTelemetry SDK, once as OTLP JSON and once as SDK span JSON lines.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag.entry import main as entry_main
from graphrag.tracing.otel import read_spans, traces_from_file, traces_from_spans

FIXTURES = Path(__file__).parent / "fixtures" / "otel"


class RealSpanTests(unittest.TestCase):
    def check(self, name: str) -> None:
        traces = traces_from_file(FIXTURES / name)
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace.query, "How do sessions retry?")
        self.assertTrue(trace.answer)
        self.assertEqual(sorted(item.label for item in trace.items), ["Proxy config", "Retry policy"])
        self.assertTrue(all(item.content and item.score is not None for item in trace.items))
        self.assertIn("retriever", {span.kind for span in trace.spans})
        self.assertTrue(all(span.start_ms is not None and span.start_ms >= 0 for span in trace.spans))

    def test_otlp_json(self):
        self.check("otlp.json")

    def test_sdk_span_json_lines(self):
        self.check("sdk.jsonl")

    def test_both_shapes_agree(self):
        a = traces_from_file(FIXTURES / "otlp.json")[0]
        b = traces_from_file(FIXTURES / "sdk.jsonl")[0]
        self.assertEqual([item.id for item in a.items], [item.id for item in b.items])
        self.assertEqual(len(a.spans), len(b.spans))


class ShapeTests(unittest.TestCase):
    def test_metadata_json_and_error_status(self):
        span = {
            "name": "retrieve", "context": {"trace_id": "t1", "span_id": "s1"}, "parent_id": None,
            "start_time": "2026-10-18T10:00:00Z", "end_time": "2026-10-18T10:00:01Z",
            "status": {"status_code": "ERROR"},
            "attributes": {
                "openinference.span.kind": "RETRIEVER", "input.value": "q",
                "retrieval.documents.0.document.id": "d0",
                "retrieval.documents.0.document.content": "alpha beta",
                "retrieval.documents.0.document.metadata": json.dumps({"url": "https://x/d0"}),
            },
        }
        trace = traces_from_spans(read_spans(json.dumps(span)))[0]
        self.assertEqual(trace.items[0].source_uri, "https://x/d0")
        self.assertEqual(trace.spans[0].status, "error")
        self.assertEqual(trace.spans[0].end_ms, 1000.0)

    def test_traces_without_retrieval_are_skipped(self):
        span = {"name": "llm", "context": {"trace_id": "t", "span_id": "s"},
                "attributes": {"openinference.span.kind": "LLM"}}
        self.assertEqual(traces_from_spans(read_spans(json.dumps(span))), [])

    def test_command_writes_trace_and_report(self):
        with TemporaryDirectory() as folder:
            self.assertEqual(entry_main(["import-otel", str(FIXTURES / "otlp.json"), "--out", folder]), 0)
            written = sorted(path.name for path in Path(folder).iterdir())
            self.assertEqual(written, ["otel-01-how-do-sessions-retry.json",
                                       "otel-01-how-do-sessions-retry.report.md"])

    def test_command_reports_a_file_with_nothing_to_import(self):
        with TemporaryDirectory() as folder:
            empty = Path(folder) / "empty.json"
            empty.write_text("{}")
            self.assertEqual(entry_main(["import-otel", str(empty), "--out", folder]), 1)


if __name__ == "__main__":
    unittest.main()
