"""The LlamaIndex tracer against a real query engine. Skipped without llama-index-core."""

from __future__ import annotations

import unittest

try:
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
except ImportError:  # pragma: no cover - optional extra
    VectorStoreIndex = None


@unittest.skipIf(VectorStoreIndex is None, "llama-index-core is not installed")
class LlamaIndexTracerTests(unittest.TestCase):
    def setUp(self):
        Settings.embed_model = MockEmbedding(embed_dim=8)
        Settings.llm = MockLLM(max_tokens=16)
        documents = [
            Document(text="Sessions retry failed requests with exponential backoff.",
                     metadata={"title": "Retry policy", "url": "https://example.com/retry"}),
            Document(text="Proxies are read from HTTPS_PROXY.", metadata={"title": "Proxy config"}),
        ]
        self.engine = VectorStoreIndex.from_documents(documents).as_query_engine(similarity_top_k=2)

    def test_a_query_becomes_a_trace_with_its_retrieved_nodes(self):
        from graphrag.tracing_llamaindex import LlamaIndexTracer

        with LlamaIndexTracer() as tracer:
            response = self.engine.query("How do sessions retry?")
        trace = tracer.finish()

        self.assertEqual(trace.query, "How do sessions retry?")
        self.assertEqual(trace.answer, str(response))
        self.assertEqual(trace.producer, "llamaindex")
        self.assertEqual(len(trace.retrievals), 1)
        labels = sorted(item.label for item in trace.items)
        self.assertEqual(labels, ["Proxy config", "Retry policy"])
        retry = next(item for item in trace.items if item.label == "Retry policy")
        self.assertEqual(retry.source_uri, "https://example.com/retry")
        self.assertIsNotNone(retry.score)
        self.assertTrue(any(span.kind == "retriever" for span in trace.spans))
        self.assertTrue(all(span.end_ms is not None for span in trace.spans))

    def test_detached_tracer_records_nothing_more(self):
        from graphrag.tracing_llamaindex import LlamaIndexTracer

        with LlamaIndexTracer() as tracer:
            self.engine.query("first")
        self.engine.query("second")
        self.assertEqual(len(tracer.finish().retrievals), 1)

    def test_the_trace_saves_and_reports(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from graphrag.tracing import save, score_overlaps
        from graphrag.tracing.report import render_markdown
        from graphrag.tracing_llamaindex import LlamaIndexTracer

        with LlamaIndexTracer() as tracer:
            self.engine.query("How do sessions retry?")
        trace = score_overlaps(tracer.finish(answer="Sessions retry failed requests with exponential backoff."))
        with TemporaryDirectory() as folder:
            save(trace, Path(folder) / "li.json")
        self.assertIn("**Retrieved 2, used 1.**", render_markdown(trace))


if __name__ == "__main__":
    unittest.main()
