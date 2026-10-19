"""The DSPy tracer against a real retrieve-then-answer program. Skipped without dspy."""

from __future__ import annotations

import unittest

try:
    import dspy
    from dspy.utils.dummies import DummyLM
except ImportError:  # pragma: no cover - optional extra
    dspy = None

CORPUS = [
    "Retry policy | Sessions retry failed requests with exponential backoff.",
    "Proxy config | Proxies are read from HTTPS_PROXY.",
    "Changelog | Version two dropped an old TLS setting.",
]
ANSWER = "Sessions retry failed requests with exponential backoff."


def embed(texts):
    """Word-presence vectors over a fixed vocabulary: deterministic, no model."""
    vocabulary = ["retry", "sessions", "proxy", "proxies", "tls", "version", "backoff", "requests"]
    return [[float(word in text.lower()) + 0.01 for word in vocabulary] for text in texts]


if dspy is not None:
    class RAG(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retrieve = dspy.retrievers.Embeddings(corpus=CORPUS, embedder=dspy.Embedder(embed), k=2)
            self.answer = dspy.Predict("context, question -> answer")

        def forward(self, question):
            context = self.retrieve(question).passages
            return self.answer(context=context, question=question)


@unittest.skipIf(dspy is None, "dspy is not installed")
class DSPyTracerTests(unittest.TestCase):
    def setUp(self):
        dspy.configure(lm=DummyLM([{"answer": ANSWER}] * 4))

    def test_a_run_becomes_a_trace_with_its_retrieved_passages(self):
        from graphrag.tracing_dspy import DSPyTracer

        with DSPyTracer() as tracer:
            prediction = RAG()(question="How do sessions retry?")
        trace = tracer.finish()

        self.assertEqual(prediction.answer, ANSWER)
        self.assertEqual(trace.query, "How do sessions retry?")
        self.assertEqual(trace.answer, ANSWER)
        self.assertEqual(trace.producer, "dspy")
        self.assertEqual(len(trace.retrievals), 1)
        self.assertEqual(trace.retrievals[0].query, "How do sessions retry?")
        self.assertEqual(trace.items[0].label, "Retry policy")
        self.assertEqual(len(trace.items), 2)
        kinds = {span.kind for span in trace.spans}
        self.assertTrue({"chain", "llm"} <= kinds)
        # The passages are tied to the predictor call that received them.
        predictor = next(span for span in trace.spans if span.id == trace.retrievals[0].span_id)
        self.assertEqual(predictor.name, "Predict")

    def test_used_and_unused_after_scoring(self):
        from graphrag.tracing import is_used, score_overlaps
        from graphrag.tracing_dspy import DSPyTracer

        with DSPyTracer() as tracer:
            RAG()(question="How do sessions retry?")
        trace = score_overlaps(tracer.finish())
        verdicts = {item.label: is_used(item.overlap) for item in trace.items}
        self.assertTrue(verdicts["Retry policy"])

    def test_record_captures_a_search_dspy_cannot_see(self):
        from graphrag.tracing_dspy import DSPyTracer

        with DSPyTracer() as tracer:
            tracer.record("retry", [{"id": "d1", "title": "Retry policy", "text": "Sessions retry.",
                                     "url": "https://example.com/retry"}], scores=[0.9])
        item = tracer.finish().items[0]
        self.assertEqual((item.id, item.label, item.source_uri, item.score),
                         ("d1", "Retry policy", "https://example.com/retry", 0.9))

    def test_the_tracer_detaches_on_exit(self):
        from graphrag.tracing_dspy import DSPyTracer

        with DSPyTracer() as tracer:
            pass
        self.assertNotIn(tracer.callback, dspy.settings.get("callbacks") or [])

    def test_a_tracer_error_does_not_break_the_program(self):
        from graphrag.tracing_dspy import DSPyTracer

        with DSPyTracer() as tracer:
            tracer._add = None  # every retrieval now raises inside the callback
            prediction = RAG()(question="How do sessions retry?")
        self.assertEqual(prediction.answer, ANSWER)


if __name__ == "__main__":
    unittest.main()
