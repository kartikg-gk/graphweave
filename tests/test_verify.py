from __future__ import annotations

import unittest

from graphrag.tracing import capture
from graphrag.verify import context_of, similarity, verify

ITEMS = [
    {"id": "retry", "label": "Retry policy", "content": "Sessions retry failed requests with exponential backoff."},
    {"id": "proxy", "label": "Proxy config", "content": "Proxies are read from environment variables."},
    {"id": "noise", "label": "Changelog", "content": "Version two dropped an old TLS setting."},
]


def fake_model(question: str, context: str) -> str:
    """Answers from the retry item when present; otherwise says it cannot."""
    if "exponential backoff" in context:
        return "Sessions retry failed requests with exponential backoff."
    return "The context does not say how sessions retry."


class VerifyTests(unittest.TestCase):
    def test_only_the_item_the_answer_needed_is_needed(self):
        result = verify(capture("How do sessions retry?", ITEMS), fake_model)
        needed = {item.id: item.needed for item in result.items}
        self.assertEqual(needed, {"retry": True, "proxy": False, "noise": False})
        self.assertEqual(result.noise, 0.0)

    def test_agreement_with_word_overlap(self):
        result = verify(capture("How do sessions retry?", ITEMS), fake_model)
        self.assertEqual(result.agreement, 1.0)
        self.assertTrue(next(item for item in result.items if item.id == "retry").overlap_says_used)

    def test_a_noisy_model_raises_the_bar(self):
        answers = iter([
            "Sessions retry with backoff.",                       # baseline
            "They retry failed calls, waiting longer each time.",  # same context, different words
            "Sessions retry, waiting longer each time.",           # without retry: drifted, but within noise
            "Sessions retry with backoff.",
            "Sessions retry with backoff.",
        ])
        result = verify(capture("q", ITEMS), lambda question, context: next(answers))
        self.assertGreater(result.noise, 0.5)
        self.assertFalse(any(item.needed for item in result.items))

    def test_the_model_sees_every_item_but_the_removed_one(self):
        contexts = []
        verify(capture("q", ITEMS), lambda question, context: contexts.append(context) or "x")
        self.assertEqual(len(contexts), len(ITEMS) + 2)
        self.assertNotIn("Retry policy", contexts[2])
        self.assertIn("Retry policy", contexts[3])

    def test_similarity_bounds(self):
        self.assertEqual(similarity("a b c", "a b c"), 1.0)
        self.assertEqual(similarity("", ""), 1.0)
        self.assertEqual(similarity("alpha beta", "gamma delta"), 0.0)
        self.assertIn("[Retry policy]", context_of(capture("q", ITEMS).items))


if __name__ == "__main__":
    unittest.main()
