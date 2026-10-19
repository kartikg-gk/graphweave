"""How alike two surface forms are.

One embedder, one scoring path, no abstraction over either. Text becomes a
dense vector from all-MiniLM-L6-v2 run through ONNX Runtime; vectors are L2-normalised; a score is
a dot product against the candidate matrix. That is the whole module.

There is deliberately no embedder interface and no way to select one by name.
An interface with a single usable implementation is not flexibility — it is
unexercised code plus indirection.

What this replaced, and why the replacement is not free
------------------------------------------------------

An earlier version hashed character n-grams into a fixed number of buckets.
That caught orthographic drift exactly — ``notification-service`` against
``notification_service`` scored 1.0 — because it compared characters.

A learned model compares meaning, and on this corpus that is not uniformly an
improvement. Measured against the same twenty surface forms, the model reads
``#413`` and ``#414`` as near-identical: they *are* near-identical as text, and
for an identifier near-identical text means definitely different. The n-gram
scorer kept every consecutive ticket pair apart; this one does not. See
``test_similarity.py``, which pins the wrong merges rather than hiding them.

The model earns its place on drift no character overlap can reach — "the
billing box" against ``payment_service`` — which is why it is here.

Normalisation
-------------

**The model returns raw vectors. Normalisation happens in ``Similarity``,
once.** ``normalize_embeddings`` is deliberately not passed to the library: the
guarantee is ours, enforced in one place, and a stub embedder in a test can
then return hand-built numbers of any magnitude and still drive real
thresholds.

Loading the model
-----------------

The model is loaded once per process and reused. Loading costs roughly a second
and a suite constructs many ``Similarity`` objects, so a per-instance load would
dominate the run.

The model runs through ONNX Runtime (``fastembed``), not torch. Measured on
593 texts from the demo graph against the sentence-transformers build of the
same model: the largest difference in any vector element was 2.3e-7, no
pairwise score moved by more than 6e-7, and every top-10 neighbour set and
every decision at thresholds 0.5 to 0.92 was identical. So the identity string
is unchanged and stores written by the torch build still load.

What it changed: install 225 MB instead of 1.4 GB, model load about 1 s
instead of 13 s, peak memory 347 MB instead of 794 MB while embedding the
same texts at the same speed. That needs ``BATCH_SIZE`` small: at fastembed's
default of 256 the peak was 2.5 GB, more than the demo server has.

The test seam
-------------

``Similarity`` accepts an embedder so tests can substitute fixed vectors. That
is the only seam, and it exists for tests rather than for configuration —
production constructs ``Similarity()`` and gets the one embedder.
"""

from __future__ import annotations

import math
from typing import Sequence

#: Its output size is the vector dimension, which is therefore taken from the
#: model rather than configured — there is no dimension to choose and nothing
#: to tune.
MODEL_NAME = "all-MiniLM-L6-v2"
#: Texts per ONNX call. Larger batches pad every text to the longest one and
#: memory grows with it (2.5 GB at 256); 8 was as fast as any size measured.
BATCH_SIZE = 8

_model = None


def _load_model():
    """The one model instance, loaded on first use and kept.

    A module-level cache rather than a fixture, so production gets the same
    reuse the tests do. Loading is not thread-safe here; nothing in this
    project embeds concurrently.
    """
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(f"sentence-transformers/{MODEL_NAME}")
    return _model


class MiniLMEmbedder:
    """Text to a dense vector, via all-MiniLM-L6-v2.

    Everything here is one call into someone else's library. There is no
    comparison logic to go untested, because comparison happens in
    ``Similarity`` on whatever vectors come back.

    ``identity`` names the model and its dimension, so a store built by a
    different model — or by the n-gram embedder this replaced — is rejected
    rather than silently compared.
    """

    #: The model's output size. Fixed by the model, not configured.
    dimension = 384

    @property
    def identity(self) -> str:
        return f"{MODEL_NAME}-{self.dimension}"

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Batches through the model, which is much faster than one at a time."""
        if not texts:
            return []
        rows = _load_model().embed(list(texts), batch_size=BATCH_SIZE)
        return [[float(value) for value in row] for row in rows]


#: The earlier name, kept so existing imports still work.
SentenceTransformerEmbedder = MiniLMEmbedder


def _normalised(vector: Sequence[float]) -> list[float]:
    """An L2-normalised copy. A zero vector stays zero rather than dividing."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return list(vector)
    return [value / magnitude for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class Similarity:
    """Normalise, then dot product against the candidate matrix.

    Scoring a whole candidate list in one call is deliberate: it lets the
    uncached candidates go through the model as one batch, and it means
    resolution never builds an N-by-N matrix it would throw most of away.

    Normalised vectors are cached by text. Resolution scores the same candidate
    repeatedly as the canonical set grows, and re-embedding it each time is the
    one avoidable cost in the loop.
    """

    def __init__(self, embedder=None) -> None:
        # The test seam, and the only one. Production passes nothing.
        self.embedder = (
            embedder if embedder is not None else MiniLMEmbedder()
        )
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        """Length of every vector this instance stores."""
        return self.embedder.dimension

    @property
    def identity(self) -> str:
        """Which embedding scheme every stored vector was produced by.

        **Recorded because a dimension check cannot detect a different
        embedder.** Two models can output the same width — 384 is a common
        one — so a store written by one and read by the other passes every
        length check in this module. The vectors are the right shape, the dot
        products are in range, and the scores mean nothing. There is no
        symptom to notice.

        The identity is the only thing that separates those two stores.
        ``test_a_store_from_a_different_embedder_is_refused`` asserts the equal
        dimension explicitly before asserting the rejection, so the reader can
        see the length check passing and doing no work.

        Both checks run on load and neither implies the other: a different
        model can share a dimension, and a different dimension can arrive under
        an identity string someone forgot to bump.
        """
        return getattr(
            self.embedder,
            "identity",
            f"{type(self.embedder).__name__}-{self.dimension}",
        )

    def _checked(self, vector: Sequence[float], text: str) -> list[float]:
        """Refuse a vector of the wrong length, at the point it is written.

        Checked on every write rather than once at construction: an embedder
        that returns a short vector for one input and a correct one for the
        next would otherwise poison the store silently, and the resulting
        scores are not obviously wrong — just wrong.
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"{self.identity} produced a vector of length {len(vector)} "
                f"for {text!r}, but the declared dimension is {self.dimension}"
            )
        return list(vector)

    def _fill_cache(self, texts: Sequence[str]) -> None:
        """Embed everything not already cached, in one batch where possible."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if not missing:
            return

        batch = getattr(self.embedder, "embed_many", None)
        vectors = (
            batch(missing) if batch is not None else [self.embedder.embed(t) for t in missing]
        )
        for text, vector in zip(missing, vectors):
            self._cache[text] = _normalised(self._checked(vector, text))

    def vector(self, text: str) -> list[float]:
        """The normalised vector for ``text``."""
        self._fill_cache([text])
        return self._cache[text]

    def scores(self, text: str, candidates: Sequence[str]) -> list[float]:
        """Similarity of ``text`` to each candidate, in order, -1.0 to 1.0."""
        if not candidates:
            return []

        self._fill_cache([text, *candidates])
        query = self._cache[text]
        return [_dot(query, self._cache[candidate]) for candidate in candidates]

    def export_vectors(self) -> dict:
        """The stored vectors, tagged with what produced them.

        Plain data, so a caller can write it to a file. The tag travels with
        the vectors because it is worthless anywhere else.
        """
        return {
            "identity": self.identity,
            "dimension": self.dimension,
            "vectors": dict(self._cache),
        }

    def load_vectors(self, payload: dict) -> None:
        """Adopt a stored set, refusing one this embedder did not produce.

        This is where a stale store is caught. Both checks matter and neither
        implies the other: a different model can share a dimension and mean
        something else entirely, and a different dimension can arrive under an
        identity string someone forgot to bump.
        """
        identity = payload.get("identity")
        if identity != self.identity:
            raise ValueError(
                f"stored vectors were produced by {identity!r}, but this "
                f"scorer embeds with {self.identity!r}; the store is stale"
            )

        vectors = payload.get("vectors", {})
        for text, vector in vectors.items():
            self._checked(vector, text)

        self._cache.update({text: list(v) for text, v in vectors.items()})
