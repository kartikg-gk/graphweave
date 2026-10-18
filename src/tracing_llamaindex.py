"""LlamaIndex adapter: observe queries and emit a framework-neutral trace.

    from graphrag.tracing import save, score_overlaps
    from graphrag.tracing_llamaindex import LlamaIndexTracer

    with LlamaIndexTracer() as tracer:
        response = query_engine.query("How do sessions retry?")
    save(score_overlaps(tracer.finish()))

It listens to LlamaIndex's instrumentation events, so it works with any query
engine, retriever or chat engine built on ``llama-index-core`` without changing
how they are constructed. Each retrieval becomes a ``Retrieval`` with its nodes
as items; the query and the synthesized answer are taken from the query events.

**Pure observation.** A failure inside the handler is swallowed, so a tracer
bug cannot break the query it is watching. Nothing LlamaIndex-shaped survives
into the trace: nodes are turned into plain items the moment they arrive.

Lives beside ``tracing_langgraph`` rather than inside ``tracing/``, which stays
standard-library only; install with ``pip install "graphweave[llamaindex]"``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from pydantic import PrivateAttr

from .tracing import (
    ARM_VECTOR,
    KIND_CHAIN,
    KIND_DOCUMENT,
    KIND_RETRIEVER,
    PRODUCER_UNKNOWN,
    STATUS_OK,
    Retrieval,
    Span,
    Trace,
    TraceItem,
)

#: Node metadata keys that name a source, in priority order.
LABEL_KEYS = ("title", "file_name", "filename", "name", "source")
#: Node metadata keys that hold a link or path to the source.
URI_KEYS = ("url", "source_url", "link", "file_path", "path")


def _query_text(value: Any) -> str:
    """A query may arrive as a string or a QueryBundle."""
    if value is None:
        return ""
    text = getattr(value, "query_str", value)
    return str(text)


def _answer_text(response: Any) -> str | None:
    if response is None:
        return None
    text = getattr(response, "response", response)
    return None if text is None else str(text)


def _item(node_with_score: Any, index: int) -> TraceItem:
    node = getattr(node_with_score, "node", node_with_score)
    metadata = dict(getattr(node, "metadata", None) or {})
    node_id = str(getattr(node, "node_id", None) or getattr(node, "id_", None) or f"node-{index}")
    try:
        content = node.get_content()
    except Exception:  # noqa: BLE001 - a node type without text still gets an item
        content = str(getattr(node, "text", "") or "")
    label = next((str(metadata[key]) for key in LABEL_KEYS if metadata.get(key)), None)
    uri = next((str(metadata[key]) for key in URI_KEYS if metadata.get(key)), None)
    score = getattr(node_with_score, "score", None)
    return TraceItem(
        id=node_id,
        content=content,
        source="llamaindex",
        label=label,
        kind=KIND_DOCUMENT,
        source_uri=uri,
        score=None if score is None else float(score),
        vector_score=None if score is None else float(score),
        metadata={key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))},
    )


class _Handler(BaseEventHandler):
    """The object LlamaIndex calls; forwards every event to the tracer."""

    _tracer: Any = PrivateAttr()

    def __init__(self, tracer: "LlamaIndexTracer", **data: Any) -> None:
        super().__init__(**data)
        self._tracer = tracer

    @classmethod
    def class_name(cls) -> str:
        return "GraphweaveTraceHandler"

    def handle(self, event: Any, **kwargs: Any) -> None:
        try:
            self._tracer._on_event(event)
        except Exception:  # noqa: BLE001 - observation must never break the query
            pass


class LlamaIndexTracer:
    """Collects LlamaIndex query and retrieval events into one ``Trace``."""

    def __init__(self, *, producer: str = "llamaindex") -> None:
        self.producer = producer
        self.started_at = datetime.now(timezone.utc)
        self._began = perf_counter()
        self._lock = threading.Lock()
        self._handler = _Handler(self)
        self._dispatcher = None
        self.retrievals: list[Retrieval] = []
        self.spans: list[Span] = []
        self._open: dict[str, Span] = {}
        self._first_query: str | None = None
        self._last_answer: str | None = None

    # -- attaching ---------------------------------------------------------

    def attach(self, dispatcher: Any = None) -> "LlamaIndexTracer":
        """Start listening on ``dispatcher`` (the root dispatcher by default)."""
        self._dispatcher = dispatcher or get_dispatcher()
        self._dispatcher.add_event_handler(self._handler)
        return self

    def detach(self) -> None:
        """Stop listening. Safe to call twice."""
        if self._dispatcher is not None:
            handlers = self._dispatcher.event_handlers
            if self._handler in handlers:
                handlers.remove(self._handler)
            self._dispatcher = None

    def __enter__(self) -> "LlamaIndexTracer":
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # -- events ------------------------------------------------------------

    def _now_ms(self) -> float:
        return (perf_counter() - self._began) * 1000

    def _start(self, key: str, name: str, kind: str) -> None:
        span = Span(id=f"{kind}-{len(self.spans) + 1}", name=name, kind=kind,
                    start_ms=self._now_ms(), status="running")
        self.spans.append(span)
        self._open[key] = span

    def _end(self, key: str) -> Span | None:
        span = self._open.pop(key, None)
        if span is not None:
            span.end_ms = self._now_ms()
            span.status = STATUS_OK
        return span

    def _on_event(self, event: Any) -> None:
        kind = type(event).__name__
        key = str(getattr(event, "span_id", None) or kind)
        with self._lock:
            if kind == "QueryStartEvent":
                text = _query_text(getattr(event, "query", None))
                if self._first_query is None and text:
                    self._first_query = text
                self._start(f"query:{key}", "query", KIND_CHAIN)
            elif kind == "QueryEndEvent":
                self._end(f"query:{key}")
                answer = _answer_text(getattr(event, "response", None))
                if answer:
                    self._last_answer = answer
            elif kind == "RetrievalStartEvent":
                self._start(f"retrieve:{key}", "retrieve", KIND_RETRIEVER)
            elif kind == "RetrievalEndEvent":
                span = self._end(f"retrieve:{key}")
                nodes = list(getattr(event, "nodes", None) or [])
                self.retrievals.append(Retrieval(
                    query=_query_text(getattr(event, "str_or_query_bundle", None)),
                    span_id=span.id if span else None,
                    arm=ARM_VECTOR,
                    items=[_item(node, index) for index, node in enumerate(nodes)],
                ))
            elif kind == "SynthesizeEndEvent":
                answer = _answer_text(getattr(event, "response", None))
                if answer:
                    self._last_answer = answer

    # -- result ------------------------------------------------------------

    def finish(self, query: str | None = None, answer: str | None = None) -> Trace:
        """The trace so far. ``query`` and ``answer`` default to what was observed."""
        with self._lock:
            for span in self._open.values():
                span.end_ms = self._now_ms()
                span.status = STATUS_OK
            self._open.clear()
            latency = max((span.end_ms or 0.0 for span in self.spans), default=None)
            return Trace(
                query=query if query is not None else (self._first_query or ""),
                answer=answer if answer is not None else self._last_answer,
                producer=self.producer or PRODUCER_UNKNOWN,
                started_at=self.started_at,
                duration_ms=latency,
                retrievals=list(self.retrievals),
                spans=list(self.spans),
            )
