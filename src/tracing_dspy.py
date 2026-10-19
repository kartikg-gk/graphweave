"""DSPy adapter: observe a program run and emit a framework-neutral trace.

    from graphrag.tracing import save, score_overlaps
    from graphrag.tracing_dspy import DSPyTracer

    with DSPyTracer() as tracer:
        prediction = program(question="How do sessions retry?")
    save(score_overlaps(tracer.finish()))

It is a DSPy callback, so it works with any module without changing how the
program is built. Every module call becomes a span. Retrieved items are taken
from what the model was actually given: a list passed to a module as
``context``, ``passages`` or ``documents`` becomes a ``Retrieval`` with one
item per entry, however it was fetched. A module that returns ``passages``
(``dspy.Retrieve`` and anything shaped like it) is recorded the same way. The
same list seen twice is recorded once.

DSPy's own ``dspy.retrievers.Embeddings`` does not report to callbacks, which
is why the model's input is read rather than the retriever's output. For a
search whose results never reach a module input, call
``tracer.record(query, passages)``.

The question is the outermost module's first text input, and the answer is its
``answer`` output, or its last text output when there is no ``answer``.

**Pure observation.** A failure inside the callback is swallowed, so a tracer
bug cannot break the program it is watching. Nothing DSPy-shaped survives into
the trace: passages are turned into plain items the moment they arrive.

Lives beside ``tracing_langgraph`` rather than inside ``tracing/``, which stays
standard-library only; install with ``pip install "graphweave[dspy]"``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterable

import dspy
from dspy.utils.callback import BaseCallback

from .tracing import (
    ARM_VECTOR,
    KIND_CHAIN,
    KIND_DOCUMENT,
    KIND_LLM,
    KIND_RETRIEVER,
    KIND_TOOL,
    PRODUCER_UNKNOWN,
    STATUS_OK,
    Retrieval,
    Span,
    Trace,
    TraceItem,
)
from .tracing.schema import STATUS_ERROR

#: Input names that hold the question, in priority order.
QUESTION_KEYS = ("question", "query", "q", "input")
#: Module inputs that hold retrieved passages.
CONTEXT_KEYS = ("context", "passages", "documents", "docs")
#: Passage dict keys that hold text, a label and a link.
TEXT_KEYS = ("long_text", "text", "content", "passage")
LABEL_KEYS = ("title", "name", "source")
URI_KEYS = ("url", "source_url", "link", "path")


def _fields(value: Any) -> dict[str, Any]:
    """A Prediction or Example as a plain dict; anything else as nothing."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value.items())
    except Exception:  # noqa: BLE001 - not a DSPy record
        return {}


def _arguments(inputs: dict[str, Any]) -> dict[str, Any]:
    """A module call's keyword arguments. DSPy passes ``{"args": ..., "kwargs": ...}``."""
    if "kwargs" in inputs and isinstance(inputs["kwargs"], dict):
        return inputs["kwargs"]
    return inputs


def _question(inputs: dict[str, Any]) -> str | None:
    for key in QUESTION_KEYS:
        if isinstance(inputs.get(key), str) and inputs[key].strip():
            return inputs[key]
    return next((value for value in inputs.values() if isinstance(value, str) and value.strip()), None)


def _answer(outputs: Any) -> str | None:
    fields = _fields(outputs)
    if isinstance(fields.get("answer"), str):
        return fields["answer"]
    texts = [value for key, value in fields.items() if isinstance(value, str) and key != "reasoning"]
    return texts[-1] if texts else None


def _item(passage: Any, index: int, score: Any = None) -> TraceItem:
    if isinstance(passage, dict):
        text = next((str(passage[key]) for key in TEXT_KEYS if passage.get(key)), "")
        label = next((str(passage[key]) for key in LABEL_KEYS if passage.get(key)), None)
        uri = next((str(passage[key]) for key in URI_KEYS if passage.get(key)), None)
        item_id = str(passage.get("id") or passage.get("pid") or f"passage-{index}")
        metadata = {key: value for key, value in passage.items()
                    if key not in TEXT_KEYS and isinstance(value, (str, int, float, bool))}
    else:
        text = str(getattr(passage, "long_text", passage))
        label = uri = None
        item_id = f"passage-{index}"
        metadata = {}
    if label is None and text:
        # DSPy's own corpora put the title before " | ".
        head = text.split(" | ", 1)[0]
        label = head if head != text and len(head) <= 120 else None
    number = None if score is None else float(score)
    return TraceItem(
        id=item_id,
        content=text,
        source="dspy",
        label=label,
        kind=KIND_DOCUMENT,
        source_uri=uri,
        score=number,
        vector_score=number,
        metadata=metadata,
    )


class _Callback(BaseCallback):
    """The object DSPy calls; forwards every event to the tracer."""

    def __init__(self, tracer: "DSPyTracer") -> None:
        self._tracer = tracer

    def _safely(self, handler, *args: Any) -> None:
        try:
            handler(*args)
        except Exception:  # noqa: BLE001 - observation must never break the program
            pass

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safely(self._tracer._module_start, call_id, instance, inputs)

    def on_module_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safely(self._tracer._module_end, call_id, outputs, exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        self._safely(self._tracer._start, call_id, "lm", KIND_LLM)

    def on_lm_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safely(self._tracer._end, call_id, exception)

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        name = getattr(instance, "name", None) or "tool"
        self._safely(self._tracer._start, call_id, str(name), KIND_TOOL)

    def on_tool_end(self, call_id: str, outputs: Any, exception: BaseException | None = None) -> None:
        self._safely(self._tracer._end, call_id, exception)


class DSPyTracer:
    """Collects DSPy module, LM and retrieval calls into one ``Trace``."""

    def __init__(self, *, producer: str = "dspy") -> None:
        self.producer = producer
        self.started_at = datetime.now(timezone.utc)
        self._began = perf_counter()
        self._lock = threading.Lock()
        self.callback = _Callback(self)
        self.retrievals: list[Retrieval] = []
        self.spans: list[Span] = []
        self._open: dict[str, Span] = {}
        self._queries: dict[str, str | None] = {}
        self._depth = 0
        self._question: str | None = None
        self._answer: str | None = None
        self._context = None
        self._seen: set[tuple[str, ...]] = set()

    # -- attaching ---------------------------------------------------------

    def __enter__(self) -> "DSPyTracer":
        callbacks = list(dspy.settings.get("callbacks") or []) + [self.callback]
        self._context = dspy.context(callbacks=callbacks)
        self._context.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._context is not None:
            self._context.__exit__(*exc)
            self._context = None

    # -- events ------------------------------------------------------------

    def _now_ms(self) -> float:
        return (perf_counter() - self._began) * 1000

    def _start(self, call_id: str, name: str, kind: str) -> Span:
        with self._lock:
            span = Span(id=f"{kind}-{len(self.spans) + 1}", name=name, kind=kind,
                        start_ms=self._now_ms(), status="running")
            self.spans.append(span)
            self._open[call_id] = span
            return span

    def _end(self, call_id: str, exception: BaseException | None = None) -> Span | None:
        with self._lock:
            span = self._open.pop(call_id, None)
            if span is not None:
                span.end_ms = self._now_ms()
                span.status = STATUS_ERROR if exception is not None else STATUS_OK
            return span

    def _module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        span = self._start(call_id, type(instance).__name__, KIND_CHAIN)
        arguments = _arguments(inputs or {})
        question = _question(arguments)
        with self._lock:
            self._queries[call_id] = question
            if self._depth == 0 and question and self._question is None:
                self._question = question
            nested = self._depth > 0
            self._depth += 1
        if nested:
            passages = next((arguments[key] for key in CONTEXT_KEYS
                             if isinstance(arguments.get(key), (list, tuple)) and arguments[key]), None)
            if passages is not None:
                self._add(question, passages, None, span)

    def _module_end(self, call_id: str, outputs: Any, exception: BaseException | None) -> None:
        span = self._end(call_id, exception)
        with self._lock:
            self._depth = max(self._depth - 1, 0)
            query = self._queries.pop(call_id, None)
            outermost = self._depth == 0
        fields = _fields(outputs)
        passages = fields.get("passages")
        if exception is None and isinstance(passages, (list, tuple)):
            if span is not None:
                span.kind = KIND_RETRIEVER
            self._add(query, passages, fields.get("scores"), span)
        elif outermost and exception is None:
            answer = _answer(outputs)
            if answer:
                self._answer = answer

    def _add(self, query: str | None, passages: Iterable[Any], scores: Any, span: Span | None) -> None:
        scores = list(scores) if isinstance(scores, (list, tuple)) else []
        items = [_item(passage, index, scores[index] if index < len(scores) else None)
                 for index, passage in enumerate(passages)]
        key = tuple(item.content for item in items)
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            self.retrievals.append(Retrieval(
                query=query or "",
                span_id=span.id if span else None,
                arm=ARM_VECTOR,
                items=items,
            ))

    # -- manual recording ----------------------------------------------------

    def record(self, query: str, passages: Iterable[Any], scores: Iterable[float] | None = None) -> None:
        """Record a retrieval DSPy cannot see, such as a plain search function."""
        span = Span(id=f"{KIND_RETRIEVER}-{len(self.spans) + 1}", name="retrieve",
                    kind=KIND_RETRIEVER, start_ms=self._now_ms(), end_ms=self._now_ms(),
                    status=STATUS_OK)
        with self._lock:
            self.spans.append(span)
        self._add(query, list(passages), list(scores) if scores is not None else None, span)

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
                query=query if query is not None else (self._question or ""),
                answer=answer if answer is not None else self._answer,
                producer=self.producer or PRODUCER_UNKNOWN,
                started_at=self.started_at,
                duration_ms=latency,
                retrievals=list(self.retrievals),
                spans=list(self.spans),
            )
