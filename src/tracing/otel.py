"""Read OpenTelemetry spans into traces: no code change on the traced app.

Apps already instrumented for AI tracing usually follow the OpenInference
conventions: each span carries ``openinference.span.kind`` (``RETRIEVER``,
``LLM``, ``CHAIN``, ``TOOL``...), a retriever span lists what it found under
``retrieval.documents.<n>.document.{id,content,score,metadata}``, and the
question and answer sit in ``input.value`` and ``output.value``. This module
turns each OpenTelemetry trace into one graphweave ``Trace``.

Two file shapes are read, each as one JSON document or one per line:

* **OTLP JSON** — ``{"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}``,
  what an OpenTelemetry Collector file exporter writes. Attributes are a list
  of ``{"key", "value": {"stringValue" | "intValue" | ...}}``.
* **SDK span JSON** — what ``span.to_json()`` and the console exporter print:
  ``{"name", "context": {"trace_id", "span_id"}, "parent_id", "attributes"}``
  with plain attribute values. A JSON list of these is read too.

Standard library only, like the rest of ``tracing``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .capture import KIND_CHAIN, KIND_LLM, KIND_RETRIEVER, KIND_TOOL
from .schema import ARM_UNKNOWN, STATUS_ERROR, STATUS_OK, Retrieval, Span, Trace, TraceItem

KIND_ATTRIBUTE = "openinference.span.kind"
DOCUMENT_KEY = re.compile(r"^retrieval\.documents\.(\d+)\.document\.(id|content|score|metadata)$")
SPAN_KINDS = {"RETRIEVER": KIND_RETRIEVER, "LLM": KIND_LLM, "TOOL": KIND_TOOL}
PRODUCER = "opentelemetry"


@dataclass
class _RawSpan:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    start_ns: int | None
    end_ns: int | None
    error: bool
    attributes: dict[str, Any] = field(default_factory=dict)


# -- reading the two shapes ---------------------------------------------------


def _otlp_value(value: dict) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(item) for item in value["arrayValue"].get("values", [])]
    return None


def _from_otlp(document: dict) -> Iterable[_RawSpan]:
    for resource in document.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", resource.get("instrumentationLibrarySpans", [])):
            for span in scope.get("spans", []):
                attributes = {item["key"]: _otlp_value(item.get("value", {}))
                              for item in span.get("attributes", [])}
                start, end = span.get("startTimeUnixNano"), span.get("endTimeUnixNano")
                yield _RawSpan(
                    trace_id=str(span.get("traceId", "")),
                    span_id=str(span.get("spanId", "")),
                    parent_id=span.get("parentSpanId") or None,
                    name=str(span.get("name", "")),
                    start_ns=int(start) if start else None,
                    end_ns=int(end) if end else None,
                    error=(span.get("status") or {}).get("code") in (2, "STATUS_CODE_ERROR"),
                    attributes=attributes,
                )


def _iso_ns(value: Any) -> int | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(moment.timestamp() * 1_000_000_000)


def _from_sdk(span: dict) -> _RawSpan:
    context = span.get("context") or {}
    status = (span.get("status") or {}).get("status_code", "")
    return _RawSpan(
        trace_id=str(context.get("trace_id", "")),
        span_id=str(context.get("span_id", "")),
        parent_id=span.get("parent_id") or None,
        name=str(span.get("name", "")),
        start_ns=_iso_ns(span.get("start_time")),
        end_ns=_iso_ns(span.get("end_time")),
        error=str(status).upper() == "ERROR",
        attributes=dict(span.get("attributes") or {}),
    )


def _documents(text: str) -> list[Any]:
    text = text.strip()
    if not text:
        return []
    try:
        whole = json.loads(text)
        return whole if isinstance(whole, list) else [whole]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_spans(text: str) -> list[_RawSpan]:
    """Every span in a file's text, whichever shape it is."""
    spans: list[_RawSpan] = []
    for document in _documents(text):
        if isinstance(document, dict) and "resourceSpans" in document:
            spans.extend(_from_otlp(document))
        elif isinstance(document, dict) and "context" in document:
            spans.append(_from_sdk(document))
    return spans


# -- spans to traces ----------------------------------------------------------


def _items(attributes: dict[str, Any]) -> list[TraceItem]:
    found: dict[int, dict[str, Any]] = {}
    for key, value in attributes.items():
        match = DOCUMENT_KEY.match(key)
        if match:
            found.setdefault(int(match.group(1)), {})[match.group(2)] = value
    items = []
    for index in sorted(found):
        fields = found[index]
        metadata = fields.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata = metadata if isinstance(metadata, dict) else {}
        score = fields.get("score")
        items.append(TraceItem(
            id=str(fields.get("id") or f"document-{index}"),
            content=str(fields.get("content") or ""),
            source=PRODUCER,
            label=next((str(metadata[key]) for key in ("title", "file_name", "name") if metadata.get(key)), None),
            source_uri=next((str(metadata[key]) for key in ("url", "source", "file_path") if metadata.get(key)), None),
            score=None if score is None else float(score),
            metadata={key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool))},
        ))
    return items


def _trace(spans: list[_RawSpan]) -> Trace:
    starts = [span.start_ns for span in spans if span.start_ns is not None]
    origin = min(starts) if starts else None
    ordered = sorted(spans, key=lambda span: span.start_ns or 0)

    def ms(value: int | None) -> float | None:
        return None if value is None or origin is None else (value - origin) / 1_000_000

    ids = {span.span_id for span in spans}
    roots = [span for span in ordered if not span.parent_id or span.parent_id not in ids]
    root = roots[0] if roots else ordered[0]
    query = str(root.attributes.get("input.value") or "")
    answer = root.attributes.get("output.value")
    if not answer:
        llm_outputs = [span.attributes.get("output.value") for span in ordered
                       if span.attributes.get(KIND_ATTRIBUTE) == "LLM" and span.attributes.get("output.value")]
        answer = llm_outputs[-1] if llm_outputs else None

    retrievals = []
    trace_spans = []
    for span in ordered:
        kind = SPAN_KINDS.get(str(span.attributes.get(KIND_ATTRIBUTE, "")).upper(), KIND_CHAIN)
        trace_spans.append(Span(
            id=span.span_id, name=span.name, kind=kind,
            parent_id=span.parent_id if span.parent_id in ids else None,
            start_ms=ms(span.start_ns), end_ms=ms(span.end_ns),
            status=STATUS_ERROR if span.error else STATUS_OK,
        ))
        if kind == KIND_RETRIEVER:
            retrievals.append(Retrieval(
                query=str(span.attributes.get("input.value") or query),
                span_id=span.span_id, arm=ARM_UNKNOWN, items=_items(span.attributes),
            ))
            if not query:
                query = retrievals[-1].query

    ends = [span.end_ns for span in spans if span.end_ns is not None]
    duration = ms(max(ends)) if ends and origin is not None else None
    return Trace(query=query, answer=None if answer is None else str(answer), producer=PRODUCER,
                 duration_ms=duration, retrievals=retrievals, spans=trace_spans)


def traces_from_spans(spans: list[_RawSpan]) -> list[Trace]:
    """One trace per OpenTelemetry trace id that retrieved something."""
    groups: dict[str, list[_RawSpan]] = {}
    for span in spans:
        groups.setdefault(span.trace_id, []).append(span)
    traces = [_trace(group) for group in groups.values()]
    return [trace for trace in traces if trace.retrievals]


def traces_from_file(path: str | Path) -> list[Trace]:
    return traces_from_spans(read_spans(Path(path).read_text(encoding="utf-8")))
