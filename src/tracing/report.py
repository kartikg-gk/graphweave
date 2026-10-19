"""A Markdown report of one trace, written beside it for an agent to read.

The JSON is for the viewer; this is for whatever reads text. An assistant can
open the report and answer from it without parsing the trace format: what was
asked, what was answered, which retrieved items the answer drew on, which it
did not, and where each link in the graph came from.

The used/unused split is the same measurement the viewer applies: the words an
item shares with the answer, over the smaller of the two, judged at ``DEFAULT_THRESHOLD``. An item
the trace already measured keeps its number; one it did not is measured here.
"""

from __future__ import annotations

from pathlib import Path

from .classify import DEFAULT_THRESHOLD, is_used, overlap_score
from .schema import Trace, TraceEdge, TraceItem
from .store import load

#: Links read from a platform's own records: who wrote, reviewed, fixed,
#: reported or changed something. Anything else was read out of prose.
RECORD_RELATIONS = frozenset({
    "AUTHORED", "AUTHORED_BY", "RESOLVES", "REVIEWED", "TOUCHES", "PART_OF", "REPORTED",
})
#: Links inferred from text: something named in a body, or names close together.
TEXT_RELATIONS = frozenset({"MENTIONS", "CO_OCCURS"})

SOURCE_RECORD = "record"
SOURCE_TEXT = "text"
SOURCE_UNKNOWN = "unknown"

#: Characters of an item's content quoted in the report.
EXCERPT_CHARS = 160


def link_source(relation: str) -> str:
    """Where a link came from: a platform record, the text, or unknown."""
    name = (relation or "").upper()
    if name in RECORD_RELATIONS:
        return SOURCE_RECORD
    if name in TEXT_RELATIONS:
        return SOURCE_TEXT
    return SOURCE_UNKNOWN


def _overlap(item: TraceItem, answer: str | None) -> float | None:
    if item.overlap is not None:
        return item.overlap
    if not answer:
        return None
    return overlap_score(item.content, answer)


def _excerpt(text: str) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= EXCERPT_CHARS else flat[: EXCERPT_CHARS - 1] + "…"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _item_line(item: TraceItem, overlap: float | None) -> str:
    name = item.label or item.id
    title = f"[{name}]({item.source_uri})" if item.source_uri else name
    parts = [f"overlap {_number(overlap)}", f"score {_number(item.score)}"]
    if item.graph_score is not None:
        parts.append(f"graph {_number(item.graph_score)}")
    detail = f"  \n  {_excerpt(item.content)}" if item.content else ""
    return f"- **{title}** ({', '.join(parts)}){detail}"


def _edge_line(edge: TraceEdge, labels: dict[str, str]) -> str:
    weight = "" if edge.weight is None else f", weight {edge.weight:.2f}"
    return (
        f"- {labels.get(edge.source, edge.source)} —{edge.relation}→ "
        f"{labels.get(edge.target, edge.target)} (from {link_source(edge.relation)}{weight})"
    )


def render_markdown(trace: Trace, *, threshold: float = DEFAULT_THRESHOLD) -> str:
    """The trace as Markdown: answer, used items, unused items, links."""
    items = list(trace.items)
    scored = [(item, _overlap(item, trace.answer)) for item in items]
    used = [(item, value) for item, value in scored if is_used(value, threshold)]
    unused = [(item, value) for item, value in scored if is_used(value, threshold) is False]
    unmeasured = [item for item, value in scored if value is None]

    lines = [f"# Trace: {trace.query}", ""]
    if trace.answer:
        lines += ["## Answer", "", trace.answer.strip(), ""]
    if trace.answer:
        lines += [
            f"**Retrieved {len(items)}, used {len(used)}.** An item counts as used when the words it "
            f"shares with the answer are at least {threshold:.0%} of the smaller of the two.",
            "",
        ]
    else:
        lines += [f"**Retrieved {len(items)}.** No answer was recorded, so nothing is marked used.", ""]

    for title, group in (("Used", used), ("Not used", unused)):
        if group:
            lines += [f"## {title} ({len(group)})", ""]
            lines += [_item_line(item, value) for item, value in sorted(group, key=lambda pair: -(pair[1] or 0))]
            lines.append("")
    if unmeasured:
        lines += [f"## Not measured ({len(unmeasured)})", ""]
        lines += [_item_line(item, None) for item in unmeasured]
        lines.append("")

    edges = list(trace.graph_edges or trace.edges)
    if edges:
        named = [*(trace.graph_nodes or []), *items]
        labels = {item.id: item.label or item.id for item in named}
        record = sum(link_source(edge.relation) == SOURCE_RECORD for edge in edges)
        text = sum(link_source(edge.relation) == SOURCE_TEXT for edge in edges)
        lines += [
            f"## Links ({len(edges)}: {record} from records, {text} from text)",
            "",
            *[_edge_line(edge, labels) for edge in edges],
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def report_path(trace_path: str | Path) -> Path:
    """Where a trace's report goes: beside it, ``.report.md`` instead of ``.json``."""
    path = Path(trace_path)
    return path.with_name(f"{path.stem}.report.md")


def write_report(trace: Trace, trace_path: str | Path, *, threshold: float = DEFAULT_THRESHOLD) -> Path:
    """Write the report beside a saved trace and return its path."""
    destination = report_path(trace_path)
    destination.write_text(render_markdown(trace, threshold=threshold), encoding="utf-8")
    return destination


def report_file(trace_path: str | Path, *, threshold: float = DEFAULT_THRESHOLD) -> Path:
    """Load a trace from disk and write its report beside it."""
    return write_report(load(trace_path), trace_path, threshold=threshold)
