"""Leave-one-out: check the used/unused verdict against what the answer needed.

The viewer calls an item *used* when enough of its words reach the answer. That
is a cheap guess, and it can be wrong both ways: a paraphrased item looks
unused, and an item that shares words by chance looks used. This measures
instead. The question is answered from all the retrieved items, then once more
with each item removed. An item the answer depended on changes the answer when
it is gone.

Models do not answer identically twice even with nothing removed, so the
baseline is asked twice and the difference between those two answers sets the
noise floor. An item counts as **needed** only when removing it moves the
answer further than that, by ``MARGIN``.

This is slow and costs model calls (``len(items) + 2`` per trace), so it is a
separate, offline command and never part of recording or viewing a trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from .tracing import Trace, TraceItem, is_used, overlap_score, tokens

#: How much further than the noise floor an answer must move to count as changed.
MARGIN = 0.15
#: Characters of each item put in front of the model.
ITEM_CHARS = 1500

Ask = Callable[[str, str], str]

PROMPT = (
    "Answer the question using only the context below. If the context does not "
    "contain the answer, say so in one sentence.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def similarity(a: str, b: str) -> float:
    """Word-set Jaccard similarity of two answers, 1.0 for identical word sets."""
    left, right = tokens(a), tokens(b)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def context_of(items: list[TraceItem]) -> str:
    return "\n\n".join(f"[{item.label or item.id}]\n{(item.content or '')[:ITEM_CHARS]}" for item in items)


@dataclass
class ItemVerdict:
    id: str
    label: str | None
    overlap: float | None
    overlap_says_used: bool | None
    answer_moved: float
    needed: bool


@dataclass
class Verification:
    query: str
    baseline: str
    noise: float
    items: list[ItemVerdict] = field(default_factory=list)

    @property
    def agreement(self) -> float | None:
        judged = [item for item in self.items if item.overlap_says_used is not None]
        if not judged:
            return None
        return sum(item.overlap_says_used == item.needed for item in judged) / len(judged)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["agreement"] = self.agreement
        return data


def verify(trace: Trace, ask: Ask, *, margin: float = MARGIN) -> Verification:
    """Run leave-one-out over a trace's retrieved items with ``ask(question, context)``."""
    items = list(trace.items)
    question = trace.query
    baseline = ask(question, context_of(items))
    again = ask(question, context_of(items))
    noise = 1.0 - similarity(baseline, again)

    result = Verification(query=question, baseline=baseline, noise=round(noise, 3))
    for index, item in enumerate(items):
        rest = items[:index] + items[index + 1:]
        without = ask(question, context_of(rest))
        moved = 1.0 - similarity(baseline, without)
        overlap = overlap_score(item.content, baseline) if baseline else None
        result.items.append(ItemVerdict(
            id=item.id,
            label=item.label,
            overlap=None if overlap is None else round(overlap, 3),
            overlap_says_used=is_used(overlap),
            answer_moved=round(moved, 3),
            needed=moved > noise + margin,
        ))
    return result


def openai_ask(model: str, *, api_key: str, base_url: str | None = None, max_tokens: int = 300) -> Ask:
    """An ``ask`` backed by any OpenAI-compatible endpoint, at temperature 0."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    def ask(question: str, context: str) -> str:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=max_tokens,
            messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        )
        return (response.choices[0].message.content or "").strip()

    return ask


def render(result: Verification) -> str:
    lines = [
        f"Question: {result.query}",
        f"Noise floor (two identical runs differ by): {result.noise:.2f}",
        "",
        f"{'item':40}  overlap  overlap-verdict  answer-moved  needed",
    ]
    for item in result.items:
        verdict = "-" if item.overlap_says_used is None else ("used" if item.overlap_says_used else "unused")
        name = (item.label or item.id)[:40]
        lines.append(f"{name:40}  {item.overlap if item.overlap is not None else '-':>7}  {verdict:>15}  "
                     f"{item.answer_moved:>12.2f}  {'yes' if item.needed else 'no'}")
    if result.agreement is not None:
        lines += ["", f"Word overlap agreed with leave-one-out on {result.agreement:.0%} of items."]
    return "\n".join(lines)
