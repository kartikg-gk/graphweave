"""Tests for M3, the used-vs-ignored classifier.

The threshold is pinned here rather than by a live run. Three fixtures do that
work: one item that overlaps the answer heavily, one that shares nothing, and
one built to land exactly on the threshold so the comparison direction is
nailed down and cannot drift silently.

The boundary fixtures use distinct nonsense tokens on purpose — with a
ten-word answer, an item sharing two of those words scores exactly 0.20, which
makes the assertions exact instead of approximate.
"""

from __future__ import annotations

import pytest

from src.tracing import (
    capture,
    DEFAULT_THRESHOLD,
    MIN_TOKEN_LENGTH,
    is_used,
    used_count,
    Trace,
    TraceItem,
    score_overlaps,
    overlap_score,
    tokens,
)


def _trace(*, query: str = "", answer=None, items=(), edges=(), **kwargs):
    """Build a flat trace the way a one-shot producer does.

    Schema 3 nests items under a Retrieval, and ``capture`` is the supported
    way to make one from a flat list — so these tests exercise the real path
    instead of assembling the dataclass by hand.
    """
    return capture(query, items, answer, edges=edges, **kwargs)



# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# Ten distinct content words, none of them stopwords under any list.
#
# That is deliberate and it is a limitation worth naming: because no stopword
# list touches these words, the three boundary fixtures score identically
# under a 20-word list and a 96-word one. They pin the comparison direction
# and the exact cutoff, which is their job, and they say nothing whatever
# about the stopword list.
#
# Making them stopword-sensitive would be a mistake — the arithmetic is only
# exact because each item has exactly ten counted tokens, and a fixture whose
# denominator moves when the list is edited cannot land on 0.20 by
# construction. So the list is covered by a separate fixture below
# (STOPWORD_SENSITIVE_ANSWER) rather than by weakening these.
BOUNDARY_ANSWER = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"


def boundary_item(shared: int) -> str:
    """A ten-token item, ``shared`` of whose tokens appear in the answer.

    The score divides by the ITEM's token count, so the denominator has to be
    fixed on the item side for the arithmetic to be exact. Every item here is
    padded to exactly ten tokens; ``shared`` of them come from the answer and
    the rest are filler that appears nowhere else, so the score is ``shared /
    10`` by construction.

    Padding rather than shortening: an item of ``shared`` tokens and nothing
    else would score 1.0 whatever ``shared`` was, which is the shape that
    makes a boundary fixture useless.
    """
    words = BOUNDARY_ANSWER.split()[:shared]
    filler = [f"filler{n:02d}" for n in range(10 - shared)]
    return " ".join(words + filler)


ABOVE = boundary_item(3)   # 0.30
AT = boundary_item(2)      # 0.20, exactly on the threshold
BELOW = boundary_item(1)   # 0.10

REALISTIC_ANSWER = (
    "Alice rewrote the auth middleware token expiry check, which fixed the "
    "login failures for expired tokens."
)

HEAVY = (
    "Rewrote the auth middleware token expiry check so expired tokens no "
    "longer cause login failures."
)

UNRELATED = "Document how to rotate service account credentials for staging."


def boundary_trace() -> Trace:
    """Items scoring exactly 0.30, 0.20 and 0.10 against the answer."""
    return _trace(
        query="q",
        answer=BOUNDARY_ANSWER,
        items=[
            TraceItem(id="above", content=ABOVE, source="stub"),
            TraceItem(id="at", content=AT, source="stub"),
            TraceItem(id="below", content=BELOW, source="stub"),
        ],
    )


# --------------------------------------------------------------------------
# the score itself
# --------------------------------------------------------------------------


def test_an_item_sharing_nothing_scores_zero():
    assert overlap_score(UNRELATED, REALISTIC_ANSWER) == 0.0


def test_an_item_repeating_the_answer_scores_one():
    assert overlap_score(REALISTIC_ANSWER, REALISTIC_ANSWER) == 1.0


def test_a_heavily_overlapping_item_scores_high():
    assert overlap_score(HEAVY, REALISTIC_ANSWER) > 0.5


def test_the_boundary_fixtures_score_exactly_as_designed():
    assert overlap_score(ABOVE, BOUNDARY_ANSWER) == pytest.approx(0.30)
    assert overlap_score(AT, BOUNDARY_ANSWER) == pytest.approx(0.20)
    assert overlap_score(BELOW, BOUNDARY_ANSWER) == pytest.approx(0.10)


def test_the_boundary_fixture_lands_exactly_on_the_threshold():
    """Not approximately. The comparison direction depends on it being exact."""
    assert overlap_score(AT, BOUNDARY_ANSWER) == DEFAULT_THRESHOLD


def test_scoring_ignores_case():
    assert overlap_score(AT.upper(), BOUNDARY_ANSWER) == pytest.approx(0.20)


def test_scoring_ignores_punctuation():
    punctuated = ", ".join(AT.split()) + "!"

    assert overlap_score(punctuated, BOUNDARY_ANSWER) == pytest.approx(0.20)


def test_repeating_a_word_does_not_inflate_the_score():
    """Tokens are a set, so a repeat changes neither side of the ratio."""
    repeated = BELOW + " " + BELOW.split()[0]

    assert overlap_score(repeated, BOUNDARY_ANSWER) == pytest.approx(0.10)


def test_stopwords_do_not_count_as_overlap():
    assert overlap_score("the and of to with", REALISTIC_ANSWER) == 0.0


def test_an_empty_answer_scores_zero():
    assert overlap_score(HEAVY, "") == 0.0


def test_an_empty_item_scores_zero():
    assert overlap_score("", REALISTIC_ANSWER) == 0.0


# --------------------------------------------------------------------------
# the threshold, applied at read time
# --------------------------------------------------------------------------


def test_the_default_threshold_is_pinned():
    assert DEFAULT_THRESHOLD == 0.2


def test_scoring_records_the_measurement_and_no_verdict():
    """The whole point of schema 2: a trace stores what was measured."""
    trace = score_overlaps(boundary_trace())

    assert [item.overlap for item in trace.items] == pytest.approx([0.30, 0.20, 0.10])
    assert not hasattr(trace.items[0], "used")


def test_an_overlap_exactly_on_the_threshold_counts_as_used():
    assert is_used(0.2, 0.2) is True


def test_an_overlap_just_below_the_threshold_is_ignored():
    assert is_used(0.1, 0.2) is False


def test_an_unmeasured_item_is_unclassified_not_ignored():
    assert is_used(None) is None


def test_the_same_trace_reclassifies_at_a_different_threshold():
    """An archived trace is not frozen at the cutoff current when written."""
    trace = score_overlaps(boundary_trace())

    strict = [is_used(item.overlap, 0.25) for item in trace.items]
    loose = [is_used(item.overlap, 0.05) for item in trace.items]

    assert strict == [True, False, False]
    assert loose == [True, True, True]


def test_used_count_applies_the_threshold_for_a_consumer():
    trace = score_overlaps(boundary_trace())

    assert used_count(trace, 0.2) == 2
    assert used_count(trace, 0.25) == 1
    assert used_count(trace, 0.05) == 3


def test_the_three_deliberate_fixtures_land_where_intended():
    trace = _trace(
        query="q",
        answer=REALISTIC_ANSWER,
        items=[
            TraceItem(id="heavy", content=HEAVY, source="stub"),
            TraceItem(id="none", content=UNRELATED, source="stub"),
        ],
    )

    score_overlaps(trace)

    by_id = {item.id: item for item in trace.items}
    assert is_used(by_id["heavy"].overlap) is True
    assert is_used(by_id["none"].overlap) is False


def test_a_trace_with_no_answer_is_left_unmeasured():
    trace = _trace(query="q", items=[TraceItem(id="a", content=HEAVY, source="stub")])

    score_overlaps(trace)

    assert trace.items[0].overlap is None
    assert is_used(trace.items[0].overlap) is None


def test_a_trace_with_an_empty_answer_is_left_unmeasured():
    trace = _trace(
        query="q", answer="", items=[TraceItem(id="a", content=HEAVY, source="stub")]
    )

    score_overlaps(trace)

    assert trace.items[0].overlap is None


def test_score_overlaps_returns_the_trace_it_was_given():
    trace = boundary_trace()

    assert score_overlaps(trace) is trace


def test_a_trace_with_no_items_scores_cleanly():
    trace = score_overlaps(_trace(query="q", answer=REALISTIC_ANSWER))

    assert trace.items == []


# --------------------------------------------------------------------------
# capture then score_overlaps, the order the pipeline uses
# --------------------------------------------------------------------------


def stub_retriever(query: str) -> list[dict]:
    return [
        {"id": "doc:1", "content": HEAVY, "source": "stub", "score": 0.9},
        {"id": "doc:2", "content": UNRELATED, "source": "stub", "score": 0.4},
    ]


def test_capture_leaves_classification_alone_and_classify_fills_it_in():
    trace = capture("q", stub_retriever("q"), REALISTIC_ANSWER)
    assert all(is_used(item.overlap) is None for item in trace.items)

    score_overlaps(trace)

    assert [is_used(item.overlap) for item in trace.items] == [True, False]


def test_the_viewer_reports_how_many_of_the_retrieved_items_were_used():
    from src.tracing import render

    trace = score_overlaps(capture("q", stub_retriever("q"), REALISTIC_ANSWER))

    assert "1 of 2" in render(trace)


# --------------------------------------------------------------------------
# tokenization
#
# A code corpus lives on identifiers and issue refs. A plain word-character
# split shatters exactly the tokens that carry the signal.
# --------------------------------------------------------------------------


def test_an_identifier_survives_as_one_token():
    """`payment_service` split in two becomes two common, useless words."""
    # One item token, and the answer contains it.
    assert overlap_score("payment_service", "payment_service failed") == 1.0
    # A bare "payment" is a different token, so a word-character split loses it.
    assert overlap_score("payment", "payment_service failed") == 0.0


def test_an_issue_reference_survives_as_one_token():
    # The item has two tokens, fixes and #412, one of which is in the answer.
    assert overlap_score("fixes #412", "see #412 and #999") == pytest.approx(0.5)
    # The bare number is a different token, so a plain digit split loses the ref.
    assert overlap_score("412", "see #412 and #999") == 0.0


def test_a_hyphenated_name_survives_as_one_token():
    assert overlap_score("feature-flag", "feature-flag rollout") == 1.0
    assert overlap_score("feature", "feature-flag rollout") == 0.0


def test_short_tokens_are_dropped():
    """Two-character fragments match everywhere and say nothing."""
    assert overlap_score("id os go", "id os go alpha") == 0.0


def test_a_token_at_the_minimum_length_is_kept():
    assert MIN_TOKEN_LENGTH == 3
    assert overlap_score("abc", "abc") == 1.0


def test_repetition_does_not_inflate_the_score():
    """Tokens are a set: saying a word twenty times is saying it once."""
    once = overlap_score("alpha", "alpha bravo")
    many = overlap_score("alpha alpha alpha alpha", "alpha bravo")

    assert once == many == 1.0


def test_repetition_in_the_answer_does_not_change_the_score():
    assert overlap_score("alpha", "alpha alpha alpha") == 1.0


def test_the_ratio_divides_by_the_smaller_side():
    """A long item that supplied the whole answer scores 1.0, not 1/201."""
    long_item = "alpha " + " ".join(f"unrelated{n}" for n in range(200))

    assert overlap_score(long_item, "alpha") == 1.0


def test_a_short_item_echoed_in_a_long_answer_scores_high():
    answer = "alpha " + " ".join(f"word{n}" for n in range(50))

    assert overlap_score("alpha bravo", answer) == pytest.approx(0.5)


SUPPLIED_ANSWER = "auth middleware rejected expired tokens expiry comparison strict"


def supplying_document(unused_tokens: int) -> str:
    """A file that supplied the entire answer, plus unrelated content."""
    tail = " ".join(f"unrelated_symbol_{n}" for n in range(unused_tokens))
    return f"{SUPPLIED_ANSWER} {tail}"


@pytest.mark.parametrize("unused_tokens", [0, 50, 300, 2000])
def test_a_long_document_that_supplied_the_answer_is_used(unused_tokens):
    """Length no longer decides the verdict: the file supplied every answer token."""
    score = overlap_score(supplying_document(unused_tokens), SUPPLIED_ANSWER)

    assert score == pytest.approx(1.0)
    assert is_used(score) is True


def test_a_long_document_that_supplied_little_is_unused():
    document = "auth " + " ".join(f"unrelated_symbol_{n}" for n in range(300))

    assert overlap_score(document, SUPPLIED_ANSWER) == pytest.approx(1 / 8)
    assert is_used(overlap_score(document, SUPPLIED_ANSWER)) is False


# --------------------------------------------------------------------------
# a fixture that actually depends on the stopword list
#
# The boundary fixtures at the top of this file deliberately avoid stopwords,
# so they score identically under a 20-word list and a 96-word one. This one
# is built the opposite way.
#
# The list-sensitive words live in the ITEM, not the answer. When the item is
# the smaller side the score divides by its token count, so a stopword list can only move the denominator
# by removing item tokens — words dropped from the answer change the
# intersection at most, and usually nothing. An earlier version of this
# fixture put them in the answer and stopped detecting anything the moment the
# denominator changed sides.
# --------------------------------------------------------------------------

#: Plain content words, none of them on any list.
STOPWORD_SENSITIVE_ANSWER = "alpha bravo charlie delta"

#: One answer token, three fillers, then four words a long list removes and a
#: short one keeps. Under the long list the item is 4 tokens and scores 0.25;
#: under the short list it is 8 and scores 0.125.
STOPWORD_SENSITIVE_ITEM = "alpha filler1 filler2 filler3 which about when who"

# Both lists below are SYNTHETIC and deliberately not the shipped ``STOP``.
# They exist to prove the mechanism — that list length moves the denominator
# and can move a verdict — which has to stay true whichever list ships. Do not
# "correct" them to match ``STOP``: doing so couples this test to a choice it
# is not testing, and the next revert would silently gut it. What the project
# actually ships is pinned in ``test_text.py``.
SHORT_LIST = frozenset("a an the and or but if in on at to of for with from by as is are was".split())
LONG_LIST = SHORT_LIST | {"which", "about", "when", "who"}


def scored_with(content: str, answer: str, stop: frozenset[str]) -> float:
    from src.tracing._text import tokens

    answer_tokens = tokens(answer, stop=stop)
    item_tokens = tokens(content, stop=stop)
    if not answer_tokens or not item_tokens:
        return 0.0
    return len(answer_tokens & item_tokens) / len(item_tokens)


def test_the_stopword_list_changes_the_denominator():
    """A longer list removes item tokens, so the score rises."""
    long_score = scored_with(
        STOPWORD_SENSITIVE_ITEM, STOPWORD_SENSITIVE_ANSWER, LONG_LIST
    )
    short_score = scored_with(
        STOPWORD_SENSITIVE_ITEM, STOPWORD_SENSITIVE_ANSWER, SHORT_LIST
    )

    assert long_score == pytest.approx(0.25)
    assert short_score == pytest.approx(0.125)


def test_the_stopword_list_can_flip_a_verdict():
    """The effect that matters: same text, same threshold, different verdict.

    Written against explicit lists rather than the shipped ``STOP`` so it
    keeps testing the mechanism no matter which list ships. What the project
    actually ships is pinned in ``test_text.py``.
    """
    long_score = scored_with(
        STOPWORD_SENSITIVE_ITEM, STOPWORD_SENSITIVE_ANSWER, LONG_LIST
    )
    short_score = scored_with(
        STOPWORD_SENSITIVE_ITEM, STOPWORD_SENSITIVE_ANSWER, SHORT_LIST
    )

    assert is_used(long_score) is True
    assert is_used(short_score) is False


def test_a_stopword_only_item_scores_zero_under_both_lists():
    """The floor does not move with the list."""
    for stop in (SHORT_LIST, LONG_LIST):
        assert scored_with("the and of to", STOPWORD_SENSITIVE_ANSWER, stop) == 0.0
