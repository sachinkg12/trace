"""Conservatively route requested table rows to their owning papers.

Multi-paper comparison questions normally ask for one value from each method's
own paper.  Broadcasting every requested row to every selected paper is unsafe:
a paper's related-work comparison table can contain the same method with a
different setting, and first-non-null assembly may then keep that comparator
value instead of the value reported by the method's own paper.

This module performs a deterministic, fail-open routing step.  A row is assigned
to one paper only when the paper title or its early pages identify that row with
a clear margin.  Ambiguous or unsupported rows remain assigned to every paper,
preserving the pre-routing recall floor.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from littraceqa.answer.scorer_contract import normalize_text


_TITLE_SCORE = 10_000
_TITLE_ACRONYM_SCORE = 9_000
_EARLY_PAGE_SCORES = (4_000, 2_500, 1_500)
_MIN_ROUTE_SCORE = 1_500
_MIN_ROUTE_MARGIN = 500
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TITLE_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
    "the", "to", "towards", "via", "with",
}


@dataclass(frozen=True)
class RowRoute:
    """One expected row's conservative routing decision."""

    expected_key: tuple[str, ...]
    paper_ids: tuple[str, ...]
    status: str
    scores: dict[str, int]


def _label_aliases(expected_key: tuple[str, ...]) -> list[str]:
    """Return precise-to-broad aliases for the row's primary entity label."""

    raw = next((value.strip() for value in expected_key if value.strip()), "")
    if not raw:
        return []
    without_qualifier = _PAREN_RE.sub("", raw).strip()
    candidates = [raw, without_qualifier]

    # Descriptive planner labels often append a role after the named method
    # (``StreamNet bridge layers``). The leading printed entity is useful for
    # routing, but only when it is identifier-like; ordinary prose must not be
    # reduced to a generic first word.
    first = without_qualifier.split()[0] if without_qualifier.split() else ""
    identifier_like = (
        any(character.isupper() for character in first[1:])
        or "-" in first
        or first.isupper()
    )
    if identifier_like:
        candidates.append(first)

    # A qualified model family (``ECM-XL``) can be introduced in its own paper
    # as the base method (``ECM``).  Restrict this alias to an acronym-like
    # prefix so compound names such as ``Vision-LM-0.5`` never collapse to a
    # generic first token.
    prefix = re.split(r"[-/]", first, maxsplit=1)[0]
    if len(prefix) >= 3 and prefix.isupper():
        candidates.append(prefix)

    aliases: list[str] = []
    for candidate in candidates:
        normalized = normalize_text(candidate)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _acronym(text: str) -> str:
    tokens = [
        token
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _TITLE_STOPWORDS
    ]
    return "".join(token[0].casefold() for token in tokens)


def _title_acronyms(title: str) -> set[str]:
    """High-precision acronyms for the full title and its pre-colon name.

    Arbitrary title-token windows are deliberately excluded.  For example,
    ``time consistency models`` appears as a suffix in several papers and
    would falsely make all of them owners of ``TCM``; only the actual title
    ``Truncated Consistency Models`` has the full-title acronym ``TCM``.
    """

    out = {_acronym(title)}
    if ":" in title:
        out.add(_acronym(title.split(":", 1)[0]))
    return {candidate for candidate in out if len(candidate) >= 3}


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack
    ) is not None


def _paper_score(ctx, paper_id: str, aliases: Iterable[str]) -> int:
    aliases = list(aliases)
    if not aliases:
        return 0
    title = str((getattr(ctx, "paper_titles", None) or {}).get(paper_id) or "")
    normalized_title = normalize_text(title)
    score = 0
    if any(_contains_phrase(normalized_title, alias) for alias in aliases):
        score = max(score, _TITLE_SCORE)

    acronyms = _title_acronyms(title)
    for alias in aliases:
        compact = re.sub(r"[^a-z0-9]+", "", alias)
        if len(compact) >= 3 and compact in acronyms:
            score = max(score, _TITLE_ACRONYM_SCORE)

    parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
    pages = list(getattr(parsed, "pages", None) or [])
    for page_score, page in zip(_EARLY_PAGE_SCORES, pages[:3]):
        normalized = normalize_text(getattr(page, "text", "") or "")
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            score = max(score, page_score)
    return score


def route_expected_keys(ctx, expected_keys: list[tuple[str, ...]]) -> list[RowRoute]:
    """Route each expected key, preserving all papers when confidence is low.

    The result always covers every expected key.  A confident decision contains
    one paper id; a fail-open decision contains the complete input paper list.
    """

    paper_ids = tuple(dict.fromkeys(getattr(ctx, "paper_ids", None) or []))
    if len(paper_ids) <= 1:
        return [
            RowRoute(key, paper_ids, "single_paper", {pid: 0 for pid in paper_ids})
            for key in expected_keys
        ]

    routes: list[RowRoute] = []
    for key in expected_keys:
        aliases = _label_aliases(key)
        scores = {pid: _paper_score(ctx, pid, aliases) for pid in paper_ids}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], paper_ids.index(item[0])))
        top_id, top_score = ranked[0]
        second_score = ranked[1][1]
        if (
            aliases
            and top_score >= _MIN_ROUTE_SCORE
            and top_score - second_score >= _MIN_ROUTE_MARGIN
        ):
            routes.append(RowRoute(key, (top_id,), "owned", scores))
        else:
            routes.append(RowRoute(key, paper_ids, "ambiguous", scores))
    return routes


def expected_keys_by_paper(ctx, expected_keys: list[tuple[str, ...]]):
    """Return per-paper expected keys plus JSON-friendly diagnostics."""

    routes = route_expected_keys(ctx, expected_keys)
    paper_ids = list(dict.fromkeys(getattr(ctx, "paper_ids", None) or []))
    by_paper: dict[str, list[tuple[str, ...]]] = {pid: [] for pid in paper_ids}
    for route in routes:
        for paper_id in route.paper_ids:
            by_paper[paper_id].append(route.expected_key)
    diagnostics = [
        {
            "expected_key": list(route.expected_key),
            "paper_ids": list(route.paper_ids),
            "status": route.status,
            "scores": dict(route.scores),
        }
        for route in routes
    ]
    return by_paper, diagnostics


def source_attested_expected_keys(
    ctx,
    expected_keys_by_paper: dict[str, list[tuple[str, ...]]],
) -> list[tuple[str, ...]]:
    """Expected keys printed together in an assigned paper source.

    This is the precision boundary for null-cell row recovery. Every component
    must be known and token-boundary present in the paper title or on one parsed
    page. Components spread across different pages are not treated as one
    scorer row, and an empty composite component never acts as a wildcard.
    Output order follows selected-paper order and then planned-key order.
    """

    attested: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    paper_order = list(dict.fromkeys(getattr(ctx, "paper_ids", None) or []))
    for paper_id in paper_order:
        title = str((getattr(ctx, "paper_titles", None) or {}).get(paper_id) or "")
        parsed = (getattr(ctx, "parsed_by_id", None) or {}).get(paper_id)
        sources = [normalize_text(title)]
        sources.extend(
            normalize_text(getattr(page, "text", "") or "")
            for page in (getattr(parsed, "pages", None) or [])
        )
        for expected_key in expected_keys_by_paper.get(paper_id, []):
            normalized = tuple(normalize_text(value) for value in expected_key)
            if not normalized or any(not value for value in normalized):
                continue
            if expected_key in seen:
                continue
            if any(
                all(_contains_phrase(source, value) for value in normalized)
                for source in sources
            ):
                seen.add(expected_key)
                attested.append(expected_key)
    return attested
