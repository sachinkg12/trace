"""Fail-closed selection of a single reporting paper for a multi-row table.

Some table questions enumerate datasets or settings that are answer rows, not
papers.  This opt-in selector recognizes only the unusually strong case where
one ordinary property-retrieval passage contains every planned row identity and
an informative value concept.  It never retrieves or creates candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping

from littraceqa.retrieval.cardinality import explicit_target_paper_count


_SAFE_SECTION_KINDS = frozenset({"experiments", "results", "other"})
_MULTI_PAPER_RE = re.compile(
    r"\b(?:papers|studies|works|articles|publications)\b", re.IGNORECASE
)
_UNAMBIGUOUS_MULTI_SOURCE_RE = re.compile(
    r"\b(?:respectively|each\s+(?:paper|study|work)|"
    r"different\s+(?:papers|studies|works)|"
    r"(?:from|in|by)\s+[^,;?.]{1,80}\s+(?:paper|study|work)\s+and\s+"
    r"(?:from|in|by)\s+[^,;?.]{1,80}\s+(?:paper|study|work))\b",
    re.IGNORECASE,
)
_COORDINATED_OWNER_PAIR_RE = re.compile(
    r"\b(?:according\s+to|based\s+on|from|in|by|using)\s+"
    r"(?P<left>[^,;?.]{1,60}?)\s+(?:and|as\s+well\s+as)\s+"
    r"(?P<right>[^,;?.]{1,60}?)"
    r"(?=,\s*(?:what|which|how)\b|[;?.]|$)",
    re.IGNORECASE,
)
_TOGETHER_OWNER_PAIR_RE = re.compile(
    r"\b(?:according\s+to|based\s+on|from|in|by|using)\s+"
    r"(?P<left>[^,;?.]{1,60}?)\s+together\s+with\s+"
    r"(?P<right>[^,;?.]{1,60}?)"
    r"(?=,\s*(?:what|which|how)\b|[;?.]|$)",
    re.IGNORECASE,
)
_BETWEEN_OWNER_PAIR_RE = re.compile(
    r"\b(?:between|across)\s+(?P<left>[^,;?.]{1,60}?)\s+and\s+"
    r"(?P<right>[^,;?.]{1,60}?)"
    r"(?=,\s*(?:what|which|how)\b|[;?.]|$)",
    re.IGNORECASE,
)
_COMPARE_OWNER_PAIR_RE = re.compile(
    r"\b(?:compare|comparing)\s+(?P<left>[^,;?.]{1,60}?)\s+"
    r"(?:with|to)\s+(?P<right>[^,;?.]{1,60}?)"
    r"(?=,\s*(?:what|which|how)\b|[;?.]|$)",
    re.IGNORECASE,
)
_VERSUS_OWNER_PAIR_RE = re.compile(
    r"(?:^|[,;]\s*)(?P<left>[^,;?.]{1,60}?)\s+"
    r"(?:versus|vs\.?)\s+(?P<right>[^,;?.]{1,60}?)"
    r"(?=[,;?.]|\s+(?:reports?|shows?|what|which|how)\b|$)",
    re.IGNORECASE,
)
_NON_OWNER_CONTEXT_RE = re.compile(
    r"\b(?:related\s+work|prior\s+work|previous\s+work|baseline|"
    r"reproduced\s+from|adapted\s+from|taken\s+from|reported\s+by|"
    r"copied\s+from|copy\s+of|sourced?\s+from|source\s*:|"
    r"according\s+to|cites?|cited|citation)\b",
    re.IGNORECASE,
)
_AUTHORIAL_TABLE_RE = re.compile(
    r"\b(?:(?:we|our|this\s+(?:paper|study|work))\b[^.]{0,80}\b"
    r"(?:report|present|show|summarize)|"
    r"(?:table|results?)\s+(?:reports?|presents?|shows?))\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"\[[^\]\n]{1,80}\]|"
    r"\([^()\n]{0,80}\b(?:19|20)\d{2}[a-z]?[^()\n]{0,40}\)",
    re.IGNORECASE,
)
_LABEL_NUMBER_RE = re.compile(
    r"\b(?:table|page|figure|fig\.?|appendix)\s*[A-Z]?\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
_CELL_ATOM_RE = re.compile(
    r"(?<![\w.])(?:[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\d*\.\d+)"
    r"(?:\s*%)?|true|false|yes|no|[-–—])(?![\w.])",
    re.IGNORECASE | re.UNICODE,
)
_TABLE_CUE_RE = re.compile(
    r"(?:^|\b)(?:table\s*[A-Z]?\d+(?:\.\d+)*|table\s+caption|caption\s*:)",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(
    # Every sentence punctuation mark is a boundary except a decimal point
    # with a digit on both sides. Thus "3. Results" is bounded but "3.1" is not.
    r"[!?]|(?<!\d)\.|\.(?!\d)",
    re.UNICODE,
)
_POST_GRID_BOUNDARY_RE = re.compile(
    r"(?:\n\s*\n|(?:^|\s)#{1,6}\s*[A-Z]\b|"
    r"\b(?:table|figure)\s+[A-Z]?\d+(?:\.\d+)*\s+"
    r"(?:shows?|presents?|reports?|summarizes?))",
    re.IGNORECASE,
)
_GENERIC_ROW_TOKENS = frozenset({
    "alpha", "baseline", "beta", "candidate", "comparison", "dataset",
    "delta", "epsilon", "evaluation", "first", "framework", "gamma",
    "method", "model", "ours", "proposed", "result", "second", "setting",
    "paper", "publication", "report", "source", "study", "system", "task",
    "test", "third", "variant", "work",
})
_OWNER_ROW_NOUNS = frozenset({
    "article", "paper", "publication", "report", "source", "study", "work"
})
_OWNER_SCHEMA_TOKENS = _OWNER_ROW_NOUNS | frozenset({
    "author", "authors", "bibliography", "citation", "citations", "title",
    "titles", "venue", "venues",
})
_GENERIC_SINGLE_VALUE_NAMES = frozenset({
    "answer", "metric", "number", "result", "score", "value",
})
_NUMERIC_TYPES = frozenset({"float", "integer", "number", "numeric"})
_BOOLEAN_TYPES = frozenset({"bool", "boolean"})
_STRING_TYPES = frozenset({"str", "string", "text"})
_MIN_ABSOLUTE_MARGIN = 5.0
_MIN_RELATIVE_MARGIN = 0.15


@dataclass(frozen=True)
class TableContainerSelection:
    selected_paper_id: str | None
    reason: str
    expected_row_keys: tuple[str, ...] = ()
    value_concepts: tuple[str, ...] = ()
    property_margin: float | None = None
    scores: tuple[dict[str, Any], ...] = ()
    selected_support: dict[str, Any] | None = None

    @property
    def applied(self) -> bool:
        return self.selected_paper_id is not None

    def to_trace(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "selected_paper_id": self.selected_paper_id,
            "expected_row_keys": list(self.expected_row_keys),
            "value_concepts": list(self.value_concepts),
            "property_margin": self.property_margin,
            "scores": [dict(item) for item in self.scores],
            "selected_support": (
                dict(self.selected_support)
                if self.selected_support is not None else None
            ),
        }


@dataclass(frozen=True)
class _ValueColumn:
    name: str
    signature: tuple[str, ...]
    value_type: str
    row_header_signatures: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _TokenSpan:
    token: str
    start: int
    end: int


def _tokens(value: Any) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _contains_sequence(text_tokens: tuple[str, ...], key: tuple[str, ...]) -> bool:
    if not key or len(key) > len(text_tokens):
        return False
    width = len(key)
    return any(
        text_tokens[position:position + width] == key
        for position in range(len(text_tokens) - width + 1)
    )


def _token_spans(text: str) -> tuple[_TokenSpan, ...]:
    return tuple(
        _TokenSpan(
            unicodedata.normalize("NFKC", match.group(0)).casefold(),
            match.start(),
            match.end(),
        )
        for match in re.finditer(r"[^\W_]+", text, flags=re.UNICODE)
    )


def _sequence_spans(
    spans: tuple[_TokenSpan, ...], signature: tuple[str, ...]
) -> tuple[tuple[int, int], ...]:
    if not signature or len(signature) > len(spans):
        return ()
    width = len(signature)
    return tuple(
        (spans[position].start, spans[position + width - 1].end)
        for position in range(len(spans) - width + 1)
        if tuple(item.token for item in spans[position:position + width])
        == signature
    )


def _informative_row_key(value: str) -> tuple[str, ...] | None:
    tokens = _tokens(value)
    if (
        not tokens
        or len(tokens) > 8
        or _OWNER_ROW_NOUNS.intersection(tokens)
    ):
        return None
    informative = tuple(
        token for token in tokens
        if token not in _GENERIC_ROW_TOKENS and len(token) >= 2
    )
    if not informative:
        return None
    if len(tokens) == 1:
        token = informative[0]
        if len(token) < 5 and not any(ord(char) > 127 for char in token):
            return None
    return tokens


def _target_value(target: Any, field: str) -> Any:
    return target.get(field) if isinstance(target, Mapping) else getattr(
        target, field, None
    )


def _expected_rows(plan: Any, question: str) -> tuple[str, ...] | None:
    targets = list(getattr(plan, "targets", ()) or ())
    constraints = [
        str(_target_value(target, "text") or _target_value(target, "key") or "").strip()
        for target in targets
        if str(_target_value(target, "role") or "").casefold() == "constraint"
    ]
    named_methods = list(getattr(plan, "named_methods", ()) or ())
    target_surfaces = [
        str(_target_value(target, "text") or _target_value(target, "key") or "").strip()
        for target in targets
    ]
    # Planner role labels are not owner proof: table rows are often emitted as
    # role=target. Prefer explicit constraint rows, then named rows, then compact
    # target surfaces; the raw-question grammar independently vetoes multi-owner
    # questions before this helper is used.
    raw = (
        constraints if len(constraints) >= 3
        else named_methods if len(named_methods) >= 3
        else target_surfaces
    )
    cleaned = [str(value).strip() for value in raw if str(value).strip()]
    signatures = [_informative_row_key(value) for value in cleaned]
    if len(cleaned) < 3 or any(signature is None for signature in signatures):
        return None
    normalized = [" ".join(signature or ()) for signature in signatures]
    if len(normalized) != len(set(normalized)):
        return None
    concrete = [signature or () for signature in signatures]
    if any(
        left != right and _contains_sequence(right, left)
        for left in concrete
        for right in concrete
    ):
        return None
    question_tokens = _tokens(question)
    if any(
        not _contains_sequence(question_tokens, signature or ())
        for signature in signatures
    ):
        return None
    return tuple(cleaned)


def _has_multi_source_grammar(
    question: str,
    expected_rows: Iterable[str],
) -> bool:
    if _UNAMBIGUOUS_MULTI_SOURCE_RE.search(question or ""):
        return True
    row_signatures = {_tokens(row) for row in expected_rows}
    for match in (
        *_COORDINATED_OWNER_PAIR_RE.finditer(question or ""),
        *_TOGETHER_OWNER_PAIR_RE.finditer(question or ""),
        *_BETWEEN_OWNER_PAIR_RE.finditer(question or ""),
        *_COMPARE_OWNER_PAIR_RE.finditer(question or ""),
        *_VERSUS_OWNER_PAIR_RE.finditer(question or ""),
    ):
        coordinated = (
            _tokens(match.group("left")),
            _tokens(match.group("right")),
        )
        # "in RowA and RowB" can introduce answer rows, not source owners.
        # Exempt only exact admitted row surfaces; partial/fuzzy containment is
        # deliberately insufficient. Serial row lists containing commas do not
        # match this bounded two-owner grammar at all.
        if row_signatures and all(item in row_signatures for item in coordinated):
            continue
        return True
    return False


def _value_concepts(
    schema: list[dict] | None,
    _question: str,
    _expected_rows: Iterable[str],
) -> _ValueColumn | None:
    if not isinstance(schema, list):
        return None
    row_columns = [column for column in schema if column.get("is_row_key")]
    value_columns = [column for column in schema if not column.get("is_row_key")]
    if len(row_columns) != 1 or len(value_columns) != 1:
        return None
    row_header = _tokens(row_columns[0].get("name"))
    if not row_header or _OWNER_SCHEMA_TOKENS.intersection(row_header):
        return None
    row_header_variants = {row_header}
    last = row_header[-1]
    if last.endswith("ies") and len(last) > 3:
        row_header_variants.add((*row_header[:-1], f"{last[:-3]}y"))
    elif last.endswith("s") and len(last) > 3:
        row_header_variants.add((*row_header[:-1], last[:-1]))
    else:
        row_header_variants.add((*row_header[:-1], f"{last}s"))
    column = value_columns[0]
    name = str(column.get("name") or "").strip()
    signature = _tokens(name)
    if not signature:
        return None
    # A single generic header is not an identity. A multi-token header remains
    # usable as an exact unit (for example, "evaluation result"), even if its
    # individual words are generic; there is no question-token fallback.
    if len(signature) == 1 and signature[0] in _GENERIC_SINGLE_VALUE_NAMES:
        return None
    value_type = str(column.get("type") or "").strip().casefold()
    if value_type not in _NUMERIC_TYPES | _BOOLEAN_TYPES | _STRING_TYPES:
        return None
    return _ValueColumn(
        name, signature, value_type, tuple(sorted(row_header_variants))
    )


def _mask_identity_spans(
    text: str,
    signatures: Iterable[tuple[str, ...]],
) -> str:
    spans = _token_spans(text)
    intervals = [
        interval
        for signature in signatures
        for interval in _sequence_spans(spans, signature)
    ]
    if not intervals:
        return text
    chars = list(text)
    for start, end in intervals:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _cell_atoms(
    segment: str,
    identity_signatures: Iterable[tuple[str, ...]],
) -> tuple[tuple[str, ...], bool]:
    cleaned = _mask_identity_spans(segment, identity_signatures)
    for pattern in (_CITATION_RE, _LABEL_NUMBER_RE):
        cleaned = pattern.sub(
            lambda match: " " * (match.end() - match.start()), cleaned
        )
    atoms: list[str] = []
    first_start: int | None = None
    for match in _CELL_ATOM_RE.finditer(cleaned):
        raw = match.group(0).strip()
        numeric = raw.rstrip("%").replace(",", "")
        if numeric.isdigit() and 1900 <= int(numeric) <= 2099:
            continue
        if first_start is None:
            first_start = match.start()
        atoms.append(raw.casefold())
    has_lexical_prefix = bool(
        _tokens(cleaned[:first_start]) if first_start is not None else ()
    )
    return tuple(atoms), has_lexical_prefix


def _local_header_pairs(
    text: str,
    spans: tuple[_TokenSpan, ...],
    value_column: _ValueColumn,
) -> tuple[tuple[int, int], ...]:
    value_headers = _sequence_spans(spans, value_column.signature)
    row_headers = tuple(
        interval
        for signature in value_column.row_header_signatures
        for interval in _sequence_spans(spans, signature)
    )
    pairs: list[tuple[int, int]] = []
    for row_start, row_end in row_headers:
        for value_start, value_end in value_headers:
            if row_start > value_start:
                continue
            start, end = row_start, max(row_end, value_end)
            between = text[start:end]
            if end - start <= 240 and not _SENTENCE_BOUNDARY_RE.search(between):
                pairs.append((start, end))
    return tuple(sorted(set(pairs)))


def _bounded_grid_rows(
    text: str,
    spans: tuple[_TokenSpan, ...],
    row_signatures: tuple[tuple[str, ...], ...],
    value_column: _ValueColumn,
) -> tuple[tuple[tuple[int, int], ...], int] | None:
    occurrences = tuple(
        _sequence_spans(spans, signature)
        for signature in row_signatures
    )
    if any(not items for items in occurrences):
        return None
    for _header_start, header_end in _local_header_pairs(
        text, spans, value_column
    ):
        chosen: list[tuple[int, int]] = []
        cursor = header_end
        for items in occurrences:
            eligible = [item for item in items if item[0] >= cursor]
            if not eligible:
                chosen = []
                break
            selected = eligible[0]
            chosen.append(selected)
            cursor = selected[1]
        if not chosen or chosen[0][0] - header_end > 320:
            continue
        rows = tuple(chosen)
        if any(
            left[1] > right[0] for left, right in zip(rows, rows[1:])
        ):
            continue

        tail_start = rows[-1][1]
        boundary_candidates = [len(text)]
        boundary_match = _POST_GRID_BOUNDARY_RE.search(text, tail_start)
        if boundary_match is not None:
            boundary_candidates.append(boundary_match.start())
        # A later repetition belongs to caption/prose, not the first grid.
        for items in occurrences:
            boundary_candidates.extend(
                start for start, _end in items if start >= tail_start
            )
        grid_end = min(boundary_candidates)
        # Inside the bounded grid, every expected row remains unambiguous.
        if any(
            sum(
                header_end <= start < grid_end
                for start, _end in items
            ) != 1
            for items in occurrences
        ):
            continue
        return rows, grid_end
    return None


def _table_shape_attested(
    text: str,
    row_signatures: tuple[tuple[str, ...], ...],
    value_column: _ValueColumn,
) -> tuple[bool, int]:
    spans = _token_spans(text)
    bounded = _bounded_grid_rows(
        text, spans, row_signatures, value_column
    )
    if bounded is None:
        return False, 0
    ordered, grid_end = bounded
    identities = (
        *row_signatures,
        value_column.signature,
        *value_column.row_header_signatures,
    )
    row_atoms: list[tuple[str, ...]] = []
    for index, (_start, end) in enumerate(ordered):
        next_start = (
            ordered[index + 1][0]
            if index + 1 < len(ordered) else grid_end
        )
        # A bounded row-local interval cannot borrow a cell from distant prose.
        segment = text[end:min(next_start, end + 320)]
        sentence_boundary = _SENTENCE_BOUNDARY_RE.search(segment)
        if sentence_boundary is not None:
            segment = segment[:sentence_boundary.start()]
        atoms, has_lexical_prefix = _cell_atoms(segment, identities)
        if has_lexical_prefix:
            return False, min((len(items) for items in row_atoms), default=0)
        if index < len(ordered) - 1 and len(atoms) < 3:
            return False, min((len(items) for items in row_atoms), default=0)
        if index < len(ordered) - 1 and not any(
            atom not in {"-", "–", "—"} for atom in atoms
        ):
            return False, 0
        row_atoms.append(atoms)
    preceding_widths = [len(items) for items in row_atoms[:-1]]
    if not preceding_widths or max(preceding_widths) - min(preceding_widths) > 1:
        return False, min(preceding_widths, default=0)
    modal_width = sorted(
        Counter(preceding_widths).items(), key=lambda item: (-item[1], item[0])
    )[0][0]
    final_atoms = row_atoms[-1]
    if len(final_atoms) < modal_width:
        return False, len(final_atoms)
    final_atoms = final_atoms[:modal_width]
    if not any(atom not in {"-", "–", "—"} for atom in final_atoms):
        return False, 0
    row_atoms[-1] = final_atoms
    widths = [len(items) for items in row_atoms]
    stable_width = max(widths) - min(widths) <= 1
    stronger_repetition = max(widths) == min(widths) and min(widths) >= 4
    if not stable_width or not (
        _TABLE_CUE_RE.search(text) or stronger_repetition
    ):
        return False, min(widths)
    return True, min(widths)


def _property_signal(candidate: Any):
    signals = [
        signal
        for signal in (getattr(candidate, "route_signals", ()) or ())
        if getattr(signal, "route", None) == "property"
        and getattr(signal, "group_key", None) is None
        and getattr(signal, "role", "target") == "target"
        and isinstance(getattr(signal, "rank", None), int)
        and not isinstance(getattr(signal, "rank", None), bool)
        and isinstance(getattr(signal, "score", None), (int, float))
        and not isinstance(getattr(signal, "score", None), bool)
        and math.isfinite(float(signal.score))
    ]
    return signals[0] if len(signals) == 1 else None


def select_table_container(
    candidates: Iterable[Any],
    plan: Any,
    *,
    question: str,
    answer_types: Iterable[str],
    table_schema: list[dict] | None,
    pool: Any,
    baseline_paper_ids: Iterable[str],
) -> TableContainerSelection:
    """Return one existing full-table container, or an abstention reason."""

    answer_type_list = list(answer_types or ())
    if answer_type_list != ["table"]:
        return TableContainerSelection(None, "not_table_answer")
    expected_rows = _expected_rows(plan, question)
    if (
        (explicit_target_paper_count(question) or 0) > 1
        or _MULTI_PAPER_RE.search(question or "")
        or _has_multi_source_grammar(question, expected_rows or ())
    ):
        return TableContainerSelection(None, "explicit_multiple_papers")
    if expected_rows is None:
        return TableContainerSelection(None, "insufficient_informative_rows")
    value_column = _value_concepts(table_schema, question, expected_rows)
    if value_column is None:
        return TableContainerSelection(
            None, "no_informative_value_concept", expected_rows
        )
    concepts = (value_column.name,)

    row_signatures = tuple(_tokens(value) for value in expected_rows)
    candidate_list = list(candidates)
    baseline_ids = tuple(dict.fromkeys(
        str(paper_id) for paper_id in baseline_paper_ids
    ))
    baseline_id_set = set(baseline_ids)
    property_scores: list[tuple[float, int, str]] = []
    full: list[tuple[Any, Any, dict]] = []
    score_trace: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidate_list):
        signal = _property_signal(candidate)
        if signal is None:
            continue
        if candidate.paper_id not in baseline_id_set or signal.rank > 2:
            continue
        if pool is None or pool.by_id(candidate.paper_id) is None:
            continue
        property_scores.append((float(signal.score), signal.rank, candidate.paper_id))
        best_coverage = 0
        concept_hit = False
        best_local_cells = 0
        authorial_attested = False
        qualifying: list[dict] = []
        for support in (getattr(candidate, "support", ()) or ()):
            if (
                not isinstance(support, Mapping)
                or support.get("source") != "passage"
                or support.get("route") != "property"
                or support.get("group_key") is not None
                or support.get("role", "target") != "target"
                or support.get("section_kind") not in _SAFE_SECTION_KINDS
            ):
                continue
            passage_score = support.get("score")
            if (
                not isinstance(passage_score, (int, float))
                or isinstance(passage_score, bool)
                or not math.isfinite(float(passage_score))
                or not math.isclose(
                    float(passage_score), float(signal.score), rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                continue
            text_tokens = _tokens(support.get("text"))
            raw_text = str(support.get("text") or "")
            if _NON_OWNER_CONTEXT_RE.search(raw_text):
                continue
            coverage = sum(
                _contains_sequence(text_tokens, signature)
                for signature in row_signatures
            )
            passage_concept_hit = _contains_sequence(
                text_tokens, value_column.signature
            )
            table_shape, local_cells = _table_shape_attested(
                raw_text, row_signatures, value_column
            )
            best_coverage = max(best_coverage, coverage)
            concept_hit = concept_hit or passage_concept_hit
            best_local_cells = max(best_local_cells, local_cells)
            authorial_attested = authorial_attested or bool(
                _AUTHORIAL_TABLE_RE.search(raw_text)
            )
            if (
                coverage == len(row_signatures)
                and passage_concept_hit
                and table_shape
            ):
                qualifying.append(dict(support))
        score_trace.append({
            "paper_id": candidate.paper_id,
            "candidate_position": position,
            "property_rank": signal.rank,
            "property_score": float(signal.score),
            "row_coverage": best_coverage,
            "row_count": len(row_signatures),
            "value_concept_hit": concept_hit,
            "local_cell_count": best_local_cells,
            "authorial_attested": authorial_attested,
            "baseline_ranked": candidate.paper_id in baseline_id_set,
            "qualifying_passages": len(qualifying),
        })
        if len(qualifying) == 1:
            full.append((candidate, signal, qualifying[0]))

    compact_trace = tuple(sorted(
        score_trace,
        key=lambda item: (item["property_rank"], item["candidate_position"]),
    )[:10])
    if len(full) != 1:
        return TableContainerSelection(
            None, "full_coverage_not_unique", expected_rows, concepts,
            scores=compact_trace,
        )
    winner, signal, selected_support = full[0]
    if signal.rank != 0:
        return TableContainerSelection(
            None, "full_coverage_not_property_rank_zero", expected_rows,
            concepts, scores=compact_trace,
        )
    ordered_scores = sorted(property_scores, reverse=True)
    if len(ordered_scores) < 2 or ordered_scores[0][2] != winner.paper_id:
        return TableContainerSelection(
            None, "property_lead_not_proven", expected_rows, concepts,
            scores=compact_trace,
        )
    margin = ordered_scores[0][0] - ordered_scores[1][0]
    required_margin = max(
        _MIN_ABSOLUTE_MARGIN,
        abs(ordered_scores[0][0]) * _MIN_RELATIVE_MARGIN,
    )
    if margin < required_margin:
        return TableContainerSelection(
            None, "property_margin_too_small", expected_rows, concepts,
            property_margin=margin, scores=compact_trace,
        )
    return TableContainerSelection(
        winner.paper_id,
        "unique_full_table_container",
        expected_rows,
        concepts,
        property_margin=margin,
        scores=compact_trace,
        selected_support={
            "source": selected_support.get("source"),
            "route": selected_support.get("route"),
            "group_key": selected_support.get("group_key"),
            "role": selected_support.get("role"),
            "chunk_id": selected_support.get("chunk_id"),
            "page": selected_support.get("page"),
            "section_kind": selected_support.get("section_kind"),
            "text_sha256": hashlib.sha256(
                str(selected_support.get("text") or "").encode("utf-8")
            ).hexdigest(),
        },
    )
