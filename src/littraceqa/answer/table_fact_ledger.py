"""Immutable, source-attested facts for explicitly requested table rows.

The legacy table assembler merges the first non-null value it sees.  That is
fast, but it loses the distinction between a requested row, a comparison row,
and two repeated leaf headers under different datasets or settings.  This
module is the data-contract layer for a safer opt-in path: requested rows exist
before extraction and every candidate cell retains its complete source path.

Nothing here changes scorer-facing predictions.  Runtime composition and the
development gate decide whether facts may be assembled later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table_assemble import matches_expected_key
from littraceqa.answer.table_route import route_expected_keys
from littraceqa.answer.table_value_contract import build_cell_value_contract


_PAREN_RE = re.compile(r"\(([^()]*)\)")
_QUALIFIER_RE = re.compile(
    r"\b(?:conditional|epoch|inference|ipc|iteration|nfe|parameter|resolution|"
    r"shot|split|step|train|training|unconditional|voting)\w*\b|"
    r"\b\d+(?:\.\d+)?\s*[kmb]\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_HEADER_STOPWORDS = frozenset({
    "a", "an", "and", "at", "by", "for", "from", "in", "of", "on",
    "test", "the", "to", "value", "values", "with", "without",
})
_EXCLUSIVE_HEADER_FAMILIES = (
    frozenset({
        "cifar", "imagenet", "modelnet", "omnidata", "scifact",
        "hotpotqa", "nfcorpus", "climate", "sparc", "cosql",
    }),
)
_SOURCE_TYPES = frozenset({
    "text_span",
    "table",
    "figure",
    "equation_algorithm",
    "citation_context",
})
_OPTIONAL_RESULT_TOKENS = frozenset({"accuracy"})
_GLOBAL_QUALIFIER_RE = re.compile(
    r"\bIPC\s*[=:]\s*\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_OPERATIONAL_PHRASE_RE = re.compile(
    r"\b(?:IPC\s*[=:]\s*\d+(?:\.\d+)?|"
    r"\d+(?:\.\d+)?\s*[kmb]?\s+(?:training\s+)?"
    r"(?:steps?|iterations?|epochs?|NFE)|"
    r"(?:zero|one|two|three|four|five|\d+)[- ]shot)\b",
    re.IGNORECASE,
)
_MODEL_CONTEXT_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+){1,5}\b"
)
_DESCRIPTOR_COLUMN_RE = re.compile(
    r"\b(?:hyperparameter|measurement|metric|property|quantity|statistic)\b",
    re.IGNORECASE,
)
_GENERIC_VALUE_COLUMNS = frozenset({"answer", "result", "value", "values"})
_GENERIC_SINGLE_ALIAS_TOKENS = frozenset({
    "accuracy", "answer", "average", "baseline", "method", "model",
    "result", "score", "setting", "value",
})


def _tuple_text(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value or "").strip() for value in values)


def _normalized_tuple(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(normalize_text(value) for value in values)


def _dedup_tuples(values: Iterable[Sequence[Any]]) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for value in values:
        text = _tuple_text(value)
        normalized = _normalized_tuple(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(text)
    return tuple(out)


def _source_alias_component(source_value: Any, requested_value: Any) -> bool:
    """Whether a concise frozen/source label can anchor one requested key.

    Scorer-key matching is deliberately tolerant, but descriptive planner keys
    can prepend a role (``average IoU gain from ...``) and therefore fail its
    prefix relation.  For *search aliases only*, admit a source label whose
    meaningful tokens are a subset of the requested description.  A one-token
    alias must look like an actual identifier (``PISA``/``ECM-XL``), never a
    generic metric such as ``accuracy``.
    """

    source = str(source_value or "").strip()
    requested = str(requested_value or "").strip()
    if not source or not requested:
        return False
    source_tokens = _token_signature(source)
    requested_tokens = _token_signature(requested)
    if not source_tokens or not source_tokens.issubset(requested_tokens):
        return False
    if len(source_tokens) >= 2:
        return True
    token = next(iter(source_tokens))
    raw_tokens = _TOKEN_RE.findall(source)
    raw = raw_tokens[0] if len(raw_tokens) == 1 else ""
    compound_identifier = (
        len(token) >= 5
        and token not in _GENERIC_SINGLE_ALIAS_TOKENS
        and re.search(
            rf"(?<![A-Za-z0-9]){re.escape(token)}[-/]",
            requested,
            flags=re.IGNORECASE,
        ) is not None
    )
    return (
        len(token) >= 3
        and bool(raw)
        and (
            raw.isupper()
            or "-" in source
            or any(character.isupper() for character in raw[1:])
            or compound_identifier
        )
    )


def _source_alias_matches(
    source_key: Sequence[Any],
    requested_key: Sequence[Any],
    row_key_cols: Sequence[str],
) -> bool:
    if matches_expected_key(source_key, requested_key, row_key_cols):
        return True
    if len(source_key) != len(row_key_cols) or len(requested_key) != len(
        row_key_cols
    ):
        return False
    return all(
        _source_alias_component(source, requested)
        for source, requested in zip(source_key, requested_key, strict=True)
    )


def _unique_source_alias_target(
    source_key: Sequence[Any],
    aliases_by_position: Sequence[Sequence[Sequence[Any]]],
    row_key_cols: Sequence[str],
) -> int | None:
    matches = [
        position
        for position, aliases in enumerate(aliases_by_position)
        if any(
            _source_alias_matches(source_key, alias, row_key_cols)
            for alias in aliases
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _split_identity_and_qualifiers(
    expected_key: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove only clearly operational parentheticals from row identity.

    Parentheses may be identity-bearing (for example a dataset variant), so a
    generic ``strip every parenthetical`` rule is too destructive.  Training
    budgets, steps/NFE, splits, and similar operational constraints are safe to
    store separately and recover the released q_028 ``ECM-XL`` contract.
    """

    identity: list[str] = []
    qualifiers: list[str] = []
    for raw_value in _tuple_text(expected_key):
        removable: list[tuple[int, int]] = []
        for match in _PAREN_RE.finditer(raw_value):
            content = match.group(1).strip()
            if content and _QUALIFIER_RE.search(content):
                qualifiers.append(content)
                removable.append(match.span())
        value = raw_value
        for start, end in reversed(removable):
            value = value[:start] + value[end:]
        cleaned = re.sub(r"\s+", " ", value).strip()
        # A key consisting only of an operational parenthetical is malformed,
        # but retaining the raw key is safer than crashing the answerer or
        # creating an empty wildcard target.
        identity.append(cleaned or raw_value)
    return tuple(identity), tuple(dict.fromkeys(qualifiers))


def _target_id(position: int, raw_key: Sequence[Any]) -> str:
    payload = json.dumps(
        list(_tuple_text(raw_key)), ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"row-{position:03d}-{digest}"


def _source_contains(source: str, value: Any) -> bool:
    needle = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    haystack = unicodedata.normalize("NFKC", source or "").casefold()
    if not needle:
        return False
    needle = re.sub(r"\s+", " ", needle)
    haystack = re.sub(r"\s+", " ", haystack)
    if needle in haystack:
        return True
    # Geometry packets interleave every printed PDF word with its x-coordinate:
    # ``x=300 \"25.8\" | x=331 \"±\" | x=340 \"0.4\"``.  Searching the
    # serialized packet directly therefore cannot prove a visibly contiguous
    # scalar.  Reconstruct only the quoted source words (never model output)
    # and retry the same exact/compact presence test on that printed stream.
    printed_words = re.findall(r'"([^"\n]*)"', source or "")
    if printed_words:
        printed = unicodedata.normalize(
            "NFKC", " ".join(printed_words)
        ).casefold()
        printed = re.sub(r"\s+", " ", printed)
        if needle in printed:
            return True
        percent = re.fullmatch(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))%", needle
        )
        if percent and re.search(
            rf"(?<![\d.]){re.escape(percent.group(1))}\s*%(?!\w)",
            printed,
        ):
            return True
        compact_printed = re.sub(r"[^a-z0-9]+", "", printed)
        compact_needle = re.sub(r"[^a-z0-9]+", "", needle)
        if len(compact_needle) >= 5 and compact_needle in compact_printed:
            return True
    # Recover labels split by PDF line breaks/hyphenation.  Short acronyms and
    # scalar values use the exact path above to avoid accidental substrings.
    compact_needle = re.sub(r"[^a-z0-9]+", "", needle)
    compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
    return len(compact_needle) >= 5 and compact_needle in compact_haystack


def _token_signature(value: Any) -> frozenset[str]:
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(str(value or "").casefold()):
        if token in _HEADER_STOPWORDS:
            continue
        if token in {"iter", "iters", "iteration", "iterations"}:
            token = "iteration"
        elif token in {"acc", "oa", "performance"}:
            token = "accuracy"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token == "nfe":
            token = "step"
        tokens.add(token)
    return frozenset(tokens)


def _question_qualifiers(
    question: str, identity: Sequence[str]
) -> tuple[str, ...]:
    """Recover explicit row-local and global operational constraints.

    Enumeration models sometimes shorten ``ECM-XL (100k iterations)`` to
    ``ECM-XL``.  The row identity should stay concise, but extraction must not
    lose the adjacent budget.  IPC is a table-wide constraint and therefore
    applies to every requested row.
    """

    qualifiers = [
        match.group(0) for match in _GLOBAL_QUALIFIER_RE.finditer(question)
    ]
    for value in identity:
        anchor = str(value or "").strip()
        if not anchor:
            continue
        match = re.search(re.escape(anchor), question, flags=re.IGNORECASE)
        if match is None:
            continue
        suffix = question[match.end():]
        parenthetical = re.match(r"\s*\(([^()]*)\)", suffix)
        if parenthetical is None:
            continue
        content = parenthetical.group(1).strip()
        if content and _QUALIFIER_RE.search(content):
            qualifiers.append(
                re.sub(r"^with\s+", "", content, flags=re.IGNORECASE)
            )
    return tuple(dict.fromkeys(qualifiers))


def _source_covers_tokens(source: str, value: Any) -> bool:
    required = _token_signature(value)
    return bool(required) and required.issubset(_token_signature(source))


def _source_covers_header_tokens(source: str, value: Any) -> bool:
    tokens = _token_signature(value)
    required = tokens - _OPTIONAL_RESULT_TOKENS
    return bool(tokens) and required.issubset(_token_signature(source))


def header_path_compatible(
    header_path: Sequence[str],
    column_name: str,
    required_terms: Sequence[str] = (),
) -> bool:
    """Require the leaf metric and every explicit qualifier in one path."""

    path = tuple(str(item or "").strip() for item in header_path)
    if not path or any(not item for item in path):
        return False
    path_tokens = _token_signature(" ".join(path))
    column_tokens = _token_signature(column_name)
    generic_value_column = normalize_text(column_name) in {"value", "values"}
    if not column_tokens and not generic_value_column:
        return False
    # Generic value columns intentionally carry no metric identity.  The
    # complete source header path and row target must supply that identity;
    # requiring the literal token "value" made every such fact impossible.
    requirements = [
        *(() if generic_value_column else (column_name,)),
        *required_terms,
    ]
    for requirement in requirements:
        tokens = _token_signature(requirement)
        if tokens and not tokens.issubset(path_tokens):
            return False
        for family in _EXCLUSIVE_HEADER_FAMILIES:
            selected = tokens.intersection(family)
            conflicting = path_tokens.intersection(family) - selected
            if selected and conflicting:
                return False
    return bool(path_tokens)


def complete_header_path(
    header_path: Sequence[str],
    column_name: str,
    required_terms: Sequence[str],
    source_text: str,
) -> tuple[str, ...] | None:
    """Restore omitted parent labels only when the same source proves them.

    Readers often return a visible leaf path such as ``Tiny ImageNet -> 10``
    while omitting the parent labels ``Test Performance`` and ``IPC``.  A
    parent may be restored from the bounded source packet only when every
    dataset-family token and numeric child from the requirement is already in
    the returned path.  This keeps sibling paths such as CIFAR-10 and ImageNet
    mutually exclusive.
    """

    completed = tuple(str(item or "").strip() for item in header_path)
    if not completed or any(not item for item in completed):
        return None
    path_tokens = _token_signature(" ".join(completed))
    for requirement in (column_name, *required_terms):
        required = _token_signature(requirement)
        if not required or required.issubset(path_tokens):
            continue
        for family in _EXCLUSIVE_HEADER_FAMILIES:
            selected = required.intersection(family)
            if selected and not selected.issubset(path_tokens):
                return None
            if selected and (path_tokens.intersection(family) - selected):
                return None
        numeric_children = {
            token for token in required if any(char.isdigit() for char in token)
        }
        if not numeric_children.issubset(path_tokens):
            return None
        source_required = required - _OPTIONAL_RESULT_TOKENS
        if not source_required.issubset(_token_signature(source_text)):
            return None
        completed = (*completed, str(requirement).strip())
        path_tokens = _token_signature(" ".join(completed))
    return (
        completed
        if header_path_compatible(completed, column_name, required_terms)
        else None
    )


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    value_type: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("column name must be non-empty")


@dataclass(frozen=True)
class ClaimTuple:
    """Typed question-side context that one source cell must jointly satisfy.

    Row identity alone is insufficient for dense scientific tables: the same
    method can occur under several models, steps, datasets, or quantities.  The
    tuple stays immutable from ledger construction through source admission so
    a short printed alias cannot silently discard those qualifiers.
    """

    model_method: tuple[str, ...] = ()
    dataset_metric: tuple[str, ...] = ()
    step_split_budget: tuple[str, ...] = ()
    quantity: tuple[str, ...] = ()
    numeric_answer_required: bool = False

    @property
    def header_requirements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.model_method, *self.step_split_budget)))

    @property
    def source_requirements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.quantity))


@dataclass(frozen=True)
class RowTarget:
    target_id: str
    expected_key: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...]
    qualifiers: tuple[str, ...]
    owner_papers: tuple[str, ...]
    claim: ClaimTuple = field(default_factory=ClaimTuple)

    def __post_init__(self) -> None:
        if not self.target_id or not self.expected_key or not any(self.expected_key):
            raise ValueError("row target requires an id and non-empty key")
        if not self.aliases:
            raise ValueError("row target requires at least one alias")

    @property
    def header_requirements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.qualifiers, *self.claim.header_requirements)))


@dataclass(frozen=True)
class RowAttestation:
    target_id: str
    paper_id: str
    printed_row_key: tuple[str, ...]
    page: int
    source_type: str
    object_id: str | None
    quote: str
    bbox: tuple[float, float, float, float] | None = None

    @property
    def source_row_key(self) -> tuple[Any, ...]:
        return (
            self.paper_id,
            self.page,
            self.object_id or "",
            _normalized_tuple(self.printed_row_key),
        )


@dataclass(frozen=True)
class CellFact:
    target_id: str
    paper_id: str
    printed_row_key: tuple[str, ...]
    column_name: str
    header_path: tuple[str, ...]
    raw_value: str
    typed_value: object
    page: int
    source_type: str
    object_id: str | None
    quote: str
    bbox: tuple[float, float, float, float] | None
    verifier_families: frozenset[str]
    scorer_candidates: tuple[object, ...] = ()
    unit: str | None = None
    unit_origin: str = "none"

    @property
    def source_row_key(self) -> tuple[Any, ...]:
        return (
            self.paper_id,
            self.page,
            self.object_id or "",
            _normalized_tuple(self.printed_row_key),
        )


@dataclass(frozen=True)
class TableFactLedger:
    row_key_cols: tuple[str, ...]
    value_columns: tuple[ColumnSpec, ...]
    targets: tuple[RowTarget, ...]
    attestations: tuple[RowAttestation, ...] = ()
    facts: tuple[CellFact, ...] = ()

    def __post_init__(self) -> None:
        if not self.row_key_cols or any(not value for value in self.row_key_cols):
            raise ValueError("ledger requires row-key columns")
        if len(set(self.row_key_cols)) != len(self.row_key_cols):
            raise ValueError("row-key columns must be unique")
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target ids must be unique")
        targets_by_id = {target.target_id: target for target in self.targets}
        known_targets = set(targets_by_id)
        known_columns = {column.name for column in self.value_columns}
        if len(known_columns) != len(self.value_columns):
            raise ValueError("value columns must be unique")
        if known_columns.intersection(self.row_key_cols):
            raise ValueError("row-key and value columns must be disjoint")
        for target in self.targets:
            if len(target.expected_key) != len(self.row_key_cols):
                raise ValueError("target key width does not match schema")
            if any(len(alias) != len(self.row_key_cols) for alias in target.aliases):
                raise ValueError("target alias width does not match schema")
        for item in [*self.attestations, *self.facts]:
            if item.target_id not in known_targets:
                raise ValueError(f"unknown target id: {item.target_id}")
            if len(item.printed_row_key) != len(self.row_key_cols):
                raise ValueError("printed row key width does not match schema")
            target = targets_by_id[item.target_id]
            if target.owner_papers and item.paper_id not in target.owner_papers:
                raise ValueError("fact paper is outside the target search scope")
            if not _target_accepts_key(self, target, item.printed_row_key):
                raise ValueError("printed row key does not match its target")
        for fact in self.facts:
            if fact.column_name not in known_columns:
                raise ValueError(f"unknown value column: {fact.column_name}")

    def target(self, target_id: str) -> RowTarget:
        try:
            return next(
                target for target in self.targets if target.target_id == target_id
            )
        except StopIteration as error:
            raise KeyError(target_id) from error

    def column(self, column_name: str) -> ColumnSpec:
        try:
            return next(
                column for column in self.value_columns if column.name == column_name
            )
        except StopIteration as error:
            raise KeyError(column_name) from error

    def with_attestation(self, attestation: RowAttestation) -> "TableFactLedger":
        if attestation in self.attestations:
            return self
        return replace(self, attestations=(*self.attestations, attestation))

    def with_fact(self, fact: CellFact) -> "TableFactLedger":
        if fact in self.facts:
            return self
        return replace(self, facts=(*self.facts, fact))

    def to_dict(self) -> dict[str, Any]:
        def target_dict(item: RowTarget) -> dict[str, Any]:
            return {
                "target_id": item.target_id,
                "expected_key": list(item.expected_key),
                "aliases": [list(alias) for alias in item.aliases],
                "qualifiers": list(item.qualifiers),
                "owner_papers": list(item.owner_papers),
                "claim": asdict(item.claim),
            }

        def attestation_dict(item: RowAttestation) -> dict[str, Any]:
            payload = asdict(item)
            payload["printed_row_key"] = list(item.printed_row_key)
            payload["bbox"] = list(item.bbox) if item.bbox is not None else None
            return payload

        def fact_dict(item: CellFact) -> dict[str, Any]:
            payload = asdict(item)
            payload["printed_row_key"] = list(item.printed_row_key)
            payload["header_path"] = list(item.header_path)
            payload["bbox"] = list(item.bbox) if item.bbox is not None else None
            payload["verifier_families"] = sorted(item.verifier_families)
            return payload

        attestations = sorted(
            self.attestations,
            key=lambda item: (item.source_row_key, item.target_id, item.quote),
        )
        facts = sorted(
            self.facts,
            key=lambda item: (
                item.source_row_key,
                item.target_id,
                item.column_name,
                repr(item.typed_value),
            ),
        )
        return {
            "row_key_cols": list(self.row_key_cols),
            "value_columns": [asdict(column) for column in self.value_columns],
            "targets": [target_dict(target) for target in self.targets],
            "attestations": [attestation_dict(item) for item in attestations],
            "facts": [fact_dict(item) for item in facts],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, payload: str) -> "TableFactLedger":
        raw = json.loads(payload)
        return cls(
            row_key_cols=tuple(raw["row_key_cols"]),
            value_columns=tuple(ColumnSpec(**item) for item in raw["value_columns"]),
            targets=tuple(
                RowTarget(
                    target_id=item["target_id"],
                    expected_key=tuple(item["expected_key"]),
                    aliases=tuple(tuple(alias) for alias in item["aliases"]),
                    qualifiers=tuple(item["qualifiers"]),
                    owner_papers=tuple(item["owner_papers"]),
                    claim=ClaimTuple(**{
                        key: (
                            tuple(value)
                            if key != "numeric_answer_required"
                            else bool(value)
                        )
                        for key, value in item.get("claim", {}).items()
                    }),
                )
                for item in raw["targets"]
            ),
            attestations=tuple(
                RowAttestation(
                    **{
                        **item,
                        "printed_row_key": tuple(item["printed_row_key"]),
                        "bbox": tuple(item["bbox"]) if item["bbox"] is not None else None,
                    }
                )
                for item in raw["attestations"]
            ),
            facts=tuple(
                CellFact(
                    **{
                        **item,
                        "printed_row_key": tuple(item["printed_row_key"]),
                        "header_path": tuple(item["header_path"]),
                        "bbox": tuple(item["bbox"]) if item["bbox"] is not None else None,
                        "verifier_families": frozenset(item["verifier_families"]),
                        "scorer_candidates": tuple(
                            item.get("scorer_candidates", ())
                        ),
                        "unit": item.get("unit"),
                        "unit_origin": item.get("unit_origin", "none"),
                    }
                )
                for item in raw["facts"]
            ),
        )


def build_table_fact_ledger(
    plan,
    ctx,
    *,
    additional_qualifiers: Mapping[tuple[str, ...], Sequence[str]] | None = None,
    include_complete_frozen_rows: bool = False,
) -> TableFactLedger:
    """Create immutable row slots for requested and missing frozen rows.

    Planner labels are search hints, while a frozen scorer row may retain the
    exact label printed in the source (for example ``ground-truth prompts`` vs
    ``w/ Ground Truth``).  A null-bearing frozen row is therefore admitted as
    an alias of a matching requested target.  If no requested target matches,
    it becomes its own bounded repair target.  Non-null frozen rows do not
    expand extraction work merely because a replay floor was supplied.
    """

    if not isinstance(include_complete_frozen_rows, bool):
        raise ValueError("include_complete_frozen_rows must be a boolean")
    raw_keys = [_tuple_text(key) for key in (plan.expected_keys or [])]
    aliases_by_position: list[list[tuple[str, ...]]] = [
        [identity, raw_key]
        for raw_key in raw_keys
        for identity, _qualifiers in [_split_identity_and_qualifiers(raw_key)]
    ]
    value_names = tuple(str(column["name"]) for column in plan.value_cols)
    frozen_rows = [
        row
        for row in (getattr(ctx, "frozen_table_rows", None) or [])
        if isinstance(row, Mapping)
    ]
    for row in frozen_rows:
        frozen_key = _tuple_text(row.get(column) for column in plan.row_key_cols)
        if not frozen_key or any(not value for value in frozen_key):
            continue
        matched_position = _unique_source_alias_target(
            frozen_key, aliases_by_position, plan.row_key_cols
        )
        if matched_position is not None:
            aliases_by_position[matched_position].append(frozen_key)
            continue
        # Complete rows do not normally create extraction work.  A caller that
        # explicitly enables cell replacement must opt in so replacement
        # behavior does not depend on whether the non-deterministic planner
        # happened to enumerate the same frozen row in this run.
        if (
            not include_complete_frozen_rows
            and not any(row.get(column) is None for column in value_names)
        ):
            continue
        raw_keys.append(frozen_key)
        aliases_by_position.append([frozen_key])

    routes = route_expected_keys(ctx, raw_keys) if raw_keys else []
    targets: list[RowTarget] = []
    for position, (raw_key, aliases, route) in enumerate(
        zip(raw_keys, aliases_by_position, routes, strict=True)
    ):
        identity, parsed_qualifiers = _split_identity_and_qualifiers(raw_key)
        extras = tuple(
            str(value).strip()
            for value in (additional_qualifiers or {}).get(raw_key, ())
            if str(value).strip()
        )
        question_qualifiers = _question_qualifiers(
            str(getattr(ctx, "question", "") or ""), identity
        )
        qualifiers = tuple(dict.fromkeys(
            (*parsed_qualifiers, *question_qualifiers, *extras)
        ))
        deduped_aliases = _dedup_tuples((identity, raw_key, *aliases))
        question = str(getattr(ctx, "question", "") or "")
        targets.append(RowTarget(
            target_id=_target_id(position, raw_key),
            expected_key=identity,
            aliases=deduped_aliases,
            qualifiers=qualifiers,
            owner_papers=tuple(route.paper_ids),
            claim=_claim_tuple(
                raw_key=raw_key,
                aliases=deduped_aliases,
                qualifiers=qualifiers,
                question=question,
                row_key_cols=plan.row_key_cols,
                value_columns=plan.value_cols,
                frozen_rows=frozen_rows,
                strict_replacement=include_complete_frozen_rows,
            ),
        ))
    return TableFactLedger(
        row_key_cols=tuple(plan.row_key_cols),
        value_columns=tuple(
            ColumnSpec(
                name=str(column["name"]),
                value_type=str(column.get("type") or "string").casefold(),
            )
            for column in plan.value_cols
        ),
        targets=tuple(targets),
    )


def _target_accepts_key(
    ledger: TableFactLedger, target: RowTarget, printed_row_key: Sequence[Any]
) -> bool:
    return any(
        matches_expected_key(printed_row_key, alias, ledger.row_key_cols)
        for alias in target.aliases
    )


def _valid_bbox(bbox: Sequence[Any] | None) -> bool:
    if bbox is None:
        return True
    if len(bbox) != 4:
        return False
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in values) and (
        values[0] <= values[2] and values[1] <= values[3]
    )


def make_row_attestation(
    ledger: TableFactLedger,
    *,
    target_id: str,
    paper_id: str,
    printed_row_key: Sequence[Any],
    page: int,
    source_type: str,
    object_id: str | None,
    quote: str,
    source_text: str,
    identity_source_text: str | None = None,
    bbox: Sequence[Any] | None = None,
) -> RowAttestation | None:
    """Create an attestation only when the complete printed key is present."""

    try:
        target = ledger.target(target_id)
    except KeyError:
        return None
    printed = _tuple_text(printed_row_key)
    if (
        not paper_id
        or (target.owner_papers and paper_id not in target.owner_papers)
        or not isinstance(page, int)
        or isinstance(page, bool)
        or page < 1
        or source_type not in _SOURCE_TYPES
        or len(printed) != len(ledger.row_key_cols)
        or any(not value for value in printed)
        or not _target_accepts_key(ledger, target, printed)
        or not quote.strip()
        or not _valid_bbox(bbox)
        or not all(
            _source_contains(identity_source_text or source_text, value)
            for value in printed
        )
    ):
        return None
    return RowAttestation(
        target_id=target_id,
        paper_id=paper_id,
        printed_row_key=printed,
        page=page,
        source_type=source_type,
        object_id=str(object_id).strip() if object_id else None,
        quote=quote.strip(),
        bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
    )


def make_cell_fact_with_reasons(
    ledger: TableFactLedger,
    *,
    target_id: str,
    paper_id: str,
    printed_row_key: Sequence[Any],
    column_name: str,
    header_path: Sequence[str],
    raw_value: str,
    typed_value: object,
    page: int,
    source_type: str,
    object_id: str | None,
    quote: str,
    native_packet_text: str,
    verifier_families: Iterable[str],
    row_identity_source_text: str | None = None,
    allow_attributed_row_identity: bool = False,
    required_header_terms: Sequence[str] = (),
    question: str = "",
    bbox: Sequence[Any] | None = None,
) -> tuple[CellFact | None, tuple[str, ...]]:
    """Validate one source-bound cell candidate and explain every rejection.

    The scorer-facing producer still fails closed.  The additional reason tuple
    exists solely so a development gate can distinguish a missing extraction
    from a correct two-view proposal rejected by a later source contract.
    Keeping validation and diagnostics in one function prevents the diagnostic
    path from drifting away from production admission.
    """

    try:
        target = ledger.target(target_id)
    except KeyError:
        return None, ("unknown_target",)
    try:
        column = ledger.column(column_name)
    except KeyError:
        return None, ("unknown_column",)
    printed = _tuple_text(printed_row_key)
    path = tuple(str(value or "").strip() for value in header_path)
    families = frozenset(str(value).strip() for value in verifier_families if str(value).strip())
    contract = build_cell_value_contract(
        raw_value,
        column_type=column.value_type,
        column_name=column_name,
        header_path=path,
        question=question,
    )
    raw = contract.source_literal if contract is not None else ""
    reasons: list[str] = []
    if not paper_id:
        reasons.append("paper_id_missing")
    if target.owner_papers and paper_id not in target.owner_papers:
        reasons.append("paper_outside_target_scope")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        reasons.append("page_invalid")
    if source_type not in _SOURCE_TYPES:
        reasons.append("source_type_invalid")
    if len(printed) != len(ledger.row_key_cols):
        reasons.append("row_key_width")
    elif any(not value for value in printed):
        reasons.append("row_key_empty")
    elif not _target_accepts_key(ledger, target, printed):
        reasons.append("row_key_mismatch")
    if not raw:
        reasons.append("value_contract_invalid")
    if not quote.strip():
        reasons.append("quote_missing")
    if not native_packet_text.strip():
        reasons.append("native_packet_missing")
    if not families:
        reasons.append("verifier_family_missing")
    if not _valid_bbox(bbox):
        reasons.append("bbox_invalid")
    if not header_path_compatible(path, column_name, required_header_terms):
        reasons.append("header_path_incompatible")
    if not isinstance(allow_attributed_row_identity, bool):
        reasons.append("attributed_row_identity_invalid")
    identity_source = row_identity_source_text or native_packet_text
    if (
        not allow_attributed_row_identity
        and not all(_source_contains(identity_source, value) for value in printed)
    ):
        reasons.append("row_key_not_visible")
    if not all(
        _source_covers_header_tokens(native_packet_text, value)
        for value in path
    ):
        reasons.append("header_path_not_visible")
    if raw and not _source_contains(native_packet_text, raw):
        reasons.append("raw_value_not_visible_in_native")
    if raw and not _source_contains(quote, raw):
        reasons.append("raw_value_not_visible_in_quote")
    missing_claim_terms = [
        value
        for value in target.claim.source_requirements
        if not _source_covers_tokens(native_packet_text, value)
    ]
    if missing_claim_terms:
        reasons.append("claim_quantity_not_visible")
    if (
        target.claim.numeric_answer_required
        and normalize_text(column_name) in _GENERIC_VALUE_COLUMNS
        and raw
        and re.search(r"\d", raw) is None
    ):
        reasons.append("claim_numeric_value_required")
    if reasons:
        return None, tuple(reasons)
    if column.value_type == "number":
        coerced = contract.source_value if contract is not None else None
        if (
            coerced is None
            or isinstance(typed_value, bool)
            or not isinstance(typed_value, (int, float))
            or not math.isfinite(float(typed_value))
            or not math.isclose(float(coerced), float(typed_value))
        ):
            return None, ("number_type_mismatch",)
    elif (
        contract is None
        or not isinstance(typed_value, str)
        or not typed_value.strip()
        or normalize_text(typed_value) != normalize_text(contract.source_value)
    ):
        return None, ("string_type_mismatch",)
    return CellFact(
        target_id=target_id,
        paper_id=paper_id,
        printed_row_key=printed,
        column_name=column_name,
        header_path=path,
        raw_value=raw,
        typed_value=typed_value,
        page=page,
        source_type=source_type,
        object_id=str(object_id).strip() if object_id else None,
        quote=quote.strip(),
        bbox=tuple(float(value) for value in bbox) if bbox is not None else None,
        verifier_families=families,
        scorer_candidates=contract.scorer_candidates,
        unit=contract.unit,
        unit_origin=contract.unit_origin.value,
    ), ()


def make_cell_fact(
    ledger: TableFactLedger,
    *,
    target_id: str,
    paper_id: str,
    printed_row_key: Sequence[Any],
    column_name: str,
    header_path: Sequence[str],
    raw_value: str,
    typed_value: object,
    page: int,
    source_type: str,
    object_id: str | None,
    quote: str,
    native_packet_text: str,
    verifier_families: Iterable[str],
    row_identity_source_text: str | None = None,
    allow_attributed_row_identity: bool = False,
    required_header_terms: Sequence[str] = (),
    question: str = "",
    bbox: Sequence[Any] | None = None,
) -> CellFact | None:
    """Validate one source-bound cell candidate; return ``None`` on doubt."""

    fact, _reasons = make_cell_fact_with_reasons(
        ledger,
        target_id=target_id,
        paper_id=paper_id,
        printed_row_key=printed_row_key,
        column_name=column_name,
        header_path=header_path,
        raw_value=raw_value,
        typed_value=typed_value,
        page=page,
        source_type=source_type,
        object_id=object_id,
        quote=quote,
        native_packet_text=native_packet_text,
        verifier_families=verifier_families,
        row_identity_source_text=row_identity_source_text,
        allow_attributed_row_identity=allow_attributed_row_identity,
        required_header_terms=required_header_terms,
        question=question,
        bbox=bbox,
    )
    return fact


def _looks_scalar(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?(?:\s*(?:%|±\s*\d+(?:\.\d+)?))?",
        text,
    ) is not None


def _model_method_context(
    raw_key: Sequence[Any], aliases: Sequence[Sequence[Any]]
) -> tuple[str, ...]:
    exact_alias_components = {
        normalize_text(value)
        for alias in aliases
        for value in alias
        if str(value or "").strip()
    }
    out: list[str] = []
    for value in _tuple_text(raw_key):
        for match in _MODEL_CONTEXT_RE.finditer(value):
            candidate = match.group(0).strip()
            # Model/version context is intentionally narrow.  Plain method
            # aliases such as ECM-XL belong to row identity; Mistral-7B-v0.1
            # and Llama-3.1-8B carry the disambiguating numeric version.
            if (
                not any(character.isdigit() for character in candidate)
                or normalize_text(candidate) in exact_alias_components
            ):
                continue
            out.append(candidate)
    return tuple(dict.fromkeys(out))


def _matching_frozen_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    row_key_cols: Sequence[str],
    aliases: Sequence[Sequence[Any]],
) -> list[Mapping[str, Any]]:
    normalized_aliases = {
        _normalized_tuple(alias)
        for alias in aliases
        if any(str(value or "").strip() for value in alias)
    }
    exact = [
        row
        for row in frozen_rows
        if _normalized_tuple(row.get(column) for column in row_key_cols)
        in normalized_aliases
    ]
    if exact:
        return exact
    matches: list[Mapping[str, Any]] = []
    for row in frozen_rows:
        key = _tuple_text(row.get(column) for column in row_key_cols)
        if any(
            _source_alias_matches(key, alias, row_key_cols)
            for alias in aliases
        ):
            matches.append(row)
    # A tolerant alias is only useful when it identifies one frozen row.  The
    # earlier iCT/iCT-deep failure came from allowing one concise prefix to
    # borrow context from several rows before one-to-one assignment.
    return matches if len(matches) == 1 else []


def _short_alias_context(
    raw_key: Sequence[Any], aliases: Sequence[Sequence[Any]]
) -> tuple[str, ...]:
    raw = _tuple_text(raw_key)
    raw_signature = _token_signature(" ".join(raw))
    if not raw_signature:
        return ()
    shortened = any(
        signature and signature < raw_signature
        for alias in aliases
        for signature in (_token_signature(" ".join(_tuple_text(alias))),)
    )
    return raw if shortened else ()


def _claim_tuple(
    *,
    raw_key: Sequence[Any],
    aliases: Sequence[Sequence[Any]],
    qualifiers: Sequence[str],
    question: str,
    row_key_cols: Sequence[str],
    value_columns: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, Any]],
    strict_replacement: bool,
) -> ClaimTuple:
    operations = [
        match.group(0).strip()
        for value in (*_tuple_text(raw_key), *qualifiers)
        for match in _OPERATIONAL_PHRASE_RE.finditer(value)
    ]
    metrics = [
        str(column.get("name") or "").strip()
        for column in value_columns
        if str(column.get("name") or "").strip()
        and normalize_text(column.get("name")) not in _GENERIC_VALUE_COLUMNS
        and _DESCRIPTOR_COLUMN_RE.search(
            str(column.get("name") or "").replace("_", " ")
        ) is None
    ]
    descriptors: list[str] = []
    matched_frozen = _matching_frozen_rows(frozen_rows, row_key_cols, aliases)
    for row in matched_frozen:
        for column in value_columns:
            name = str(column.get("name") or "").strip()
            value = row.get(name)
            if (
                not name
                or _DESCRIPTOR_COLUMN_RE.search(name.replace("_", " ")) is None
                or not isinstance(value, str)
                or _looks_scalar(value)
            ):
                continue
            descriptors.append(value.strip())
    if strict_replacement and any(
        all(
            row.get(str(column.get("name") or "")) is not None
            for column in value_columns
        )
        for row in matched_frozen
    ):
        descriptors.extend(_short_alias_context(raw_key, aliases))
    requires_numeric = (
        re.search(r"\bhow many\b", question or "", flags=re.IGNORECASE)
        is not None
        and any(
            normalize_text(column.get("name")) in _GENERIC_VALUE_COLUMNS
            for column in value_columns
        )
    )
    return ClaimTuple(
        model_method=_model_method_context(raw_key, aliases),
        dataset_metric=tuple(dict.fromkeys(metrics)),
        step_split_budget=tuple(dict.fromkeys(operations)),
        quantity=tuple(dict.fromkeys(descriptors)),
        numeric_answer_required=requires_numeric,
    )
