"""Source-preserving value contracts for scorer-facing table cells.

Scientific tables express units in three different places: inside the cell
(``80%``), in a header (``Accuracy (%)``), or only in surrounding prose.  A
single string therefore cannot tell us both what the PDF printed and which
representation a schema expects.  This module keeps those concerns separate:

* ``source_value`` is a conservative, transport-normalized transcription;
* ``scorer_candidates`` records deterministic representation alternatives;
* ``value_for`` chooses a policy explicitly, never as an extraction side
  effect.

The production default remains ``source``.  ``schema_canonical`` is an opt-in
development policy: numeric schemas emit a real number and string schemas
whose unit is declared by the column/header emit the bare scalar.
``header_unit_explicit`` is the inverse, equally explicit representation: a
bare numeric string inherits a percent unit printed in its table header.  It
never promotes a unit mentioned only in question prose.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import re
import unicodedata
from typing import Any, Sequence

from littraceqa.answer.scorer_contract import normalize_text
from littraceqa.answer.table import _coerce_number


_PM_RE = re.compile(r"\s*(?:±|\\pm)\s*", re.IGNORECASE)
_PERCENT_SUFFIX_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:%|percent)\s*$",
    re.IGNORECASE,
)
_PERCENT_UNIT_RE = re.compile(r"(?:%|\bpercent(?:age)?\b)", re.IGNORECASE)
_SUPERSCRIPT_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾]+")
_SUPERSCRIPT_CHARS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")",
})
_SCIENTIFIC_MARKS = frozenset("^%±×≈≤≥{}[]\\")
_ALNUM_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class UnitOrigin(enum.Enum):
    NONE = "none"
    CELL = "cell"
    HEADER = "header"
    COLUMN = "column"
    QUESTION = "question"


@dataclass(frozen=True)
class CellValueContract:
    source_literal: str
    source_value: object
    scorer_candidates: tuple[object, ...]
    unit: str | None
    unit_origin: UnitOrigin

    def value_for(self, policy: str = "source") -> object:
        """Return one explicit scorer representation.

        ``source`` is the immutable floor.  ``schema_canonical`` may select a
        different already-recorded candidate only when the schema/header, not
        question prose, declares the unit.
        """

        if policy == "source":
            return self.source_value
        if policy not in {"schema_canonical", "header_unit_explicit"}:
            raise ValueError(
                "cell value policy must be 'source', 'schema_canonical', or "
                "'header_unit_explicit'"
            )
        if policy == "header_unit_explicit":
            if (
                self.unit == "%"
                and self.unit_origin in {UnitOrigin.HEADER, UnitOrigin.COLUMN}
            ):
                explicit = next(
                    (
                        candidate
                        for candidate in self.scorer_candidates
                        if isinstance(candidate, str)
                        and _PERCENT_SUFFIX_RE.fullmatch(candidate) is not None
                    ),
                    None,
                )
                if explicit is not None:
                    return explicit
            return self.source_value
        if (
            self.unit == "%"
            and self.unit_origin in {UnitOrigin.HEADER, UnitOrigin.COLUMN}
        ):
            bare = next(
                (
                    candidate
                    for candidate in self.scorer_candidates
                    if isinstance(candidate, str)
                    and _PERCENT_SUFFIX_RE.fullmatch(candidate) is None
                ),
                None,
            )
            if bare is not None:
                return bare
        return self.source_value


def _expand_unicode_superscripts(value: str) -> str:
    """Convert visible Unicode exponents before NFKC erases their role."""

    def replace(match: re.Match[str]) -> str:
        exponent = match.group(0).translate(_SUPERSCRIPT_CHARS)
        return f"^{exponent}" if len(exponent) == 1 else f"^{{{exponent}}}"

    return _SUPERSCRIPT_RE.sub(replace, value)


def clean_scientific_literal(raw_value: Any) -> str:
    """Transport-normalize a printed value without deleting math semantics.

    Unicode NFKC maps ``²`` to the ordinary digit ``2``.  Doing NFKC first
    therefore changes ``O(K²)`` into ``O(K2)``, which is a different string
    under the official scorer.  Expand visible superscripts first, then apply
    the established whitespace and uncertainty normalization.
    """

    raw = "" if raw_value is None else str(raw_value)
    value = unicodedata.normalize("NFKC", _expand_unicode_superscripts(raw)).strip()
    value = _PM_RE.sub("±", value)
    value = re.sub(r"\s+", " ", value)
    # PyMuPDF can retain a layout gap on either side of a superscript span.
    value = re.sub(r"(?<=[A-Za-z0-9)])\s+\^", "^", value)
    value = re.sub(r"\^\s+", "^", value)
    value = re.sub(r"\s+(?=[),;])", "", value)
    return value


def _clean_literal(raw_value: Any) -> str:
    return clean_scientific_literal(raw_value)


def replacement_preserves_information(
    incumbent: Any,
    candidate: Any,
    *,
    column_type: str = "string",
) -> bool:
    """Reject a replacement that is only a lossy shortening of the floor.

    This is deliberately one-sided.  A genuinely different source scalar is
    still eligible for the existing dual-verifier replacement policy.  The
    guard only blocks candidates whose alphanumeric payload is contained in
    the incumbent while units, compound clauses, or scientific notation have
    disappeared. Numeric evaluator columns are already compared by numeric
    value and do not need string-typography protection.
    """

    if str(column_type or "string").casefold() == "number":
        return True
    if not isinstance(incumbent, str) or not isinstance(candidate, str):
        return True
    old = clean_scientific_literal(incumbent)
    new = clean_scientific_literal(candidate)
    if not old or not new or normalize_text(old) == normalize_text(new):
        return True

    old_atoms = _ALNUM_RE.findall(old.casefold())
    new_atoms = _ALNUM_RE.findall(new.casefold())
    if not old_atoms or not new_atoms:
        return True
    old_compact = "".join(old_atoms)
    new_compact = "".join(new_atoms)

    # ``200 sessions`` -> ``200`` and compound-value truncations are not new
    # facts; they are strictly less informative views of the incumbent.
    strict_payload_subset = (
        len(new_compact) < len(old_compact) and new_compact in old_compact
    )
    # ``O(K^2)`` -> ``O(K2)`` has the same alphanumeric skeleton but loses a
    # scorer-significant scientific mark.
    lost_marks = {
        mark for mark in _SCIENTIFIC_MARKS if mark in old and mark not in new
    }
    same_payload_lost_notation = (
        new_compact == old_compact and bool(lost_marks)
    )
    return not (strict_payload_subset or same_payload_lost_notation)


def _string_source_value(raw_value: Any) -> str | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    if isinstance(raw_value, int):
        return str(raw_value)
    if isinstance(raw_value, float):
        return format(raw_value, ".15g") if math.isfinite(raw_value) else None
    if isinstance(raw_value, str):
        cleaned = _clean_literal(raw_value)
        return cleaned or None
    return None


def _unit_origin(
    source_literal: str,
    column_name: str,
    header_path: Sequence[str],
    question: str,
) -> UnitOrigin:
    if _PERCENT_SUFFIX_RE.fullmatch(source_literal):
        return UnitOrigin.CELL
    if _PERCENT_UNIT_RE.search(" ".join(str(value) for value in header_path)):
        return UnitOrigin.HEADER
    if _PERCENT_UNIT_RE.search(column_name):
        return UnitOrigin.COLUMN
    if _PERCENT_UNIT_RE.search(question):
        return UnitOrigin.QUESTION
    return UnitOrigin.NONE


def _dedup_candidates(values: Sequence[object]) -> tuple[object, ...]:
    output: list[object] = []
    seen: set[tuple[str, object]] = set()
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                continue
            key = ("number", float(value))
        elif isinstance(value, str):
            if not value.strip():
                continue
            key = ("string", normalize_text(value))
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return tuple(output)


def build_cell_value_contract(
    raw_value: Any,
    *,
    column_type: str,
    column_name: str,
    header_path: Sequence[str] = (),
    question: str = "",
) -> CellValueContract | None:
    """Build a typed, source-preserving contract for one printed scalar."""

    literal = _clean_literal(raw_value)
    if not literal:
        return None
    origin = _unit_origin(literal, column_name, header_path, question)
    percent = _PERCENT_SUFFIX_RE.fullmatch(literal)
    bare_percent = percent.group(1) if percent is not None else None
    declared_type = str(column_type or "string").casefold()

    if declared_type == "number":
        # Numeric evaluator columns compare numbers, not unit-bearing strings.
        # Accept a printed percent scalar without changing its magnitude.
        source_value = _coerce_number(
            bare_percent if bare_percent is not None else literal
        )
        if source_value is None:
            return None
        candidates = _dedup_candidates((source_value,))
    else:
        source_value = _string_source_value(raw_value)
        if source_value is None:
            return None
        scalar = bare_percent
        if scalar is None and origin in {
            UnitOrigin.HEADER,
            UnitOrigin.COLUMN,
            UnitOrigin.QUESTION,
        }:
            scalar_match = re.fullmatch(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", source_value
            )
            scalar = scalar_match.group(0) if scalar_match else None
        variants: list[object] = [source_value]
        if scalar is not None:
            variants.extend((scalar, f"{scalar}%"))
        candidates = _dedup_candidates(variants)

    return CellValueContract(
        source_literal=literal,
        source_value=source_value,
        scorer_candidates=candidates,
        unit="%" if origin is not UnitOrigin.NONE else None,
        unit_origin=origin,
    )
