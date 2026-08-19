"""Deterministic source objects tied to the exact bytes of one PDF.

The ledger is an identity and extraction boundary, not a relevance model.  It
records physical pages, embedded page labels, typed text/table/caption blocks,
and stable quote anchors without deciding which block answers a question.
Downstream resolvers can therefore ground a quote to a scorer-facing locator
without trusting an LLM-generated page number or silently mixing PDF versions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from littraceqa.answer.scorer_contract import normalize_visible_id
from littraceqa.evidence import normalize_object_id
from littraceqa.localize.pymupdf_runtime import serialized_pymupdf


_CAPTION_RE = re.compile(
    r"^\s*(table|tab\.|figure|fig\.)\s*"
    r"(\d+\.\d+|[A-Z]\d+|\d+[a-z]?)\s*(?=[:.\-–—])",
    re.IGNORECASE,
)
_ALGORITHM_RE = re.compile(
    r"^\s*(algorithm|alg\.)\s*(\d+[a-z]?)\s*(?=[:.\-–—])",
    re.IGNORECASE,
)
_EQUATION_LINE_RE = re.compile(
    r"(?m)^\s*(?=\S)(?=.*(?:=|≤|≥|\+|−|-|∑|\\sum|\\mathcal))"
    r".+?\(\s*(\d+[a-z]?)\s*\)\s*$",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(r"^\s*\[\s*(\d+[a-z]?)\s*\](?!\d)", re.IGNORECASE)
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:references|bibliography)\s*$", re.IGNORECASE
)
_ANCHOR_CHARS = 240
_RAW_SUPERSCRIPT_FLAG = 1


class SourceLedgerBuildError(ValueError):
    """The exact PDF bytes could not produce a trustworthy source ledger."""


@dataclass(frozen=True)
class PdfManifest:
    paper_id: str
    sha256: str
    page_count: int
    source_url: str | None
    parser_name: str
    parser_version: str


@dataclass(frozen=True)
class SourceBlock:
    block_id: str
    paper_id: str
    physical_page: int
    printed_page_label: str | None
    source_type: str
    object_id: str | None
    bbox: tuple[float, float, float, float] | None
    text: str
    normalized_text: str
    prefix: str
    suffix: str


@dataclass(frozen=True)
class SourceLedger:
    manifest: PdfManifest
    blocks: tuple[SourceBlock, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation in deterministic order."""
        return {
            "manifest": asdict(self.manifest),
            "blocks": [asdict(block) for block in self.blocks],
        }

    def to_json(self) -> str:
        """Serialize without platform- or insertion-order-dependent spacing."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceLedger":
        """Rehydrate a ledger while restoring tuple-valued fields."""
        manifest_value = value.get("manifest")
        block_values = value.get("blocks")
        if not isinstance(manifest_value, Mapping) or not isinstance(block_values, list):
            raise SourceLedgerBuildError("invalid serialized source ledger")
        try:
            manifest = PdfManifest(**dict(manifest_value))
            blocks = tuple(
                SourceBlock(
                    **{
                        **dict(item),
                        "bbox": (
                            tuple(item["bbox"])
                            if item.get("bbox") is not None
                            else None
                        ),
                    }
                )
                for item in block_values
                if isinstance(item, Mapping)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceLedgerBuildError("invalid serialized source ledger") from exc
        if len(blocks) != len(block_values):
            raise SourceLedgerBuildError("invalid serialized source ledger block")
        return cls(manifest=manifest, blocks=blocks)


@runtime_checkable
class SourceLedgerBuilder(Protocol):
    def build(
        self,
        paper_id: str,
        pdf_bytes: bytes,
        *,
        source_url: str | None = None,
    ) -> SourceLedger: ...


_BUILDERS: dict[str, Callable[..., SourceLedgerBuilder]] = {}


def register_source_ledger_builder(name: str):
    """Register a source-ledger producer behind the composition seam."""
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("source-ledger builder name must be non-empty")

    def decorate(builder: Callable[..., SourceLedgerBuilder]):
        if clean_name in _BUILDERS:
            raise ValueError(f"source-ledger builder {clean_name!r} already registered")
        _BUILDERS[clean_name] = builder
        return builder

    return decorate


def build_source_ledger_builder(name: str, **kwargs: Any) -> SourceLedgerBuilder:
    """Construct a registered builder; concrete names stay outside callers."""
    if name not in _BUILDERS:
        raise KeyError(
            f"no source-ledger builder registered as {name!r}; have {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](**kwargs)


@dataclass(frozen=True)
class _BlockDraft:
    physical_page: int
    printed_page_label: str | None
    source_type: str
    object_id: str | None
    bbox: tuple[float, float, float, float] | None
    text: str


def normalize_anchor_text(value: Any) -> str:
    """Normalize quote anchors without discarding numeric or math content."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\u00ad", "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def canonical_source_object_id(raw: str, kind: str) -> str:
    """Convert printed label typography to the evaluator's canonical key.

    The vendored evaluator intentionally accepts only narrow clean forms.  A
    physically printed ``Eq. (6)`` or ``Reference [24]`` therefore first goes
    through the submission emitter's typography normalizer, then through the
    evaluator primitive that defines equality.
    """
    if kind not in {"table", "figure", "equation", "algorithm", "citation"}:
        raise ValueError(f"unsupported source object kind: {kind!r}")
    emitted = normalize_object_id(raw, kind)
    scorer_prefix = "equation" if kind == "algorithm" else kind
    return normalize_visible_id(emitted, scorer_prefix)


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _raw_span_text(span: Mapping[str, Any]) -> tuple[str, bool]:
    """Reconstruct one PyMuPDF raw span without flattening superscripts."""

    chars = span.get("chars")
    if isinstance(chars, list):
        text = "".join(
            str(item.get("c") or "")
            for item in chars
            if isinstance(item, Mapping)
        )
    else:
        text = str(span.get("text") or "")
    try:
        superscript = bool(int(span.get("flags", 0)) & _RAW_SUPERSCRIPT_FLAG)
    except (TypeError, ValueError):
        superscript = False
    stripped = text.strip()
    if (
        superscript
        and stripped
        and len(stripped) <= 8
        and re.fullmatch(r"[A-Za-z0-9+\-=()]+", stripped)
    ):
        exponent = f"^{stripped}" if len(stripped) == 1 else f"^{{{stripped}}}"
        text = f"{text[:len(text) - len(text.lstrip())]}{exponent}{text[len(text.rstrip()):]}"
    return text.replace("\x00", ""), superscript


def _lossless_rawdict_blocks(rawdict: Mapping[str, Any]) -> dict[
    tuple[float, float, float, float], str
]:
    """Return block text with character geometry and exponent roles intact.

    PyMuPDF's ordinary ``blocks`` view is still the structural source of
    truth.  This companion view only replaces a block when both APIs expose
    the same rectangle and the same alphanumeric payload.
    """

    output: dict[tuple[float, float, float, float], str] = {}
    for block in rawdict.get("blocks", ()):
        if not isinstance(block, Mapping) or block.get("type", 0) != 0:
            continue
        block_bbox = _bbox(block.get("bbox"))
        if block_bbox is None:
            continue
        lines: list[str] = []
        for line in block.get("lines", ()):
            if not isinstance(line, Mapping):
                continue
            pieces: list[str] = []
            previous_bbox: tuple[float, float, float, float] | None = None
            for span in line.get("spans", ()):
                if not isinstance(span, Mapping):
                    continue
                text, superscript = _raw_span_text(span)
                if not text:
                    continue
                span_bbox = _bbox(span.get("bbox"))
                if (
                    pieces
                    and not superscript
                    and not pieces[-1].endswith((" ", "\t"))
                    and not text.startswith((" ", "\t", ")", "]", "}", ",", ".", ";", ":"))
                    and previous_bbox is not None
                    and span_bbox is not None
                    and span_bbox[0] - previous_bbox[2] > 1.0
                ):
                    pieces.append(" ")
                pieces.append(text)
                if span_bbox is not None:
                    previous_bbox = span_bbox
            line_text = "".join(pieces).strip()
            if line_text:
                lines.append(line_text)
        text = "\n".join(lines).strip()
        if text:
            output[block_bbox] = text
    return output


def _prefer_lossless_text(
    ordinary: str,
    bbox: tuple[float, float, float, float] | None,
    lossless_blocks: Mapping[tuple[float, float, float, float], str],
) -> str:
    if bbox is None:
        return ordinary
    candidate = lossless_blocks.get(bbox)
    if not candidate:
        return ordinary
    ordinary_payload = re.sub(r"[^A-Za-z0-9]+", "", ordinary).casefold()
    candidate_payload = re.sub(r"[^A-Za-z0-9]+", "", candidate).casefold()
    if ordinary_payload != candidate_payload:
        return ordinary
    candidate = re.sub(r"(?<=[A-Za-z0-9)])\s+\^", "^", candidate)
    candidate = re.sub(r"\^\s+", "^", candidate)
    candidate = re.sub(r"\s+(?=[),;])", "", candidate)
    return candidate


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        result = tuple(round(float(item), 3) for item in value)
    except (TypeError, ValueError):
        return None
    return result  # type: ignore[return-value]


def _caption_identity(text: str) -> tuple[str, str] | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    match = _CAPTION_RE.match(first_line)
    if match is None:
        return None
    label, number = match.groups()
    if label.casefold().startswith("tab"):
        return "table", canonical_source_object_id(f"Table {number}", "table")
    return "figure", canonical_source_object_id(f"Figure {number}", "figure")


def _equation_identity(text: str) -> str | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    algorithm = _ALGORITHM_RE.match(first_line)
    if algorithm is not None:
        return canonical_source_object_id(
            f"Algorithm {algorithm.group(2)}", "algorithm"
        )
    equation = _EQUATION_LINE_RE.search(text)
    if equation is not None:
        return canonical_source_object_id(
            f"Equation {equation.group(1)}", "equation"
        )
    return None


def _citation_identity(text: str, references_started: bool) -> str | None:
    if not references_started:
        return None
    match = _REFERENCE_RE.match(text)
    if match is None:
        return None
    return canonical_source_object_id(f"Citation {match.group(1)}", "citation")


def _typed_text_draft(
    *,
    physical_page: int,
    printed_page_label: str | None,
    bbox: tuple[float, float, float, float] | None,
    text: str,
    references_started: bool,
) -> _BlockDraft:
    caption = _caption_identity(text)
    if caption is not None:
        source_type, object_id = caption
    else:
        citation_id = _citation_identity(text, references_started)
        equation_id = _equation_identity(text)
        if citation_id is not None:
            source_type, object_id = "citation_context", citation_id
        elif equation_id is not None:
            source_type, object_id = "equation_algorithm", equation_id
        else:
            source_type, object_id = "text_span", None
    return _BlockDraft(
        physical_page=physical_page,
        printed_page_label=printed_page_label,
        source_type=source_type,
        object_id=object_id,
        bbox=bbox,
        text=text,
    )


def _reference_entry_drafts(
    *,
    physical_page: int,
    printed_page_label: str | None,
    bbox: tuple[float, float, float, float] | None,
    text: str,
    references_started: bool,
) -> tuple[list[_BlockDraft], bool]:
    """Split reference entries only after a physically visible heading.

    PDF text extraction often places a ``References`` heading and many
    bibliography entries in one block.  Keeping that whole block would expose
    only the first visible citation ID.  This narrow splitter preserves each
    entry and leaves ordinary prose blocks intact.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    contains_heading = any(_REFERENCES_HEADING_RE.fullmatch(line) for line in lines)
    citation_starts = sum(_REFERENCE_RE.match(line) is not None for line in lines)
    if not contains_heading and (not references_started or citation_starts < 2):
        return [
            _typed_text_draft(
                physical_page=physical_page,
                printed_page_label=printed_page_label,
                bbox=bbox,
                text=text,
                references_started=references_started,
            )
        ], references_started

    output: list[_BlockDraft] = []
    current: list[str] = []
    in_references = references_started

    def flush() -> None:
        if not current:
            return
        entry = "\n".join(current)
        output.append(_typed_text_draft(
            physical_page=physical_page,
            printed_page_label=printed_page_label,
            bbox=bbox,
            text=entry,
            references_started=in_references,
        ))
        current.clear()

    for line in lines:
        if _REFERENCES_HEADING_RE.fullmatch(line):
            flush()
            in_references = True
            output.append(_typed_text_draft(
                physical_page=physical_page,
                printed_page_label=printed_page_label,
                bbox=bbox,
                text=line,
                references_started=True,
            ))
            continue
        if in_references and _REFERENCE_RE.match(line):
            flush()
        current.append(line)
    flush()
    return output, in_references


def _table_text(table: Any) -> str:
    try:
        rows = table.extract()
    except Exception:  # noqa: BLE001 -- malformed optional table fails closed
        return ""
    if not isinstance(rows, (list, tuple)):
        return ""
    output: list[str] = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        values = [
            # PDF glyph grouping often splits one decimal inside a cell as
            # ``93.\n8`` or ``93 . 8``.  This is transport normalization of
            # adjacent numeric glyphs, not a model inference; preserving the
            # split would make an exact printed scalar impossible to prove.
            re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", _clean_text(value))
            for value in row
        ]
        if any(values):
            output.append("\t".join(values))
    return "\n".join(output).strip()


def _nearby_table_caption(
    table_bbox: tuple[float, float, float, float] | None,
    captions: list[_BlockDraft],
) -> str | None:
    if table_bbox is None or not captions:
        return None
    x0, y0, x1, y1 = table_bbox

    def distance(block: _BlockDraft) -> tuple[float, float, str]:
        assert block.bbox is not None
        bx0, by0, bx1, by1 = block.bbox
        horizontal_gap = max(0.0, max(x0, bx0) - min(x1, bx1))
        vertical_gap = min(abs(y0 - by1), abs(by0 - y1))
        return horizontal_gap, vertical_gap, block.object_id or ""

    candidates = [
        block
        for block in captions
        if block.source_type == "table" and block.bbox is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=distance)
    horizontal_gap, vertical_gap, _object_id = distance(best)
    if horizontal_gap > 72 or vertical_gap > 180:
        return None
    return best.object_id


def _native_table_drafts(
    page: Any,
    *,
    physical_page: int,
    printed_page_label: str | None,
    captions: list[_BlockDraft],
) -> list[_BlockDraft]:
    output: list[_BlockDraft] = []
    seen: set[tuple[tuple[float, float, float, float] | None, str]] = set()
    for strategy in ("lines", "text"):
        try:
            finder = page.find_tables(
                strategy=strategy,
                min_words_vertical=2,
                min_words_horizontal=1,
            )
        except Exception:  # noqa: BLE001 -- one detector is optional
            continue
        for table in getattr(finder, "tables", ()):
            text = _table_text(table)
            table_bbox = _bbox(getattr(table, "bbox", None))
            if not text:
                continue
            object_id = _nearby_table_caption(table_bbox, captions)
            # Borderless text detection can segment ordinary prose as a grid.
            # Require a caption unless line geometry independently supports it.
            if strategy == "text" and object_id is None:
                continue
            key = (table_bbox, normalize_anchor_text(text))
            if key in seen:
                continue
            seen.add(key)
            output.append(_BlockDraft(
                physical_page=physical_page,
                printed_page_label=printed_page_label,
                source_type="table",
                object_id=object_id,
                bbox=table_bbox,
                text=text,
            ))
    return output


def _page_label(page: Any, has_embedded_labels: bool) -> str | None:
    if not has_embedded_labels:
        return None
    try:
        label = str(page.get_label() or "").strip()
    except Exception:  # noqa: BLE001 -- optional embedded metadata
        return None
    return label or None


def _draft_sort_key(block: _BlockDraft) -> tuple[Any, ...]:
    bbox = block.bbox or (float("inf"),) * 4
    return (
        block.physical_page,
        bbox[1],
        bbox[0],
        block.source_type,
        block.object_id or "",
        normalize_anchor_text(block.text),
    )


def _block_id(paper_id: str, block: _BlockDraft, ordinal: int) -> str:
    identity = json.dumps(
        {
            "paper_id": paper_id,
            "physical_page": block.physical_page,
            "source_type": block.source_type,
            "object_id": block.object_id,
            "bbox": block.bbox,
            "text": normalize_anchor_text(block.text),
            "ordinal": ordinal,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{paper_id}:{block.physical_page}:{digest}"


def _materialize_blocks(
    paper_id: str, drafts: list[_BlockDraft]
) -> tuple[SourceBlock, ...]:
    ordered = sorted(drafts, key=_draft_sort_key)
    output: list[SourceBlock] = []
    for position, block in enumerate(ordered):
        previous_block = ordered[position - 1] if position > 0 else None
        following_block = ordered[position + 1] if position + 1 < len(ordered) else None
        previous = (
            previous_block.text
            if previous_block is not None
            and previous_block.physical_page == block.physical_page
            else ""
        )
        following = (
            following_block.text
            if following_block is not None
            and following_block.physical_page == block.physical_page
            else ""
        )
        output.append(SourceBlock(
            block_id=_block_id(paper_id, block, position),
            paper_id=paper_id,
            physical_page=block.physical_page,
            printed_page_label=block.printed_page_label,
            source_type=block.source_type,
            object_id=block.object_id,
            bbox=block.bbox,
            text=block.text,
            normalized_text=normalize_anchor_text(block.text),
            prefix=normalize_anchor_text(previous)[-_ANCHOR_CHARS:],
            suffix=normalize_anchor_text(following)[:_ANCHOR_CHARS],
        ))
    return tuple(output)


@register_source_ledger_builder("pymupdf")
class PyMuPdfSourceLedgerBuilder:
    """Build a source ledger using only the supplied PDF byte snapshot."""

    def __init__(self, *, include_native_tables: bool = True):
        if not isinstance(include_native_tables, bool):
            raise ValueError("include_native_tables must be boolean")
        self._include_native_tables = include_native_tables

    @serialized_pymupdf
    def build(
        self,
        paper_id: str,
        pdf_bytes: bytes,
        *,
        source_url: str | None = None,
    ) -> SourceLedger:
        clean_paper_id = str(paper_id or "").strip()
        if not clean_paper_id:
            raise SourceLedgerBuildError("paper_id must be non-empty")
        if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
            raise SourceLedgerBuildError("pdf_bytes must be non-empty bytes")
        snapshot = bytes(pdf_bytes)
        try:
            import fitz

            with fitz.open(stream=snapshot, filetype="pdf") as document:
                if document.page_count < 1:
                    raise SourceLedgerBuildError("PDF contains no pages")
                try:
                    has_embedded_labels = bool(document.get_page_labels())
                except Exception:  # noqa: BLE001 -- optional embedded metadata
                    has_embedded_labels = False
                drafts: list[_BlockDraft] = []
                references_started = False
                for page_index in range(document.page_count):
                    page = document.load_page(page_index)
                    physical_page = page_index + 1
                    printed_page_label = _page_label(page, has_embedded_labels)
                    page_drafts: list[_BlockDraft] = []
                    try:
                        lossless_blocks = _lossless_rawdict_blocks(
                            page.get_text("rawdict", sort=True)
                        )
                    except Exception:  # noqa: BLE001 -- optional rich text view
                        lossless_blocks = {}
                    raw_blocks = page.get_text("blocks", sort=True)
                    for raw in raw_blocks:
                        if not isinstance(raw, (list, tuple)) or len(raw) < 5:
                            continue
                        block_bbox = _bbox(raw[:4])
                        text = _prefer_lossless_text(
                            _clean_text(raw[4]), block_bbox, lossless_blocks
                        )
                        if not text:
                            continue
                        typed, references_started = _reference_entry_drafts(
                            physical_page=physical_page,
                            printed_page_label=printed_page_label,
                            bbox=block_bbox,
                            text=text,
                            references_started=references_started,
                        )
                        page_drafts.extend(typed)
                    captions = [
                        block
                        for block in page_drafts
                        if block.source_type in {"table", "figure"}
                    ]
                    if self._include_native_tables:
                        page_drafts.extend(_native_table_drafts(
                            page,
                            physical_page=physical_page,
                            printed_page_label=printed_page_label,
                            captions=captions,
                        ))
                    drafts.extend(page_drafts)
                blocks = _materialize_blocks(clean_paper_id, drafts)
                parser_version = str(getattr(fitz, "VersionBind", "unknown"))
                page_count = int(document.page_count)
        except SourceLedgerBuildError:
            raise
        except Exception as exc:  # noqa: BLE001 -- invalid snapshot fails closed
            raise SourceLedgerBuildError(
                f"could not parse PDF snapshot for {clean_paper_id!r}"
            ) from exc
        manifest = PdfManifest(
            paper_id=clean_paper_id,
            sha256=hashlib.sha256(snapshot).hexdigest(),
            page_count=page_count,
            source_url=(str(source_url).strip() or None) if source_url else None,
            parser_name=(
                "pymupdf-source-ledger+native-tables"
                if self._include_native_tables
                else "pymupdf-source-ledger-text-only"
            ),
            parser_version=parser_version,
        )
        return SourceLedger(manifest=manifest, blocks=blocks)
