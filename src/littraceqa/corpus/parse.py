"""Level-1 corpus parse: PDF bytes -> ONE deterministic parsed artifact.

No LLM, no vision. Turns each downloaded PDF into a single structured JSON
artifact (pages, sections, captions, acronyms, references) that the higher
passages/objects/aliases/citations indexes are built from. Everything here is
deterministic (regex + PyMuPDF page text) so the same PDF always yields the
same artifact -- a hard requirement for a reproducible submission.

Design mirrors the corpus DOWNLOADER (downloader.py / gcs_backend.py) so the
same run works local OR streams artifacts to the bucket:

  * PDF-source seam (`PdfSource`): where PDF bytes are READ from --
    `LocalDirPdfSource` (a dir of `{paper_id}.pdf`) now, `GcsPdfSource`
    (reads `gs://<bucket>/<pdf_prefix>/{paper_id}.pdf` through a `GcsBackend`)
    for the VM. DIP seam, injected into `CorpusParser`.
  * Artifact-sink seam (`ArtifactSink`): where the artifact JSON is WRITTEN --
    `LocalArtifactSink` (`<dir>/{paper_id}.json`) or `GcsArtifactSink`
    (`gs://<bucket>/<parsed_prefix>/{paper_id}.json`). Distinct from the
    download `StorageBackend` because that one hardcodes a `.pdf` suffix; this
    writes `.json`. Same exists/write shape so resume works either way.
  * Parse manifest: the SAME append/flush-safe `Manifest` / `GcsManifest` as the
    downloader (they're generic paper_id-keyed logs), so a 27k parse resumes.

Resume + versioning: a paper is skipped ONLY when the manifest says it was
`parsed` at the CURRENT `PARSER_VERSION`. A `failed` paper, a missing row, or a
`parser_version` bump all re-parse. Total-degradation: any single sub-extraction
that raises degrades to `[]`; a whole-PDF extraction failure returns a minimal
artifact with a non-null `error` marker; the run NEVER raises on one bad PDF.

Scorer landmines encoded here (verified against `vendor/evaluate.py`):
  * Pages are 1-BASED (`pages[0]["page"] == 1`), matching the gold evidence page
    convention (`normalize_visible_id`/`coarse_evidence_key` compare on that).
  * Each caption stores BOTH `scorer_visible_id` (the scorer's canonical form,
    e.g. lowercase `"table 4"` from `evaluate.normalize_visible_id`) AND
    `parser_visible_id` (the raw matched label text, e.g. `"Tab. 4"`).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable

from littraceqa.answer.references import parse_reference_list
from littraceqa.corpus.downloader import STATUS_FAILED, Manifest
from littraceqa.localize.interfaces import PageUnit, ParsedPdf
from littraceqa.localize.pymupdf_runtime import serialized_pymupdf

# Bump this string on any change to the extraction logic that should force a
# re-parse of the whole corpus (resume compares against it).
PARSER_VERSION = "level1-v2"

STATUS_PARSED = "parsed"
PARSE_MANIFEST_NAME = "parse_manifest.jsonl"


# --------------------------------------------------------------------------- #
# Page-text extraction seam (default: PyMuPDF). Injectable so tests need no
# real PDF and the whole-PDF degradation path is exercisable.
# --------------------------------------------------------------------------- #
@serialized_pymupdf
def extract_pages_pymupdf(pdf_bytes: bytes) -> list[str]:
    """Return the plain text of each page, in order, via PyMuPDF. Page order is
    the source order; `parse_pdf` assigns the 1-based page numbers."""
    import fitz  # PyMuPDF; lazy so importing this module stays cheap

    out: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i in range(doc.page_count):
            out.append(doc.load_page(i).get_text("text"))
    return out


PageExtractor = Callable[[bytes], list[str]]


# --------------------------------------------------------------------------- #
# Caption detection -- BOTH scorer-canonical and parser-raw visible ids.
# --------------------------------------------------------------------------- #
# `^(Table|Tab.?|Figure|Fig.?)\s*(\d+)` at line start. `Table`/`Figure` are
# listed before the `Tab.?`/`Fig.?` abbreviations so the full word wins.
_CAPTION_RE = re.compile(r"^\s*(Table|Tab\.?|Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)


def _scorer_visible_id(source_type: str, num: str) -> str:
    """The scorer's canonical visible id (`evaluate.normalize_visible_id`):
    lowercase `"<prefix> <n>"`, e.g. `"table 4"` / `"figure 2"`. `source_type`
    is already the lowercase prefix (`"table"`/`"figure"`)."""
    return f"{source_type} {num}"


def _detect_captions(pages: list[dict]) -> list[dict]:
    caps: list[dict] = []
    for page in pages:
        for line in (page["text"] or "").splitlines():
            m = _CAPTION_RE.match(line)
            if not m:
                continue
            label, num = m.group(1), m.group(2)
            source_type = "table" if label.lower().startswith("tab") else "figure"
            caps.append({
                "source_type": source_type,
                "scorer_visible_id": _scorer_visible_id(source_type, num),
                "parser_visible_id": m.group(0).strip(),
                "page": page["page"],
                "caption": line.strip(),
            })
    return caps


# --------------------------------------------------------------------------- #
# Acronym detection -- forward `Full Name (ABBR)` and reverse `ABBR, our name`.
# --------------------------------------------------------------------------- #
# Forward: a run of Title-case words, then `(ABBR)`.
_ACRO_FWD_RE = re.compile(
    r"((?:[A-Z][A-Za-z'’\-]+\s+){0,6}[A-Z][A-Za-z'’\-]+)\s*"
    r"\(([A-Za-z][A-Za-z0-9\-]*)\)"
)
# Reverse: `ABBR, [our] some full name`.
_ACRO_REV_RE = re.compile(
    r"\b([A-Z]{2,}[A-Za-z0-9\-]*)\s*,\s+"
    r"((?:[A-Za-z][A-Za-z'’\-]*\s+){0,6}[A-Za-z][A-Za-z'’\-]*)"
)
_FILLER = {"our", "the", "a", "an", "its", "their"}


def _uppercase_count(s: str) -> int:
    return sum(1 for c in s if c.isupper())


def _valid_abbr(abbr: str) -> bool:
    return len(abbr) >= 2 and _uppercase_count(abbr) >= 2


def _trim_expansion(words: list[str], abbr: str) -> str:
    """Trim a candidate expansion to the last `len(uppercase letters)` words --
    aligns `Latent Consistency Models (LCM)` to the 3 words matching L, C, M and
    drops any leading run-on text swept in by the greedy word capture."""
    k = _uppercase_count(abbr)
    return " ".join(words[-k:] if 0 < k <= len(words) else words)


def _detect_acronyms(pages: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (alias, expanded-lower)
    for page in pages:
        # Newlines -> spaces: PDF text wraps definitions across lines.
        text = re.sub(r"\s+", " ", page["text"] or "")

        for m in _ACRO_FWD_RE.finditer(text):
            phrase, abbr = m.group(1).strip(), m.group(2).strip()
            if not _valid_abbr(abbr):
                continue
            expanded = _trim_expansion(phrase.split(), abbr)
            if not expanded or expanded.upper() == abbr:
                continue
            key = (abbr, expanded.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"alias": abbr, "expanded": expanded, "page": page["page"],
                        "support_text": m.group(0).strip(), "source": "body_definition"})

        for m in _ACRO_REV_RE.finditer(text):
            abbr = m.group(1).strip()
            if not _valid_abbr(abbr):
                continue
            words = [w for w in m.group(2).split() if w.lower() not in _FILLER]
            k = _uppercase_count(abbr)
            words = words[:k] if 0 < k <= len(words) else words
            expanded = " ".join(words)
            if not expanded or expanded.upper() == abbr:
                continue
            key = (abbr, expanded.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"alias": abbr, "expanded": expanded, "page": page["page"],
                        "support_text": m.group(0).strip(), "source": "body_definition"})
    return out


# --------------------------------------------------------------------------- #
# Section-kind detection.
# --------------------------------------------------------------------------- #
# Ordered kind patterns: first match wins. `references` is checked before the
# looser body headings so a lone "References" line classifies correctly.
_SECTION_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("related_work", r"related\s+works?|background\s+and\s+related\s+work"),
    ("references", r"references|bibliography"),
    ("experiments", r"experiments?|experimental\s+set(?:up|tings?)?"),
    ("results", r"results?(?:\s+and\s+(?:discussion|analysis|ablations?))?"),
    ("other", r"introduction|method(?:s|ology)?|approach|conclusions?|"
              r"discussion|preliminaries|ablations?|limitations?|"
              r"acknowledg(?:e)?ments?|appendix"),
)
# A heading line: an optional leading number (`3`, `4.1`, `A`), then a known
# section name, on its own line (trailing colon/period tolerated).
_HEADING_RE = re.compile(
    r"^\s*(?:(?:\d+|[A-Z])(?:\.\d+)*\.?\s+)?(?P<name>.+?)\s*[:.]?\s*$")
_SECTION_KIND_RES = tuple(
    (kind, re.compile(rf"^(?:{pat})$", re.IGNORECASE))
    for kind, pat in _SECTION_KIND_PATTERNS
)


def _classify_heading(name: str) -> str | None:
    for kind, rx in _SECTION_KIND_RES:
        if rx.match(name):
            return kind
    return None


def _detect_sections(pages: list[dict]) -> list[dict]:
    sections: list[dict] = []
    for page in pages:
        for line in (page["text"] or "").splitlines():
            stripped = line.strip()
            # A heading is a short standalone line; skip prose.
            if not stripped or len(stripped) > 60:
                continue
            m = _HEADING_RE.match(stripped)
            if not m:
                continue
            kind = _classify_heading(m.group("name").strip())
            if kind is None:
                continue
            sections.append({"title": stripped, "page": page["page"], "kind": kind})
    return sections


# --------------------------------------------------------------------------- #
# References -- reuse the answer-layer `parse_reference_list`.
# --------------------------------------------------------------------------- #
def _detect_references(paper_id: str, pages: list[dict]) -> list[dict]:
    parsed = ParsedPdf(
        paper_id=paper_id,
        pages=[PageUnit(page=p["page"], text=p["text"] or "") for p in pages],
    )
    entries = parse_reference_list(parsed)   # [] if it doesn't parse
    return [{"index": i + 1, "raw": entry} for i, entry in enumerate(entries)]


# --------------------------------------------------------------------------- #
# Quality metadata -- deterministic signals so a downstream layer can decide to
# escalate (e.g. to GROBID) from the artifact/manifest ALONE, no re-parse.
# --------------------------------------------------------------------------- #
# A numeric citation marker version tag. Bumped INDEPENDENTLY of PARSER_VERSION:
# this heuristic decides trust flags/warnings only -- it does NOT change the
# extracted page text, captions, or aliases, so a change here must NOT force a
# full corpus re-parse. `parser_version` stays "level1-v2"; downstream code that
# only cares about the trust flag can key off `reference_quality_version`.
REFERENCE_QUALITY_VERSION = "ref-quality-v2"

# An inline numeric citation marker: "[12]" or a grouped "[3, 4, 5]". Rough
# proxy for how many things the paper cites inline (one match per bracket group).
# This is the CVPR/IEEE-style numeric convention.
_CITATION_MARKER_RE = re.compile(r"\[\d{1,3}(?:\s*,\s*\d{1,3})*\]")

# Author-year in-text citations (ACL/EMNLP/NAACL convention), which emit ZERO
# numeric `[N]` markers and so used to slip through the numeric-only heuristic.
# These match citation-SHAPED author-year mentions, NOT every 4-digit year, so
# ordinary experiment prose ("trained in 2024 on 2020 data", "the 2019
# baseline") does not inflate the count:
#   1. a parenthetical cite: "(Smith et al., 2024)", "(Brown, 2022; Lee, 2023)".
#   2. a narrative cite: "Vaswani et al. (2017)", "Smith and Jones (2021)".
AUTHOR_YEAR_PATTERNS = [
    re.compile(
        r'''\( [^()]{0,120}? [A-Z][A-Za-z'’-]+
            (?: \s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'’-]+|&\s*[A-Z][A-Za-z'’-]+) )?
            ,?\s* (?:19|20)\d{2}[a-z]? [^()]{0,120}? \)''',
        re.VERBOSE,
    ),
    re.compile(
        r'''\b [A-Z][A-Za-z'’-]+
            (?: \s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'’-]+|&\s*[A-Z][A-Za-z'’-]+) )?
            \s* \( (?:19|20)\d{2}[a-z]? \)''',
        re.VERBOSE,
    ),
]


def _compute_quality(pages: list[dict], sections: list[dict], captions: list[dict],
                     acronyms: list[dict], references: list[dict]) -> dict:
    """Deterministic quality block (no LLM). `reference_quality` is a coarse
    trust flag; `parse_warnings` are string codes a scan can filter on."""
    page_count = len(pages)
    text_pages_extracted = sum(1 for p in pages if (p["text"] or "").strip())
    empty_text_pages = any(not (p["text"] or "").strip() for p in pages)
    caption_count = len(captions)
    figure_count = sum(1 for c in captions if c.get("source_type") == "figure")
    table_count = sum(1 for c in captions if c.get("source_type") == "table")
    reference_count = len(references)
    # Numeric [N] markers (CVPR/IEEE) AND author-year cites (ACL/EMNLP/NAACL):
    # summing both makes the trust flag venue-agnostic instead of only firing on
    # numeric-citation papers.
    body_citation_markers = sum(
        len(_CITATION_MARKER_RE.findall(p["text"] or "")) for p in pages)
    author_year_citation_markers = sum(
        len(pat.findall(p["text"] or "")) for p in pages for pat in AUTHOR_YEAR_PATTERNS)
    total_citation_markers = body_citation_markers + author_year_citation_markers
    has_references_section = any(s.get("kind") == "references" for s in sections)

    # Venue-agnostic refs-vs-length rule: a real >=6-page paper has a real
    # bibliography; parsing <5 refs means we under-collected regardless of the
    # citation convention (catches ACL papers with zero numeric markers).
    few_refs_for_length = (page_count >= 6 and reference_count < 5)
    low_refs = (total_citation_markers >= 10
                and reference_count < 0.3 * total_citation_markers)
    no_ref_section = (not has_references_section and total_citation_markers >= 10)

    warnings: list[str] = []
    if reference_count == 0:
        warnings.append("zero_references")
    elif few_refs_for_length:  # zero_references already subsumes this case
        warnings.append("few_references_for_paper_length")
    if low_refs:
        warnings.append("reference_count_low_vs_citation_markers")
    if no_ref_section:
        warnings.append("no_references_section_detected")
    if caption_count == 0:
        warnings.append("zero_captions")
    if empty_text_pages:
        warnings.append("empty_text_pages")

    suspicious = (reference_count == 0 or few_refs_for_length
                  or low_refs or no_ref_section)
    return {
        "text_pages_extracted": text_pages_extracted,
        "caption_count": caption_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "acronym_candidate_count": len(acronyms),
        "reference_count": reference_count,
        "body_citation_markers": body_citation_markers,
        "author_year_citation_markers": author_year_citation_markers,
        "total_citation_markers": total_citation_markers,
        "reference_quality": "suspicious" if suspicious else "plausible",
        "reference_quality_version": REFERENCE_QUALITY_VERSION,
        "parse_warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# parse_pdf: the whole artifact.
# --------------------------------------------------------------------------- #
def _minimal_artifact(paper_id: str, error: str) -> dict:
    return {
        "paper_id": paper_id, "parser_version": PARSER_VERSION, "page_count": 0,
        "pages": [], "sections": [], "captions": [], "acronyms": [],
        "references": [], "quality": _compute_quality([], [], [], [], []),
        "error": error,
    }


def _safe(fn, *args):
    """Run a sub-extraction; any failure degrades that field to []."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 -- one broken field must not fail the artifact
        return []


def parse_pdf(paper_id: str, pdf_bytes: bytes, *,
              page_extractor: PageExtractor = extract_pages_pymupdf) -> dict:
    """Parse `pdf_bytes` into ONE deterministic artifact dict (see module docs
    for the schema). Never raises: a whole-PDF extraction failure returns a
    minimal artifact whose `error` is non-null; each sub-extraction degrades to
    `[]` independently. `error` is `None` on a successful parse."""
    try:
        page_texts = page_extractor(pdf_bytes)
    except Exception as exc:  # noqa: BLE001 -- degrade the whole PDF, never raise
        return _minimal_artifact(paper_id, repr(exc))

    # Pages are 1-BASED to match the gold evidence page convention (landmine).
    pages = [{"page": i + 1, "text": text or ""} for i, text in enumerate(page_texts)]
    sections = _safe(_detect_sections, pages)
    captions = _safe(_detect_captions, pages)
    acronyms = _safe(_detect_acronyms, pages)
    references = _safe(_detect_references, paper_id, pages)
    return {
        "paper_id": paper_id,
        "parser_version": PARSER_VERSION,
        "page_count": len(pages),
        "pages": pages,
        "sections": sections,
        "captions": captions,
        "acronyms": acronyms,
        "references": references,
        "quality": _compute_quality(pages, sections, captions, acronyms, references),
        "error": None,
    }


# --------------------------------------------------------------------------- #
# PDF-source seam (READ side).
# --------------------------------------------------------------------------- #
@runtime_checkable
class PdfSource(Protocol):
    def read_pdf(self, paper_id: str) -> bytes | None: ...


class LocalDirPdfSource:
    """Reads `<dir>/{paper_id}.pdf` (the download layer's on-disk layout)."""

    def __init__(self, pdf_dir: str | pathlib.Path):
        self._dir = pathlib.Path(pdf_dir)

    def read_pdf(self, paper_id: str) -> bytes | None:
        path = self._dir / f"{paper_id}.pdf"
        return path.read_bytes() if path.exists() else None


class GcsPdfSource:
    """Reads `gs://<bucket>/<prefix>/{paper_id}.pdf` through a `GcsBackend`'s
    generic blob access (keeps all bucket access funnelled through one place)."""

    def __init__(self, backend, prefix: str = "pdfs"):
        self._backend = backend
        self._prefix = prefix.strip("/")

    def read_pdf(self, paper_id: str) -> bytes | None:
        name = f"{self._prefix}/{paper_id}.pdf" if self._prefix else f"{paper_id}.pdf"
        return self._backend.read_blob(name)


# --------------------------------------------------------------------------- #
# Artifact-sink seam (WRITE side).
# --------------------------------------------------------------------------- #
@runtime_checkable
class ArtifactSink(Protocol):
    def exists(self, paper_id: str) -> bool: ...
    def write(self, paper_id: str, data: bytes) -> None: ...


class LocalArtifactSink:
    """Writes each artifact to `<dir>/{paper_id}.json` (atomic temp+rename)."""

    def __init__(self, out_dir: str | pathlib.Path):
        self._dir = pathlib.Path(out_dir)

    def _path(self, paper_id: str) -> pathlib.Path:
        return self._dir / f"{paper_id}.json"

    def exists(self, paper_id: str) -> bool:
        return self._path(paper_id).exists()

    def write(self, paper_id: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(paper_id).with_suffix(".json.part")
        tmp.write_bytes(data)
        os.replace(tmp, self._path(paper_id))


class GcsArtifactSink:
    """Writes each artifact to `gs://<bucket>/<prefix>/{paper_id}.json` through
    a `GcsBackend`'s generic blob access (uploads are atomic in GCS)."""

    def __init__(self, backend, prefix: str = "parsed"):
        self._backend = backend
        self._prefix = prefix.strip("/")

    def _name(self, paper_id: str) -> str:
        return f"{self._prefix}/{paper_id}.json" if self._prefix else f"{paper_id}.json"

    def exists(self, paper_id: str) -> bool:
        return self._backend.read_blob(self._name(paper_id)) is not None

    def write(self, paper_id: str, data: bytes) -> None:
        self._backend.write_blob(self._name(paper_id), data)


# --------------------------------------------------------------------------- #
# Run summary.
# --------------------------------------------------------------------------- #
@dataclass
class ParseSummary:
    by_status: Counter = field(default_factory=Counter)
    skipped: int = 0
    processed: int = 0
    total_pages: int = 0
    total_captions: int = 0
    total_acronyms: int = 0
    total_references: int = 0

    @property
    def parsed(self) -> int:
        return self.by_status.get(STATUS_PARSED, 0)

    @property
    def failed(self) -> int:
        return self.by_status.get(STATUS_FAILED, 0)

    def render(self) -> str:
        avg = (self.total_pages / self.parsed) if self.parsed else 0.0
        lines = [
            "Corpus parse summary:",
            f"  processed={self.processed} skipped(resumed)={self.skipped}",
            f"  parsed={self.parsed} failed={self.failed}",
            f"  avg_pages={avg:.1f}",
            f"  totals: captions={self.total_captions} "
            f"acronyms={self.total_acronyms} references={self.total_references}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CorpusParser: iterate the pool, parse each PDF, write artifact + manifest.
# --------------------------------------------------------------------------- #
def _manifest_row(paper_id, status, page_count, n_captions, n_acronyms,
                  n_references, error, reference_quality="suspicious") -> dict:
    # `reference_quality` is carried on the row so a scan of the manifest ALONE
    # can list papers needing GROBID, without opening each artifact.
    return {
        "paper_id": paper_id, "status": status, "parser_version": PARSER_VERSION,
        "page_count": page_count, "n_captions": n_captions,
        "n_acronyms": n_acronyms, "n_references": n_references,
        "reference_quality": reference_quality, "error": error,
    }


class CorpusParser:
    """Parses every paper in the pool into an artifact via injected seams.

    Resume source of truth is the manifest: a paper is skipped ONLY when it was
    `parsed` at the CURRENT `PARSER_VERSION`; a `failed` row, a missing row, or a
    version bump re-parses. One bad PDF is recorded `failed` and the run
    continues."""

    def __init__(self, *, source: PdfSource, sink: ArtifactSink, manifest: Manifest,
                 page_extractor: PageExtractor = extract_pages_pymupdf):
        self._source = source
        self._sink = sink
        self._manifest = manifest
        self._page_extractor = page_extractor

    def _already_done(self, prior: dict | None) -> bool:
        return bool(prior and prior.get("status") == STATUS_PARSED
                    and prior.get("parser_version") == PARSER_VERSION)

    def _parse_one(self, paper_id: str) -> tuple[dict, dict]:
        """Return (artifact, manifest_row) for one paper. Never raises."""
        try:
            pdf_bytes = self._source.read_pdf(paper_id)
        except Exception as exc:  # noqa: BLE001 -- a source read error is a failure, not a crash
            pdf_bytes = None
            read_error = repr(exc)
        else:
            read_error = None

        if pdf_bytes is None:
            art = _minimal_artifact(paper_id, read_error or "pdf not found in source")
            row = _manifest_row(paper_id, STATUS_FAILED, 0, 0, 0, 0, art["error"],
                                art["quality"]["reference_quality"])
            return art, row

        art = parse_pdf(paper_id, pdf_bytes, page_extractor=self._page_extractor)
        status = STATUS_FAILED if art["error"] else STATUS_PARSED
        row = _manifest_row(
            paper_id, status, art["page_count"], len(art["captions"]),
            len(art["acronyms"]), len(art["references"]), art["error"],
            art["quality"]["reference_quality"],
        )
        return art, row

    def run(self, papers: Iterable, *, max_papers: int | None = None) -> ParseSummary:
        done = self._manifest.load()
        summary = ParseSummary()
        for paper in papers:
            paper_id = paper.paper_id
            if self._already_done(done.get(paper_id)):
                summary.skipped += 1
                continue
            if max_papers is not None and summary.processed >= max_papers:
                break

            art, row = self._parse_one(paper_id)
            # Always persist the artifact (even the minimal error one) so the
            # bucket/dir is self-describing; the manifest status drives resume.
            self._sink.write(paper_id, json.dumps(art).encode("utf-8"))
            self._manifest.append(row)

            summary.processed += 1
            summary.by_status[row["status"]] += 1
            if row["status"] == STATUS_PARSED:
                summary.total_pages += row["page_count"]
                summary.total_captions += row["n_captions"]
                summary.total_acronyms += row["n_acronyms"]
                summary.total_references += row["n_references"]
        return summary


# --------------------------------------------------------------------------- #
# `python -m littraceqa.corpus.parse` delegates to the CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin shim
    from littraceqa.corpus.parse_cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
