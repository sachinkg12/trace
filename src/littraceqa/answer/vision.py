"""Vision answering path: read a cited FIGURE or TABLE page as an image.

Root cause this addresses: ~15/55 dev questions have their answer INSIDE a
figure, and TABLE text extracted by PyMuPDF is column-major-scrambled -- so
"which system has the lowest value" style questions return a stray number
instead of the row label. In both cases the answer isn't recoverable from
the parser's TEXT. Gemini 2.5 Flash is multimodal: this module rasterizes
the cited page (`localize.render.render_page_png`) from the on-disk PDF
cache and hands the PNG to the model so it can SEE the intact figure/table.

This is an ADDITIVE path, kept behind the same strategy interface: the
freeform / MC strategies call in here ONLY when the located evidence
contains a `source_type in ("figure", "table")` item. Total-degradation
contract, exactly like the rest of the answer pipeline: no cached PDF, a
render failure, an LLM error, or a `NOT_IN_FIGURE` reply all return `None`,
and the caller falls back to its existing text path. No evidence is
fabricated -- a vision answer grounds on the very page it was read from.
"""
from __future__ import annotations

import pathlib
import re

from littraceqa.answer.interfaces import AnswerContext
from littraceqa.localize.fetch_local import DEFAULT_CACHE_DIR
from littraceqa.localize.interfaces import LocatedEvidence
from littraceqa.localize.render import render_page_png, render_table_object_png

# Source types whose answer lives in pixels, not in extracted text.
_VISUAL_SOURCE_TYPES = ("figure", "table")

# Sentinel the model returns when the image doesn't contain the answer; we
# treat it as "fall back to the text path", never as an emitted answer.
_NOT_IN_FIGURE = "NOT_IN_FIGURE"

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
# Wrapping quotes the LLM sometimes adds (straight + curly, single + double);
# stripped so the emitted answer matches the scorer's normalize_text (which
# collapses whitespace and strips wrapping quotes) instead of losing exact
# match to a leading space / stray quote.
_WRAP_QUOTES = "\"'“”‘’"


def _trim_answer(s: str) -> str:
    """Strip surrounding whitespace and wrapping quotes from a model answer."""
    s = (s or "").strip()
    while len(s) >= 2 and s[0] in _WRAP_QUOTES and s[-1] in _WRAP_QUOTES:
        s = s[1:-1].strip()
    return s.strip(_WRAP_QUOTES).strip()

# Coordinator-specified prompt: handles both "which <X>? -> label" and
# "numeric value" answers, keeps printed precision, and forbids hallucination.
_VISION_SYSTEM = (
    "You are reading an image of a figure or table from an academic paper to "
    "answer ONE question. Read it carefully: axis labels and scales, the "
    "legend, every plotted series/bar/point, subfigure labels like '(a)'/'(f)', "
    "table row and column headers, and any printed numbers. Answer ONLY from "
    "what is visibly in the image -- never guess or use outside knowledge. If "
    "the question asks 'which <model/method/curve/system/task>...', return the "
    "exact label/name as printed. If it asks for a numeric value, read it off "
    "the axis/point/cell and return the number exactly as shown (keep printed "
    "precision, e.g. '14.70'). If the image does not contain enough to answer, "
    "reply exactly: NOT_IN_FIGURE. Respond with ONLY the answer -- no "
    "explanation."
)


def first_visual_evidence(evidence: list[LocatedEvidence]) -> LocatedEvidence | None:
    """The first figure/table evidence item, or None. This is the sole
    trigger for the vision path."""
    for ev in evidence or []:
        if ev.source_type in _VISUAL_SOURCE_TYPES:
            return ev
    return None


def _load_cached_pdf(paper_id: str, cache_dir: pathlib.Path | None) -> bytes | None:
    """Read `<cache_dir>/{paper_id}.pdf` from disk; None if absent/unreadable
    (never raises). Mirrors `LocalPdfFetcher`'s on-disk cache layout."""
    if not paper_id:
        return None
    base = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path = base / f"{paper_id}.pdf"
    try:
        if not path.exists():
            return None
        return path.read_bytes()
    except Exception:  # noqa: BLE001 -- disk error degrades to None
        return None


def render_evidence_png(ev: LocatedEvidence, *,
                        pdf_bytes: bytes | None = None,
                        cache_dir: pathlib.Path | None = None,
                        dpi: int = 150,
                        focus_table: bool = False,
                        focus_dpi: int = 250) -> bytes | None:
    """Render the evidence's cited page (1-indexed gold page) to PNG; None on any
    failure.

    SHARED PDF SOURCE (FIX 1): prefer the in-memory `pdf_bytes` the runner
    retained when it fetched this paper for localization (this is what makes the
    GCS/streaming corpus path work -- those bytes never land on disk). Fall back
    to the on-disk `<cache_dir>/{paper_id}.pdf` cache only when no bytes are in
    hand, so the old local-cache path is byte-for-byte unchanged."""
    pdf = pdf_bytes if pdf_bytes else _load_cached_pdf(ev.paper_id, cache_dir)
    if pdf is None:
        return None
    if focus_table and ev.source_type == "table" and ev.object_id:
        focused = render_table_object_png(
            pdf, ev.page, ev.object_id, dpi=focus_dpi
        )
        if focused is not None:
            return focused
    return render_page_png(pdf, ev.page, dpi=dpi)


def _prompt(ctx: AnswerContext, ev: LocatedEvidence) -> str:
    where = f"page {ev.page}"
    if ev.object_id:
        where += f", {ev.object_id}"
    return (
        f"Question: {ctx.question}\n\n"
        f"(The image is {where} of paper {ev.paper_id}. Read the relevant "
        f"{ev.source_type} and answer.)"
    )


def _clean(response: str) -> str:
    return _trim_answer(_FENCE_RE.sub("", (response or "").strip()))


def vision_answer_text(ctx: AnswerContext, ev: LocatedEvidence, *,
                       cache_dir: pathlib.Path | None = None,
                       dpi: int = 150) -> str | None:
    """Render the cited figure/table page, ask the model to read it, and
    return the raw answer text. Returns None (fall back to the text path)
    when the page can't be rendered, the model errors, or it replies
    NOT_IN_FIGURE / empty. Never raises."""
    # Prefer the runner-retained bytes for this paper (the GCS-safe shared PDF
    # source); render falls back to the on-disk cache when none are present.
    pdf_bytes = (getattr(ctx, "pdf_bytes_by_id", None) or {}).get(ev.paper_id)
    png = render_evidence_png(ev, pdf_bytes=pdf_bytes, cache_dir=cache_dir, dpi=dpi)
    if png is None:
        return None
    vision_llm = getattr(ctx, "vision_llm", None) or ctx.llm
    try:
        response = vision_llm.complete(
            _prompt(ctx, ev), system=_VISION_SYSTEM, temperature=0.0, images=[png]
        )
    except Exception:  # noqa: BLE001 -- any LLM failure degrades, never crashes
        return None
    text = _clean(response)
    if not text:
        return None
    # NOT_IN_FIGURE (tolerating trailing punctuation / case) => no answer.
    if text.upper().strip(".!").strip() == _NOT_IN_FIGURE:
        return None
    return text
