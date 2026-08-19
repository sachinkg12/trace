"""PyMuPDF parser: PDF bytes -> page-indexed ParsedPdf.

Registered as `"pymupdf"`. Google Document AI swaps in later behind the
same `PdfParser` Protocol by registration. v1 emits per-page text plus the
`Table N` / `Figure N` caption labels detected in that text; structured
table-cell extraction is deferred to the answer pipeline (#7).
"""
from __future__ import annotations

import re

from littraceqa.evidence import pdf_page_to_gold
from littraceqa.localize.interfaces import (
    DetectedObject, PageUnit, ParsedPdf, register_parser,
)
from littraceqa.localize.pymupdf_runtime import serialized_pymupdf

# Label at line start or after whitespace, then a number the scorer treats as
# a visible id: bare `4`, appendix `A1`, or dotted `4.1`.
_LABELS = {"table": r"[Tt]ab(?:le|\.)?", "figure": r"[Ff]ig(?:ure|\.)?"}
# Order matters: dotted (4.1) and appendix (A1) forms must be tried
# BEFORE bare \d+[a-z]? or `Table 4.1` matches `4` and truncates to
# `Table 4` — mismatching gold (see evidence.normalize_object_id).
_NUM = r"(\d+\.\d+|[A-Z]\d+|\d+[a-z]?)"


def detect_objects(text: str, page: int) -> list[DetectedObject]:
    """Find distinct `Table N`/`Figure N` caption labels in `text`.

    One `DetectedObject` per distinct `(source_type, canonical-number)` on the
    page; `object_id` is stored as canonical `Table N` / `Figure N` (raw label
    variants like `Tab. 4` collapse here), leaving `make_evidence` to do the
    scorer-mirrored final normalization at emit time.
    """
    out: list[DetectedObject] = []
    seen: set[tuple[str, str]] = set()  # (source_type, number)
    for source_type, label in _LABELS.items():
        prefix = "Table" if source_type == "table" else "Figure"
        for m in re.finditer(rf"(?:(?<=\s)|^){label}\s*{_NUM}\b", text):
            num = m.group(1)
            if (source_type, num) in seen:
                continue
            seen.add((source_type, num))
            out.append(DetectedObject(source_type=source_type,
                                      object_id=f"{prefix} {num}", page=page))
    return out


@register_parser("pymupdf")
class PyMuPdfParser:
    @serialized_pymupdf
    def parse(self, paper_id: str, pdf_bytes: bytes) -> ParsedPdf:
        import fitz  # PyMuPDF; imported lazily so the package import is cheap
        pages: list[PageUnit] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i in range(doc.page_count):
                text = doc.load_page(i).get_text("text")
                gold_page = pdf_page_to_gold(i)
                pages.append(PageUnit(page=gold_page, text=text,
                                      objects=detect_objects(text, gold_page)))
        return ParsedPdf(paper_id=paper_id, pages=pages)
