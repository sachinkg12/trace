"""Defensive PDF page -> PNG renderer for the vision answering path.

A figure question's answer lives in the figure's pixels, which the text
parser (`parse_pymupdf`) never captures. `render_page_png` rasterizes the
cited page so a multimodal LLM can SEE it. Total-degradation contract,
mirroring the rest of `localize/`: it NEVER raises -- bad bytes, a
corrupt/empty PDF, or an out-of-range page all return `None`, and the
caller falls back to the existing text path.

Page numbering is the gold 1-indexed convention (see
`littraceqa.evidence.pdf_page_to_gold`: gold = pymupdf0 + 1), so page 1 is
the first page; it is converted to PyMuPDF's 0-index internally.
"""
from __future__ import annotations

from typing import Sequence

from littraceqa.localize.pymupdf_runtime import serialized_pymupdf


def select_table_clip(
    table_boxes: Sequence[Sequence[float]],
    caption_boxes: Sequence[Sequence[float]],
    page_box: Sequence[float],
    *,
    padding: float = 12.0,
) -> tuple[float, float, float, float] | None:
    """Choose the table geometrically nearest an exact caption occurrence.

    Horizontal separation is weighted more heavily than vertical separation:
    two-column papers often place different tables at the same height, while a
    table caption normally sits immediately above or below its own grid.
    ``None`` makes callers fall back to the full-page renderer.
    """

    if not table_boxes or not caption_boxes or len(page_box) != 4:
        return None

    def gap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, a0 - b1, b0 - a1)

    candidates = []
    for table_position, raw_table in enumerate(table_boxes):
        if len(raw_table) != 4:
            continue
        table = tuple(float(value) for value in raw_table)
        if table[2] <= table[0] or table[3] <= table[1]:
            continue
        for caption_position, raw_caption in enumerate(caption_boxes):
            if len(raw_caption) != 4:
                continue
            caption = tuple(float(value) for value in raw_caption)
            x_gap = gap(table[0], table[2], caption[0], caption[2])
            y_gap = gap(table[1], table[3], caption[1], caption[3])
            center_gap = abs(
                (table[0] + table[2]) - (caption[0] + caption[2])
            ) + abs((table[1] + table[3]) - (caption[1] + caption[3]))
            candidates.append((
                3.0 * x_gap + y_gap + 0.001 * center_gap,
                table_position,
                caption_position,
                table,
                caption,
            ))
    if not candidates:
        return None
    _, _, _, table, caption = min(candidates)
    page = tuple(float(value) for value in page_box)
    return (
        max(page[0], min(table[0], caption[0]) - padding),
        max(page[1], min(table[1], caption[1]) - padding),
        min(page[2], max(table[2], caption[2]) + padding),
        min(page[3], max(table[3], caption[3]) + padding),
    )


@serialized_pymupdf
def render_page_png(pdf_bytes: bytes, page_1indexed: int, dpi: int = 150) -> bytes | None:
    """Rasterize the 1-indexed `page_1indexed` of `pdf_bytes` to PNG bytes.

    Returns `None` (never raises) on bad input, a page out of range, or any
    rendering failure."""
    if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
        return None
    if not isinstance(page_1indexed, int) or page_1indexed < 1:
        return None
    try:
        import fitz  # PyMuPDF; imported lazily so the package import is cheap

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page0 = page_1indexed - 1
            if page0 < 0 or page0 >= doc.page_count:
                return None
            pix = doc.load_page(page0).get_pixmap(dpi=dpi)
            png = pix.tobytes("png")
        if not (isinstance(png, (bytes, bytearray)) and png):
            return None
        return bytes(png)
    except Exception:  # noqa: BLE001 -- render is best-effort; degrade to None
        return None


@serialized_pymupdf
def render_page_clip_png(
    pdf_bytes: bytes,
    page_1indexed: int,
    clip: Sequence[float],
    *,
    dpi: int = 250,
) -> bytes | None:
    """Render a validated physical-page rectangle, failing closed.

    Unlike ``render_table_object_png``, this function does not rediscover a
    table from a caption.  The caller supplies a source-derived rectangle (for
    example, a word-geometry window around an exact requested row).
    """

    if (
        not isinstance(pdf_bytes, (bytes, bytearray))
        or not pdf_bytes
        or not isinstance(page_1indexed, int)
        or isinstance(page_1indexed, bool)
        or page_1indexed < 1
        or not isinstance(clip, (list, tuple))
        or len(clip) != 4
    ):
        return None
    try:
        import fitz

        values = tuple(float(value) for value in clip)
        if values[2] <= values[0] or values[3] <= values[1]:
            return None
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page0 = page_1indexed - 1
            if page0 < 0 or page0 >= document.page_count:
                return None
            page = document.load_page(page0)
            clipped = fitz.Rect(values) & page.rect
            if clipped.is_empty or clipped.is_infinite:
                return None
            png = page.get_pixmap(dpi=dpi, clip=clipped).tobytes("png")
        return bytes(png) if isinstance(png, (bytes, bytearray)) and png else None
    except Exception:  # noqa: BLE001 -- a focused render is best-effort
        return None


@serialized_pymupdf
def render_table_object_png(
    pdf_bytes: bytes,
    page_1indexed: int,
    object_id: str,
    *,
    dpi: int = 250,
) -> bytes | None:
    """Render a detected table and its exact caption as a focused PNG.

    Detection or caption failures return ``None`` so callers can retain the
    full-page path.  No guessed crop is emitted.
    """

    if (
        not isinstance(pdf_bytes, (bytes, bytearray))
        or not pdf_bytes
        or not isinstance(page_1indexed, int)
        or page_1indexed < 1
        or not isinstance(object_id, str)
        or not object_id.strip()
    ):
        return None
    try:
        import fitz

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page0 = page_1indexed - 1
            if page0 < 0 or page0 >= doc.page_count:
                return None
            page = doc.load_page(page0)
            captions = page.search_for(object_id.strip())
            tables = page.find_tables().tables
            clip_values = select_table_clip(
                [table.bbox for table in tables],
                [tuple(rect) for rect in captions],
                tuple(page.rect),
            )
            if clip_values is None:
                return None
            pix = page.get_pixmap(dpi=dpi, clip=fitz.Rect(clip_values))
            png = pix.tobytes("png")
        return bytes(png) if isinstance(png, (bytes, bytearray)) and png else None
    except Exception:  # noqa: BLE001 -- focused render is an optional route
        return None
