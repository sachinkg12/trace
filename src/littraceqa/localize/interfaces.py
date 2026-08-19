"""Evidence-localization contracts: data model, Protocols, and registries.

Dispatch is OCP (mirrors littraceqa.retrieval.interfaces): each backend
registers itself; the `build_*` functions look it up by name and never
change when a new backend is added. Protocols are DIP seams so the
service depends on interfaces, not concrete PyMuPDF/Gemini/HTTP classes
(local-first now; Google Document AI / Vertex swap in later by
registration, zero edits to callers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Callable

from littraceqa.retrieval.interfaces import Paper


@dataclass(frozen=True)
class DetectedObject:
    """A caption label (`Table N` / `Figure N`) the parser found, with the
    1-indexed (gold-convention) page it appears on. `object_id` is the RAW
    label; scorer normalization happens at emit time in `make_evidence`."""
    source_type: str
    object_id: str
    page: int


@dataclass(frozen=True)
class PageUnit:
    page: int                       # 1-indexed (gold convention)
    text: str
    objects: list[DetectedObject] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedPdf:
    paper_id: str
    pages: list[PageUnit]

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page(self, page: int) -> PageUnit | None:
        """Look up a page by its 1-indexed `.page` value; None if absent
        (never raises). Scans by attribute, not list position, so a dropped
        page can't cause a silent off-by-one. Single page-lookup for the
        whole subsystem."""
        for unit in self.pages:
            if unit.page == page:
                return unit
        return None


@dataclass(frozen=True)
class LocatedEvidence:
    paper_id: str
    source_type: str
    page: int                       # 1-indexed (gold convention)
    object_id: str | None
    quote: str
    confidence: float


@runtime_checkable
class PdfFetcher(Protocol):
    def fetch(self, paper_id: str, pdf_url: str | None) -> bytes | None: ...


@runtime_checkable
class PdfParser(Protocol):
    def parse(self, paper_id: str, pdf_bytes: bytes) -> ParsedPdf: ...


@runtime_checkable
class EvidenceLocalizer(Protocol):
    def locate(self, question: str, paper: Paper, parsed: ParsedPdf) -> list[LocatedEvidence]: ...


def _make_registry(kind: str):
    reg: dict[str, Callable] = {}

    def register(name: str):
        def deco(cls):
            if name in reg:
                raise ValueError(f"{kind} {name!r} already registered")
            reg[name] = cls
            return cls
        return deco

    def build(name: str, **kwargs):
        if name not in reg:
            raise KeyError(f"no {kind} registered as {name!r}; have {sorted(reg)}")
        return reg[name](**kwargs)

    return register, build


register_fetcher, build_fetcher = _make_registry("fetcher")
register_parser, build_parser = _make_registry("parser")
register_localizer, build_localizer = _make_registry("localizer")
