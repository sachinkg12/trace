"""Evidence-localization orchestrator: fetch -> parse (cached) -> localize
-> scorer-shaped evidence dicts, degrading to [] at every failure.

Every stage is an injected Protocol (DIP): the same service runs on
local PyMuPDF+Gemini now and on Document AI+Vertex later with no edits.
"""
from __future__ import annotations

import threading

from littraceqa.evidence import make_evidence
from littraceqa.localize.interfaces import (
    EvidenceLocalizer, LocatedEvidence, ParsedPdf, PdfFetcher, PdfParser,
)
from littraceqa.retrieval.interfaces import Paper


class EvidenceLocalizationService:
    def __init__(self, fetcher: PdfFetcher, parser: PdfParser, localizer: EvidenceLocalizer):
        self._fetcher = fetcher
        self._parser = parser
        self._localizer = localizer
        self._parsed_cache: dict[str, ParsedPdf] = {}
        self._raw_cache: dict[str, bytes] = {}
        # Guards the check-then-act on `_parsed_cache`. Under
        # `run_experiment(workers>1)` many question threads call `parsed_for`
        # concurrently, some for the SAME paper_id; an unlocked read/insert races
        # (lost updates / a torn view of the dict). We do NOT hold the lock across
        # the slow fetch+parse (that would serialize every distinct paper); we
        # only guard the cache read and the insert (double-checked). Distinct
        # object from the reranker's lock and never acquired nested with it, so no
        # deadlock is possible.
        self._cache_lock = threading.Lock()

    def parsed_for(self, paper: Paper) -> ParsedPdf | None:
        paper_id = paper.paper_id
        with self._cache_lock:
            cached = self._parsed_cache.get(paper_id)
            raw = self._raw_cache.get(paper_id)
        if cached is not None:
            return cached
        # fetch AND parse are wrapped: no fetcher/parser backend (local disk
        # errors, or a future Document AI / Vertex client) may breach the
        # service's total-degradation contract by raising out of here. Done
        # OUTSIDE the lock so slow I/O for different papers runs concurrently.
        try:
            if raw is None:
                raw = self._fetcher.fetch(paper_id, paper.pdf_url)
            if raw is None:
                return None
            # Keep one canonical byte string per paper. This lets downstream
            # vision reuse the exact PDF that was parsed instead of fetching it
            # a second time (especially important for streamed GCS PDFs).
            with self._cache_lock:
                raw = self._raw_cache.setdefault(paper_id, raw)
            parsed = self._parser.parse(paper_id, raw)
        except Exception:
            return None
        # Double-checked insert: if a racing thread already cached this paper
        # while we parsed, discard our duplicate parse and return the CANONICAL
        # object so every caller for this paper_id sees the same one. Worst case
        # is a rare, harmless double-parse; the cache is never corrupted.
        with self._cache_lock:
            existing = self._parsed_cache.get(paper_id)
            if existing is not None:
                return existing
            self._parsed_cache[paper_id] = parsed
        return parsed

    def raw_for(self, paper: Paper) -> bytes | None:
        """Return cached source PDF bytes, fetching/parsing once if needed.

        The parsed and raw caches deliberately share the same population path,
        so answer-time vision renders the same document version as localization.
        Like the rest of this service, backend failures degrade to ``None``.
        """
        with self._cache_lock:
            raw = self._raw_cache.get(paper.paper_id)
        if raw is not None:
            return raw
        if self.parsed_for(paper) is None:
            return None
        with self._cache_lock:
            return self._raw_cache.get(paper.paper_id)

    def locate_located(self, question: str, paper: Paper) -> list[LocatedEvidence]:
        """Return the RICH `LocatedEvidence` (with quotes/confidence) the
        answer pipeline (#7) and dev-eval loop (#8) ground against, reusing
        the parsed cache -- no re-fetch, no second localizer instantiation.
        Degrades to [] at every failure, exactly like `locate_evidence`."""
        parsed = self.parsed_for(paper)
        if parsed is None:
            return []
        try:
            return self._localizer.locate(question, paper, parsed)
        except Exception:
            return []

    def locate_evidence(self, question: str, paper: Paper) -> list[dict]:
        parsed = self.parsed_for(paper)
        if parsed is None:
            return []
        try:
            located = self._localizer.locate(question, paper, parsed)
        except Exception:
            return []
        out: list[dict] = []
        for ev in located:
            try:
                out.append(make_evidence(ev.paper_id, ev.source_type, ev.page, ev.object_id))
            except (ValueError, KeyError):
                continue
        return out
