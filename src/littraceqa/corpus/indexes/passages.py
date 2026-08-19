"""Passages index: chunk each artifact's per-page text into BM25-searchable
passage records.

Chunking rule (scorer-aware): a chunk NEVER crosses a page boundary -- chunking
runs INSIDE one page's text -- because the gold evidence page convention is
1-based per page and a passage that straddled two pages could not be attributed
to a single evidence page. Within a page, text is split on paragraph then
sentence boundaries and greedily packed up to ~`max_tokens` (default 300, a soft
cap; an over-long single sentence is hard-split on word windows so no chunk runs
away). Each chunk carries the `section_kind` ACTIVE on its page (the most recent
section heading at or before that page), so a retrieved passage knows whether it
came from e.g. `results` vs `related_work`.

BM25 over the chunk text reuses `BM25TextIndex` (-> `retrieval.bm25`).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from littraceqa.corpus.indexes.text_index import BM25TextIndex

DEFAULT_MAX_TOKENS = 300

# Sentence boundary: end punctuation followed by whitespace. Deliberately simple
# and deterministic (no NLP dependency) -- good enough to keep chunks near
# sentence edges.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class PassageRecord:
    chunk_id: str
    paper_id: str
    page: int  # 1-based, matches the gold evidence page convention
    section_kind: str | None
    text: str


def _split_units(text: str) -> list[str]:
    """Paragraph-then-sentence units within one page's text (order preserved)."""
    units: list[str] = []
    for para in _PARA_SPLIT_RE.split(text or ""):
        para = para.strip()
        if not para:
            continue
        for sent in _SENT_SPLIT_RE.split(para):
            sent = sent.strip()
            if sent:
                units.append(sent)
    return units


def chunk_page_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[str]:
    """Chunk ONE page's text into <=~`max_tokens`-token passages on sentence /
    paragraph boundaries. Never spans across the caller's page boundary because
    it only ever sees a single page's text."""
    # Hard-split any single unit longer than the cap so one runaway sentence
    # cannot exceed the token budget.
    units: list[str] = []
    for u in _split_units(text):
        words = u.split()
        if len(words) <= max_tokens:
            units.append(u)
        else:
            for i in range(0, len(words), max_tokens):
                units.append(" ".join(words[i:i + max_tokens]))

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for u in units:
        n = len(u.split())
        if cur and cur_tokens + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _section_kind_for_page(sections: list[dict], page: int) -> str | None:
    """The section kind ACTIVE on `page`: the kind of the last section heading
    whose page is <= `page`. `None` if no heading has appeared yet."""
    kind: str | None = None
    best_page = -1
    for s in sections or []:
        sp = s.get("page")
        if sp is None or sp > page:
            continue
        # Later-or-equal page (and later in list on ties) wins -> nearest heading.
        if sp >= best_page:
            best_page = sp
            kind = s.get("kind")
    return kind


def _records_for_artifact(artifact: dict, max_tokens: int) -> list[PassageRecord]:
    paper_id = artifact.get("paper_id", "")
    sections = artifact.get("sections") or []
    records: list[PassageRecord] = []
    for page in artifact.get("pages") or []:
        page_no = page.get("page")
        text = page.get("text") or ""
        kind = _section_kind_for_page(sections, page_no)
        for n, chunk in enumerate(chunk_page_text(text, max_tokens)):
            records.append(PassageRecord(
                chunk_id=f"{paper_id}:p{page_no}:c{n}",
                paper_id=paper_id,
                page=page_no,
                section_kind=kind,
                text=chunk,
            ))
    return records


class PassagesIndex:
    """BM25-searchable chunked-passage index over a set of parsed artifacts."""

    def __init__(self, records: list[PassageRecord]):
        self.records = list(records)
        self._by_id = {r.chunk_id: r for r in self.records}
        self._by_paper: dict[str, list[PassageRecord]] = defaultdict(list)
        for record in self.records:
            self._by_paper[record.paper_id].append(record)
        self._bm25 = BM25TextIndex(
            [r.chunk_id for r in self.records],
            [r.text for r in self.records],
        )

    @classmethod
    def build(cls, artifacts, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> "PassagesIndex":
        records: list[PassageRecord] = []
        for art in artifacts:
            records.extend(_records_for_artifact(art, max_tokens))
        return cls(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "PassagesIndex":
        """Reconstruct from persisted `all_records()` dicts (rebuilds BM25 from
        the stored chunk text -- no re-chunking / no artifact re-read)."""
        return cls([PassageRecord(**r) for r in records])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._bm25.search(query, k)

    def get(self, chunk_id: str) -> PassageRecord | None:
        return self._by_id.get(chunk_id)

    def search_paper(
        self, paper_id: str, query: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """BM25 only this paper's chunks, retaining exact page-bearing ids."""
        records = self._by_paper.get(paper_id, [])
        if not records or k <= 0:
            return []
        index = BM25TextIndex(
            [record.chunk_id for record in records],
            [record.text for record in records],
        )
        return index.search(query, k)

    def all_records(self) -> list[dict]:
        return [asdict(r) for r in self.records]
