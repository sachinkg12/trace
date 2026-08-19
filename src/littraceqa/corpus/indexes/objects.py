"""Objects index: one BM25-searchable record per caption (table / figure).

Each parsed caption becomes an object record keyed by
`{paper_id}:{source_type}:{num}` (the number pulled from the scorer-canonical
visible id, e.g. `"table 4"` -> `4`). BOTH visible ids are carried: the
`scorer_visible_id` (lowercase canonical, what the gold evidence compares on)
and the `parser_visible_id` (the raw matched label, e.g. `"Tab. 4"`), so a
downstream evidence check can match either form.

Lookup is by (paper_id, scorer_visible_id) -- the exact pair the scorer's
`normalize_visible_id` produces -- and BM25 over the caption text finds the
right object from a free-text query.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from littraceqa.corpus.indexes.text_index import BM25TextIndex

_NUM_RE = re.compile(r"(\d+)")


def _normalize_visible_id(visible_id: str) -> str:
    """Scorer-style normalization: collapse whitespace, lowercase. Mirrors the
    canonical form captions are stored in so `get` is robust to caller casing."""
    return re.sub(r"\s+", " ", (visible_id or "").strip()).lower()


def _num_of(scorer_visible_id: str) -> str:
    m = _NUM_RE.search(scorer_visible_id or "")
    return m.group(1) if m else ""


@dataclass(frozen=True)
class ObjectRecord:
    object_key: str
    paper_id: str
    source_type: str  # "table" | "figure"
    scorer_visible_id: str
    parser_visible_id: str
    page: int
    caption: str


def _records_for_artifact(artifact: dict) -> list[ObjectRecord]:
    paper_id = artifact.get("paper_id", "")
    out: list[ObjectRecord] = []
    for cap in artifact.get("captions") or []:
        source_type = cap.get("source_type", "")
        scorer_id = _normalize_visible_id(cap.get("scorer_visible_id", ""))
        num = _num_of(scorer_id)
        out.append(ObjectRecord(
            object_key=f"{paper_id}:{source_type}:{num}",
            paper_id=paper_id,
            source_type=source_type,
            scorer_visible_id=scorer_id,
            parser_visible_id=cap.get("parser_visible_id", ""),
            page=cap.get("page"),
            caption=cap.get("caption", ""),
        ))
    return out


class ObjectsIndex:
    """BM25-over-captions + exact (paper_id, scorer_visible_id) lookup."""

    def __init__(self, records: list[ObjectRecord]):
        self.records = list(records)
        self._by_key = {r.object_key: r for r in self.records}
        self._by_paper: dict[str, list[ObjectRecord]] = defaultdict(list)
        for record in self.records:
            self._by_paper[record.paper_id].append(record)
        # (paper_id, normalized scorer_visible_id) -> record. Last write wins if a
        # caption line is duplicated across pages; the object key still disambiguates.
        self._by_visible: dict[tuple[str, str], ObjectRecord] = {
            (r.paper_id, r.scorer_visible_id): r for r in self.records
        }
        self._bm25 = BM25TextIndex(
            [r.object_key for r in self.records],
            [r.caption for r in self.records],
        )

    @classmethod
    def build(cls, artifacts) -> "ObjectsIndex":
        records: list[ObjectRecord] = []
        for art in artifacts:
            records.extend(_records_for_artifact(art))
        return cls(records)

    @classmethod
    def from_records(cls, records: list[dict]) -> "ObjectsIndex":
        """Reconstruct from persisted `all_records()` dicts (rebuilds BM25 +
        lookups from the stored captions -- no artifact re-read)."""
        return cls([ObjectRecord(**r) for r in records])

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self._bm25.search(query, k)

    def get(self, paper_id: str, scorer_visible_id: str) -> ObjectRecord | None:
        return self._by_visible.get((paper_id, _normalize_visible_id(scorer_visible_id)))

    def get_by_key(self, object_key: str) -> ObjectRecord | None:
        return self._by_key.get(object_key)

    def search_paper(
        self, paper_id: str, query: str, k: int = 5
    ) -> list[tuple[str, float]]:
        """BM25 only this paper's captions, retaining exact visible IDs/pages."""
        records = self._by_paper.get(paper_id, [])
        if not records or k <= 0:
            return []
        index = BM25TextIndex(
            [record.object_key for record in records],
            [record.caption for record in records],
        )
        return index.search(query, k)

    def all_records(self) -> list[dict]:
        return [asdict(r) for r in self.records]
