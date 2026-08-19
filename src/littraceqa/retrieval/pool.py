"""Pool loader over the real paper-metadata JSONL.

Hardens the prototype in `analysis/common.py` (`load_metadata_index`,
`paper_text`) into a typed `Paper`-based loader plus a small in-memory
index. Kept intentionally thin: this is a data-access layer, not a
retriever -- retrievers (Task 3+) consume `PoolIndex` via `Retriever`
Protocol implementations.
"""

from __future__ import annotations

import json
import pathlib
import warnings

from littraceqa.retrieval.interfaces import Paper

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_POOL_PATH = REPO_ROOT / "data" / "paper_metadata.jsonl"


def _record_to_paper(rec: dict) -> Paper:
    """Map a raw JSONL record's field names onto `Paper`.

    The metadata schema (`paper_id`, `title`, `abstract`, `authors`,
    `venue`, `year`, `pdf_url`, plus fields `Paper` doesn't track --
    `track`, `award`, `source_url`, `arxiv_id`, `doi`, `openreview_id`,
    `anthology_id`) maps 1:1 by name onto the `Paper` attributes we keep;
    `abstract` is occasionally `null` in the source data, so it is
    normalized to `""` since `Paper.abstract` is a non-optional `str`.
    """
    return Paper(
        paper_id=rec["paper_id"],
        title=rec.get("title") or "",
        abstract=rec.get("abstract") or "",
        venue=rec.get("venue"),
        year=rec.get("year"),
        authors=list(rec.get("authors") or []),
        pdf_url=rec.get("pdf_url"),
    )


def load_pool(path: str | pathlib.Path = DEFAULT_POOL_PATH) -> list[Paper]:
    """Load every record in the metadata JSONL into a `Paper` list.

    Duplicate `paper_id`s are not expected in the real pool (verified: 0
    across 27,487 records), but if a rerun of the crawl introduces one,
    the FIRST occurrence is kept and a warning is emitted -- silent
    overwrite would non-deterministically depend on line order.
    """
    papers: list[Paper] = []
    seen: set[str] = set()
    dup_count = 0
    with pathlib.Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            paper_id = rec["paper_id"]
            if paper_id in seen:
                dup_count += 1
                continue
            seen.add(paper_id)
            papers.append(_record_to_paper(rec))
    if dup_count:
        warnings.warn(
            f"load_pool: skipped {dup_count} duplicate paper_id record(s); "
            "kept first occurrence of each.",
            stacklevel=2,
        )
    return papers


class PoolIndex:
    """`paper_id -> Paper` lookup over a loaded pool."""

    def __init__(self, papers: list[Paper]):
        self._by_id: dict[str, Paper] = {p.paper_id: p for p in papers}

    def by_id(self, paper_id: str) -> Paper | None:
        return self._by_id.get(paper_id)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id.keys())

    def doc_text(self, paper_id: str) -> str:
        """title + " " + abstract for the given paper_id."""
        paper = self._by_id[paper_id]
        return f"{paper.title} {paper.abstract}"
