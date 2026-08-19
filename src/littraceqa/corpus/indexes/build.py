"""Build all FOUR Level-2 indexes from parsed artifacts + the pool, persist
their record stores as JSONL, and report counts.

Artifact-source seam (`ArtifactSource`) mirrors the parse layer's
`PdfSource`/`ArtifactSink` DIP: `LocalDirArtifactSource` reads `{paper_id}.json`
from a local dir; `GcsArtifactSource` streams the same shape from
`gs://<bucket>/<prefix>/{paper_id}.json` through a `GcsBackend` (lazy
google-cloud-storage import -- importing this module never needs the lib; only
constructing a real backend does). Both yield plain artifact dicts, so
`build_indexes` is source-agnostic.

Build order is dependency-driven: passages, objects, then aliases (needs the
pool for title/abstract mining), then relations (needs the aliases index + the
pool to resolve baseline mentions). Relations is only built when a `pool` is
supplied (it cannot resolve mentions without one); passages/objects/aliases are
always built.

Persistence is plain JSONL, one record per line, from each index's
`all_records()`. Reload is symmetric: `<Index>.from_records(...)` rebuilds the
index (incl. its BM25) from the stored records with NO artifact / PDF re-read --
`load_indexes(dir)` wires all four back up so retrieval loads without rebuilding.

SCALE NOTE (26k). Passages dominate: ~26k papers x an est. 40-80 chunks/paper
=> ~1-2M passage records, plus a large aliases record count (many mined
acronyms/coinages per paper). This first pass materializes the artifact list and
all record lists in memory, which is fine for the sample/dev runs and is the
simplest correct approach; on the full 26k VM run expect the passages + objects
BM25 term postings to be the memory driver (order GBs). If that proves tight, the
seam to chunk is per-index streaming: `iter_artifacts()` already yields lazily,
so passages/objects could be built + flushed to JSONL in artifact batches without
holding every record at once. We LOG the counts + an approximate footprint so the
VM run surfaces the real numbers rather than guessing; we do not pre-chunk here.
"""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, runtime_checkable

from littraceqa.corpus.indexes.aliases import AliasesIndex
from littraceqa.corpus.indexes.objects import ObjectsIndex
from littraceqa.corpus.indexes.passages import PassagesIndex
from littraceqa.corpus.indexes.relations import RelationsIndex

PASSAGES_STORE = "passages.jsonl"
OBJECTS_STORE = "objects.jsonl"
ALIASES_STORE = "aliases.jsonl"
RELATIONS_STORE = "relations.jsonl"


# --------------------------------------------------------------------------- #
# Artifact-source seam (READ side): where parsed artifact JSON is read from.
# --------------------------------------------------------------------------- #
@runtime_checkable
class ArtifactSource(Protocol):
    def iter_artifacts(self) -> Iterator[dict]: ...


class LocalDirArtifactSource:
    """Iterates `{paper_id}.json` artifacts in a local directory (sorted for a
    deterministic build order). Silently skips unreadable/oversized junk files."""

    def __init__(self, artifacts_dir: str | pathlib.Path):
        self._dir = pathlib.Path(artifacts_dir)

    def iter_artifacts(self) -> Iterator[dict]:
        for path in sorted(self._dir.glob("*.json")):
            try:
                yield json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue


class GcsArtifactSource:
    """Streams `gs://<bucket>/<prefix>/{paper_id}.json` artifacts through a
    `GcsBackend` (mirrors the parse layer's `GcsPdfSource`). Lists the blobs
    under `<prefix>/`, downloads each, and yields the parsed dict. A malformed
    or unreadable blob is skipped so one bad artifact never aborts the build.

    Construction accepts an injected `backend` (real or a test fake's client
    wrapped by `GcsBackend`); the lazy google-cloud-storage import lives in
    `GcsBackend`, so importing THIS module never needs the lib."""

    def __init__(self, backend, prefix: str = "parsed"):
        self._backend = backend
        self._prefix = prefix.strip("/")

    def iter_artifacts(self) -> Iterator[dict]:
        names = sorted(self._backend.list_blob_names(self._prefix))
        for name in names:
            if not name.endswith(".json"):
                continue
            data = self._backend.read_blob(name)
            if not data:
                continue
            try:
                yield json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue


class ListArtifactSource:
    """Wraps an in-memory list of artifact dicts (tests / already-loaded)."""

    def __init__(self, artifacts: Iterable[dict]):
        self._artifacts = list(artifacts)

    def iter_artifacts(self) -> Iterator[dict]:
        yield from self._artifacts


@dataclass
class BuildReport:
    n_artifacts: int = 0
    n_passages: int = 0
    n_objects: int = 0
    n_aliases: int = 0
    n_paper_specific: int = 0
    n_generic: int = 0
    n_relations: int = 0
    n_relations_resolved: int = 0

    def approx_mem_mb(self) -> float:
        """Very rough in-memory footprint of the persisted records (~1KB per
        record, dominated by passage chunk text). A signpost for the 26k run,
        not an exact measurement."""
        n = self.n_passages + self.n_objects + self.n_aliases + self.n_relations
        return round(n * 1024 / (1024 * 1024), 1)

    def render(self) -> str:
        return "\n".join([
            "Index build summary:",
            f"  artifacts={self.n_artifacts}",
            f"  passages={self.n_passages}",
            f"  objects={self.n_objects}",
            f"  aliases={self.n_aliases} "
            f"(paper_specific={self.n_paper_specific} generic={self.n_generic})",
            f"  relations={self.n_relations} (resolved={self.n_relations_resolved} "
            f"nil={self.n_relations - self.n_relations_resolved})",
            f"  approx_record_mem~={self.approx_mem_mb()}MB "
            f"(scale note: at 26k expect ~1-2M passages; watch BM25 postings mem)",
        ])


@dataclass
class BuiltIndexes:
    passages: PassagesIndex
    objects: ObjectsIndex
    aliases: AliasesIndex
    relations: RelationsIndex | None
    report: BuildReport


@dataclass
class LoadedIndexes:
    """The four indexes reconstructed from a persisted JSONL store (no artifact
    re-read). `relations` is None only if the store had no relations.jsonl."""
    passages: PassagesIndex
    objects: ObjectsIndex
    aliases: AliasesIndex
    relations: RelationsIndex | None


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_indexes(source: ArtifactSource, pool=None, *,
                  out_dir: str | pathlib.Path | None = None,
                  use_fuzzy: bool = True) -> BuiltIndexes:
    """Build passages + objects + aliases (+ relations when `pool` is given)
    from `source`, optionally persisting each record store as JSONL under
    `out_dir`. `pool` (a `PoolIndex` or {paper_id: Paper}) supplies title/abstract
    text for the aliases index and is REQUIRED for the relations index (baseline
    mention resolution); without a pool, relations is skipped (None)."""
    artifacts = list(source.iter_artifacts())

    # Dependency order: passages, objects, aliases (needs pool), relations
    # (needs the aliases index + pool).
    passages = PassagesIndex.build(artifacts)
    objects = ObjectsIndex.build(artifacts)
    aliases = AliasesIndex.build(artifacts, pool)
    relations = (
        RelationsIndex.build(artifacts, aliases, pool, use_fuzzy=use_fuzzy)
        if pool is not None else None
    )

    alias_recs = aliases.all_records()
    n_paper_specific = sum(1 for r in alias_recs if r["alias_scope"] == "paper_specific")
    rel_recs = relations.all_records() if relations is not None else []
    n_resolved = sum(1 for r in rel_recs if r["resolution_status"] == "resolved")
    report = BuildReport(
        n_artifacts=len(artifacts),
        n_passages=len(passages.records),
        n_objects=len(objects.records),
        n_aliases=len(alias_recs),
        n_paper_specific=n_paper_specific,
        n_generic=len(alias_recs) - n_paper_specific,
        n_relations=len(rel_recs),
        n_relations_resolved=n_resolved,
    )

    if out_dir is not None:
        out = pathlib.Path(out_dir)
        _write_jsonl(out / PASSAGES_STORE, passages.all_records())
        _write_jsonl(out / OBJECTS_STORE, objects.all_records())
        _write_jsonl(out / ALIASES_STORE, alias_recs)
        if relations is not None:
            _write_jsonl(out / RELATIONS_STORE, rel_recs)

    return BuiltIndexes(passages, objects, aliases, relations, report)


def load_indexes(out_dir: str | pathlib.Path) -> LoadedIndexes:
    """Reconstruct all four indexes from the persisted JSONL stores under
    `out_dir` -- rebuilds each index (incl. BM25) from its records with NO
    artifact / PDF re-read, so retrieval loads without rebuilding. `relations`
    is None when no `relations.jsonl` was written (pool-less build)."""
    out = pathlib.Path(out_dir)
    passages = PassagesIndex.from_records(_read_jsonl(out / PASSAGES_STORE))
    objects = ObjectsIndex.from_records(_read_jsonl(out / OBJECTS_STORE))
    aliases = AliasesIndex.from_records(_read_jsonl(out / ALIASES_STORE))
    rel_path = out / RELATIONS_STORE
    relations = (
        RelationsIndex.from_records(_read_jsonl(rel_path)) if rel_path.exists() else None
    )
    return LoadedIndexes(passages, objects, aliases, relations)


# --------------------------------------------------------------------------- #
# CLI: `python -m littraceqa.corpus.build_indexes`
# --------------------------------------------------------------------------- #
class _MaxArtifactSource:
    """Caps a wrapped source at `max_n` artifacts (dev / smoke runs)."""

    def __init__(self, inner: ArtifactSource, max_n: int):
        self._inner = inner
        self._max = max_n

    def iter_artifacts(self) -> Iterator[dict]:
        for i, art in enumerate(self._inner.iter_artifacts()):
            if i >= self._max:
                break
            yield art


def _build_source(args) -> ArtifactSource:
    if args.gcs_bucket:
        from littraceqa.corpus.gcs_backend import GcsBackend  # lazy: only real GCS runs

        backend = GcsBackend(args.gcs_bucket, prefix=args.gcs_parsed_prefix)
        source: ArtifactSource = GcsArtifactSource(backend, prefix=args.gcs_parsed_prefix)
    else:
        source = LocalDirArtifactSource(args.artifacts_src)
    if args.max is not None:
        source = _MaxArtifactSource(source, args.max)
    return source


def main(argv: list[str] | None = None) -> int:
    import argparse

    from littraceqa.retrieval.pool import PoolIndex, load_pool

    parser = argparse.ArgumentParser(
        description="Build + persist the four Level-2 indexes (passages, objects, "
                    "aliases, relations) from parsed artifacts, local dir or GCS.")
    parser.add_argument("artifacts_src",
                        help="dir of parsed {paper_id}.json artifacts "
                             "(ignored when --gcs-bucket is given)")
    parser.add_argument("out_dir",
                        help="dir to write the JSONL record stores into")
    parser.add_argument("--gcs-bucket", default=None,
                        help="read artifacts from gs://<bucket>/<prefix>/ instead "
                             "of a local dir")
    parser.add_argument("--gcs-parsed-prefix", default="parsed",
                        help="blob prefix for parsed artifacts (default: parsed)")
    parser.add_argument("--pool", default=None,
                        help="paper_metadata.jsonl (title/abstract source for "
                             "aliases; REQUIRED to build the relations index)")
    parser.add_argument("--max", type=int, default=None,
                        help="cap artifacts processed (dev / smoke runs)")
    parser.add_argument("--no-fuzzy", action="store_true",
                        help="disable BM25 fuzzy baseline resolution in relations")
    args = parser.parse_args(argv)

    pool = PoolIndex(load_pool(args.pool)) if args.pool else None
    source = _build_source(args)
    built = build_indexes(source, pool, out_dir=args.out_dir,
                          use_fuzzy=not args.no_fuzzy)
    print(built.report.render())
    if pool is None:
        print("  (relations index skipped: no --pool given)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
