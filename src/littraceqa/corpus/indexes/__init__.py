"""Level-2 local index builders over the Level-1 parsed artifacts.

Consumes the deterministic per-paper artifacts produced by
`littraceqa.corpus.parse` (pages / captions / acronyms) and turns them into
searchable LOCAL indexes: a passages BM25 index, an objects (captions) index,
and an aliases (acronym) index. Deliberately BM25-first: NO OpenSearch and NO
dense embeddings yet (dense retrieval is a later layer); this keeps the whole
build dependency-free and reproducible.

All three reuse the verified BM25 implementation in `retrieval.bm25` (tokenizer
+ scoring) via the small `BM25TextIndex` wrapper, so tokenization is never
reinvented. The artifact-source seam (`ArtifactSource`) mirrors the parse
layer's `PdfSource`/`ArtifactSink` DIP pattern so the builders read from a local
dir now and GCS later with zero edits.
"""
