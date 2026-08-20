# Reproduction protocol

1. Check out an immutable release commit and create the Python environment in
   the README.
2. Download the public benchmark files with `scripts/fetch_data.py`.
3. Download source PDFs from the metadata records' original URLs.
4. Parse PDFs and build the four persisted indexes.
5. Export `GEMINI_API_KEY` and the exact checked-out Git revision.
6. Run the full-generation command from the README without a parent output.
7. Retain the generated manifest, which records the source revision, active
   configuration, input and pool hashes, output hash, trace hash, index sizes,
   dense model, and failure count.

The configuration uses temperature zero and a fixed seed, but hosted Gemini services are not
immutable. Exact byte-level output reproducibility is therefore not guaranteed
across service revisions. The validator, provenance hashes, deterministic
retrieval indexes, and fixed random seed make differences observable.

The released command has no previous-output, component-replay, query allowlist,
or per-query override interface. The source tree contains no submitted JSONL.
This makes the public reproducibility boundary narrower and easier to audit.
