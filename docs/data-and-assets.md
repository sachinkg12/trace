# Data and assets

The official LitTraceQA dataset is hosted at
<https://huggingface.co/datasets/LitTraceQA/LitTraceQA>. Its annotations and
benchmark files are released under CC BY-NC 4.0. The dataset does not
redistribute paper PDFs; those remain subject to the original publishers'
terms.

`scripts/fetch_data.py` downloads only public benchmark files from that
official repository. None are committed to TRACE. The corpus downloader then
uses each metadata record's original URL. Set `OPENREVIEW_TOKEN` only when an
OpenReview source requires authenticated access.

Expected local assets are:

```text
data/test.jsonl
data/validation.jsonl
data/validation_inputs.jsonl
data/paper_metadata.jsonl
data/pdfs/{paper_id}.pdf
data/parsed/{paper_id}.json
data/indexes/{passages,objects,aliases,relations}.jsonl
data/cache/pool_emb.npy
data/cache/pool_emb.ids.json
```

The full persisted index used by the system contains millions of records and
requires several gigabytes. Building it is a corpus-scale preprocessing step,
not part of the short prediction command.

