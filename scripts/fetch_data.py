"""Download public LitTraceQA assets from the official Hugging Face dataset."""
from __future__ import annotations

import hashlib
import pathlib

import requests

BASE = "https://huggingface.co/datasets/LitTraceQA/LitTraceQA/resolve/main"
FILES = {
    "data/validation.jsonl": "data/validation.jsonl",
    "data/validation_inputs.jsonl": "data/validation_inputs.jsonl",
    "data/test.jsonl": "data/test.jsonl",
    "data/paper_metadata.jsonl": "data/paper_metadata.jsonl",
    "data/sample_submission.jsonl": "data/sample_submission.jsonl",
    "schema/input.schema.json": "schema/input.schema.json",
    "schema/submission.schema.json": "schema/submission.schema.json",
    "schema/littraceqa.schema.json": "schema/littraceqa.schema.json",
    "data/tools/evaluate.py": "scripts/evaluate.py",
}


def download(relative_destination: str, relative_source: str) -> None:
    destination = pathlib.Path(relative_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with requests.get(
        f"{BASE}/{relative_source}", stream=True, timeout=180
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
    print(f"{destination}  sha256={digest.hexdigest()}")


def main() -> None:
    for destination, source in FILES.items():
        download(destination, source)


if __name__ == "__main__":
    main()
