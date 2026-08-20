#!/usr/bin/env python3
"""Fail-closed audit for the public TRACE source tree."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".jsonl", ".npy", ".npz", ".pdf", ".tar", ".tgz", ".zip"
}
FORBIDDEN_PARTS = {
    "artifacts", "outputs", "data", "__pycache__", ".pytest_cache"
}
PRIVATE_MARKERS = (
    "/" + "Users/",
    "/" + "private/",
    "Original" + "_Work",
    "E" + "B1",
    ".co" + "dex/",
    "Cur" + "sor/",
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{30,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
RELEASED_ID = re.compile(r"(?:ltqa_[0-9a-f]{16}|q_[0-9]{3})(?![0-9a-f])")
REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "configs/trace-littraceqa.yaml",
    "docs/index.html",
    "docs/assets/trace-architecture.svg",
    "results/official-test-0.760613.json",
    "src/littraceqa/_vendor/evaluate.py",
    "src/littraceqa/_vendor/validate_submission.py",
}


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    )
    return [ROOT / item.decode() for item in raw.split(b"\0") if item]


def main() -> int:
    files = tracked_files()
    relative = {path.relative_to(ROOT).as_posix() for path in files}
    problems: list[str] = []
    digest = hashlib.sha256()

    missing = sorted(REQUIRED - relative)
    if missing:
        problems.append(f"missing required files: {', '.join(missing)}")

    for path in files:
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        if path.is_symlink():
            problems.append(f"symlink: {rel_text}")
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            problems.append(f"generated/binary artifact: {rel_text}")
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            problems.append(f"forbidden directory: {rel_text}")
        if rel.name == ".env":
            problems.append("tracked .env")

        payload = path.read_bytes()
        digest.update(rel_text.encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        if b"\0" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        if any(marker.casefold() in text.casefold() for marker in PRIVATE_MARKERS):
            problems.append(f"private local path marker: {rel_text}")
        if RELEASED_ID.search(text):
            problems.append(f"released query identifier: {rel_text}")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            problems.append(f"credential-shaped string: {rel_text}")

    if problems:
        for problem in sorted(set(problems)):
            print(f"FAIL {problem}")
        return 1

    print(f"PASS tracked_files={len(files)}")
    print(f"PASS tree_sha256={digest.hexdigest()}")
    print("PASS no credentials, private paths, released IDs, generated artifacts, or symlinks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
