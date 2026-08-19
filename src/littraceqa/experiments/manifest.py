"""`RunManifest`: everything needed to REPRODUCE (or invalidate) a run.

Captured at run start and written verbatim to `manifest.json`. Deliberately
records nothing secret: the config/params are scrubbed of API-key/token-shaped
keys before capture, and the environment is snapshotted via `pip freeze` to a
sibling `env-lock.txt` (a package list -- no secrets), NOT by dumping
`os.environ`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from littraceqa.corpus.parse import PARSER_VERSION, REFERENCE_QUALITY_VERSION
from littraceqa.experiments.config import RunConfig
from littraceqa.experiments.rundir import ENV_LOCK_NAME, MANIFEST_NAME
from littraceqa.retrieval.pool import DEFAULT_POOL_PATH

# Keys whose NAME looks secret-bearing are dropped from any config/params copy
# that lands in the manifest. Matches api_key, apikey, token, secret, password,
# and gemini_api_key etc. -- substring, case-insensitive.
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|token|secret|password|passwd|credential)", re.IGNORECASE
)

# Value-SHAPE detectors: a secret can hide as a benign-KEYED string value
# (e.g. params.note = "AIzaSy...."), which the key-name scrub above would miss.
# We redact string VALUES matching a known credential shape.
_REDACTED = "[REDACTED]"
_AIZA_RE = re.compile(r"AIza[0-9A-Za-z_\-]{35}")  # Google API key
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")  # bearer/JWT
_PEM_RE = re.compile(r"-----BEGIN")  # PEM private key / cert block
# Generic high-entropy backstop: a >=32-char run of [A-Za-z0-9_-]. We only
# treat such a run as secret if it is NOT pure-hex (so a 40-hex git SHA or a
# 64-hex sha256 survives) AND mixes letters with digits (so a plain word or a
# path segment survives). Paths as a whole are split on `/` and `.`, which are
# outside the char class, so their short segments never trip this.
_LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{32,}")
_HEX_ONLY_RE = re.compile(r"\A[0-9a-fA-F]+\Z")


def _value_is_hard_secret(s: str) -> bool:
    """High-CONFIDENCE credential shapes only: an AIza Google key, a JWT/bearer,
    or a PEM block. Used as the manifest WRITE-GUARD (defence in depth).

    Unlike `_value_looks_secret`, this does NOT apply the generic high-entropy
    backstop, because that backstop false-positives on legitimate hyphenated
    provenance strings -- notably `platform.platform()` values like
    "Linux-6.1.0-...-x86_64-with-glibc2.36" (hyphens/underscores live inside the
    token class, so a version string reads as one long mixed-alnum run). The
    write-guard must never block a manifest that carries no real credential; the
    Gemini key (AIza...) and OpenReview token (JWT) ARE caught by these shapes."""
    return bool(_AIZA_RE.search(s) or _JWT_RE.search(s) or _PEM_RE.search(s))


def _value_looks_secret(s: str) -> bool:
    """True if the string carries a credential SHAPE (see the patterns above).

    Used to REDACT values inside user-supplied config/params (where a stray
    credential is plausible and over-redaction is acceptable). Deliberately
    spares pure-hex hashes (git SHA / sha256) and ordinary file paths so real
    provenance survives. NOTE: the generic backstop here can flag a hyphenated
    version/platform string, so it must NOT be used as a whole-manifest guard --
    use `_value_is_hard_secret` for that (see write_manifest)."""
    if _value_is_hard_secret(s):
        return True
    for tok in _LONG_TOKEN_RE.findall(s):
        if _HEX_ONLY_RE.match(tok):
            continue  # hex hash (git SHA / sha256) -- provenance, not a secret
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            return True  # mixed-class high-entropy token -> credential-shaped
    return False


def _scrub(obj: Any) -> Any:
    """Recursively drop secret-NAMED keys AND redact secret-SHAPED string
    VALUES from dict/list structures. Key-name scrubbing keeps a legitimate
    `top_k`/`k`; value-shape scrubbing catches a credential smuggled under a
    benign key (e.g. `note: "AIzaSy..."`)."""
    if isinstance(obj, dict):
        return {
            k: _scrub(v)
            for k, v in obj.items()
            if not _SECRET_KEY_RE.search(str(k))
        }
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, str) and _value_looks_secret(obj):
        return _REDACTED
    return obj


def sha256_file(path: str | pathlib.Path) -> str:
    """Streaming sha256 of a file (the pool is ~47MB)."""
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """`git rev-parse HEAD`, or "unknown" if unavailable (non-git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _index_checksums(
    index_paths: dict[str, str], index_uris: dict[str, str]
) -> dict[str, Any]:
    """For each declared index: sha256 the local file IF a path is given (and
    exists); otherwise record the remote URI + note "remote". A declared path
    that is missing on disk is recorded as an error string rather than crashing
    manifest capture."""
    out: dict[str, Any] = {}
    for name, path in index_paths.items():
        p = pathlib.Path(path)
        if p.exists():
            out[name] = {"path": str(path), "sha256": sha256_file(p)}
        else:
            out[name] = {"path": str(path), "sha256": None, "note": "missing"}
    for name, uri in index_uris.items():
        if name not in out:
            out[name] = {"uri": str(uri), "location": "remote"}
    return out


@dataclass(frozen=True)
class RunManifest:
    """Immutable provenance record for one run. Serialize with `to_dict()`."""

    git_commit: str
    python_version: str
    platform: str
    parser_version: str
    reference_quality_version: str
    pool_path: str
    pool_sha256: str
    index_checksums: dict[str, Any]
    config: dict[str, Any]
    model_params: dict[str, Any]
    random_seed: int
    timestamp: str
    env_lock_file: str = ENV_LOCK_NAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_manifest(
    config: RunConfig,
    *,
    pool_path: str | pathlib.Path | None = None,
    timestamp: str | None = None,
) -> RunManifest:
    """Capture a `RunManifest` for `config` at run start.

    - git_commit via `git rev-parse HEAD`.
    - python_version + platform from the interpreter.
    - parser_version / reference_quality_version imported from
      `littraceqa.corpus.parse` (the index/parse build-version constants).
    - pool_sha256: sha256 of the pool JSONL (default `data/paper_metadata.jsonl`).
    - index_checksums: sha256 per declared local index path, else remote URI.
    - config: the FULL resolved YAML mapping, verbatim but SCRUBBED of
      secret-named keys.
    - model_params: `config.params`, scrubbed.
    - random_seed, timestamp (ISO-8601 UTC).
    """
    pool_path = pathlib.Path(pool_path) if pool_path is not None else DEFAULT_POOL_PATH
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    return RunManifest(
        git_commit=git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        parser_version=PARSER_VERSION,
        reference_quality_version=REFERENCE_QUALITY_VERSION,
        pool_path=str(pool_path),
        pool_sha256=sha256_file(pool_path),
        index_checksums=_index_checksums(config.index_paths, config.index_uris),
        config=_scrub(config.raw),
        model_params=_scrub(config.params),
        random_seed=config.seed,
        timestamp=ts,
    )


def write_manifest(manifest: RunManifest, run_dir: str | pathlib.Path) -> pathlib.Path:
    """Write `manifest.json` into `run_dir`. Asserts (defence in depth) that no
    secret-shaped substring survived into the serialized text before it hits
    disk -- the manifest must never carry an API key."""
    path = pathlib.Path(run_dir) / MANIFEST_NAME
    text = json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False)
    lowered = text.lower()
    for needle in ("api_key", "apikey", "gemini_api_key", "secret", "-----begin"):
        if needle in lowered:
            raise AssertionError(
                f"refusing to write manifest: secret-shaped token {needle!r} present"
            )
    # Defence in depth: also refuse a HIGH-CONFIDENCE value-SHAPE leak (an
    # AIza key / JWT / PEM block) that slipped past the key-name scrub. We use
    # `_value_is_hard_secret` (NOT the generic entropy backstop) so a legitimate
    # hyphenated provenance string -- e.g. platform "…-x86_64-with-glibc2.36" --
    # never false-positives and blocks a clean run. Pure-hex hashes are spared.
    if _value_is_hard_secret(text):
        raise AssertionError(
            "refusing to write manifest: secret-shaped VALUE present"
        )
    path.write_text(text)
    return path


def capture_env_lock(run_dir: str | pathlib.Path) -> pathlib.Path:
    """Snapshot the interpreter's installed packages to `env-lock.txt` via
    `pip freeze` (a package list, no secrets). On failure, writes a short note
    rather than aborting the run."""
    path = pathlib.Path(run_dir) / ENV_LOCK_NAME
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        body = out.stdout if out.returncode == 0 else f"# pip freeze failed:\n{out.stderr}"
    except (OSError, subprocess.SubprocessError) as exc:
        body = f"# pip freeze unavailable: {exc}\n"
    path.write_text(body)
    return path
