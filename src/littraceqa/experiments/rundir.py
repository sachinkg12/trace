"""Run-directory allocation under `outputs/dev55/<commit>-<config-name>/`.

REFUSES to clobber an existing NON-EMPTY run dir: reproducibility depends on
a run's artifacts never being silently overwritten. Default behaviour is to
suffix (`-2`, `-3`, ...) to the first free slot; `on_exist="raise"` instead
errors out. An existing EMPTY dir is reused as-is.
"""
from __future__ import annotations

import pathlib

# Filenames written into a run dir. Single source of truth so tests and the
# driver agree on the layout.
MANIFEST_NAME = "manifest.json"
ENV_LOCK_NAME = "env-lock.txt"
PREDICTIONS_NAME = "predictions.jsonl"
TRACES_NAME = "traces.jsonl"
SCORES_NAME = "scores.json"
SUMMARY_NAME = "summary.json"
LOG_NAME = "run.log"

DEFAULT_BASE = pathlib.Path("outputs/dev55")


def _slug(commit: str, config_name: str) -> str:
    """`<short-commit>-<config-name>`; short commit keeps the path readable
    while the FULL commit is still recorded in the manifest."""
    short = (commit or "unknown")[:12]
    return f"{short}-{config_name}"


def _is_nonempty_dir(path: pathlib.Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def prepare_run_dir(
    base: str | pathlib.Path,
    commit: str,
    config_name: str,
    *,
    on_exist: str = "suffix",
) -> pathlib.Path:
    """Create and return a fresh run directory under `base`.

    - Non-existent / existing-but-empty -> used directly.
    - Existing and NON-EMPTY -> `on_exist="suffix"` picks the first free
      `<slug>-N` (N>=2); `on_exist="raise"` raises `FileExistsError`.
    Never overwrites an existing non-empty directory.
    """
    base = pathlib.Path(base)
    slug = _slug(commit, config_name)
    candidate = base / slug

    if _is_nonempty_dir(candidate):
        if on_exist == "raise":
            raise FileExistsError(
                f"run dir already exists and is non-empty: {candidate} "
                "(refusing to overwrite)"
            )
        if on_exist != "suffix":
            raise ValueError(f"on_exist must be 'suffix' or 'raise', got {on_exist!r}")
        n = 2
        while _is_nonempty_dir(base / f"{slug}-{n}"):
            n += 1
        candidate = base / f"{slug}-{n}"

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate
