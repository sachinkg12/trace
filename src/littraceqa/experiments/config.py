"""`RunConfig`: a declarative, YAML-loaded description of ONE experiment.

Deliberately PERMISSIVE. The harness (Build A) only implements `path: "old"`,
but a config carries a free-form `params` dict and optional `index` block so
the follow-up planner/cascade build (Build B) can add `path: "new"` /
`path: "union"` and richer params WITHOUT changing this schema or breaking
existing configs. Unknown top-level keys are preserved verbatim in `raw` so
they survive into the run manifest for provenance.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

KNOWN_PATHS = frozenset({"old", "new", "union"})


@dataclass(frozen=True)
class RunConfig:
    """Resolved experiment configuration.

    Fields:
    - name: run identifier (used in the run-dir name).
    - path: which runner to build -- "old" | "new" | "union". Only "old" is
      implemented in this harness; the factory raises NotImplementedError for
      the others (the Build-B extension point).
    - params: free-form pass-through dict (strategies, cascade_stages, model
      names, k, ...) handed to the runner factory. Schema-permissive.
    - seed: random seed applied to `random`/`numpy` at startup.
    - index_paths: name -> local index file path; checksummed into the
      manifest when present.
    - index_uris: name -> remote (e.g. gs://) URI; recorded verbatim + noted
      "remote" in the manifest when no local path is given.
    - raw: the FULL loaded YAML mapping, kept verbatim for the manifest copy.
    """

    name: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    index_paths: dict[str, str] = field(default_factory=dict)
    index_uris: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def parse_config(raw: dict[str, Any]) -> RunConfig:
    """Build a `RunConfig` from an already-loaded mapping.

    Validates only the two load-bearing fields (`name`, `path`); everything
    else is permissive so Build B can extend the schema freely. Note that an
    UNIMPLEMENTED but KNOWN path ("new"/"union") parses fine here -- it is the
    runner FACTORY (`make_runner`) that raises NotImplementedError, keeping the
    "not yet built" decision in the composition root rather than in the schema.
    """
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping (got %r)" % type(raw).__name__)

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("config 'name' is required and must be a non-empty string")

    path = raw.get("path")
    if path not in KNOWN_PATHS:
        raise ValueError(
            f"config 'path' must be one of {sorted(KNOWN_PATHS)}, got {path!r}"
        )

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("config 'params' must be a mapping if present")

    # seed: top-level `seed` wins; else params.seed; else 0.
    seed = raw.get("seed", params.get("seed", 0))
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"config 'seed' must be an integer, got {seed!r}")

    index = raw.get("index") or {}
    index_paths = index.get("paths") or {} if isinstance(index, dict) else {}
    index_uris = index.get("uris") or {} if isinstance(index, dict) else {}
    if not isinstance(index_paths, dict) or not isinstance(index_uris, dict):
        raise ValueError("config 'index.paths' / 'index.uris' must be mappings")

    return RunConfig(
        name=name,
        path=path,
        params=dict(params),
        seed=seed,
        index_paths={str(k): str(v) for k, v in index_paths.items()},
        index_uris={str(k): str(v) for k, v in index_uris.items()},
        raw=raw,
    )


def load_config(path: str | pathlib.Path) -> RunConfig:
    """Load and parse a `RunConfig` from a YAML file."""
    text = pathlib.Path(path).read_text()
    raw = yaml.safe_load(text)
    return parse_config(raw)
