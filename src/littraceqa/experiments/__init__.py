"""Reproducible dev-55 experiment harness.

Runs a declarative `RunConfig` (loaded from YAML) against the 55 dev input
records and produces a fully-traced, independently reproducible run
directory: a captured `RunManifest` (git commit, versions, checksums,
verbatim config), the submission `predictions.jsonl`, a per-question
`traces.jsonl`, the evaluator `scores.json`, an `env-lock.txt`, and a
`run.log`.

This package is MEASUREMENT infrastructure. It wraps the OLD pipeline path
(`build_pipeline`) as "rung 1" via `OldPathRunner`; the planner->router->
cascade "new"/"union" paths are a SEPARATE follow-up build (Build B) that
plugs into the same `DevRunner` seam and the same `make_runner` factory.
"""
