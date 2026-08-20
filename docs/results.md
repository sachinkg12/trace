# Results

## Selected clean-track system

The official 71-question LitTraceQA test submission selected for the paper
scored:

| Metric | Score |
|---|---:|
| Composite | **0.757968** |
| Paper precision / recall / F1 | 0.9859 / 0.9683 / 0.9728 |
| Evidence precision / recall / F1 | 0.6594 / 0.7582 / 0.6847 |
| Multiple-choice accuracy | 0.9800 |
| Table row F1 | 0.5185 |
| Table cell accuracy, macro / micro | 0.3508 / 0.4023 |

The exact vector is in
[`../results/official-test-0.757968.json`](../results/official-test-0.757968.json).

## Interpretation

The result is strongest on paper identity and multiple-choice selection. The
remaining gap is concentrated in exact evidence-locator identity and table
observation units: a source-correct fact can still miss when its object type,
page, printed identifier, row key, or cell assignment differs from the
organizer contract.

## Reproducibility boundary

This repository implements the generic full-generation system without a
previous-prediction input or per-question overrides. It does not redistribute
the submitted JSONL, benchmark inputs, or source PDFs.

Hosted Gemini endpoints are an external dependency. Consequently, the release
supports source- and artifact-level auditability, but does not promise that a
future remote inference run will be byte-identical.

A historical adaptive artifact scored 0.792252 after repeated score-guided
diagnosis. It is useful error-analysis history, but is explicitly excluded from
the selected reproducible claim and from this source release.
