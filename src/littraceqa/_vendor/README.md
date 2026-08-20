# Official LitTraceQA validator and evaluator primitives

`validate_submission.py` and `evaluate.py` are copied from the official
LitTraceQA public dataset release. The production pipeline imports the
evaluator's normalization and evidence-key functions so generation and final
validation use exactly the same identity contract. These files are not
covered by TRACE's MIT license. LitTraceQA benchmark files are distributed
under CC BY-NC 4.0; see
<https://huggingface.co/datasets/LitTraceQA/LitTraceQA/blob/main/LICENSE.md>.
