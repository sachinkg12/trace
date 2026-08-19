"""Turnkey CLI: `python -m littraceqa.pipeline run <input.jsonl> <output.jsonl>`.

Wires `build_pipeline` (the composition root) to `run_submission` (the
input-JSONL -> submission-JSONL runner) behind a thin argparse front end --
no logic lives here beyond argument parsing and dispatch.
"""
from __future__ import annotations

import argparse
import sys

from littraceqa.pipeline.build import build_pipeline
from littraceqa.pipeline.evaluate import run_submission


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Path to the input JSONL of query records.")
    parser.add_argument("output", help="Path to write the submission JSONL to.")
    parser.add_argument(
        "--eval",
        metavar="GOLD_JSONL",
        default=None,
        help="Optional gold JSONL: after writing the submission, score it "
        "against this file and print the scorecard.",
    )
    parser.add_argument(
        "--use-expander",
        action="store_true",
        help="Enable seed-kNN recall expansion for multi-paper questions.",
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="Top-N seed candidates per question (default: 3)."
    )
    parser.add_argument(
        "--llm", default="gemini", help="Registered LLM backend name (default: gemini)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littraceqa.pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline over an input JSONL.")
    _add_run_args(run_parser)

    args = parser.parse_args(argv)

    if args.command == "run":
        pipeline = build_pipeline(
            llm_name=args.llm, top_n_seeds=args.top_n, use_expander=args.use_expander
        )
        run_submission(args.input, args.output, pipeline)
        print(f"Wrote submission to {args.output}")

        if args.eval:
            # `evaluate_dev` is added by Task 4's dev-eval loop; imported
            # lazily here so `run` (without --eval) never requires it.
            from littraceqa.pipeline.evaluate import evaluate_dev

            scorecard = evaluate_dev(args.output, args.eval)
            print(scorecard)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
