"""Command-line interface for the Phase-1 pipeline."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sanskrit-nlp",
        description="Sanskrit grantha digitization and translation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a Sanskrit sentence.")
    analyze.add_argument("text", help="Sanskrit text to analyze.")

    tokenize = subparsers.add_parser("tokenize", help="Tokenize Devanagari aksharas.")
    tokenize.add_argument("text", help="Devanagari text to tokenize.")

    metrics = subparsers.add_parser("metrics", help="Evaluate OCR/MT metrics.")
    metrics.add_argument("reference", help="Reference text.")
    metrics.add_argument("hypothesis", help="Hypothesis text.")
    metrics.add_argument(
        "--kind",
        choices=["ocr", "mt"],
        default="ocr",
        help="Which metric family to run.",
    )

    subparsers.add_parser("version", help="Print version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        from sanskrit_nlp import __version__

        print(f"sanskrit-nlp {__version__}")
        return 0

    if args.command == "tokenize":
        from sanskrit_nlp.sandhi.rules import tokenize_aksharas

        aksharas = tokenize_aksharas(args.text)
        for ak in aksharas:
            print(f"{ak.start}:{ak.end}\t{ak.base}\t{ak.vowel}\t{ak.modifier}")
        return 0

    if args.command == "analyze":
        from sanskrit_nlp.pipeline import analyze_text

        result = analyze_text(args.text)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "metrics":
        from sanskrit_nlp.nlp.eval import evaluate_translation
        from sanskrit_nlp.ocr.metrics import evaluate_ocr

        if args.kind == "ocr":
            metrics = evaluate_ocr(args.reference, args.hypothesis).to_dict()
        else:
            metrics = evaluate_translation(args.reference, args.hypothesis).to_dict()
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
