"""Command-line entry point: ``lagrange "x^3 - 3x"``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .analysis import analyze
from .config import DEFAULT_WINDOW
from .formatting import DISPLAYS
from .plotting import plot
from .report import report

PROMPT = "Enter a function, e.g. x^3 - 3x: "
VAR_PROMPT = "Variable to differentiate with respect to [x]: "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lagrange",
        description="Analyze and plot a real function of one real variable.",
        epilog='example: lagrange "x/(x^2 - 4)"',
    )
    parser.add_argument("function", nargs="?",
                        help="the function, e.g. 'x^3 - 3x' (asked for if omitted)")
    parser.add_argument("variable", nargs="?", default=None,
                        help="the variable to differentiate with respect to (default: x)")
    parser.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                        default=DEFAULT_WINDOW,
                        help="x-range used when f has infinitely many features "
                             f"(default: {DEFAULT_WINDOW[0]:g} {DEFAULT_WINDOW[1]:g})")
    parser.add_argument("--display", choices=DISPLAYS, default="text",
                        help="how to write the math: text (plain, the default), pretty "
                             "(Unicode, 2-D), latex (LaTeX source) or image (typeset inline; "
                             "needs iTerm2, Kitty, WezTerm or Ghostty)")
    parser.add_argument("--no-plot", action="store_true", help="print the report only")
    parser.add_argument("--save", metavar="PATH", help="write the plot to PATH instead of showing it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    text = args.function if args.function is not None else input(PROMPT)
    variable = args.variable if args.variable is not None else (
        "x" if args.function is not None else input(VAR_PROMPT).strip() or "x")

    try:
        result = analyze(text, variable, window=tuple(args.window))
    except (ValueError, SyntaxError) as error:
        print(f"lagrange: {error}", file=sys.stderr)
        return 2

    report(result, args.display)
    if not args.no_plot:
        import matplotlib.pyplot as plt

        figure = plot(result)
        if args.save:
            figure.savefig(args.save, dpi=150)
            print(f"\nPlot written to {args.save}")
        else:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
