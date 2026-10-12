"""The text report: everything the analysis found, written out in words."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import sympy as sp
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm

from . import terminal_image
from .formatting import Display, display_form, display_mode, fmt, fmt_bound, fmt_set
from .models import Analysis


def report(a: Analysis, display: Display = "text") -> None:
    """Print the analysis.

    ``display`` picks how the math is written: ``"text"`` (plain SymPy syntax; LaTeX when
    run in a notebook), ``"pretty"`` (Unicode, with f, f′, f″ laid out in 2-D), ``"latex"``
    (LaTeX source) or ``"image"`` (every line with math typeset inline, in iTerm2, Kitty,
    WezTerm or Ghostty; elsewhere it falls back to ``"pretty"``).
    """
    proto = None
    if display == "image" and (proto := terminal_image.protocol()) is None:
        print("lagrange: this terminal cannot show inline images (try iTerm2, Kitty, WezTerm "
              "or Ghostty); using --display pretty instead", file=sys.stderr)
        display = "pretty"

    for label, expr in (("f(x)", a.f), ("f'(x)", a.df), ("f''(x)", a.d2f)):
        _show_math(label, display_form(expr), display, proto)

    if display != "image":
        with display_mode(display):
            for line in _body(a):
                print(line)
        return
    # Typeset each line from its LaTeX form; fall back to its Unicode form.
    with display_mode("latex"):
        tex_lines = list(_body(a))
    with display_mode("pretty"):
        pretty_lines = list(_body(a))
    for tex, pretty in zip(tex_lines, pretty_lines, strict=True):
        try:
            if "$" not in tex:
                raise ValueError("no math to typeset")
            terminal_image.show(tex, proto)
        except ValueError:
            print(pretty)


def _body(a: Analysis) -> Iterator[str]:
    """The report after f, f′ and f″, line by line, in the current display mode."""
    def section(title: str, lines: list[str], empty: str = "none") -> Iterator[str]:
        yield ""
        yield title
        for line in lines or [empty]:
            yield f"  {line}"

    yield ""
    yield f"Domain: {fmt_set(a.domain)}"
    if a.window:
        yield (f"Note: some equations have infinitely many solutions or had to be solved "
               f"numerically, so results are listed for x in [{a.window[0]:g}, {a.window[1]:g}].")

    yield from section("Critical points", [
        f"x = {fmt(p.x)}: {p.reason} → {p.kind}, f = {fmt(p.y)}  [{p.test}]"
        for p in a.critical_points
    ], "none" if not a.df.is_zero else "f is constant: every point is critical")
    yield from section("Local maxima", [str(p) for p in a.maxima])
    yield from section("Local minima", [str(p) for p in a.minima])

    yield from section("Monotonicity", [f"increasing on {fmt_set(a.increasing)}",
                                        f"decreasing on {fmt_set(a.decreasing)}"])
    yield from section("Concavity", [f"concave up on   {fmt_set(a.concave_up)}",
                                     f"concave down on {fmt_set(a.concave_down)}"])
    yield from section("Inflection points", [str(p) for p in a.inflection_points])

    yield from section("Singularities", [
        f"x = {fmt(s.x)}: {s.kind}; " + ", ".join(
            f"f → {fmt(v)} as x → {fmt_bound(s.x)}{side}"
            for v, side in ((s.left, "⁻"), (s.right, "⁺")) if v is not None)
        for s in a.singularities
    ])
    yield from section("Asymptotes", [str(s) for s in a.asymptotes])


def _show_math(label: str, expr: sp.Expr, display: Display,
               proto: terminal_image.Protocol | None) -> None:
    """Write ``label = expr`` in the chosen display mode."""
    match display:
        case "latex":
            print(f"${label} = {sp.latex(expr)}$")
            return
        case "pretty":
            _show_pretty(label, expr)
            return
        case "image":
            try:
                terminal_image.show(f"${label} = {sp.latex(expr)}$", proto)
            except ValueError:  # beyond what mathtext can typeset
                _show_pretty(label, expr)
            return
    try:
        from IPython import get_ipython
        from IPython.display import Math, display
        if get_ipython() is not None and "Terminal" not in type(get_ipython()).__name__:
            display(Math(f"{label} = {sp.latex(expr)}"))
            return
    except ImportError:
        pass
    print(f"{label:<6} = {sp.sstr(expr)}")


def _show_pretty(label: str, expr: sp.Expr) -> None:
    """The expression in 2-D Unicode, with the label on its baseline."""
    form = PrettyPrinter({"use_unicode": True, "wrap_line": False})._print(expr)
    form = prettyForm(*form.left(f"{label:<6} = "))
    print("\n".join(line.rstrip() for line in form.picture))
