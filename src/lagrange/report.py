"""The text report: everything the analysis found, written out in words."""

from __future__ import annotations

import sympy as sp

from .formatting import display_form, fmt, fmt_bound, fmt_set
from .models import Analysis


def report(a: Analysis) -> None:
    """Print the analysis (as LaTeX in a notebook, as plain text elsewhere)."""
    _show_math("f(x)", display_form(a.f))
    _show_math("f'(x)", display_form(a.df))
    _show_math("f''(x)", display_form(a.d2f))

    def section(title: str, lines: list[str], empty: str = "none") -> None:
        print(f"\n{title}")
        for line in lines or [empty]:
            print(f"  {line}")

    print(f"\nDomain: {fmt_set(a.domain)}")
    if a.window:
        print(f"Note: some equations have infinitely many solutions or had to be solved "
              f"numerically, so results are listed for x in [{a.window[0]:g}, {a.window[1]:g}].")

    section("Critical points", [
        f"x = {fmt(p.x)}: {p.reason} → {p.kind}, f = {fmt(p.y)}  [{p.test}]"
        for p in a.critical_points
    ], "none" if not a.df.is_zero else "f is constant: every point is critical")
    section("Local maxima", [str(p) for p in a.maxima])
    section("Local minima", [str(p) for p in a.minima])

    section("Monotonicity", [f"increasing on {fmt_set(a.increasing)}",
                             f"decreasing on {fmt_set(a.decreasing)}"])
    section("Concavity", [f"concave up on   {fmt_set(a.concave_up)}",
                          f"concave down on {fmt_set(a.concave_down)}"])
    section("Inflection points", [str(p) for p in a.inflection_points])

    section("Singularities", [
        f"x = {fmt(s.x)}: {s.kind}; " + ", ".join(
            f"f → {fmt(v)} as x → {fmt_bound(s.x)}{side}"
            for v, side in ((s.left, "⁻"), (s.right, "⁺")) if v is not None)
        for s in a.singularities
    ])
    section("Asymptotes", [str(s) for s in a.asymptotes])


def _show_math(label: str, expr: sp.Expr) -> None:
    """Pretty LaTeX in a notebook, plain text elsewhere."""
    try:
        from IPython import get_ipython
        from IPython.display import Math, display
        if get_ipython() is not None and "Terminal" not in type(get_ipython()).__name__:
            display(Math(f"{label} = {sp.latex(expr)}"))
            return
    except ImportError:
        pass
    print(f"{label:<6} = {sp.sstr(expr)}")
