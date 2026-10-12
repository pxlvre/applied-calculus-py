"""Turning user input such as 'x^3 - 3x' into a real-valued SymPy expression."""

from __future__ import annotations

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = (
    *standard_transformations,
    implicit_multiplication,  # 2x  -> 2*x
    implicit_application,     # sin x -> sin(x)
    convert_xor,              # x^2 -> x**2
)


def parse_function(text: str, var: str = "x") -> tuple[sp.Expr, sp.Symbol]:
    """Parse text such as 'x^3 - 3x' or 'sin x / x' into a real-valued expression."""
    x = sp.Symbol(var, real=True)
    names = {var: x, "ln": sp.log} | ({} if var == "e" else {"e": sp.E})
    expr = parse_expr(text, local_dict=names, transformations=TRANSFORMATIONS)
    if extra := expr.free_symbols - {x}:
        raise ValueError(f"Unknown symbol(s) {', '.join(map(str, extra))}; "
                         f"the only variable allowed is {var}.")
    return _real_odd_roots(expr), x


def _real_odd_roots(expr: sp.Expr) -> sp.Expr:
    """Make x**(p/q) with odd q real for negative x, as in ordinary calculus.

    SymPy uses the complex principal root, so (-8)**(1/3) would not be -2.
    """
    def is_odd_root(e: sp.Basic) -> bool:
        return (e.is_Pow and e.exp.is_Rational and not e.exp.is_Integer
                and e.exp.q % 2 == 1 and e.base.is_nonnegative is not True)

    def real_root(e: sp.Pow) -> sp.Expr:
        magnitude = sp.Abs(e.base) ** e.exp
        return magnitude if e.exp.p % 2 == 0 else sp.sign(e.base) * magnitude

    return expr.replace(is_odd_root, real_root)
