"""Rendering SymPy objects the way a textbook would write them."""

from __future__ import annotations

import sympy as sp


def fmt(v: sp.Expr | None) -> str:
    """Exact form when it is short, plus a decimal approximation when useful."""
    if v is None:
        return "undefined"
    if v in (sp.oo, -sp.oo):
        return "∞" if v == sp.oo else "−∞"
    if v.is_Integer or v.is_Rational:
        return sp.sstr(v)
    if v.is_Float:
        return f"{float(v):.6g}"
    return f"{sp.sstr(v)} ≈ {float(sp.N(v)):.6g}"


def fmt_set(s: sp.Set) -> str:
    """Write a set the way a textbook would: (−∞, 0) ∪ (0, ∞)."""
    match s:
        case _ if s is sp.S.EmptySet:
            return "∅"
        case _ if s == sp.S.Reals:
            return "(−∞, ∞)"
        case sp.Interval():
            lo = "(" if s.left_open else "["
            hi = ")" if s.right_open else "]"
            return f"{lo}{fmt_bound(s.start)}, {fmt_bound(s.end)}{hi}"
        case sp.Union():
            return " ∪ ".join(fmt_set(a) for a in sorted(s.args, key=_sort_key))
        case sp.FiniteSet():
            return "{" + ", ".join(fmt_bound(p) for p in s) + "}"
        case sp.ImageSet():
            (n,) = s.lamda.variables
            return f"{{{sp.sstr(s.lamda.expr.subs(n, sp.Symbol('k')))} : k ∈ ℤ}}"
        case sp.Complement() if s.args[0] == sp.S.Reals:
            removed = fmt_set(s.args[1])
            return f"ℝ \\ ({removed})" if isinstance(s.args[1], sp.Union) else f"ℝ \\ {removed}"
        case _:
            return sp.sstr(s)


def _sort_key(s: sp.Set) -> float:
    try:
        return float(s.inf)
    except (NotImplementedError, TypeError):
        return float("inf")


def fmt_bound(v: sp.Expr) -> str:
    if v.is_Float:
        return f"{float(v):.6g}"
    return {sp.oo: "∞", -sp.oo: "−∞"}.get(v, sp.sstr(v))


def fmt_short(v: sp.Expr) -> str:
    """Compact form for a plot annotation."""
    return sp.sstr(v) if (v.is_Rational and len(sp.sstr(v)) <= 6) else f"{float(sp.N(v)):.3g}"


def display_form(expr: sp.Expr) -> sp.Expr:
    """Undo the real-root rewrite for printing: sign(u)·|u|^(1/3) → u^(1/3)."""
    def fix(term: sp.Expr) -> sp.Expr:
        factors = sp.Mul.make_args(term)
        signs = {f.base.args[0] if f.is_Pow else f.args[0]: (f.exp if f.is_Pow else 1)
                 for f in factors if isinstance(f.base if f.is_Pow else f, sp.sign)}
        out = []
        for f in factors:
            if f.is_Pow and isinstance(f.base, sp.Abs) and f.exp.is_Rational and not f.exp.is_Integer:
                u = f.base.args[0]
                if signs.get(u, 0) % 2 == f.exp.p % 2:
                    out.append(u ** f.exp)
                    signs[u] = None  # consumed
                    continue
            out.append(f)
        consumed = {u for u, k in signs.items() if k is None}
        return sp.Mul(*(f for f in out if not (
            isinstance(f.base if f.is_Pow else f, sp.sign)
            and (f.base.args[0] if f.is_Pow else f.args[0]) in consumed)))

    return expr.replace(lambda e: e.is_Mul or (e.is_Pow and isinstance(e.base, sp.Abs)),
                        lambda e: fix(e) if e.is_Mul else fix(sp.Mul(e, evaluate=False)))
