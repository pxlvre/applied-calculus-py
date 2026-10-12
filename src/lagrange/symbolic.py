"""Thin, defensive wrappers over SymPy and NumPy.

Every helper here answers a question about an expression or a set and returns
``None`` (or a sensible fallback) instead of raising when SymPy cannot decide.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import sympy as sp
from sympy.calculus.util import AccumBounds, continuous_domain

from .config import TOLERANCE

# ── Numbers ───────────────────────────────────────────────────────────────────

def num(v: sp.Expr) -> float:
    """``v`` as a float."""
    return float(sp.N(v))


def is_finite_real(v: sp.Expr | None) -> bool:
    return (v is not None and not isinstance(v, AccumBounds)
            and v.is_extended_real is True and v.is_finite is True)


# ── Calculus ──────────────────────────────────────────────────────────────────

def domain_of(expr: sp.Expr, x: sp.Symbol) -> sp.Set:
    """Where expr is defined and continuous on ℝ.

    |u| and sign(u) are swapped for √(u²) and 1 first: same domain, but SymPy
    can handle the result (it cannot do sign(x)·|x|^(1/3), i.e. the cube root).
    Jumps and kinks of |u| and sign(u) are found separately, from their arguments.
    """
    proxy = expr.replace(sp.Abs, lambda u: sp.sqrt(u**2)).replace(sp.sign, lambda u: sp.S.One)
    for candidate in (expr, proxy):
        try:
            return continuous_domain(candidate, x, sp.S.Reals)
        except (NotImplementedError, ValueError, TypeError):
            continue
    return sp.S.Reals


def derivative(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """d/dx expr, kept in a form that still shows where it fails to exist."""
    # Delta terms only appear at points where the derivative does not exist
    # (kinks of |x|, sign(x)); away from them they are zero.
    d = sp.diff(expr, x).replace(sp.DiracDelta, lambda *_: sp.S.Zero)
    if not d.has(sp.sign, sp.Abs):
        return sp.simplify(d)
    # sign(u)² = 1 wherever it matters; simplify() would hide the points where
    # the derivative does not exist (it turns 0·∞ at x = 0 into 0).
    even_sign = lambda e: e.is_Pow and isinstance(e.base, sp.sign) and e.exp.is_even
    return sp.powsimp(d.replace(even_sign, lambda _: sp.S.One))


def limit(f: sp.Expr, x: sp.Symbol, a: sp.Expr, direction: str = "+-") -> sp.Expr | None:
    try:
        return sp.limit(f, x, a, dir=direction)
    except (NotImplementedError, ValueError, TypeError):
        return None


# ── Evaluation ────────────────────────────────────────────────────────────────

def value_at(expr: sp.Expr, x: sp.Symbol, c: sp.Expr) -> sp.Expr | None:
    """expr(c) if it is a finite real number, otherwise None."""
    try:
        v = sp.simplify(expr.subs(x, c))
        n = complex(sp.N(v))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not np.isfinite(n) or abs(n.imag) > TOLERANCE or v.has(sp.zoo, sp.nan):
        return None
    return v


def sign_at(expr: sp.Expr, x: sp.Symbol, c: sp.Expr) -> int | None:
    """Sign of expr at c: -1, 0, +1, or None where it is undefined."""
    v = value_at(expr, x, c)
    if v is None:
        return None
    n = num(v)
    return 0 if abs(n) < TOLERANCE else (1 if n > 0 else -1)


def sample(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray:
    """Evaluate expr on a numpy grid; non-real or undefined values become NaN."""
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        try:
            ys = np.broadcast_to(np.asarray(fn(xs)), xs.shape)
        except (TypeError, ValueError, ZeroDivisionError):
            ys = np.array([_safe_float(expr, x, t) for t in xs])
    if np.iscomplexobj(ys):
        ys = np.where(np.abs(ys.imag) < TOLERANCE, ys.real, np.nan)
    return ys.astype(float)


def _safe_float(expr: sp.Expr, x: sp.Symbol, t: float) -> float:
    v = value_at(expr, x, sp.Float(t))
    return np.nan if v is None else num(v)


# ── Sets ──────────────────────────────────────────────────────────────────────

def sorted_unique(points: Iterable[sp.Expr]) -> list[sp.Expr]:
    """Points sorted by value, with numerical duplicates collapsed."""
    out: list[sp.Expr] = []
    for p in sorted(points, key=num):
        if not out or abs(num(p) - num(out[-1])) > 1e-9:
            out.append(p)
    return out


def intervals(s: sp.Set) -> list[sp.Interval]:
    """The interval components of a set (isolated points are dropped)."""
    if isinstance(s, sp.Interval):
        return [s]
    if isinstance(s, sp.Union):
        return [a for a in s.args if isinstance(a, sp.Interval)]
    return []


def complement(s: sp.Set) -> sp.Set:
    """ℝ minus s, unwrapping ℝ \\ (ℝ \\ A) = A, which SymPy leaves alone."""
    if isinstance(s, sp.Complement) and s.args[0] == sp.S.Reals:
        return s.args[1]
    return sp.Complement(sp.S.Reals, s)


def restrict(domain: sp.Set, box: sp.Interval) -> sp.Set:
    """domain ∩ box as a union of intervals (needed for periodic holes, e.g. tan)."""
    holes = complement(domain).intersect(box)
    if isinstance(holes, sp.FiniteSet) or holes.is_empty:
        return sp.Complement(box, holes).doit() if not holes.is_empty else box
    return domain.intersect(box)


def contains(s: sp.Set, p: sp.Expr) -> bool:
    return s.contains(p) is sp.true


def interior(s: sp.Set, p: sp.Expr) -> bool:
    """True if p lies strictly inside one of the intervals of s."""
    return any(iv.start != p and iv.end != p and contains(iv, p) for iv in intervals(s))
