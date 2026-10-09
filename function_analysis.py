"""
Analyze a real function of one real variable.

For f(x) it finds:
  * the domain, f′ and f″
  * critical points (f′ = 0 or f′ undefined), each classified as a local
    maximum, a local minimum or neither — including the cases where f″(c) = 0
  * intervals where f is increasing / decreasing
  * intervals where f is concave up / concave down
  * inflection points
  * singularities (removable, jump, infinite, oscillating)
  * vertical, horizontal and oblique asymptotes
and plots f, f′ and f″ with all of the above marked.

Usage
-----
In a notebook:
    result = analyze("x^3 - 3x")
    report(result)
    plot(result)

From a terminal:
    python function_analysis.py "x/(x^2 - 4)"

Requires sympy, numpy and matplotlib.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import Patch
from sympy.calculus.util import AccumBounds, continuous_domain
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

# ── Settings ──────────────────────────────────────────────────────────────────

TRANSFORMATIONS = (
    *standard_transformations,
    implicit_multiplication,  # 2x  -> 2*x
    implicit_application,     # sin x -> sin(x)
    convert_xor,              # x^2 -> x**2
)
DEFAULT_WINDOW = (-10.0, 10.0)  # used when f has infinitely many features (e.g. sin)
TOLERANCE = 1e-9                # |value| below this counts as zero
MAX_DERIVATIVE_ORDER = 8        # highest order tried by the higher-order derivative test
SAMPLES = 4001                  # grid size for numeric root finding and plotting

# Palette: categorical slots from a colour-blind-validated set; ink for text.
COLORS = {
    "f": "#2a78d6", "df": "#eb6834", "d2f": "#1baf7a",
    "max": "#6250d6", "min": "#e34948", "inflection": "#008300", "neither": "#52514e",
    "asymptote": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
    "surface": "#fcfcfb", "up": "#e3eefc", "down": "#f0efec",
}


# ── Result types ──────────────────────────────────────────────────────────────

class Kind(StrEnum):
    MAXIMUM = "local maximum"
    MINIMUM = "local minimum"
    NEITHER = "neither a maximum nor a minimum"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class Point:
    x: sp.Expr
    y: sp.Expr

    def __str__(self) -> str:
        return f"({fmt(self.x)}, {fmt(self.y)})"


@dataclass(frozen=True, slots=True)
class CriticalPoint(Point):
    kind: Kind
    reason: Literal["f′ = 0", "f′ undefined"]
    test: str  # the test that decided the kind


@dataclass(frozen=True, slots=True)
class Singularity:
    x: sp.Expr
    kind: str
    left: sp.Expr | None   # limit as x → a⁻ (None if f is undefined to the left)
    right: sp.Expr | None  # limit as x → a⁺


@dataclass(frozen=True, slots=True)
class Asymptote:
    kind: Literal["vertical", "horizontal", "oblique"]
    rhs: sp.Expr  # vertical: x = rhs; otherwise y = rhs
    where: str

    def __str__(self) -> str:
        if self.kind == "vertical":
            return f"vertical   x = {sp.sstr(self.rhs)}"
        return f"{self.kind:<10} y = {sp.sstr(self.rhs)}  (as {self.where})"


@dataclass
class Analysis:
    f: sp.Expr
    x: sp.Symbol
    domain: sp.Set
    df: sp.Expr
    d2f: sp.Expr
    critical_points: list[CriticalPoint]
    increasing: sp.Set
    decreasing: sp.Set
    concave_up: sp.Set
    concave_down: sp.Set
    inflection_points: list[Point]
    singularities: list[Singularity]
    asymptotes: list[Asymptote]
    window: tuple[float, float] | None  # set when results were limited to a window

    @property
    def maxima(self) -> list[CriticalPoint]:
        return [p for p in self.critical_points if p.kind is Kind.MAXIMUM]

    @property
    def minima(self) -> list[CriticalPoint]:
        return [p for p in self.critical_points if p.kind is Kind.MINIMUM]


# ── Parsing ───────────────────────────────────────────────────────────────────

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


# ── Small helpers ─────────────────────────────────────────────────────────────

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


def _num(v: sp.Expr) -> float:
    return float(sp.N(v))


def _is_finite_real(v: sp.Expr | None) -> bool:
    return (v is not None and not isinstance(v, AccumBounds)
            and v.is_extended_real is True and v.is_finite is True)


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


def _derivative(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    # Delta terms only appear at points where the derivative does not exist
    # (kinks of |x|, sign(x)); away from them they are zero.
    d = sp.diff(expr, x).replace(sp.DiracDelta, lambda *_: sp.S.Zero)
    if not d.has(sp.sign, sp.Abs):
        return sp.simplify(d)
    # sign(u)² = 1 wherever it matters; simplify() would hide the points where
    # the derivative does not exist (it turns 0·∞ at x = 0 into 0).
    even_sign = lambda e: e.is_Pow and isinstance(e.base, sp.sign) and e.exp.is_even
    return sp.powsimp(d.replace(even_sign, lambda _: sp.S.One))


def _value(expr: sp.Expr, x: sp.Symbol, c: sp.Expr) -> sp.Expr | None:
    """expr(c) if it is a finite real number, otherwise None."""
    try:
        v = sp.simplify(expr.subs(x, c))
        n = complex(sp.N(v))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not np.isfinite(n) or abs(n.imag) > TOLERANCE or v.has(sp.zoo, sp.nan):
        return None
    return v


def _sign(expr: sp.Expr, x: sp.Symbol, c: sp.Expr) -> int | None:
    """Sign of expr at c: -1, 0, +1, or None where it is undefined."""
    v = _value(expr, x, c)
    if v is None:
        return None
    n = _num(v)
    return 0 if abs(n) < TOLERANCE else (1 if n > 0 else -1)


def _sample(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray:
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
    v = _value(expr, x, sp.Float(t))
    return np.nan if v is None else _num(v)


def _sorted_unique(points) -> list[sp.Expr]:
    out: list[sp.Expr] = []
    for p in sorted(points, key=_num):
        if not out or abs(_num(p) - _num(out[-1])) > 1e-9:
            out.append(p)
    return out


def _intervals(s: sp.Set) -> list[sp.Interval]:
    """The interval components of a set (isolated points are dropped)."""
    if isinstance(s, sp.Interval):
        return [s]
    if isinstance(s, sp.Union):
        return [a for a in s.args if isinstance(a, sp.Interval)]
    return []


def _complement(s: sp.Set) -> sp.Set:
    """ℝ minus s, unwrapping ℝ \\ (ℝ \\ A) = A, which SymPy leaves alone."""
    if isinstance(s, sp.Complement) and s.args[0] == sp.S.Reals:
        return s.args[1]
    return sp.Complement(sp.S.Reals, s)


def _restrict(domain: sp.Set, box: sp.Interval) -> sp.Set:
    """domain ∩ box as a union of intervals (needed for periodic holes, e.g. tan)."""
    holes = _complement(domain).intersect(box)
    if isinstance(holes, sp.FiniteSet) or holes.is_empty:
        return sp.Complement(box, holes).doit() if not holes.is_empty else box
    return domain.intersect(box)


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


def _contains(s: sp.Set, p: sp.Expr) -> bool:
    return s.contains(p) is sp.true


def _interior(s: sp.Set, p: sp.Expr) -> bool:
    return any(iv.start != p and iv.end != p and _contains(iv, p) for iv in _intervals(s))


# ── Locating special points ───────────────────────────────────────────────────

@dataclass
class _Finder:
    """Finds points in the real line, clipping infinite families to a window."""
    x: sp.Symbol
    window: tuple[float, float]
    clipped: bool = False

    @property
    def box(self) -> sp.Interval:
        return sp.Interval(*map(sp.nsimplify, self.window))

    def points(self, s: sp.Set) -> list[sp.Expr]:
        """Points of a discrete set, clipped to the window if there are infinitely many."""
        if s.is_empty:
            return []
        if isinstance(s, sp.FiniteSet):
            return [p for p in s if p.is_extended_real and p.is_finite]
        self.clipped = True
        inside = s.intersect(self.box)
        return list(inside) if isinstance(inside, sp.FiniteSet) else []

    def zeros(self, expr: sp.Expr, region: sp.Set) -> list[sp.Expr]:
        """Real solutions of expr = 0 inside region."""
        if expr.is_zero:
            return []
        try:
            sols = sp.solveset(expr, self.x, region)
        except (NotImplementedError, ValueError, TypeError):
            sols = None
        if sols is not None and (sols.is_empty or isinstance(sols, sp.FiniteSet) or (
                not isinstance(sols, sp.ConditionSet)
                and isinstance(sols.intersect(self.box), sp.FiniteSet | type(sp.S.EmptySet)))):
            return [p for p in self.points(sols) if _value(expr, self.x, p) is not None]
        return self._numeric_zeros(expr, region)

    def _numeric_zeros(self, expr: sp.Expr, region: sp.Set) -> list[sp.Expr]:
        """Fallback when SymPy cannot solve exactly: sign changes + bisection."""
        self.clipped = True
        xs = np.linspace(*self.window, SAMPLES)
        ys = _sample(expr, self.x, xs)
        roots = []
        for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
            if not (np.isfinite(ya) and np.isfinite(yb)) or ya * yb > 0:
                continue
            try:
                r = sp.Float(sp.nsolve(expr, self.x, (a, b), solver="bisect"), 12)
            except (ValueError, ZeroDivisionError, TypeError):
                continue
            # A sign change across a pole is not a root.
            if (v := _value(expr, self.x, r)) is not None and abs(_num(v)) < 1e-6:
                roots.append(self._exact(expr, r))
        return [r for r in _sorted_unique(roots) if _contains(region, r)]

    def _exact(self, expr: sp.Expr, r: sp.Float) -> sp.Expr:
        """Recognize a numeric root such as -2.0000000 or 1.0471976 = π/3 when it checks out exactly."""
        guess = sp.nsimplify(r, [sp.pi, sp.E], tolerance=1e-10)
        try:
            exact = guess != r and sp.simplify(expr.subs(self.x, guess)) == 0
        except (TypeError, ValueError):
            exact = False
        return guess if exact else r

    def undefined(self, expr: sp.Expr, *, with_kinks: bool = True) -> list[sp.Expr]:
        """Points where expr is undefined or discontinuous (plus kinks of |·| and sign(·))."""
        bad = _complement(domain_of(expr, self.x))
        try:
            edge = bad.boundary
        except (NotImplementedError, TypeError, ValueError):
            edge = bad
        return self.points(edge) + (self.kinks(expr) if with_kinks else [])

    def kinks(self, expr: sp.Expr) -> list[sp.Expr]:
        """Zeros of the arguments of |·|, sign(·) and Heaviside(·): possible corners."""
        return [p for atom in expr.atoms(sp.Abs, sp.sign, sp.Heaviside)
                for p in self.zeros(atom.args[0], sp.S.Reals)]


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(f: sp.Expr | str, x: sp.Symbol | str = "x",
            window: tuple[float, float] = DEFAULT_WINDOW) -> Analysis:
    """Run the full analysis. `f` may be a SymPy expression or a string."""
    if isinstance(f, str):
        f, x = parse_function(f, x if isinstance(x, str) else x.name)
    elif isinstance(x, str):
        x = sp.Symbol(x, real=True)

    df = _derivative(f, x)
    d2f = _derivative(df, x)
    domain = domain_of(f, x)
    find = _Finder(x, window)

    stationary = find.zeros(df, domain)
    # At a corner of |u| SymPy reports f′ = sign(0)·… = 0; compare one-sided limits instead.
    corners = {c for c in find.kinks(f) + find.kinks(df) if _is_corner(df, x, c)}
    stationary = [c for c in stationary if c not in corners]
    rough = find.undefined(df, with_kinks=False) + list(corners)  # f′ undefined
    d2_zeros = find.zeros(d2f, domain)
    d2_undefined = find.undefined(d2f)
    f_breaks = find.undefined(f)

    # With infinitely many features (tan, sin …) everything is restricted to the window.
    region = _restrict(domain, find.box) if find.clipped else domain
    keep = lambda pts: _sorted_unique(p for p in pts if _contains(region, p))
    stationary, rough, d2_zeros, d2_undefined = map(keep, (stationary, rough, d2_zeros, d2_undefined))
    rough = [p for p in rough if _interior(region, p) and p not in stationary]

    df_chart = _sign_chart(df, x, region, stationary + rough + f_breaks)
    d2f_chart = _sign_chart(d2f, x, region, d2_zeros + d2_undefined + rough + f_breaks)

    derivatives = [f, df, d2f]
    critical = [
        _classify(c, reason, x, derivatives, df_chart)
        for reason, pts in (("f′ = 0", stationary), ("f′ undefined", rough))
        for c in pts if _interior(region, c)
    ]
    critical.sort(key=lambda p: _num(p.x))

    inflections = [
        Point(c, _value(f, x, c))
        for c in _sorted_unique(set(d2_zeros) | set(d2_undefined))
        if _interior(region, c) and _value(f, x, c) is not None
        and _changes_sign(d2f_chart, c)
        and not (c in corners and _is_corner(df, x, c, sharp_only=True))  # no tangent line
    ]

    singularities = _singularities(f, x, domain, f_breaks)
    defined = lambda c: _contains(domain, c)
    differentiable = lambda c: defined(c) and _value(df, x, c) is not None
    return Analysis(
        f=f, x=x, domain=domain, df=df, d2f=d2f,
        critical_points=critical,
        increasing=_where(df_chart, +1, defined),
        decreasing=_where(df_chart, -1, defined),
        concave_up=_where(d2f_chart, +1, differentiable),
        concave_down=_where(d2f_chart, -1, differentiable),
        inflection_points=inflections,
        singularities=singularities,
        asymptotes=_asymptotes(f, x, domain, singularities),
        window=window if find.clipped else None,
    )


type Chart = list[tuple[sp.Interval, int | None]]


def _sign_chart(g: sp.Expr, x: sp.Symbol, region: sp.Set, cuts: list[sp.Expr]) -> Chart:
    """Split region at the cut points and record the sign of g on each piece."""
    chart: Chart = []
    for part in _intervals(region):
        inner = [c for c in _sorted_unique(cuts) if _interior(part, c)]
        for a, b in pairwise([part.start, *inner, part.end]):
            match (a.is_infinite, b.is_infinite):
                case (True, True): probe = sp.S.Zero
                case (True, False): probe = b - 1
                case (False, True): probe = a + 1
                case _: probe = (a + b) / 2
            chart.append((sp.Interval.open(a, b), _sign(g, x, probe)))
    return chart


def _where(chart: Chart, sign: int, joinable) -> sp.Set:
    """Union of the pieces with the given sign.

    A cut point c between two pieces of the same sign is included when
    joinable(c) holds, so x³ is increasing on (−∞, ∞) rather than on
    (−∞, 0) ∪ (0, ∞). For monotonicity f must be defined at c; for concavity
    f′ must be (x^(2/3) is concave down on each side of its cusp, not on ℝ).
    """
    out = sp.Union(*(iv for iv, s in chart if s == sign))
    for (left, s1), (right, s2) in pairwise(chart):
        if s1 == s2 == sign and left.end == right.start and joinable(left.end):
            out |= sp.FiniteSet(left.end)
    return out


def _neighbours(chart: Chart, c: sp.Expr) -> tuple[int | None, int | None]:
    left = next((s for iv, s in chart if iv.end == c), None)
    right = next((s for iv, s in chart if iv.start == c), None)
    return left, right


def _changes_sign(chart: Chart, c: sp.Expr) -> bool:
    left, right = _neighbours(chart, c)
    return bool(left and right and left == -right)


def _classify(c: sp.Expr, reason: str, x: sp.Symbol,
              derivatives: list[sp.Expr], df_chart: Chart) -> CriticalPoint:
    """Second-derivative test, then higher-order test, then first-derivative test."""
    y = _value(derivatives[0], x, c)
    if reason == "f′ = 0":
        for n in range(2, MAX_DERIVATIVE_ORDER + 1):
            if n == len(derivatives):
                derivatives.append(_derivative(derivatives[-1], x))
            match _sign(derivatives[n], x, c):
                case None:
                    break  # derivative undefined at c: fall back to the sign chart
                case 0:
                    continue
                case s if n % 2 == 0:
                    test = "second-derivative test" if n == 2 else f"higher-order test (f⁽{n}⁾(c) ≠ 0)"
                    return CriticalPoint(c, y, Kind.MINIMUM if s > 0 else Kind.MAXIMUM, reason, test)
                case _:
                    return CriticalPoint(c, y, Kind.NEITHER, reason,
                                         f"higher-order test (first non-zero derivative f⁽{n}⁾ is odd)")

    match _neighbours(df_chart, c):
        case (1, -1): kind = Kind.MAXIMUM
        case (-1, 1): kind = Kind.MINIMUM
        case (1, 1) | (-1, -1): kind = Kind.NEITHER
        case _: kind = Kind.UNDETERMINED
    return CriticalPoint(c, y, kind, reason, "first-derivative test")


def _limit(f: sp.Expr, x: sp.Symbol, a: sp.Expr, direction: str = "+-") -> sp.Expr | None:
    try:
        return sp.limit(f, x, a, dir=direction)
    except (NotImplementedError, ValueError, TypeError):
        return None


def _is_corner(df: sp.Expr, x: sp.Symbol, c: sp.Expr, *, sharp_only: bool = False) -> bool:
    """True if f′ has no finite two-sided limit at c (a corner, cusp or vertical tangent).

    With sharp_only, only a true corner counts: two different finite slopes.
    """
    left, right = _limit(df, x, c, "-"), _limit(df, x, c, "+")
    if not (_is_finite_real(left) and _is_finite_real(right)):
        return not sharp_only
    return sp.simplify(left - right) != 0


def _singularities(f: sp.Expr, x: sp.Symbol, domain: sp.Set,
                   candidates: list[sp.Expr]) -> list[Singularity]:
    eps = sp.Rational(1, 10**6)
    out = []
    for a in _sorted_unique(candidates):
        if _contains(domain, a):
            continue  # f is defined and continuous there: just a boundary such as 0 for √x
        left = right = None
        if domain.intersect(sp.Interval.open(a - eps, a)) is not sp.S.EmptySet:
            left = _limit(f, x, a, "-")
        if domain.intersect(sp.Interval.open(a, a + eps)) is not sp.S.EmptySet:
            right = _limit(f, x, a, "+")
        sides = [v for v in (left, right) if v is not None]
        if not sides:
            continue
        if any(v in (sp.oo, -sp.oo) for v in sides):
            kind = "infinite (vertical asymptote)"
        elif not all(_is_finite_real(v) for v in sides):
            kind = "oscillating (no limit)"
        elif len(sides) == 1:
            kind = "boundary of the domain (finite one-sided limit)"
        elif sp.simplify(left - right) == 0:
            kind = f"removable (limit {sp.sstr(left)})"
        else:
            kind = "jump"
        out.append(Singularity(a, kind, left, right))
    return out


def _asymptotes(f: sp.Expr, x: sp.Symbol, domain: sp.Set,
                singularities: list[Singularity]) -> list[Asymptote]:
    out = [Asymptote("vertical", s.x, f"x → {sp.sstr(s.x)}")
           for s in singularities if s.kind.startswith("infinite")]

    at_infinity: dict[tuple[str, sp.Expr], list[str]] = {}
    for end, label in ((sp.oo, "+∞"), (-sp.oo, "−∞")):
        try:
            if (domain.sup if end == sp.oo else domain.inf) != end:
                continue
        except NotImplementedError:  # e.g. ℝ minus the poles of tan
            pass
        if _is_finite_real(level := _limit(f, x, end)):
            at_infinity.setdefault(("horizontal", level), []).append(label)
        elif _is_finite_real(m := _limit(f / x, x, end)) and m != 0:
            if _is_finite_real(b := _limit(f - m * x, x, end)):
                at_infinity.setdefault(("oblique", m * x + b), []).append(label)
    for (kind, rhs), ends in at_infinity.items():
        if sp.simplify(f - rhs) == 0:
            continue  # f is a line itself: calling it its own asymptote says nothing
        out.append(Asymptote(kind, rhs, "x → ±∞" if len(ends) == 2 else f"x → {ends[0]}"))
    return out


# ── Text report ───────────────────────────────────────────────────────────────

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


def report(a: Analysis) -> None:
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


# ── Plot ──────────────────────────────────────────────────────────────────────

def _math_title(name: str, expr: sp.Expr) -> str:
    text = f"${name} = {sp.latex(expr)}$"
    try:
        MathTextParser("agg").parse(text)
        return text
    except ValueError:  # LaTeX that matplotlib's mathtext cannot render
        return f"{name.replace('\\prime', '′')} = {sp.sstr(expr)}"


def _x_range(a: Analysis) -> tuple[float, float]:
    if a.window:
        return a.window
    xs = [_num(p.x) for p in (*a.critical_points, *a.inflection_points, *a.singularities)]
    if not xs:
        lo, hi = -5.0, 5.0
    else:
        span = max(max(xs) - min(xs), 2.0)
        lo, hi = min(xs) - 0.6 * span, max(xs) + 0.6 * span
    if a.domain.inf.is_finite:       # don't waste space where f is undefined
        lo = max(lo, _num(a.domain.inf) - 0.05 * (hi - lo))
    if a.domain.sup.is_finite:
        hi = min(hi, _num(a.domain.sup) + 0.05 * (hi - lo))
    return lo, hi


def _y_range(ys: np.ndarray, extra: list[float]) -> tuple[float, float]:
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        return -1.0, 1.0
    # Central 80 % of the values, so a pole doesn't flatten everything else.
    lo, hi = np.percentile(finite, [10, 90])
    lo, hi = min(lo, *extra, 0.0), max(hi, *extra, 0.0)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.6 * (hi - lo) if finite.min() < lo or finite.max() > hi else 0.15 * (hi - lo)
    lo, hi = max(lo - pad, finite.min() - 0.05 * (hi - lo)), min(hi + pad, finite.max() + 0.05 * (hi - lo))
    pad = 0.0
    return lo - pad, hi + pad


def _break_at_jumps(ys: np.ndarray, xs: np.ndarray, poles: list[float],
                    ylim: tuple[float, float]) -> np.ndarray:
    """Hide the vertical line matplotlib would draw across a pole or a jump."""
    ys = ys.copy()
    dx = xs[1] - xs[0]
    for p in poles:
        ys[np.abs(xs - p) < 1.5 * dx] = np.nan
    jumps = np.abs(np.diff(ys)) > 0.8 * (ylim[1] - ylim[0])
    ys[1:][jumps] = np.nan
    return ys


def plot(a: Analysis, xlim: tuple[float, float] | None = None) -> plt.Figure:
    """Plot f, f′ and f″ on stacked panels with every feature marked."""
    x0, x1 = xlim or _x_range(a)
    xs = np.linspace(x0, x1, SAMPLES)
    # Where f, f′ or f″ may jump: don't let the line bridge those gaps.
    breaks = [_num(s.x) for s in a.singularities] + [
        _num(p.x) for p in a.critical_points if p.reason == "f′ undefined"]
    in_view = lambda pts: [p for p in pts if x0 <= _num(p.x) <= x1]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10),
                             gridspec_kw={"height_ratios": [2.2, 1, 1]},
                             facecolor=COLORS["surface"])
    panels = [("f(x)", a.f, COLORS["f"]),
              ("f^{\\prime}(x)", a.df, COLORS["df"]),
              ("f^{\\prime\\prime}(x)", a.d2f, COLORS["d2f"])]

    for ax, (name, expr, color) in zip(axes, panels):
        ys = _sample(expr, a.x, xs)
        extra = [_num(p.y) for p in in_view(a.critical_points + a.inflection_points)
                 if p.y is not None] if ax is axes[0] else []
        ylim = _y_range(ys, extra)
        ax.plot(xs, _break_at_jumps(ys, xs, breaks, ylim), color=color, lw=2, zorder=3)
        ax.set_ylim(ylim)
        ax.set_title(_math_title(name, display_form(expr)), loc="left", fontsize=11, color=COLORS["ink"])
        ax.axhline(0, color=COLORS["ink2"], lw=0.8, zorder=1)
        ax.set_facecolor(COLORS["surface"])
        ax.grid(color=COLORS["grid"], lw=0.6)
        ax.tick_params(colors=COLORS["ink2"], labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(COLORS["grid"])

    ax_f, ax_df, ax_d2f = axes
    _draw_concavity(ax_f, a, x0, x1)
    _draw_asymptotes(ax_f, a, xs)

    marker = {Kind.MAXIMUM: ("^", COLORS["max"]), Kind.MINIMUM: ("v", COLORS["min"]),
              Kind.NEITHER: ("o", COLORS["neither"]), Kind.UNDETERMINED: ("o", COLORS["neither"])}
    label_points = len(in_view(a.critical_points + a.inflection_points)) <= 10

    for p in in_view(a.critical_points):
        shape, color = marker[p.kind]
        cx = _num(p.x)
        for ax in axes:
            ax.axvline(cx, color=color, lw=0.8, ls=":", zorder=2)
        if p.y is not None:
            _dot(ax_f, cx, _num(p.y), shape, color, f"({fmt_short(p.x)}, {fmt_short(p.y)})"
                 if label_points else None)
        if p.reason == "f′ = 0":
            _dot(ax_df, cx, 0.0, "o", color)

    for p in in_view(a.inflection_points):
        cx = _num(p.x)
        for ax in axes:
            ax.axvline(cx, color=COLORS["inflection"], lw=0.8, ls=":", zorder=2)
        _dot(ax_f, cx, _num(p.y), "D", COLORS["inflection"],
             f"({fmt_short(p.x)}, {fmt_short(p.y)})" if label_points else None)
        if (v := _value(a.d2f, a.x, p.x)) is not None:
            _dot(ax_d2f, cx, _num(v), "D", COLORS["inflection"])

    for s in a.singularities:  # hollow circle where the limit is finite (a "hole")
        if s.kind.startswith(("removable", "jump", "boundary")) and x0 <= _num(s.x) <= x1:
            for v in (s.left, s.right):
                if _is_finite_real(v):
                    ax_f.plot(_num(s.x), _num(v), "o", ms=8, mfc=COLORS["surface"],
                              mec=COLORS["f"], mew=2, zorder=5)

    ax_d2f.set_xlim(x0, x1)
    ax_d2f.set_xlabel(a.x.name, color=COLORS["ink2"])
    _legend(fig, a)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fmt_short(v: sp.Expr) -> str:
    return sp.sstr(v) if (v.is_Rational and len(sp.sstr(v)) <= 6) else f"{_num(v):.3g}"


def _dot(ax, x: float, y: float, shape: str, color: str, label: str | None = None) -> None:
    ax.plot(x, y, shape, ms=9, color=color, mec=COLORS["surface"], mew=2, zorder=6)
    if label:
        ax.annotate(label, (x, y), xytext=(6, 8), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"], zorder=7)


def _draw_concavity(ax, a: Analysis, x0: float, x1: float) -> None:
    for s, color, hatch in ((a.concave_up, COLORS["up"], None),
                            (a.concave_down, COLORS["down"], "///")):
        for iv in _intervals(s):
            lo, hi = max(_num(iv.start), x0), min(_num(iv.end), x1)
            if lo < hi:
                ax.axvspan(lo, hi, color=color, hatch=hatch, ec="#cfcdc6" if hatch else None,
                           lw=0, zorder=0)


def _draw_asymptotes(ax, a: Analysis, xs: np.ndarray) -> None:
    style = dict(color=COLORS["asymptote"], lw=1.2, ls="--", zorder=2)
    for asy in a.asymptotes:
        match asy.kind:
            case "vertical":
                ax.axvline(_num(asy.rhs), **style)
            case _:
                side = xs if asy.where == "x → ±∞" else (xs[xs >= 0] if "+" in asy.where else xs[xs <= 0])
                ax.plot(side, _sample(asy.rhs, a.x, side), **style)


def _legend(fig: plt.Figure, a: Analysis) -> None:
    items = [
        (bool(a.maxima), Line2D([], [], marker="^", ls="", ms=9, color=COLORS["max"]), "local maximum"),
        (bool(a.minima), Line2D([], [], marker="v", ls="", ms=9, color=COLORS["min"]), "local minimum"),
        (any(p.kind in (Kind.NEITHER, Kind.UNDETERMINED) for p in a.critical_points),
         Line2D([], [], marker="o", ls="", ms=8, color=COLORS["neither"]), "critical, not an extremum"),
        (bool(a.inflection_points), Line2D([], [], marker="D", ls="", ms=7, color=COLORS["inflection"]),
         "inflection point"),
        (bool(a.asymptotes), Line2D([], [], ls="--", color=COLORS["asymptote"]), "asymptote"),
        (a.concave_up is not sp.S.EmptySet, Patch(fc=COLORS["up"]), "concave up"),
        (a.concave_down is not sp.S.EmptySet, Patch(fc=COLORS["down"], hatch="///", ec="#cfcdc6"),
         "concave down"),
    ]
    handles = [(h, label) for show, h, label in items if show]
    if handles:
        fig.legend(*zip(*handles), loc="lower center", ncol=min(len(handles), 4),
                   frameon=False, fontsize=9, labelcolor=COLORS["ink2"])


# ── Command line ──────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        text, var = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "x")
    else:
        text = input("Enter a function, e.g. x^3 - 3x: ")
        var = input("Variable to differentiate with respect to [x]: ").strip() or "x"
    result = analyze(text, var)
    report(result)
    plot(result)
    plt.show()


if __name__ == "__main__":
    main()
