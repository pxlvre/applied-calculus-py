"""The analysis itself: critical points, monotonicity, concavity, asymptotes."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import sympy as sp

from .config import DEFAULT_WINDOW, MAX_DERIVATIVE_ORDER
from .finder import Finder
from .models import Analysis, Asymptote, Chart, CriticalPoint, Kind, Point, Singularity
from .parsing import parse_function
from .symbolic import (
    contains,
    derivative,
    domain_of,
    interior,
    intervals,
    is_finite_real,
    limit,
    num,
    restrict,
    sign_at,
    sorted_unique,
    value_at,
)


def analyze(f: sp.Expr | str, x: sp.Symbol | str = "x",
            window: tuple[float, float] = DEFAULT_WINDOW) -> Analysis:
    """Run the full analysis. `f` may be a SymPy expression or a string."""
    if isinstance(f, str):
        f, x = parse_function(f, x if isinstance(x, str) else x.name)
    elif isinstance(x, str):
        x = sp.Symbol(x, real=True)

    df = derivative(f, x)
    d2f = derivative(df, x)
    domain = domain_of(f, x)
    find = Finder(x, window)

    stationary = find.zeros(df, domain)
    # At a corner of |u| SymPy reports f′ = sign(0)·… = 0; compare one-sided limits instead.
    corners = {c for c in find.kinks(f) + find.kinks(df) if _is_corner(df, x, c)}
    stationary = [c for c in stationary if c not in corners]
    rough = find.undefined(df, with_kinks=False) + list(corners)  # f′ undefined
    d2_zeros = find.zeros(d2f, domain)
    d2_undefined = find.undefined(d2f)
    f_breaks = find.undefined(f)

    # With infinitely many features (tan, sin …) everything is restricted to the window.
    region = restrict(domain, find.box) if find.clipped else domain
    keep = lambda pts: sorted_unique(p for p in pts if contains(region, p))
    stationary, rough, d2_zeros, d2_undefined = map(keep, (stationary, rough, d2_zeros, d2_undefined))
    rough = [p for p in rough if interior(region, p) and p not in stationary]

    df_chart = _sign_chart(df, x, region, stationary + rough + f_breaks)
    d2f_chart = _sign_chart(d2f, x, region, d2_zeros + d2_undefined + rough + f_breaks)

    derivatives = [f, df, d2f]
    critical = [
        _classify(c, reason, x, derivatives, df_chart)
        for reason, pts in (("f′ = 0", stationary), ("f′ undefined", rough))
        for c in pts if interior(region, c)
    ]
    critical.sort(key=lambda p: num(p.x))

    inflections = [
        Point(c, value_at(f, x, c))
        for c in sorted_unique(set(d2_zeros) | set(d2_undefined))
        if interior(region, c) and value_at(f, x, c) is not None
        and _changes_sign(d2f_chart, c)
        and not (c in corners and _is_corner(df, x, c, sharp_only=True))  # no tangent line
    ]

    singularities = _singularities(f, x, domain, f_breaks)
    defined = lambda c: contains(domain, c)
    differentiable = lambda c: defined(c) and value_at(df, x, c) is not None
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


# ── Sign charts ───────────────────────────────────────────────────────────────

def _sign_chart(g: sp.Expr, x: sp.Symbol, region: sp.Set, cuts: list[sp.Expr]) -> Chart:
    """Split region at the cut points and record the sign of g on each piece."""
    chart: Chart = []
    for part in intervals(region):
        inner = [c for c in sorted_unique(cuts) if interior(part, c)]
        for a, b in pairwise([part.start, *inner, part.end]):
            match (a.is_infinite, b.is_infinite):
                case (True, True): probe = sp.S.Zero
                case (True, False): probe = b - 1
                case (False, True): probe = a + 1
                case _: probe = (a + b) / 2
            chart.append((sp.Interval.open(a, b), sign_at(g, x, probe)))
    return chart


def _where(chart: Chart, sign: int, joinable: Callable[[sp.Expr], bool]) -> sp.Set:
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


# ── Critical points ───────────────────────────────────────────────────────────

def _classify(c: sp.Expr, reason: str, x: sp.Symbol,
              derivatives: list[sp.Expr], df_chart: Chart) -> CriticalPoint:
    """Second-derivative test, then higher-order test, then first-derivative test."""
    y = value_at(derivatives[0], x, c)
    if reason == "f′ = 0":
        for n in range(2, MAX_DERIVATIVE_ORDER + 1):
            if n == len(derivatives):
                derivatives.append(derivative(derivatives[-1], x))
            match sign_at(derivatives[n], x, c):
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


def _is_corner(df: sp.Expr, x: sp.Symbol, c: sp.Expr, *, sharp_only: bool = False) -> bool:
    """True if f′ has no finite two-sided limit at c (a corner, cusp or vertical tangent).

    With sharp_only, only a true corner counts: two different finite slopes.
    """
    left, right = limit(df, x, c, "-"), limit(df, x, c, "+")
    if not (is_finite_real(left) and is_finite_real(right)):
        return not sharp_only
    return sp.simplify(left - right) != 0


# ── Singularities and asymptotes ──────────────────────────────────────────────

def _singularities(f: sp.Expr, x: sp.Symbol, domain: sp.Set,
                   candidates: list[sp.Expr]) -> list[Singularity]:
    eps = sp.Rational(1, 10**6)
    out = []
    for a in sorted_unique(candidates):
        if contains(domain, a):
            continue  # f is defined and continuous there: just a boundary such as 0 for √x
        left = right = None
        if domain.intersect(sp.Interval.open(a - eps, a)) is not sp.S.EmptySet:
            left = limit(f, x, a, "-")
        if domain.intersect(sp.Interval.open(a, a + eps)) is not sp.S.EmptySet:
            right = limit(f, x, a, "+")
        sides = [v for v in (left, right) if v is not None]
        if not sides:
            continue
        if any(v in (sp.oo, -sp.oo) for v in sides):
            kind = "infinite (vertical asymptote)"
        elif not all(is_finite_real(v) for v in sides):
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
        if is_finite_real(level := limit(f, x, end)):
            at_infinity.setdefault(("horizontal", level), []).append(label)
        elif (is_finite_real(m := limit(f / x, x, end)) and m != 0
                and is_finite_real(b := limit(f - m * x, x, end))):
            at_infinity.setdefault(("oblique", m * x + b), []).append(label)
    for (kind, rhs), ends in at_infinity.items():
        if sp.simplify(f - rhs) == 0:
            continue  # f is a line itself: calling it its own asymptote says nothing
        out.append(Asymptote(kind, rhs, "x → ±∞" if len(ends) == 2 else f"x → {ends[0]}"))
    return out
