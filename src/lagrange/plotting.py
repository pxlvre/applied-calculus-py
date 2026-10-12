"""Stacked plots of f, f′ and f″ with every feature of the analysis marked."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import Patch

from .config import SAMPLES
from .formatting import display_form, fmt_short
from .models import Analysis, Kind
from .symbolic import intervals, is_finite_real, num, sample, value_at
from .theme import COLORS


def plot(a: Analysis, xlim: tuple[float, float] | None = None) -> plt.Figure:
    """Plot f, f′ and f″ on stacked panels with every feature marked."""
    x0, x1 = xlim or _x_range(a)
    xs = np.linspace(x0, x1, SAMPLES)
    # Where f, f′ or f″ may jump: don't let the line bridge those gaps.
    breaks = [num(s.x) for s in a.singularities] + [
        num(p.x) for p in a.critical_points if p.reason == "f′ undefined"]
    in_view = lambda pts: [p for p in pts if x0 <= num(p.x) <= x1]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10),
                             gridspec_kw={"height_ratios": [2.2, 1, 1]},
                             facecolor=COLORS["surface"])
    panels = [("f(x)", a.f, COLORS["f"]),
              ("f^{\\prime}(x)", a.df, COLORS["df"]),
              ("f^{\\prime\\prime}(x)", a.d2f, COLORS["d2f"])]

    for ax, (name, expr, color) in zip(axes, panels):
        ys = sample(expr, a.x, xs)
        extra = [num(p.y) for p in in_view(a.critical_points + a.inflection_points)
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
        cx = num(p.x)
        for ax in axes:
            ax.axvline(cx, color=color, lw=0.8, ls=":", zorder=2)
        if p.y is not None:
            _dot(ax_f, cx, num(p.y), shape, color, f"({fmt_short(p.x)}, {fmt_short(p.y)})"
                 if label_points else None)
        if p.reason == "f′ = 0":
            _dot(ax_df, cx, 0.0, "o", color)

    for p in in_view(a.inflection_points):
        cx = num(p.x)
        for ax in axes:
            ax.axvline(cx, color=COLORS["inflection"], lw=0.8, ls=":", zorder=2)
        _dot(ax_f, cx, num(p.y), "D", COLORS["inflection"],
             f"({fmt_short(p.x)}, {fmt_short(p.y)})" if label_points else None)
        if (v := value_at(a.d2f, a.x, p.x)) is not None:
            _dot(ax_d2f, cx, num(v), "D", COLORS["inflection"])

    for s in a.singularities:  # hollow circle where the limit is finite (a "hole")
        if s.kind.startswith(("removable", "jump", "boundary")) and x0 <= num(s.x) <= x1:
            for v in (s.left, s.right):
                if is_finite_real(v):
                    ax_f.plot(num(s.x), num(v), "o", ms=8, mfc=COLORS["surface"],
                              mec=COLORS["f"], mew=2, zorder=5)

    ax_d2f.set_xlim(x0, x1)
    ax_d2f.set_xlabel(a.x.name, color=COLORS["ink2"])
    _legend(fig, a)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ── Ranges ────────────────────────────────────────────────────────────────────

def _x_range(a: Analysis) -> tuple[float, float]:
    if a.window:
        return a.window
    xs = [num(p.x) for p in (*a.critical_points, *a.inflection_points, *a.singularities)]
    if not xs:
        lo, hi = -5.0, 5.0
    else:
        span = max(max(xs) - min(xs), 2.0)
        lo, hi = min(xs) - 0.6 * span, max(xs) + 0.6 * span
    if a.domain.inf.is_finite:       # don't waste space where f is undefined
        lo = max(lo, num(a.domain.inf) - 0.05 * (hi - lo))
    if a.domain.sup.is_finite:
        hi = min(hi, num(a.domain.sup) + 0.05 * (hi - lo))
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
    return (max(lo - pad, finite.min() - 0.05 * (hi - lo)),
            min(hi + pad, finite.max() + 0.05 * (hi - lo)))


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


# ── Decorations ───────────────────────────────────────────────────────────────

def _math_title(name: str, expr: sp.Expr) -> str:
    text = f"${name} = {sp.latex(expr)}$"
    try:
        MathTextParser("agg").parse(text)
        return text
    except ValueError:  # LaTeX that matplotlib's mathtext cannot render
        return f"{name.replace('\\prime', '′')} = {sp.sstr(expr)}"


def _dot(ax, x: float, y: float, shape: str, color: str, label: str | None = None) -> None:
    ax.plot(x, y, shape, ms=9, color=color, mec=COLORS["surface"], mew=2, zorder=6)
    if label:
        ax.annotate(label, (x, y), xytext=(6, 8), textcoords="offset points",
                    fontsize=8.5, color=COLORS["ink"], zorder=7)


def _draw_concavity(ax, a: Analysis, x0: float, x1: float) -> None:
    for s, color, hatch in ((a.concave_up, COLORS["up"], None),
                            (a.concave_down, COLORS["down"], "///")):
        for iv in intervals(s):
            lo, hi = max(num(iv.start), x0), min(num(iv.end), x1)
            if lo < hi:
                ax.axvspan(lo, hi, color=color, hatch=hatch, ec="#cfcdc6" if hatch else None,
                           lw=0, zorder=0)


def _draw_asymptotes(ax, a: Analysis, xs: np.ndarray) -> None:
    style = {"color": COLORS["asymptote"], "lw": 1.2, "ls": "--", "zorder": 2}
    for asy in a.asymptotes:
        match asy.kind:
            case "vertical":
                ax.axvline(num(asy.rhs), **style)
            case _:
                side = xs if asy.where == "x → ±∞" else (xs[xs >= 0] if "+" in asy.where else xs[xs <= 0])
                ax.plot(side, sample(asy.rhs, a.x, side), **style)


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
