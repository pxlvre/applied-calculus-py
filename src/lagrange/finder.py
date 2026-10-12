"""Locating the special points of an expression on the real line."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .config import SAMPLES
from .symbolic import complement, contains, domain_of, num, sample, sorted_unique, value_at


@dataclass
class Finder:
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
            return [p for p in self.points(sols) if value_at(expr, self.x, p) is not None]
        return self._numeric_zeros(expr, region)

    def _numeric_zeros(self, expr: sp.Expr, region: sp.Set) -> list[sp.Expr]:
        """Fallback when SymPy cannot solve exactly: sign changes + bisection."""
        self.clipped = True
        xs = np.linspace(*self.window, SAMPLES)
        ys = sample(expr, self.x, xs)
        roots = []
        for a, b, ya, yb in zip(xs, xs[1:], ys, ys[1:]):
            if not (np.isfinite(ya) and np.isfinite(yb)) or ya * yb > 0:
                continue
            try:
                r = sp.Float(sp.nsolve(expr, self.x, (a, b), solver="bisect"), 12)
            except (ValueError, ZeroDivisionError, TypeError):
                continue
            # A sign change across a pole is not a root.
            if (v := value_at(expr, self.x, r)) is not None and abs(num(v)) < 1e-6:
                roots.append(self._exact(expr, r))
        return [r for r in sorted_unique(roots) if contains(region, r)]

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
        bad = complement(domain_of(expr, self.x))
        try:
            edge = bad.boundary
        except (NotImplementedError, TypeError, ValueError):
            edge = bad
        return self.points(edge) + (self.kinks(expr) if with_kinks else [])

    def kinks(self, expr: sp.Expr) -> list[sp.Expr]:
        """Zeros of the arguments of |·|, sign(·) and Heaviside(·): possible corners."""
        return [p for atom in expr.atoms(sp.Abs, sp.sign, sp.Heaviside)
                for p in self.zeros(atom.args[0], sp.S.Reals)]
