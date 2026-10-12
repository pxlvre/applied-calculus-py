"""The result types produced by :func:`lagrange.analysis.analyze`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import sympy as sp

from .formatting import fmt

type Chart = list[tuple[sp.Interval, int | None]]
"""A sign chart: the sign of an expression on each piece of a split region."""


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
