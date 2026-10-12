"""lagrange — analyze and plot a real function of one real variable.

    >>> from lagrange import analyze, plot, report
    >>> result = analyze("x^3 - 3x")
    >>> report(result)
    >>> plot(result)
"""

from __future__ import annotations

from .analysis import analyze
from .config import DEFAULT_WINDOW, MAX_DERIVATIVE_ORDER, SAMPLES, TOLERANCE
from .formatting import display_form, fmt, fmt_set
from .models import Analysis, Asymptote, CriticalPoint, Kind, Point, Singularity
from .parsing import parse_function
from .plotting import plot
from .report import report

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_WINDOW",
    "MAX_DERIVATIVE_ORDER",
    "SAMPLES",
    "TOLERANCE",
    "Analysis",
    "Asymptote",
    "CriticalPoint",
    "Kind",
    "Point",
    "Singularity",
    "__version__",
    "analyze",
    "display_form",
    "fmt",
    "fmt_set",
    "parse_function",
    "plot",
    "report",
]
