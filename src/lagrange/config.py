"""Tunable settings shared by the analysis and the plot."""

from __future__ import annotations

# Used when f has infinitely many features (e.g. sin) or must be solved numerically.
DEFAULT_WINDOW: tuple[float, float] = (-10.0, 10.0)

TOLERANCE = 1e-9          # |value| below this counts as zero
MAX_DERIVATIVE_ORDER = 8  # highest order tried by the higher-order derivative test
SAMPLES = 4001            # grid size for numeric root finding and plotting
