# lagrange

Analyze and plot a real function of one real variable, the way a calculus course asks you to.

For a function `f(x)` it works out:

- the domain, `f′` and `f″`
- critical points (`f′ = 0` or `f′` undefined), each classified as a local maximum, a local
  minimum or neither — including the cases where `f″(c) = 0`
- intervals where `f` is increasing / decreasing
- intervals where `f` is concave up / concave down
- inflection points
- singularities (removable, jump, infinite, oscillating)
- vertical, horizontal and oblique asymptotes

…then plots `f`, `f′` and `f″` on stacked panels with all of it marked.

Written for the *Applied Calculus with Python* course (Johns Hopkins).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[notebook]"
```

## Use it from the terminal

```bash
lagrange "x^3 - 3x"           # or: python -m lagrange "x^3 - 3x"
lagrange "x/(x^2 - 4)"
lagrange "sin(t)" t           # a different variable
lagrange "x^4 - 2x^2" --no-plot
lagrange "exp(-x^2)" --save bell.png
lagrange "exp(-x^2)" --display pretty
```

Quote the function so the shell leaves `^`, `*` and `(` alone. Input is forgiving:
`2x` means `2*x`, `sin x` means `sin(x)`, `x^2` means `x**2`, and `ln` is the natural log.
Run `lagrange` with no arguments and it asks for the function instead.

`--display` picks how the math is written:

| Mode | Looks like | Notes |
| --- | --- | --- |
| `text` (default) | `-2*x*exp(-x**2)`, `sqrt(2)/2` | plain SymPy syntax |
| `pretty` | `f, f′, f″` laid out in 2-D; `√2/2`, `ℯ^(-1/2)`, `π` inline | Unicode, any terminal |
| `latex` | `$- 2 x e^{- x^{2}}$`, `$\frac{\sqrt{2}}{2}$` | LaTeX source to paste into notes |
| `image` | every line with math typeset as an image, in LaTeX's Computer Modern | iTerm2, Kitty, WezTerm or Ghostty; falls back to `pretty` elsewhere |

`image` uses matplotlib's mathtext, so no TeX installation is needed.

Functions with infinitely many features (`sin`, `tan`, …) are reported on a window,
`[-10, 10]` by default; change it with `--window LO HI`.

## Use it from a notebook

```python
from lagrange import analyze, report, plot

result = analyze("x^3 - 3x")
report(result)   # rendered as LaTeX inside Jupyter
plot(result)
```

`analyze` returns an `Analysis` dataclass, so the results are also there to pick apart:
`result.critical_points`, `result.inflection_points`, `result.increasing`, `result.asymptotes`, …

## Layout

| Module | What lives there |
| --- | --- |
| `parsing.py` | text → a real-valued SymPy expression |
| `symbolic.py` | defensive wrappers over SymPy and NumPy (domain, derivative, limit, sets) |
| `finder.py` | locating zeros and undefined points, clipped to a window |
| `analysis.py` | sign charts, the derivative tests, singularities, asymptotes |
| `models.py` | the result types (`Analysis`, `CriticalPoint`, `Singularity`, …) |
| `formatting.py` | writing SymPy objects the way a textbook would |
| `report.py` | the text report |
| `terminal_image.py` | typeset formulas shown inline in the terminal |
| `plotting.py`, `theme.py` | the figure and its palette |
| `cli.py` | the `lagrange` command |

## Develop

```bash
pip install -e ".[dev]"
ruff check src/
```
