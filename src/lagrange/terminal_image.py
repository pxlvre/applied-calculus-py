"""Typeset LaTeX as an image shown inline in the terminal (iTerm2 and Kitty protocols)."""

from __future__ import annotations

import base64
import io
import math
import os
import re
import shutil
import struct
import sys
from typing import Literal

type Protocol = Literal["iterm", "kitty"]

FONT_SIZE = 14   # pt, for the typeset formula
DPI = 200        # rendered resolution; the terminal scales it down to fit the rows
LINE_PT = 17     # assumed height of one terminal row, in pt
CELL_PT = 8      # assumed width of one terminal column, in pt

_CHUNK = 4096    # Kitty takes the payload in chunks of at most 4096 base64 bytes


def protocol() -> Protocol | None:
    """The inline-image protocol this terminal speaks, if any."""
    if not sys.stdout.isatty():
        return None
    env = os.environ
    term, program = env.get("TERM", ""), env.get("TERM_PROGRAM", "")
    if program in ("iTerm.app", "WezTerm") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    if term in ("xterm-kitty", "xterm-ghostty") or "KITTY_WINDOW_ID" in env or program == "ghostty":
        return "kitty"
    return None


def show(line: str, proto: Protocol) -> None:
    """Typeset ``line`` (words with ``$…$`` math) and print it inline, keeping its indent.

    Uses matplotlib's mathtext, so no TeX install is needed. Raises ValueError if mathtext
    cannot parse the line.
    """
    text = line.lstrip(" ")
    indent = len(line) - len(text)
    png = _render(_to_mathtext(text))
    width_px, height_px = struct.unpack(">II", png[16:24])  # from the PNG's IHDR chunk
    rows = max(1, math.ceil(height_px * 72 / DPI / LINE_PT))
    room = shutil.get_terminal_size().columns - indent
    # Too wide for the window: fit it to the width instead and let it shrink.
    size = {"iterm": f"height={rows}", "kitty": f"r={rows}"}[proto]
    if width_px * 72 / DPI / CELL_PT > room:
        size = {"iterm": f"width={room}", "kitty": f"c={room}"}[proto]

    data = base64.standard_b64encode(png).decode("ascii")
    out = sys.stdout
    out.write(" " * indent)
    if proto == "iterm":
        out.write(f"\x1b]1337;File=inline=1;size={len(png)};{size};preserveAspectRatio=1:{data}\x07")
    else:
        chunks = [data[i:i + _CHUNK] for i in range(0, len(data), _CHUNK)]
        for i, chunk in enumerate(chunks):
            more = int(i < len(chunks) - 1)
            keys = f"a=T,f=100,{size},m={more}" if i == 0 else f"m={more}"
            out.write(f"\x1b_G{keys};{chunk}\x1b\\")
    out.write("\n")
    out.flush()


# Symbols the report writes in its prose, which the text font lacks, as mathtext.
_SYMBOLS = {"→": r"\rightarrow", "≈": r"\approx", "±": r"\pm", "∞": r"\infty", "−": "-",
            "′": "'", "″": "''", "⁻": "^{-}", "⁺": "^{+}"}
_VARIABLE = re.compile(r"(?<![A-Za-z\\])[xyf](?![A-Za-z])")


def _to_mathtext(text: str) -> str:
    """Move the prose's symbols and lone variables (x, f, f′) into math mode."""
    parts = text.split("$")
    for i in range(0, len(parts), 2):  # the prose between the $…$ pairs
        prose = "".join(f"${_SYMBOLS[c]}$" if c in _SYMBOLS else c for c in parts[i])
        parts[i] = _VARIABLE.sub(lambda m: f"${m[0]}$", prose)
    return "$".join(parts).replace("$$", "")


def _render(tex: str) -> bytes:
    """PNG bytes of ``tex`` on a transparent background, cropped to the ink."""
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties

    # Computer Modern, LaTeX's own face, for the words and the math.
    prop = FontProperties(family="cmr10", size=FONT_SIZE, math_fontfamily="cm")
    fig = Figure(figsize=(1, 1))
    fig.text(0, 0, tex, fontproperties=prop,
             color="#e8e8e8" if _dark_background() else "#111111")
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=DPI, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.03)
    return buffer.getvalue()


_dark: bool | None = None


def _dark_background() -> bool:
    """Ask the terminal for its background colour (OSC 11); assume dark if it won't say."""
    global _dark
    if _dark is None:
        rgb = _query_background()
        _dark = rgb is None or 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.5
    return _dark


def _query_background() -> tuple[float, float, float] | None:
    try:
        import select
        import termios
        import tty
    except ImportError:  # not a POSIX terminal
        return None
    try:
        with open("/dev/tty", "r+b", buffering=0) as tty_file:
            fd = tty_file.fileno()
            saved = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                tty_file.write(b"\x1b]11;?\x07")
                reply = b""
                while select.select([fd], [], [], 0.2)[0]:
                    reply += os.read(fd, 64)
                    if reply.endswith((b"\x07", b"\x1b\\")):
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    except (OSError, termios.error):
        return None
    match = re.search(rb"rgb:([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", reply, re.IGNORECASE)
    if match is None:
        return None
    r, g, b = (int(c, 16) / (16 ** len(c) - 1) for c in match.groups())
    return r, g, b
