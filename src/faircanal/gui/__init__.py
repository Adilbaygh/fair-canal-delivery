"""The desktop window: an appendix to the article, not a second way of computing.

Two rules shape everything under this package.

The window **reads**. Every number on screen comes out of a file the
published scripts wrote, and it carries that file with it. The one thing
the window starts on its own is a published script, in a separate process,
with the command printed first.

The window **never invents a verdict**. A supply level was solved, or
proved infeasible, or left without an answer, and the third is not folded
into the second. Where there is no computed number, there is a dash.

Importing this package pulls in nothing heavy. ``i18n`` and ``pages`` are
plain data and can be read on a machine with no GUI toolkit; PyQt6 is
imported only when a window or a page is actually built.
"""

from __future__ import annotations

__all__ = ["launch"]


def launch(*arguments, **keywords) -> int:
    """Open the window. Imported lazily so PyQt6 is only needed to run it."""
    from .app import launch as _launch  # noqa: PLC0415

    return _launch(*arguments, **keywords)
