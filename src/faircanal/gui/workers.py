"""Running a published script from the window without freezing it.

Nothing here changes what a script does. The command is exactly what a
reviewer would type - the interpreter running this window, then the script,
from the repository root - and the button that starts it prints that line
before it runs. The window is a convenience, never a second way of
producing the numbers.

The subprocess is configured the same way on all three platforms, with two
Windows specifics: no console window flashes up, and the child's encoding
is pinned to UTF-8 so Cyrillic output is not mangled by the console code
page.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


def child_environment() -> dict[str, str]:
    """The environment a script is run with: inherited, plus predictable I/O."""
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def _creation_flags() -> int:
    """Keep a console window from flashing up on Windows; 0 elsewhere."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def describe(command: list[str], cwd: Path) -> str:
    """The command as a reviewer would type it, with the interpreter explicit."""
    parts: list[str] = []
    for index, item in enumerate(command):
        if index == 0 and Path(item) == Path(sys.executable):
            text = "python"
        else:
            try:
                text = str(Path(item).relative_to(cwd).as_posix())
            except (ValueError, OSError):
                text = item
        parts.append(f'"{text}"' if " " in text else text)
    return " ".join(parts)


class ScriptRunner(QThread):
    """Run one command, emitting its output line by line."""

    line = pyqtSignal(str)
    finished_with = pyqtSignal(int)

    def __init__(self, command: list[str], cwd: Path) -> None:
        super().__init__()
        self.command = list(command)
        self.cwd = Path(cwd)
        self._process: subprocess.Popen[str] | None = None
        self._cancelled = False

    def run(self) -> None:  # pragma: no cover - needs a live process
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_environment(),
                creationflags=_creation_flags(),
            )
        except OSError as error:
            self.line.emit(f"{type(error).__name__}: {error}")
            self.finished_with.emit(-1)
            return

        assert self._process.stdout is not None
        for raw in self._process.stdout:
            self.line.emit(raw.rstrip("\n"))
        code = self._process.wait()
        if self._cancelled:
            self.line.emit("--- stopped by the user ---")
        self.finished_with.emit(code)

    def cancel(self) -> None:
        """Ask the child to stop; a scan that ignores it is left to finish."""
        self._cancelled = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:  # pragma: no cover - platform dependent
            pass
