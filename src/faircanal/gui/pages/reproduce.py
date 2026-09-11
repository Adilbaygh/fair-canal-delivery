"""9 · Rebuilding everything, with the command printed before it runs.

The buttons here start the published scripts as separate processes. They
pass no hidden options and read no private state: the line shown above the
log is exactly what a reviewer would type at a prompt, and running it there
gives the same files.

The scan is the long one. It writes each supply level the moment that level
finishes, so a run interrupted half way leaves a usable archive rather than
nothing, and the tables rebuild from whatever is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ... import archive as arc
from .. import widgets as w
from ..theme import COLORS
from ..workers import ScriptRunner, describe
from . import Context

#: The published steps, in the order they have to run. ``arguments`` are
#: fixed here so the button and the printed command can never disagree.
STEPS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "environment",
        "Муҳитни ёзиб олиш",
        "Record the environment",
        "record_environment.py",
        (),
    ),
    (
        "inputs",
        "Кириш маълумотларини чиқариш",
        "Export the inputs",
        "build_data.py",
        (),
    ),
    (
        "scan",
        "Тўлиқ сканер (узоқ)",
        "The full scan (long)",
        "run_experiments.py",
        (),
    ),
    (
        "tables",
        "Жадвалларни қуриш",
        "Build the tables",
        "make_tables.py",
        (),
    ),
    (
        "figures",
        "Расмларни чизиш",
        "Draw the figures",
        "make_figures.py",
        (),
    ),
    (
        "submission",
        "Журнал талабларини текшириш",
        "Check the journal requirements",
        "check_submission.py",
        (),
    ),
)


def build(context: Context) -> QWidget:
    return ReproducePage(context)


class ReproducePage(QWidget):
    def __init__(self, context: Context) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.context = context
        self.runner: ScriptRunner | None = None
        self.controller = self  # the window reaches run/stop through this

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = w.column(18)
        layout = w.body_of(body)

        layout.addWidget(
            w.page_title(
                context.pick(
                    "Ҳаммасини қайта юритиш",
                    "Reproduce everything",
                )
            )
        )
        layout.addWidget(
            w.page_lead(
                context.pick(
                    "Ҳар бир тугма эълон қилинган скриптни алоҳида жараёнда "
                    "юритади. Юритилаётган сатр журналнинг тепасида кўринади — "
                    "уни терминалда терсангиз, айнан шу файллар чиқади. Бу ойна "
                    "сонларни ҳисобламайди; у фақат ҳисоблаганни кўрсатади.",
                    "Each button runs a published script in a separate process. The "
                    "line being run is printed above the log — type it at a prompt "
                    "and you get exactly the same files. This window computes no "
                    "numbers; it only shows the ones the scripts wrote.",
                )
            )
        )

        layout.addWidget(self._steps_card())
        layout.addWidget(self._log_card())
        layout.addWidget(self._state_card())
        outer.addWidget(w.wrap_scroll(body))

    # -- the steps ----------------------------------------------------------

    def _steps_card(self) -> QWidget:
        context = self.context
        card = w.Card(
            context.pick("Босқичлар — юқоридан пастга", "The steps, top to bottom")
        )
        self.buttons: dict[str, QPushButton] = {}
        for key, uzbek, english, script, arguments in STEPS:
            row = QHBoxLayout()
            row.setSpacing(12)
            button = QPushButton(context.pick(uzbek, english))
            button.setFixedWidth(270)
            if key == "scan":
                button.setObjectName("Primary")
            button.clicked.connect(
                lambda _checked=False, s=script, a=arguments: self.run_script(s, a)
            )
            self.buttons[key] = button
            row.addWidget(button)
            row.addWidget(w.mono(f"scripts/{script}"), 1)
            card.add_layout(row)

        card.add(w.divider())
        row = QHBoxLayout()
        row.setSpacing(12)
        tests = QPushButton(context.pick("Тестларни юритиш", "Run the tests"))
        tests.setFixedWidth(270)
        tests.clicked.connect(self.run_tests)
        row.addWidget(tests)
        row.addWidget(w.mono("python -m pytest"), 1)
        card.add_layout(row)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.stop_button = QPushButton(context.ui("stop"))
        self.stop_button.setFixedWidth(270)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        card.add_layout(row)

        card.add(
            w.paragraph(
                context.pick(
                    "Сканер соатлар давом этиши мумкин. У ҳар бир манба даражасини "
                    "тугаган заҳоти ёзади, шунинг учун ярим йўлда тўхтатилган юриш "
                    "ҳам ишлатса бўладиган архив қолдиради, ва жадваллар мавжуд "
                    "нуқталардан қайта қурилади.",
                    "The scan can run for hours. It writes each supply level the "
                    "moment that level finishes, so a run stopped half way still "
                    "leaves a usable archive, and the tables rebuild from whatever "
                    "points are there.",
                )
            )
        )
        return card

    # -- the log ------------------------------------------------------------

    def _log_card(self) -> QWidget:
        context = self.context
        card = w.Card(context.pick("Юритилаётган сатр ва унинг чиқиши",
                                   "The command being run, and its output"))
        self.command_label = w.mono(context.ui("not_run"))
        card.add(self.command_label)
        self.console = QPlainTextEdit()
        self.console.setObjectName("Console")
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(260)
        card.add(self.console)

        row = QHBoxLayout()
        self.state_label = QLabel(context.ui("not_run"))
        self.state_label.setObjectName("CardBody")
        row.addWidget(self.state_label)
        row.addStretch(1)
        copy = QPushButton(context.ui("copy"))
        copy.setObjectName("Link")
        copy.clicked.connect(lambda: w.copy_to_clipboard(self.console.toPlainText()))
        row.addWidget(copy)
        card.add_layout(row)
        return card

    # -- what is on disk now ------------------------------------------------

    def _state_card(self) -> QWidget:
        context = self.context
        archive = self.context.archive
        missing = archive.missing()
        card = w.Card(
            context.pick("Ҳозир архивда нима бор", "What the archive holds right now")
        )
        rows = [
            [
                context.pick("Сканер белгилари", "Scan labels"),
                ", ".join(archive.labels()) or arc.DASH,
            ],
            [
                context.pick("Ечилган нуқталар (асосий)", "Solved points (main)"),
                str(len(archive.solved_percents(arc.MAIN_LABEL))),
            ],
            [
                context.pick("Чизилган расмлар", "Figures drawn"),
                str(sum(1 for figure in archive.figures() if figure.exists)),
            ],
            [
                context.pick("Жадвал файллари", "Table files"),
                str(len(archive.table_names())),
            ],
            [
                context.pick("Етишмайдиган файллар", "Files missing"),
                str(len(missing)),
            ],
        ]
        card.add(
            w.grid_table(
                [context.pick("Нима", "What"), context.pick("Ҳолат", "State")],
                rows,
                max_height=210,
            )
        )
        if missing:
            card.add(w.paragraph(", ".join(missing)))
        row = QHBoxLayout()
        row.addStretch(1)
        button = QPushButton(context.ui("open_folder"))
        button.setObjectName("Link")
        button.clicked.connect(lambda: w.open_in_file_manager(context.paths.results))
        row.addWidget(button)
        card.add_layout(row)
        return card

    # -- running ------------------------------------------------------------

    def run_script(self, script: str, arguments: tuple[str, ...] = ()) -> None:
        path = self.context.paths.scripts / script
        self._start([sys.executable, str(path), *arguments])

    def run_tests(self) -> None:
        self._start([sys.executable, "-m", "pytest"])

    def run_pipeline(self) -> None:
        """The menu's F5: the scan, which is the step everything else needs."""
        self.run_script("run_experiments.py")

    def _start(self, command: list[str]) -> None:
        context = self.context
        if self.runner is not None and self.runner.isRunning():
            return
        root: Path = context.paths.root
        self.console.clear()
        self.command_label.setText(describe(command, root))
        self.state_label.setText(context.ui("running"))
        self.state_label.setStyleSheet(f"color: {COLORS['accent']};")
        self._set_enabled(False)

        self.runner = ScriptRunner(command, root)
        self.runner.line.connect(self._append)
        self.runner.finished_with.connect(self._finished)
        self.runner.start()

    def _append(self, text: str) -> None:
        self.console.appendPlainText(text)

    def _finished(self, code: int) -> None:
        context = self.context
        if code == 0:
            self.state_label.setText(f"{context.ui('done')}  (exit 0)")
            self.state_label.setStyleSheet(f"color: {COLORS['filled']};")
        else:
            self.state_label.setText(f"{context.ui('failed')}  (exit {code})")
            self.state_label.setStyleSheet(f"color: {COLORS['fail']};")
        self._set_enabled(True)
        self.runner = None

    def _set_enabled(self, enabled: bool) -> None:
        for button in self.buttons.values():
            button.setEnabled(enabled)
        self.stop_button.setEnabled(not enabled)

    def stop(self) -> None:
        if self.runner is not None:
            self.runner.cancel()

    def shutdown(self) -> None:
        """Called when the page is discarded, so a run does not outlive it."""
        if self.runner is not None and self.runner.isRunning():
            self.runner.cancel()
            self.runner.wait(2000)
