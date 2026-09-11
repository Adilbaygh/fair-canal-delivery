"""The small set of widgets every page is built from.

The one that matters is :class:`ValueCard`. It cannot be constructed
without a source string, so the promise the archive makes - no number on
screen without the file it came from - is kept by the shape of the code
rather than by anybody remembering it.

The second is :func:`verdict_chip`, which is the only way a verdict reaches
the screen. It prints the word as well as the colour, so the three-way
distinction between solved, infeasible and no-verdict survives a
black-and-white printout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import archive as arc
from . import i18n
from .theme import COLORS, CRITERION_COLOR, PROVENANCE_STYLE, VERDICT_STYLE

CONTENT_WIDTH = 1040


# ----------------------------------------------------------------------- shell


def open_in_file_manager(path: Path) -> None:
    """Reveal a file or folder using whatever the platform provides."""
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def open_file(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def copy_to_clipboard(text: str) -> None:
    clipboard = QGuiApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(text)


def wrap_scroll(inner: QWidget) -> QScrollArea:
    """Put a page inside a scroll area that keeps a comfortable reading width."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    holder = QWidget()
    holder.setObjectName("Page")
    outer = QHBoxLayout(holder)
    outer.setContentsMargins(28, 24, 28, 32)
    outer.addStretch(1)
    inner.setMaximumWidth(CONTENT_WIDTH)
    outer.addWidget(inner, 20)
    outer.addStretch(1)
    area.setWidget(holder)
    return area


def column(spacing: int = 16, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QWidget:
    holder = QWidget()
    holder.setObjectName("Page")
    layout = QVBoxLayout(holder)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    return holder


def body_of(holder: QWidget) -> QVBoxLayout:
    layout = holder.layout()
    assert isinstance(layout, QVBoxLayout)
    return layout


# ---------------------------------------------------------------------- labels


def page_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("PageTitle")
    label.setWordWrap(True)
    return label


def page_lead(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("PageLead")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def paragraph(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("CardBody")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(False)
    return label


def quotation(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Quote")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def mono(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Mono")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def rich(text: str) -> str:
    """Turn the little Markdown a published caption carries into rich text.

    The captions are written for the manuscript, where ``**a**`` is bold
    and a backtick is code. They are read from the file the drawing script
    wrote, so the markers arrive with them; showing them raw would put
    asterisks on screen.
    """
    out: list[str] = []
    rest = text
    while "**" in rest:
        before, _, after = rest.partition("**")
        marked, sep, remainder = after.partition("**")
        if not sep:
            out.append(before + "**" + after)
            rest = ""
            break
        out.append(f"{before}<b>{marked}</b>")
        rest = remainder
    out.append(rest)
    joined = "".join(out)
    pieces = joined.split("`")
    for index in range(1, len(pieces), 2):
        pieces[index] = f"<code>{pieces[index]}</code>"
    return "".join(pieces)


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.Shape.NoFrame)
    return line


def chip(text: str, foreground: str, background: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Chip")
    label.setStyleSheet(f"color: {foreground}; background: {background};")
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return label


def verdict_chip(language: str, verdict: str) -> QLabel:
    """The only way a verdict reaches the screen: colour *and* the word."""
    foreground, background = VERDICT_STYLE.get(verdict, VERDICT_STYLE["missing"])
    return chip(i18n.verdict_label(language, verdict), foreground, background)


def provenance_chip(language: str, kind: str) -> QLabel:
    foreground, background = PROVENANCE_STYLE.get(
        kind, (COLORS["ink_soft"], COLORS["neutral_soft"])
    )
    return chip(i18n.provenance_label(language, kind), foreground, background)


# ----------------------------------------------------------------------- cards


class Card(QFrame):
    """A white panel with an optional title. Everything else goes inside it."""

    def __init__(self, title: str = "", subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("Card")
        # A card is as tall as what is in it. Without this a tall window
        # hands the leftover height to the cards, which then space their
        # own contents out until a table floats in the middle of nothing.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 16, 18, 18)
        self._layout.setSpacing(10)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        if title:
            heading = QLabel(title)
            heading.setObjectName("CardTitle")
            heading.setWordWrap(True)
            self._layout.addWidget(heading)
        if subtitle:
            self._layout.addWidget(paragraph(subtitle))

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_text(self, text: str) -> QLabel:
        return self.add(paragraph(text))  # type: ignore[return-value]


class ValueCard(QFrame):
    """One quantity: the value, what it is, and the file it was read from.

    ``source`` is required and is not allowed to be empty. A caller with no
    source has nothing this window is willing to display.
    """

    def __init__(
        self,
        value: str,
        caption: str,
        source: str,
        tone: str = "ink",
        note: str = "",
    ) -> None:
        # Checked before the widget exists, so the rule holds whether or not
        # there is an application running - and so a caller with no source
        # is stopped rather than half-built.
        if not source.strip():
            raise ValueError("a value card needs the file its number came from")
        super().__init__()
        self.setObjectName("NumberCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        shown = QLabel(value)
        shown.setObjectName("NumberValue")
        shown.setStyleSheet(f"color: {COLORS.get(tone, COLORS['ink'])};")
        layout.addWidget(shown)

        text = QLabel(caption)
        text.setObjectName("NumberCaption")
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(text)

        if note:
            extra = QLabel(note)
            extra.setObjectName("NumberCaption")
            extra.setStyleSheet(f"color: {COLORS['ink_faint']};")
            extra.setWordWrap(True)
            extra.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(extra)

        layout.addSpacing(2)
        origin = QLabel(source)
        origin.setObjectName("NumberSource")
        origin.setWordWrap(True)
        origin.setToolTip(source)
        layout.addWidget(origin)


def card_row(cards: Iterable[QWidget], columns: int = 3) -> QWidget:
    """Lay value cards out in a grid that stays readable when the window narrows."""
    holder = QWidget()
    holder.setObjectName("Page")
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(12)
    for index, widget in enumerate(cards):
        grid.addWidget(widget, index // columns, index % columns)
    for position in range(columns):
        grid.setColumnStretch(position, 1)
    return holder


def key_values(pairs: Iterable[tuple[str, str]], columns: int = 2) -> QWidget:
    holder = QWidget()
    holder.setObjectName("Page")
    grid = QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(7)
    for index, (key, value) in enumerate(pairs):
        row = index // columns
        base = (index % columns) * 2
        name = QLabel(key)
        name.setObjectName("CardBody")
        name.setWordWrap(True)
        shown = QLabel(value)
        shown.setWordWrap(True)
        shown.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(name, row, base)
        grid.addWidget(shown, row, base + 1)
    for position in range(columns):
        grid.setColumnStretch(position * 2 + 1, 1)
    return holder


# ---------------------------------------------------------------------- tables


def units(text: str) -> str:
    """``m^3/s`` as the file writes it, ``m³/s`` as a reader reads it."""
    return text.replace("^3", "\u00b3").replace("^2", "\u00b2")


def _looks_numeric(text: str) -> bool:
    stripped = text.strip().replace("−", "-").replace(" ", "").replace(",", "")
    if not stripped or stripped == arc.DASH:
        return False
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def grid_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    max_height: int = 340,
    tones: Sequence[Sequence[str | None]] | None = None,
    fit: bool = False,
) -> QTableWidget:
    """A read-only table. ``tones`` colours individual cells by meaning.

    ``fit`` divides the available width between the columns instead of
    sizing each to its content, which keeps a wide table of short numbers
    on screen rather than behind a horizontal scrollbar.
    """
    widget = QTableWidget(len(rows), len(header))
    widget.setHorizontalHeaderLabels(list(header))
    widget.verticalHeader().setVisible(False)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setAlternatingRowColors(False)
    widget.setWordWrap(False)
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            if column_index >= len(header):
                continue
            item = QTableWidgetItem(str(cell))
            if _looks_numeric(str(cell)):
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            if tones is not None:
                try:
                    tone = tones[row_index][column_index]
                except (IndexError, TypeError):
                    tone = None
                if tone:
                    from PyQt6.QtGui import QColor  # noqa: PLC0415

                    item.setForeground(QColor(COLORS.get(tone, COLORS["ink"])))
            widget.setItem(row_index, column_index, item)
    head = widget.horizontalHeader()
    if fit:
        # The first column names the row and is usually short; the rest carry
        # the values and should share what is left equally. Stretching every
        # column alike would spend width on the labels and then elide the
        # words that matter.
        head.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        if len(header) > 1:
            head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    else:
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setStretchLastSection(True)
    widget.resizeRowsToContents()
    # Exactly as tall as its contents, up to the cap. A minimum guessed from
    # a row count leaves a short table scrolling inside its own frame, which
    # hides the last row - and the last row is often the one that matters.
    needed = (
        widget.horizontalHeader().height()
        + sum(widget.rowHeight(index) for index in range(len(rows)))
        + 4
    )
    height = min(max_height, max(needed, 70))
    widget.setMinimumHeight(height)
    widget.setMaximumHeight(height)
    return widget


class TableCard(Card):
    """One published CSV, with its path and a button that opens it."""

    def __init__(
        self,
        table: arc.Table | None,
        language: str,
        title: str = "",
        note: str = "",
        max_height: int = 320,
    ) -> None:
        super().__init__(title or (table.source if table else ""))
        if table is None:
            self.add(paragraph(i18n.ui(language, "no_table")))
            return
        if note:
            self.add(paragraph(note))
        self.add(grid_table(table.header, table.rows, max_height))

        footer = QHBoxLayout()
        footer.setSpacing(8)
        origin = QLabel(f"{table.source}  ·  {len(table.rows)} {i18n.ui(language, 'rows_shown')}")
        origin.setObjectName("NumberSource")
        footer.addWidget(origin)
        footer.addStretch(1)
        button = QPushButton(i18n.ui(language, "open_csv"))
        button.setObjectName("Link")
        button.clicked.connect(lambda: open_file(table.path))
        footer.addWidget(button)
        self.add_layout(footer)


# --------------------------------------------------------------------- figures


class FigureCard(Card):
    """A published figure at screen size, with the caption the article prints."""

    MAX_WIDTH = 960

    def __init__(self, figure: arc.Figure, source: str, language: str) -> None:
        word = i18n.pick(language, "Расм", "Figure")
        super().__init__(f"{word} {figure.number}")
        if not figure.exists:
            self.add(paragraph(i18n.ui(language, "no_figure")))
            return

        image = QLabel()
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(figure.path))
        if not pixmap.isNull():
            if pixmap.width() > self.MAX_WIDTH:
                pixmap = pixmap.scaledToWidth(
                    self.MAX_WIDTH, Qt.TransformationMode.SmoothTransformation
                )
            image.setPixmap(pixmap)
        self.add(image)
        if figure.caption:
            self.add(paragraph(rich(figure.caption)))

        footer = QHBoxLayout()
        footer.setSpacing(8)
        origin = QLabel(source)
        origin.setObjectName("NumberSource")
        footer.addWidget(origin)
        footer.addStretch(1)
        button = QPushButton(i18n.ui(language, "open_full_size"))
        button.setObjectName("Link")
        button.clicked.connect(lambda: open_file(figure.path))
        footer.addWidget(button)
        self.add_layout(footer)


# ---------------------------------------------------------------------- charts


class Chart(QWidget):
    """A matplotlib figure inside the window, or an honest note if it is absent.

    matplotlib is already required to draw the article's figures, so
    embedding it adds no dependency; the window still has to open on a
    machine where the import fails rather than dying at start-up.
    """

    #: Margins as a fraction of the canvas, rather than a layout engine.
    #: The canvas sits in a scroll area and grows after its first paint, and
    #: a layout computed once leaves the axis labels sliced off at the new
    #: size. A fraction holds whatever the pane does.
    MARGINS = {"left": 0.10, "right": 0.985, "top": 0.94, "bottom": 0.19}

    def __init__(self, height: int = 300, margins: dict[str, float] | None = None) -> None:
        super().__init__()
        self.setObjectName("Page")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.margins = dict(self.MARGINS)
        if margins:
            self.margins.update(margins)
        self.canvas = None
        self.figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: PLC0415
            from matplotlib.figure import Figure  # noqa: PLC0415
        except Exception as error:  # pragma: no cover - depends on the machine
            self._layout.addWidget(paragraph(f"matplotlib is not available: {error}"))
            return
        self.figure = Figure(figsize=(7.6, height / 100), dpi=100, facecolor=COLORS["card"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(height)
        self._layout.addWidget(self.canvas)

    def axes(self):
        """A cleared single axis, styled like the rest of the window."""
        if self.figure is None:
            return None
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(COLORS["card"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(COLORS["border_strong"])
        ax.tick_params(colors=COLORS["ink_soft"], labelsize=9)
        ax.grid(True, color=COLORS["border"], linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        return ax

    def draw(self) -> None:
        if self.canvas is not None and self.figure is not None:
            self.figure.subplots_adjust(**self.margins)
            self.canvas.draw_idle()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt's spelling
        """Re-lay the axes when the pane grows, or the labels stay clipped."""
        super().resizeEvent(event)
        if self.canvas is not None:
            self.canvas.draw_idle()

    def say(self, text: str) -> None:
        """Put one sentence where the picture would have been."""
        ax = self.axes()
        if ax is None:
            return
        ax.axis("off")
        ax.text(
            0.5, 0.5, text, ha="center", va="center", wrap=True,
            color=COLORS["ink_soft"], fontsize=10,
        )
        self.draw()


def short_name(user: str) -> str:
    """``corning-3-user`` reads as ``3`` on an axis; the full name is elsewhere.

    The published tables spell the same thing as ``gate 3``, so the two say
    the same about the same user.
    """
    for part in user.split("-"):
        if part.isdigit():
            return part
    return user


def short_name_sort_key(user: str) -> tuple[int, str]:
    """Order users by gate number, so a chart and a table agree row for row."""
    short = short_name(user)
    return (int(short), user) if short.isdigit() else (10**6, user)


def criterion_color(code: str) -> str:
    return CRITERION_COLOR.get(code, COLORS["ink_soft"])


# ------------------------------------------------------------------- selectors


def combo(items: Iterable[tuple[str, object]], tooltip: str = "") -> QComboBox:
    box = QComboBox()
    for label, value in items:
        box.addItem(label, value)
    box.setMinimumWidth(150)
    if tooltip:
        box.setToolTip(tooltip)
    return box


def icon_size() -> QSize:  # pragma: no cover - trivial
    return QSize(16, 16)
