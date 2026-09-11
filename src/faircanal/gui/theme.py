"""Colours, fonts and one stylesheet.

Three things shaped the palette.

The study is about **who goes short**, so the two states that matter - a
share that was filled and a share that was not - get their own colours, and
the volume that is missing gets a third. Nothing carries meaning by colour
alone: every state is also spelled out in words, so the window survives a
black-and-white printout and a reader who does not separate the two hues.

The window has to read the same on Windows, macOS and Linux, so every font
is given as a family list ending in a face that ships with the platform and
covers Cyrillic.

Numbers are the content. The chrome is dark and quiet, the page is light,
and the accent is used for one thing at a time.
"""

from __future__ import annotations

#: Family lists, in the order Qt should try them. Every list ends in a face
#: that exists on the platform it targets and renders Cyrillic.
FONT_UI = (
    '"Segoe UI", "SF Pro Text", -apple-system, "Helvetica Neue", '
    '"Noto Sans", "DejaVu Sans", sans-serif'
)
FONT_MONO = (
    '"Cascadia Mono", "SF Mono", Menlo, Consolas, '
    '"DejaVu Sans Mono", "Noto Sans Mono", monospace'
)

COLORS: dict[str, str] = {
    # surfaces
    "page": "#f5f6f8",
    "card": "#ffffff",
    "sidebar": "#16233d",
    "sidebar_hover": "#22334f",
    "header": "#101a2e",
    "border": "#dadfe6",
    "border_strong": "#b8c1cd",
    # text
    "ink": "#16233d",
    "ink_soft": "#5c6879",
    "ink_faint": "#8b96a5",
    "on_dark": "#e9edf3",
    "on_dark_soft": "#96a3b8",
    # meaning
    "accent": "#2f6fb3",        # the canal, and the proposed criterion
    "accent_soft": "#e5eff8",
    "filled": "#1f7a4d",        # an order met in full, a solved point
    "filled_soft": "#e6f3ec",
    "short": "#c2703d",         # the volume somebody did not receive
    "short_soft": "#fbeee4",
    "undecided": "#8a6d1f",     # the solver returned no verdict at all
    "undecided_soft": "#faf3de",
    "fail": "#a92f2f",          # proved impossible
    "fail_soft": "#fbeaea",
    "neutral_soft": "#eef0f4",
}

#: How each verdict is drawn. The label itself is in ``i18n.VERDICT``; this
#: table only says which two colours go with it.
VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "solved": (COLORS["filled"], COLORS["filled_soft"]),
    "infeasible": (COLORS["fail"], COLORS["fail_soft"]),
    "undecided": (COLORS["undecided"], COLORS["undecided_soft"]),
    "level only": (COLORS["accent"], COLORS["accent_soft"]),
    "missing": (COLORS["ink_faint"], COLORS["neutral_soft"]),
}

#: Provenance labels, coloured the same way and equally spelled out.
PROVENANCE_STYLE: dict[str, tuple[str, str]] = {
    "observed": (COLORS["filled"], COLORS["filled_soft"]),
    "derived": (COLORS["accent"], COLORS["accent_soft"]),
    "assumed": (COLORS["short"], COLORS["short_soft"]),
}

#: One colour per criterion, used in the charts and in the legend chips so
#: that a curve and its row in a table are the same colour everywhere.
CRITERION_COLOR: dict[str, str] = {
    "B1": "#8b96a5",
    "B2": "#5c6879",
    "B3": "#c2703d",
    "B4": "#2f6fb3",
    "B5": "#7a5ea8",
    "M1": "#1f7a4d",
}


APP_STYLE = f"""
* {{
    font-family: {FONT_UI};
}}

QWidget {{
    color: {COLORS['ink']};
}}

QMainWindow, QWidget#Page, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {COLORS['page']};
}}

/* ------------------------------------------------------------------ header */

QFrame#Header {{
    background: {COLORS['header']};
    border: none;
}}

QLabel#HeaderTitle {{
    color: #ffffff;
    font-size: 17px;
    font-weight: 600;
}}

QLabel#HeaderStrip {{
    color: {COLORS['on_dark_soft']};
    font-size: 12px;
    letter-spacing: 0.2px;
}}

QPushButton#LanguageButton {{
    background: rgba(255, 255, 255, 0.10);
    color: {COLORS['on_dark']};
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 13px;
    padding: 5px 14px;
    font-size: 12px;
}}

QPushButton#LanguageButton:hover {{
    background: rgba(255, 255, 255, 0.18);
}}

/* ----------------------------------------------------------------- sidebar */

QFrame#Sidebar {{
    background: {COLORS['sidebar']};
    border: none;
}}

QPushButton#NavButton {{
    background: transparent;
    color: {COLORS['on_dark_soft']};
    border: none;
    border-left: 3px solid transparent;
    padding: 11px 16px 11px 13px;
    text-align: left;
    font-size: 13px;
}}

QPushButton#NavButton:hover {{
    background: {COLORS['sidebar_hover']};
    color: {COLORS['on_dark']};
}}

QPushButton#NavButton:checked {{
    background: {COLORS['sidebar_hover']};
    color: #ffffff;
    border-left: 3px solid {COLORS['accent']};
    font-weight: 600;
}}

QLabel#SidebarCaption {{
    color: #6f7f99;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    padding: 16px 16px 6px 16px;
}}

/* -------------------------------------------------------------------- card */

QFrame#Card {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

QLabel#CardTitle {{
    font-size: 15px;
    font-weight: 600;
}}

QLabel#CardBody {{
    color: {COLORS['ink_soft']};
    font-size: 13px;
}}

QLabel#PageTitle {{
    font-size: 23px;
    font-weight: 600;
}}

QLabel#PageLead {{
    color: {COLORS['ink_soft']};
    font-size: 14px;
}}

QLabel#SectionTitle {{
    font-size: 15px;
    font-weight: 700;
    color: {COLORS['ink']};
}}

QLabel#Quote {{
    color: {COLORS['ink_soft']};
    font-size: 13px;
    border-left: 3px solid {COLORS['border_strong']};
    padding: 2px 0 2px 12px;
}}

/* ------------------------------------------------------------ number cards */

QFrame#NumberCard {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}

QLabel#NumberValue {{
    font-size: 26px;
    font-weight: 600;
}}

QLabel#NumberCaption {{
    font-size: 12px;
    color: {COLORS['ink_soft']};
}}

QLabel#NumberSource {{
    font-family: {FONT_MONO};
    font-size: 10px;
    color: {COLORS['ink_faint']};
}}

QLabel#Chip {{
    border-radius: 9px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}}

QLabel#Mono {{
    font-family: {FONT_MONO};
    font-size: 12px;
    color: {COLORS['ink_soft']};
}}

/* ------------------------------------------------------------------ tables */

QTableWidget {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    gridline-color: {COLORS['border']};
    font-size: 12px;
    selection-background-color: {COLORS['accent_soft']};
    selection-color: {COLORS['ink']};
}}

QHeaderView::section {{
    background: {COLORS['neutral_soft']};
    color: {COLORS['ink_soft']};
    border: none;
    border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 700;
}}

QTableWidget::item {{
    padding: 4px 8px;
}}

/* ----------------------------------------------------------------- buttons */

QPushButton {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 7px;
    padding: 7px 15px;
    font-size: 13px;
}}

QPushButton:hover {{
    border-color: {COLORS['accent']};
    color: {COLORS['accent']};
}}

QPushButton:disabled {{
    color: {COLORS['ink_faint']};
    border-color: {COLORS['border']};
}}

QPushButton#Primary {{
    background: {COLORS['accent']};
    border: 1px solid {COLORS['accent']};
    color: #ffffff;
    font-weight: 600;
}}

QPushButton#Primary:hover {{
    background: #285f9b;
    color: #ffffff;
}}

QPushButton#Primary:disabled {{
    background: {COLORS['border_strong']};
    border-color: {COLORS['border_strong']};
    color: #ffffff;
}}

QPushButton#Link {{
    background: transparent;
    border: none;
    color: {COLORS['accent']};
    padding: 2px 4px;
    font-size: 12px;
    text-align: left;
}}

QPushButton#Link:hover {{
    text-decoration: underline;
}}

/* -------------------------------------------------------------- log output */

QPlainTextEdit#Console {{
    background: #0d1524;
    color: #cdd8ea;
    border: 1px solid {COLORS['header']};
    border-radius: 8px;
    font-family: {FONT_MONO};
    font-size: 12px;
    padding: 8px;
}}

/* ------------------------------------------------------------ misc widgets */

QScrollArea {{
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['border_strong']};
    border-radius: 5px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['ink_faint']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {COLORS['border_strong']};
    border-radius: 5px;
    min-width: 28px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QFrame#Divider {{
    background: {COLORS['border']};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QComboBox {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 7px;
    padding: 6px 10px;
    font-size: 13px;
}}

QComboBox:hover {{
    border-color: {COLORS['accent']};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {COLORS['border']};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS['card']};
    border: 2px solid {COLORS['accent']};
    width: 13px;
    margin: -6px 0;
    border-radius: 8px;
}}

QToolTip {{
    background: {COLORS['header']};
    color: {COLORS['on_dark']};
    border: none;
    padding: 6px 8px;
    font-size: 12px;
}}

/* ------------------------------------------------------ menu and status bar */

QMenuBar {{
    background: {COLORS['header']};
    color: {COLORS['on_dark']};
    border: none;
    padding: 2px 6px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 5px;
}}

QMenuBar::item:selected {{
    background: rgba(255, 255, 255, 0.13);
}}

QMenu {{
    background: {COLORS['card']};
    border: 1px solid {COLORS['border_strong']};
    padding: 5px;
}}

QMenu::item {{
    padding: 6px 26px 6px 20px;
    border-radius: 5px;
}}

QMenu::item:selected {{
    background: {COLORS['accent_soft']};
    color: {COLORS['ink']};
}}

QMenu::item:disabled {{
    color: {COLORS['ink_faint']};
    font-size: 10px;
    font-weight: 700;
    padding-top: 8px;
}}

QMenu::separator {{
    height: 1px;
    background: {COLORS['border']};
    margin: 5px 8px;
}}

QStatusBar {{
    background: {COLORS['card']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['ink_faint']};
    font-size: 11px;
}}

QStatusBar::item {{
    border: none;
}}
"""
