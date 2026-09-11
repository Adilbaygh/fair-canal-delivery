"""The window: a menu bar, a sectioned sidebar, one page at a time, a status line.

Pages are built the first time they are opened rather than all at start-up,
because the gallery loads four figures at 600 dpi and a reader who only
wants the certificate should not wait for them.

Switching language drops the built pages and rebuilds the one being looked
at. Pages carry their two strings inline, at the point of use, so that is
the honest way to switch: there is no table of translations to re-read.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import archive as arc
from . import i18n, pages
from . import widgets as w
from .theme import COLORS

SIDEBAR_WIDTH = 252


class MainWindow(QMainWindow):
    def __init__(self, root=None, language: str = i18n.DEFAULT_LANGUAGE) -> None:
        super().__init__()
        self.paths = arc.ProjectPaths.discover(root)
        self.language = i18n.normalise(language)
        self.archive = arc.Archive(self.paths)

        self.setWindowTitle(i18n.ui(self.language, "app_title"))
        self.resize(1340, 900)
        self.setMinimumSize(980, 640)

        central = QWidget()
        central.setObjectName("Page")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header())

        lower = QHBoxLayout()
        lower.setContentsMargins(0, 0, 0, 0)
        lower.setSpacing(0)
        self.sidebar = self._sidebar()
        lower.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        lower.addWidget(self.stack, 1)
        outer.addLayout(lower, 1)
        self.setCentralWidget(central)

        self._holders: dict[str, QWidget] = {}
        self._built: set[str] = set()
        for key in pages.PAGE_KEYS:
            holder = QWidget()
            holder.setObjectName("Page")
            layout = QVBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self._holders[key] = holder
            self.stack.addWidget(holder)

        self._build_menus()
        self._build_status_bar()
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self.toggle_language)
        self.show_page(pages.PAGE_KEYS[0])

    # ------------------------------------------------------------------ chrome

    def _header(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Header")
        bar.setFixedHeight(70)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 11, 18, 11)
        layout.setSpacing(14)

        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(i18n.ui(self.language, "app_title"))
        self.title_label.setObjectName("HeaderTitle")
        text.addWidget(self.title_label)
        self.strip_label = QLabel(i18n.ui(self.language, "headline_strip"))
        self.strip_label.setObjectName("HeaderStrip")
        text.addWidget(self.strip_label)
        layout.addLayout(text, 1)

        self.language_button = QPushButton()
        self.language_button.setObjectName("LanguageButton")
        self.language_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.language_button.clicked.connect(self.toggle_language)
        self._refresh_language_button()
        layout.addWidget(self.language_button, 0, Qt.AlignmentFlag.AlignVCenter)
        return bar

    def _sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Sidebar")
        panel.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 6, 0, 12)
        layout.setSpacing(0)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        self.section_labels: dict[str, QLabel] = {}

        seen: set[str] = set()
        for index, key in enumerate(pages.PAGE_KEYS):
            section = pages.section_of(key)
            if section not in seen:
                seen.add(section)
                caption = QLabel(pages.section_label(section, self.language))
                caption.setObjectName("SidebarCaption")
                layout.addWidget(caption)
                self.section_labels[section] = caption
            button = QPushButton(pages.nav_label(key, self.language))
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, target=key: self.show_page(target))
            self.nav_group.addButton(button, index)
            self.nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch(1)
        self.root_label = QLabel(self.paths.root.name)
        self.root_label.setObjectName("SidebarCaption")
        self.root_label.setToolTip(str(self.paths.root))
        self.root_label.setStyleSheet(f"color: {COLORS['on_dark_soft']}; font-weight: 400;")
        self.root_label.setWordWrap(True)
        layout.addWidget(self.root_label)
        return panel

    # ------------------------------------------------------------------- menus

    def _act(self, uzbek: str, english: str, slot, shortcut: str = "",
             tip: tuple[str, str] | None = None) -> QAction:
        action = QAction(self.pick(uzbek, english), self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if tip:
            text = self.pick(*tip)
            action.setStatusTip(text)
            action.setToolTip(text)
        action.triggered.connect(slot)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        bar.clear()

        file_menu = bar.addMenu(self.pick("&Файл", "&File"))
        file_menu.setToolTipsVisible(True)
        for action in (
            self._act(
                "Лойиҳа папкаси — код, маълумот ва натижалар",
                "Project folder — the code, the data and the results",
                lambda: w.open_in_file_manager(self.paths.root),
            ),
            self._act(
                "results/ — мақоладаги сонлар, расмлар ва жадваллар",
                "results/ — the numbers, figures and tables the article prints",
                lambda: w.open_in_file_manager(self.paths.results),
                tip=("Скриптлар ёзган эълон қилинган натижалар. Ойна бу ерга ҳеч "
                     "нарса ёзмайди — фақат ўқийди.",
                     "The published results the scripts wrote. This window never "
                     "writes here; it only reads."),
            ),
            self._act(
                "DATA/ — моделнинг барча кириш параметрлари",
                "DATA/ — every input the model takes",
                lambda: w.open_in_file_manager(self.paths.data),
            ),
        ):
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self._act("Чиқиш", "Quit", self.close, "Ctrl+Q"))

        run_menu = bar.addMenu(self.pick("Ҳ&исоблаш", "&Compute"))
        for action in (
            self._act("Тўлиқ сканерни бошлаш", "Start the full scan",
                      self.run_pipeline, "F5"),
            self._act("Тестларни юритиш", "Run the tests", self.run_tests, "Ctrl+T"),
            self._act("Тўхтатиш", "Stop", self.stop_running, "Esc"),
        ):
            run_menu.addAction(action)

        view = bar.addMenu(self.pick("&Кўриниш", "&View"))
        self.page_actions: dict[str, QAction] = {}
        page_group = QActionGroup(self)
        page_group.setExclusive(True)
        seen: set[str] = set()
        for index, key in enumerate(pages.PAGE_KEYS, start=1):
            section = pages.section_of(key)
            if section not in seen:
                seen.add(section)
                if len(seen) > 1:
                    view.addSeparator()
                heading = QAction(pages.section_label(section, self.language), self)
                heading.setEnabled(False)
                view.addAction(heading)
            page = pages.page_of(key)
            action = self._act(
                page.uzbek, page.english,
                lambda _checked=False, target=key: self.show_page(target),
                f"Ctrl+{index}" if index <= 9 else "",
            )
            action.setCheckable(True)
            page_group.addAction(action)
            view.addAction(action)
            self.page_actions[key] = action
        view.addSeparator()
        self.sidebar_action = self._act("Ёнбош панел", "Sidebar",
                                        self.toggle_sidebar, "Ctrl+B")
        self.sidebar_action.setCheckable(True)
        self.sidebar_action.setChecked(self.sidebar.isVisible())
        view.addAction(self.sidebar_action)

        language_menu = bar.addMenu(self.pick("&Тил", "&Language"))
        self.language_actions: dict[str, QAction] = {}
        language_group = QActionGroup(self)
        language_group.setExclusive(True)
        for code in i18n.LANGUAGES:
            action = QAction(i18n.ENDONYM[code], self)
            action.setCheckable(True)
            action.setChecked(code == self.language)
            action.triggered.connect(lambda _checked=False, c=code: self.set_language(c))
            language_group.addAction(action)
            language_menu.addAction(action)
            self.language_actions[code] = action

        help_menu = bar.addMenu(self.pick("&Ёрдам", "&Help"))
        help_menu.addAction(
            self._act("Ёрдам", "Help", lambda: self.show_page("help"), "F1")
        )
        help_menu.addSeparator()
        help_menu.addAction(self._act("Дастур ҳақида", "About", self.show_about))

    # -------------------------------------------------------------- status bar

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        self.status_state = QLabel()
        bar.addPermanentWidget(self.status_state)
        self._refresh_status()

    def _refresh_status(self) -> None:
        missing = self.archive.missing()
        if missing:
            text = f"{len(missing)} {self.pick('файл етишмайди', 'files missing')}"
            colour = COLORS["short"]
        else:
            solved = len(self.archive.solved_percents(arc.MAIN_LABEL))
            total = len(self.archive.percents(arc.MAIN_LABEL))
            text = self.pick(
                f"архив тўлиқ · {solved}/{total} нуқтада жадвал бор",
                f"archive complete · a schedule at {solved}/{total} points",
            )
            colour = COLORS["filled"]
        self.status_state.setText(text)
        self.status_state.setStyleSheet(f"color: {colour}; padding-right: 8px;")
        self.statusBar().showMessage(str(self.paths.root))

    # --------------------------------------------------------------- behaviour

    def pick(self, uzbek: str, english: str) -> str:
        return i18n.pick(self.language, uzbek, english)

    def show_page(self, key: str) -> None:
        """Open a page, building it the first time it is asked for."""
        if key not in self._holders:
            return
        if key not in self._built:
            context = pages.Context(
                paths=self.paths,
                archive=self.archive,
                language=self.language,
                goto=self.show_page,
            )
            widget = pages.build_page(key, context)
            layout = self._holders[key].layout()
            assert layout is not None
            layout.addWidget(widget)
            self._built.add(key)
        self.stack.setCurrentWidget(self._holders[key])
        button = self.nav_buttons.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        action = self.page_actions.get(key)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self._refresh_status()

    def current_key(self) -> str:
        index = self.stack.currentIndex()
        return pages.PAGE_KEYS[index] if 0 <= index < len(pages.PAGE_KEYS) else ""

    def _page_widget(self, key: str) -> QWidget | None:
        holder = self._holders.get(key)
        if holder is None:
            return None
        layout = holder.layout()
        if layout is None or not layout.count():
            return None
        item = layout.itemAt(0)
        return item.widget() if item is not None else None

    def toggle_sidebar(self) -> None:
        visible = not self.sidebar.isVisible()
        self.sidebar.setVisible(visible)
        self.sidebar_action.setChecked(visible)

    # -- running from the menu ------------------------------------------------

    def _reproduce_panel(self):
        self.show_page("reproduce")
        widget = self._page_widget("reproduce")
        return getattr(widget, "controller", None)

    def run_pipeline(self) -> None:
        panel = self._reproduce_panel()
        if panel is not None:
            panel.run_pipeline()

    def run_tests(self) -> None:
        panel = self._reproduce_panel()
        if panel is not None:
            panel.run_tests()

    def stop_running(self) -> None:
        widget = self._page_widget(self.current_key())
        controller = getattr(widget, "controller", None)
        stop = getattr(controller, "stop", None)
        if callable(stop):
            stop()

    # -- help -----------------------------------------------------------------

    def show_about(self) -> None:
        from .. import __version__  # noqa: PLC0415

        environment = self.archive.environment()
        python = (environment.get("python") or {}).get("version", "")
        platform = (environment.get("platform") or {}).get("system", "")
        QMessageBox.about(
            self,
            self.pick("Дастур ҳақида", "About"),
            self.pick(
                f"<b>Fair-Canal-Delivery {__version__}</b><br><br>"
                "Ирригация каналида келишилган ойна ичида етказилган ҳажм бўйича "
                "лексикографик адолатли тақсимот. Ойна мақоланинг иловаси: у "
                "натижаларни кўрсатади ва эълон қилинган скриптларни юритади, "
                "ҳисоблашни эса <code>scripts/</code> даги скриптлар бажаради."
                "<br><br>"
                f"Лойиҳа: {self.paths.root}<br>"
                f"Python {python} · {platform}",
                f"<b>Fair-Canal-Delivery {__version__}</b><br><br>"
                "Lexicographic max-min allocation on an irrigation canal, measured "
                "on the volume delivered inside an agreed window. This window is an "
                "appendix to the article: it shows the results and runs the "
                "published scripts, and the computing is done by the scripts in "
                "<code>scripts/</code>.<br><br>"
                f"Project: {self.paths.root}<br>"
                f"Python {python} · {platform}",
            ),
        )

    # -- language -------------------------------------------------------------

    def toggle_language(self) -> None:
        self.set_language("en" if self.language == "uz" else "uz")

    def set_language(self, language: str) -> None:
        language = i18n.normalise(language)
        if language == self.language:
            return
        current = self.current_key()
        self.language = language

        self.setWindowTitle(i18n.ui(language, "app_title"))
        self.title_label.setText(i18n.ui(language, "app_title"))
        self.strip_label.setText(i18n.ui(language, "headline_strip"))
        self._refresh_language_button()
        for key, button in self.nav_buttons.items():
            button.setText(pages.nav_label(key, language))
        for section, label in self.section_labels.items():
            label.setText(pages.section_label(section, language))
        self._build_menus()
        for code, action in self.language_actions.items():
            action.setChecked(code == language)
        self._refresh_status()

        self._discard_pages()
        self.show_page(current or pages.PAGE_KEYS[0])

    def _refresh_language_button(self) -> None:
        other = "en" if self.language == "uz" else "uz"
        self.language_button.setText(i18n.ENDONYM[other])
        self.language_button.setToolTip(
            self.pick("Тилни алмаштириш (Ctrl+L)", "Switch language (Ctrl+L)")
        )

    # -- lifecycle ------------------------------------------------------------

    def _discard_pages(self) -> None:
        for key in list(self._built):
            layout = self._holders[key].layout()
            assert layout is not None
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is None:
                    continue
                shutdown = getattr(widget, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                widget.setParent(None)
                widget.deleteLater()
        self._built.clear()

    def closeEvent(self, event) -> None:  # pragma: no cover - window lifecycle
        self._discard_pages()
        super().closeEvent(event)
