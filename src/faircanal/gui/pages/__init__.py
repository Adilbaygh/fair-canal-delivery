"""The pages, in the order the argument is made.

Three sections, because the window serves three different visits.

*The claim* walks the road the article walks: fairness is measured on the
volume actually delivered inside an agreed window; here is what the filter
does to that volume; here is what five criteria give the same eight users
at the same supply; and here is what happens when no schedule exists at
all.

*The evidence* is what can be proved and what was only measured, kept
apart. Two theorems with a witness read live out of the archive, and a
sensitivity section that says which findings survived a change of filter
and which did not.

*The apparatus* is the machinery: every input with its provenance, every
figure and table with its file, the buttons that rebuild the lot, and the
help.

**This module imports no Qt.** The registry below is plain data, so the
page list, the navigation labels and the section names can be read on a
machine that has never installed a GUI toolkit - which is what makes
``python main.py --list-pages`` work there. Only :func:`module_of` reaches
for a page module, and only that pulls PyQt6 in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ... import archive as arc
from .. import i18n

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at run time
    from PyQt6.QtWidgets import QWidget


@dataclass
class Context:
    """What a page needs: where things are, what they say, how to move on."""

    paths: arc.ProjectPaths
    archive: arc.Archive
    language: str
    goto: Callable[[str], None]

    def pick(self, uzbek: str, english: str) -> str:
        return i18n.pick(self.language, uzbek, english)

    def ui(self, key: str) -> str:
        return i18n.ui(self.language, key)


@dataclass(frozen=True)
class Page:
    """One page, described without loading it."""

    key: str
    module: str
    section: str
    uzbek: str
    english: str


#: Section captions for the sidebar and the View menu.
SECTIONS: dict[str, tuple[str, str]] = {
    "claim": ("ДАЪВО", "THE CLAIM"),
    "evidence": ("ДАЛИЛ", "THE EVIDENCE"),
    "apparatus": ("АППАРАТ", "THE APPARATUS"),
}

#: The pages, in navigation order. The number in each label is its position
#: in this tuple; a test keeps the two from drifting apart.
PAGES: tuple[Page, ...] = (
    Page("claim", "claim", "claim",
         "1 · Даъво", "1 · The claim"),
    Page("window", "delivery_window", "claim",
         "2 · Ойна ва ҳажм", "2 · Window and volume"),
    Page("comparison", "comparison", "claim",
         "3 · Бешта мезон", "3 · Five criteria"),
    Page("verdicts", "verdicts", "claim",
         "4 · Учта жавоб", "4 · Three answers"),
    Page("theorems", "theorems", "evidence",
         "5 · Теоремалар", "5 · The theorems"),
    Page("sensitivity", "sensitivity", "evidence",
         "6 · Сезгирлик", "6 · Sensitivity"),
    Page("inputs", "inputs", "apparatus",
         "7 · Кириш маълумотлари", "7 · The inputs"),
    Page("gallery", "gallery", "apparatus",
         "8 · Расм ва жадваллар", "8 · Figures and tables"),
    Page("reproduce", "reproduce", "apparatus",
         "9 · Қайта юритиш", "9 · Reproduce"),
    Page("help", "help", "apparatus",
         "10 · Ёрдам", "10 · Help"),
)

PAGE_KEYS: tuple[str, ...] = tuple(page.key for page in PAGES)

_BY_KEY: dict[str, Page] = {page.key: page for page in PAGES}


def page_of(key: str) -> Page:
    return _BY_KEY[key]


def module_of(key: str):
    """Import one page module. This is the only call here that needs PyQt6."""
    return __import__(f"{__name__}.{_BY_KEY[key].module}", fromlist=["build"])


def build_page(key: str, context: Context) -> "QWidget":
    return module_of(key).build(context)


def nav_label(key: str, language: str) -> str:
    page = _BY_KEY[key]
    return i18n.pick(language, page.uzbek, page.english)


def section_of(key: str) -> str:
    return _BY_KEY[key].section


def section_label(section: str, language: str) -> str:
    uzbek, english = SECTIONS[section]
    return i18n.pick(language, uzbek, english)
