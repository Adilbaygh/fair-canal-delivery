"""Permanent tests for the desktop window. Never deleted.

Two halves, because two different machines run them.

The first half needs no GUI toolkit at all. The page registry and the
translation tables are plain data on purpose, so a reviewer on a machine
without PyQt6 can still list the pages and this suite can still check that
every label exists in both languages. That half also asserts the property
that makes it possible: importing the registry must not pull Qt in.

The second half builds real widgets and is skipped where PyQt6 is absent.
It guards the rule the window is built on - no number on screen without the
file it came from - at the only place where it can be guarded structurally:
the card that shows a number refuses to exist without a source.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from faircanal.gui import i18n
from faircanal.gui import pages

ROOT = Path(__file__).resolve().parents[1]


def _child_environment() -> dict[str, str]:
    """A child that can import the package without an editable install."""
    environment = dict(os.environ)
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source if not existing else source + os.pathsep + existing
    )
    return environment


# ---------------------------------------------------------------------------
# No toolkit needed
# ---------------------------------------------------------------------------


def test_the_registry_does_not_drag_in_a_gui_toolkit():
    """``main.py --list-pages`` has to work where PyQt6 is not installed.

    Checked in a separate interpreter rather than in this one. On a machine
    that has the toolkit, this session has already imported it by the time
    the test runs, and a check against ``sys.modules`` here would pass on
    no evidence at all - which is the kind of test this project keeps
    finding and deleting.
    """
    script = (
        "import sys;"
        "import faircanal.gui.pages as p;"
        "import faircanal.gui.i18n;"
        "leaked=[n for n in sys.modules if n.startswith('PyQt6')];"
        "print(len(p.PAGE_KEYS), leaked)"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT), capture_output=True, text=True, env=_child_environment(),
    )
    assert finished.returncode == 0, finished.stderr
    count, leaked = finished.stdout.strip().split(" ", 1)
    assert int(count) == len(pages.PAGE_KEYS)
    assert leaked == "[]", f"importing the page registry pulled in {leaked}"


def test_the_page_list_prints_without_a_gui_toolkit():
    """The command a reviewer on a bare machine actually runs."""
    finished = subprocess.run(
        [sys.executable, "main.py", "--list-pages"],
        cwd=str(ROOT), capture_output=True, text=True, env=_child_environment(),
    )
    assert finished.returncode == 0, finished.stderr
    for key in pages.PAGE_KEYS:
        assert key in finished.stdout, key


def test_every_page_key_is_unique_and_has_a_module():
    assert len(set(pages.PAGE_KEYS)) == len(pages.PAGE_KEYS)
    modules = [page.module for page in pages.PAGES]
    assert len(set(modules)) == len(modules)


def test_every_page_module_is_a_file_next_to_the_registry():
    """A registry that names a module nobody wrote fails at the click, not here."""
    from pathlib import Path  # noqa: PLC0415

    folder = Path(pages.__file__).resolve().parent
    for page in pages.PAGES:
        assert (folder / f"{page.module}.py").is_file(), page.module


def test_every_page_belongs_to_a_named_section():
    for page in pages.PAGES:
        assert page.section in pages.SECTIONS, page.key


def test_the_number_in_a_nav_label_is_the_page_position():
    """The labels number themselves, so a reordering must renumber them.

    Without this a page moved in the tuple keeps the number it was written
    with, and the window's own cross-references ("page 5") point somewhere
    else.
    """
    for index, page in enumerate(pages.PAGES, start=1):
        for label in (page.uzbek, page.english):
            head = label.split("·", 1)[0].strip()
            assert head == str(index), f"{page.key}: {label!r} is not page {index}"


def test_every_page_is_named_in_both_languages():
    for page in pages.PAGES:
        assert page.uzbek.strip()
        assert page.english.strip()
        assert page.uzbek != page.english, page.key
        for language in i18n.LANGUAGES:
            assert pages.nav_label(page.key, language).strip()


def test_every_section_is_named_in_both_languages():
    for section in pages.SECTIONS:
        for language in i18n.LANGUAGES:
            assert pages.section_label(section, language).strip()


def test_every_frame_label_exists_in_both_languages():
    """A missing translation should fail here, not appear as an empty label."""
    for key, pair in i18n.UI.items():
        assert len(pair) == 2, key
        uzbek, english = pair
        assert uzbek.strip(), key
        assert english.strip(), key
        for language in i18n.LANGUAGES:
            assert i18n.ui(language, key).strip(), (language, key)


def test_every_verdict_word_is_translated():
    """Colour never carries a verdict on its own, so the word has to exist."""
    from faircanal import archive as arc  # noqa: PLC0415

    for verdict in (arc.SOLVED, arc.INFEASIBLE, arc.UNDECIDED,
                    arc.LEVEL_ONLY, arc.MISSING):
        assert verdict in i18n.VERDICT
        for language in i18n.LANGUAGES:
            assert i18n.verdict_label(language, verdict).strip()


def test_no_two_verdicts_share_a_word():
    """"Infeasible" and "no verdict" must not read alike in either language."""
    for index in (0, 1):
        words = [pair[index] for pair in i18n.VERDICT.values()]
        assert len(set(words)) == len(words), words


def test_every_criterion_is_named_and_explained_in_both_languages():
    from faircanal import archive as arc  # noqa: PLC0415

    for code in arc.CODES:
        assert code in i18n.CRITERION, code
        for language in i18n.LANGUAGES:
            short = i18n.criterion_short(language, code)
            assert short.startswith(code), (language, code, short)
            assert i18n.criterion_sentence(language, code).strip()


def test_every_provenance_word_is_translated():
    for kind in ("observed", "derived", "assumed"):
        for language in i18n.LANGUAGES:
            assert i18n.provenance_label(language, kind).strip()


def test_an_unknown_constraint_family_is_shown_as_the_file_wrote_it():
    """The families come from the data, so a new one must not vanish."""
    assert i18n.family_label("uz", "C9 something new") == "C9 something new"
    assert i18n.family_label("en", "C8 source availability") == "C8 source availability"
    assert i18n.family_label("uz", "C8 source availability") != "C8 source availability"


@pytest.mark.parametrize(
    "given, expected",
    [("uz", "uz"), ("en", "en"), ("UZ", "uz"), ("english", "en"),
     ("", "uz"), (None, "uz"), ("fr", "uz")],
)
def test_the_language_code_is_normalised(given, expected):
    assert i18n.normalise(given) == expected


# ---------------------------------------------------------------------------
# Needs the toolkit
# ---------------------------------------------------------------------------

try:  # noqa: SIM105 - the point is the flag, not the exception
    import PyQt6.QtWidgets  # noqa: F401
    HAS_TOOLKIT = True
except ImportError:  # pragma: no cover - depends on the machine
    HAS_TOOLKIT = False

#: Skipped, not failed. The window is an optional extra: a reviewer who
#: only wants to reproduce the numbers never installs the toolkit, and the
#: suite has to pass for them.
needs_toolkit = pytest.mark.skipif(
    not HAS_TOOLKIT, reason="the desktop window needs PyQt6"
)


@pytest.fixture(scope="module")
def application():
    """One application for the whole module; Qt allows no more than one."""
    import os  # noqa: PLC0415

    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


@needs_toolkit
def test_a_card_cannot_show_a_number_without_naming_its_file(application):
    """The window's one rule, kept by the shape of the code.

    Every figure on screen is built through this class, and it will not be
    constructed without the file the figure was read from. A page that has
    no source has nothing this window is willing to display, and that is
    enforced here rather than by anybody remembering it.
    """
    from faircanal.gui import widgets as w  # noqa: PLC0415

    with pytest.raises(ValueError):
        w.ValueCard("0.7304", "the worst-off share", "")
    with pytest.raises(ValueError):
        w.ValueCard("0.7304", "the worst-off share", "   ")


@needs_toolkit
def test_a_user_reads_as_the_gate_the_published_tables_name():
    from faircanal.gui import widgets as w  # noqa: PLC0415

    assert w.short_name("corning-3-user") == "3"
    assert w.short_name("nothing-numeric") == "nothing-numeric"
    users = ["corning-10-user", "corning-2-user", "corning-1-user"]
    assert sorted(users, key=w.short_name_sort_key) == [
        "corning-1-user", "corning-2-user", "corning-10-user",
    ]


@needs_toolkit
def test_a_caption_arrives_as_text_rather_than_as_its_markers():
    """The captions are read from the manuscript's own file, markers and all."""
    from faircanal.gui import widgets as w  # noqa: PLC0415

    assert w.rich("(**a**) Worst-off fraction") == "(<b>a</b>) Worst-off fraction"
    assert w.rich("run `make_figures.py`") == "run <code>make_figures.py</code>"
    assert w.rich("no markers here") == "no markers here"
    assert w.rich("an unclosed **marker") == "an unclosed **marker"


@needs_toolkit
def test_a_unit_is_shown_the_way_it_is_read_aloud():
    from faircanal.gui import widgets as w  # noqa: PLC0415

    assert w.units("m^3/s") == "m³/s"
    assert w.units("m^3") == "m³"
    assert w.units("m") == "m"


@needs_toolkit
def test_every_page_builds_against_the_published_archive(application):
    """The only test that constructs the window.

    It catches the failure a reviewer would meet first: a page that raises
    when it is opened. Run off-screen, so it needs no display; skipped
    where the platform plugin cannot start at all.
    """
    from faircanal import archive as arc  # noqa: PLC0415

    assert application is not None

    paths = arc.ProjectPaths.discover()
    archive = arc.Archive(paths)
    for language in i18n.LANGUAGES:
        for key in pages.PAGE_KEYS:
            context = pages.Context(
                paths=paths,
                archive=archive,
                language=language,
                goto=lambda _target: None,
            )
            widget = pages.build_page(key, context)
            assert widget is not None, (language, key)
            shutdown = getattr(widget, "shutdown", None)
            if callable(shutdown):
                shutdown()
            widget.deleteLater()


@needs_toolkit
def test_every_page_builds_when_the_archive_is_empty(application, tmp_path):
    """A fresh clone has no results, and the window still has to open.

    This is where "no number without a computation" is easiest to break: a
    page that assumes the scan has run divides by a length of zero or
    prints a blank where a dash belongs.
    """
    from faircanal import archive as arc  # noqa: PLC0415

    assert application is not None

    paths = arc.ProjectPaths(root=tmp_path)
    archive = arc.Archive(paths)
    for key in pages.PAGE_KEYS:
        context = pages.Context(
            paths=paths,
            archive=archive,
            language=i18n.DEFAULT_LANGUAGE,
            goto=lambda _target: None,
        )
        widget = pages.build_page(key, context)
        assert widget is not None, key
        shutdown = getattr(widget, "shutdown", None)
        if callable(shutdown):
            shutdown()
        widget.deleteLater()
