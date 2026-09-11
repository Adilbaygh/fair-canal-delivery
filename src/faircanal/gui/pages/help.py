"""10 · How to read this window, and what it will not do.

Short on purpose. The one thing a reader has to take away is the rule the
whole window is built on: a number appears only where somebody computed it,
and it carries the file it came from.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ... import archive as arc
from .. import widgets as w
from . import Context

SHORTCUTS: tuple[tuple[str, str, str], ...] = (
    ("Ctrl+1 … Ctrl+9", "саҳифаларга ўтиш", "go to a page"),
    ("Ctrl+L", "тилни алмаштириш", "switch the language"),
    ("Ctrl+B", "ёнбош панелни яшириш", "hide the sidebar"),
    ("F1", "шу саҳифа", "this page"),
    ("F5", "тўлиқ сканерни бошлаш", "start the full scan"),
    ("Ctrl+T", "тестларни юритиш", "run the tests"),
    ("Esc", "юритилаётган ишни тўхтатиш", "stop what is running"),
    ("Ctrl+Q", "чиқиш", "quit"),
)


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(context.pick("Ёрдам", "Help"))
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Бу ойна мақоланинг иловаси. У натижаларни кўрсатади ва эълон "
                "қилинган скриптларни юритади — ва бошқа ҳеч нарса қилмайди.",
                "This window is an appendix to the article. It shows the results and "
                "runs the published scripts — and it does nothing else.",
            )
        )
    )

    layout.addWidget(_rule(context))
    layout.addWidget(_pages(context))
    layout.addWidget(_shortcuts(context))
    layout.addWidget(_where(context))
    return w.wrap_scroll(body)


def _rule(context: Context) -> QWidget:
    card = w.Card(
        context.pick("Ойнанинг ягона қоидаси", "The one rule this window keeps")
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Ҳисобланмаган сон кўрсатилмайди.</b> Ечими бўлмаган нуқтада "
                "тире турибди, ноль эмас ва бўш катак ҳам эмас. Ечувчи ҳукм "
                "қайтармаган нуқтада ҳам тире — ва бу «ечим йўқ» дегани эмас, "
                "«ҳеч ким ҳисобламаган» дегани. Иккита ҳолат бир-бирига "
                "айлантирилмайди.",
                "<b>A number that nobody computed is not displayed.</b> A point with "
                "no schedule shows a dash — not a zero, and not a blank cell. A point "
                "where the solver returned no verdict shows a dash too, and that does "
                "not mean “infeasible”; it means “nobody computed this”. The two are "
                "never turned into one another.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Ҳар бир сон ўз файлини кўтариб юради.</b> Карточка ва жадвал "
                "остидаги кулранг сатр — сон ўқилган файл ва калит. Уни очиб, ўзингиз "
                "текширишингиз мумкин.",
                "<b>Every number carries its own file.</b> The grey line under a card "
                "or a table is the file and key the number was read from. You can "
                "open it and check.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Ойна ҳеч нарса ҳисобламайди.</b> Барча сонлар "
                "<code>results/</code> ичидаги файллардан ўқилади, ва уларни эълон "
                "қилинган скриптлар ёзади. 9-саҳифадаги тугмалар ўша скриптларни "
                "айнан терминалдагидек юритади.",
                "<b>The window computes nothing.</b> Every number is read from a file "
                "under <code>results/</code>, and the published scripts write those "
                "files. The buttons on page 9 run those same scripts exactly as a "
                "prompt would.",
            )
        )
    )
    return card


def _pages(context: Context) -> QWidget:
    card = w.Card(context.pick("Саҳифалар", "The pages"))
    rows = [
        (
            "claim",
            context.pick("1 · Даъво", "1 · The claim"),
            context.pick(
                "Учта сон ва улар нимани кўрсатади.",
                "Three numbers and what they show.",
            ),
        ),
        (
            "window",
            context.pick("2 · Ойна ва ҳажм", "2 · Window and volume"),
            context.pick(
                "Нима ўлчанади ва нима учун жадвал етарли эмас.",
                "What is measured, and why the schedule is not enough.",
            ),
        ),
        (
            "comparison",
            context.pick("3 · Бешта мезон", "3 · Five criteria"),
            context.pick(
                "Манба даражасини суриб, ким нима олишини кўринг.",
                "Drag the supply level and see who receives what.",
            ),
        ),
        (
            "verdicts",
            context.pick("4 · Учта жавоб", "4 · Three answers"),
            context.pick(
                "Ҳукмлар матрицаси ва бажарилмаслик сертификати.",
                "The matrix of verdicts, and the infeasibility certificate.",
            ),
        ),
        (
            "theorems",
            context.pick("5 · Теоремалар", "5 · The theorems"),
            context.pick(
                "Исботланган нарса, ва ундан ажратилган ўлчовлар.",
                "What is proved, kept apart from what was measured.",
            ),
        ),
        (
            "sensitivity",
            context.pick("6 · Сезгирлик", "6 · Sensitivity"),
            context.pick(
                "Фильтрни ўзгартирганда нима ўзгарди.",
                "What changed when the filter changed.",
            ),
        ),
        (
            "inputs",
            context.pick("7 · Кириш маълумотлари", "7 · The inputs"),
            context.pick(
                "Канал, чегаралар, фильтр — ҳар бирининг келиб чиқиши билан.",
                "The canal, the limits, the filter — each with its provenance.",
            ),
        ),
        (
            "gallery",
            context.pick("8 · Расм ва жадваллар", "8 · Figures and tables"),
            context.pick(
                "Мақола босадиган нарсанинг ҳаммаси.",
                "Everything the article prints.",
            ),
        ),
        (
            "reproduce",
            context.pick("9 · Қайта юритиш", "9 · Reproduce"),
            context.pick(
                "Скриптларни юритиш ва архивнинг ҳолати.",
                "Run the scripts, and see the state of the archive.",
            ),
        ),
    ]
    for key, title, text in rows:
        row = QHBoxLayout()
        row.setSpacing(12)
        button = QPushButton(title)
        button.setObjectName("Link")
        button.setMinimumWidth(190)
        button.clicked.connect(lambda _checked=False, target=key: context.goto(target))
        row.addWidget(button)
        row.addWidget(w.paragraph(text), 1)
        card.add_layout(row)
    return card


def _shortcuts(context: Context) -> QWidget:
    card = w.Card(context.pick("Тезкор тугмалар", "Keyboard shortcuts"))
    card.add(
        w.grid_table(
            [context.pick("Тугма", "Keys"), context.pick("Нима қилади", "What it does")],
            [[keys, context.pick(uzbek, english)] for keys, uzbek, english in SHORTCUTS],
            max_height=260,
        )
    )
    return card


def _where(context: Context) -> QWidget:
    paths = context.paths
    card = w.Card(
        context.pick("Файллар қаерда", "Where the files are"),
        context.pick(
            "Лойиҳа папкаси — код, кириш маълумотлари ва натижалар. Ойна ҳеч қайси "
            "файлга ёзмайди; фақат скриптлар ёзади.",
            "The project folder holds the code, the inputs and the results. The "
            "window writes to none of them; only the scripts do.",
        ),
    )
    row = QHBoxLayout()
    row.setSpacing(10)
    for path, uzbek, english in (
        (paths.root, "Лойиҳа папкаси", "Project folder"),
        (paths.results, "results/", "results/"),
        (paths.data, "DATA/", "DATA/"),
        (paths.scripts, "scripts/", "scripts/"),
    ):
        button = QPushButton(context.pick(uzbek, english))
        button.clicked.connect(
            lambda _checked=False, target=path: w.open_in_file_manager(target)
        )
        row.addWidget(button)
    row.addStretch(1)
    card.add_layout(row)

    environment = context.archive.environment()
    if environment:
        card.add(
            w.mono(
                f"{context.ui('source_prefix')}: results/environment.json  ·  "
                f"{environment.get('recorded_utc', arc.DASH)}"
            )
        )
    return card
