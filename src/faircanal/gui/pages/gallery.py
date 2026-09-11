"""8 · Every figure and table the article prints, with its file.

The captions are read from the file the drawing script writes beside the
images, so the wording here is the wording the manuscript uses. Two
captions drifted from their pictures within an hour during this study, and
they were caught because caption and code live together; the window reads
the same file rather than keeping a third copy.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ... import archive as arc
from .. import widgets as w
from . import Context

#: The published CSVs, in the order a reader meets them.
TABLES: tuple[tuple[str, str, str], ...] = (
    (
        "main_summary.csv",
        "Ҳар бир мезоннинг ҳар бир манба даражасидаги хулосаси",
        "One row per criterion per supply level",
    ),
    (
        "main_ratios.csv",
        "Ҳар бир фойдаланувчининг ҳар бир нуқтадаги улуши",
        "Every user's share at every point",
    ),
    (
        "main_certificates.csv",
        "Ҳар бир нуқтанинг сертификати ва унинг дайджести",
        "The certificate at every point, and its digest",
    ),
)


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Расмлар ва жадваллар — ҳар бири ўз файли билан",
                "Figures and tables — each one with its file",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Ҳеч ким қайта чиза олмайдиган расм — далил эмас, даъво. Шунинг учун "
                "ҳар бир расмни чизган скрипт ҳам эълон қилинган тўпламга киради, ва "
                "изоҳлар шу скрипт ёзган файлдан ўқилади.",
                "A figure nobody can redraw is a claim, not evidence. So the script "
                "that drew each one is part of the published set, and the captions "
                "are read from the file that script writes.",
            )
        )
    )

    for figure in context.archive.figures():
        layout.addWidget(
            w.FigureCard(figure, context.paths.relative(figure.path), context.language)
        )

    layout.addWidget(_tables(context))
    layout.addWidget(_folder(context))
    return w.wrap_scroll(body)


def _tables(context: Context) -> QWidget:
    card = w.Card(
        context.pick(
            "Жадваллар — сканер файлларидан ҳисобланган, қўлда терилмаган",
            "The tables — derived from the scan files, never typed",
        )
    )
    for name, uzbek, english in TABLES:
        table = context.archive.table(name, limit=200)
        card.add(
            w.TableCard(
                table,
                context.language,
                title=name,
                note=context.pick(uzbek, english),
                max_height=260,
            )
        )
    other = [
        name
        for name in context.archive.table_names()
        if name not in {item[0] for item in TABLES}
    ]
    if other:
        card.add(
            w.paragraph(
                context.pick(
                    "Сезгирлик юришлари ҳам ўз жадвалларини ёзади: "
                    + ", ".join(other),
                    "The sensitivity runs write their own tables too: "
                    + ", ".join(other),
                )
            )
        )
    return card


def _folder(context: Context) -> QWidget:
    card = w.Card(
        context.pick("Файллар қаерда", "Where the files are"),
        context.pick(
            "Расмлар журналга кетадиган ҳолда чизилган: 600 dpi PNG, ва қайта "
            "экспорт учун вектор PDF нусхаси. Изоҳлар расмнинг ичида эмас, "
            "матнда — журнал шуни талаб қилади.",
            "The figures are drawn ready for the journal: PNG at 600 dpi, with a "
            "vector PDF master kept for re-export. The captions live in the text "
            "rather than inside the image, which is what the journal asks for.",
        ),
    )
    row = QHBoxLayout()
    row.setSpacing(10)
    for path, uzbek, english in (
        (context.paths.figures, "Расмлар папкаси", "Figures folder"),
        (context.paths.tables, "Жадваллар папкаси", "Tables folder"),
        (context.paths.scan, "Сканер файллари", "Scan files"),
    ):
        button = QPushButton(context.pick(uzbek, english))
        button.clicked.connect(lambda _checked=False, target=path: w.open_in_file_manager(target))
        row.addWidget(button)
    row.addStretch(1)
    card.add_layout(row)
    card.add(w.mono(f"{context.ui('source_prefix')}: {context.paths.relative(context.paths.figures)}"))
    missing = context.archive.missing()
    if missing:
        card.add(
            w.paragraph(
                context.pick(
                    "Топилмаган файллар: " + ", ".join(missing),
                    "Files not found: " + ", ".join(missing),
                )
            )
        )
    else:
        card.add(w.paragraph(context.pick(
            "Кутилган барча файллар жойида.",
            "Every expected file is present.",
        )))
    return card
