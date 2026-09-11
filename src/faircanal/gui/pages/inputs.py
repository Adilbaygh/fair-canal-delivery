"""7 · Every input the model takes, with where each one came from.

Three words do the work on this page, and they are not interchangeable.
*Observed* means somebody else published it and the citation travels with
it. *Derived* means it follows from something observed by a stated rule.
*Assumed* means this study chose it because no published value was found,
and every result that leans on it says so.

The tables are read from the exported files rather than from the library,
because that is what a reader has: the export is checked against the live
objects by a permanent test, so it cannot quietly drift.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ... import archive as arc
from .. import widgets as w
from ..theme import COLORS
from . import Context

#: Which provenance word belongs to which block of the exported inputs, and
#: the key the word itself is read from where the file carries one.
BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("canal", "Канал геометрияси ва топология", "Canal geometry and topology"),
    ("limits", "Чегаралар", "Limits"),
    ("filter", "Фильтр", "Filter"),
    ("scenario", "Сценарий", "Scenario"),
)


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Кириш маълумотлари — ва ҳар бирининг келиб чиқиши",
                "The inputs — and where every one of them came from",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Натижалар кодни очмасдан ҳам такрорланадиган бўлиши учун моделнинг "
                "барча кириш параметрлари файлга чиқарилган. Код ҳамон ҳақиқат "
                "манбаи; бу файллар ундан ёзилади ва доимий тест уларни жонли "
                "объектлар билан солиштиради.",
                "So that the results can be reproduced without opening a Python "
                "module, every input the model takes is exported to a file. The code "
                "remains the source of truth; these files are written from it, and a "
                "permanent test rebuilds every value and fails if they disagree.",
            )
        )
    )

    layout.addWidget(_provenance(context))
    layout.addWidget(_canal(context))
    layout.addWidget(_citations(context))
    layout.addWidget(_solver(context))
    return w.wrap_scroll(body)


# ------------------------------------------------------------------ provenance


def _provenance(context: Context) -> QWidget:
    data = context.archive.inputs()
    card = w.Card(
        context.pick("Учта сўз", "Three words"),
        context.pick(
            "Ҳар бир блок биттасини олади, ва бу учтаси бир-бирининг ўрнини "
            "боса олмайди.",
            "Every block carries one of them, and the three are not "
            "interchangeable.",
        ),
    )
    for kind, uzbek, english in (
        ("observed",
         "бошқа биров эълон қилган ва иқтибос қилинган",
         "published by somebody else, and cited"),
        ("derived",
         "кузатилган нарсадан эълон қилинган формула билан ҳисобланган",
         "computed from something observed by a stated formula"),
        ("assumed",
         "бу тадқиқот танлаган, чунки эълон қилинган қиймат топилмади",
         "chosen by this study, because no published value was found"),
    ):
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(w.provenance_chip(context.language, kind))
        row.addWidget(w.paragraph(context.pick(uzbek, english)), 1)
        card.add_layout(row)

    if not data:
        card.add(
            w.paragraph(
                context.pick(
                    "Кириш файллари топилмади. «Қайта юритиш» саҳифасидан "
                    "чиқаринг.",
                    "The exported inputs were not found. Export them from the "
                    "“Reproduce” page.",
                )
            )
        )
        return card

    card.add(w.divider())
    for key, uzbek, english in BLOCKS:
        block = data.get(key) or {}
        text = str(
            block.get("provenance")
            or block.get("geometry_provenance")
            or ""
        )
        row = QHBoxLayout()
        row.setSpacing(8)
        heading = w.paragraph(f"<b>{context.pick(uzbek, english)}</b>")
        heading.setMinimumWidth(230)
        heading.setMaximumWidth(230)
        row.addWidget(heading)
        for word in _words_in(text):
            row.addWidget(w.provenance_chip(context.language, word))
        row.addStretch(1)
        card.add_layout(row)
        note = _note(text)
        if note:
            card.add(w.paragraph(note))
    card.add(w.mono(f"{context.ui('source_prefix')}: DATA/inputs.json"))
    return card


def _words_in(text: str) -> list[str]:
    """Which provenance words a block actually carries, in a fixed order.

    A block is allowed to carry more than one, and one of them does: the
    limits are part derived and part assumed. Showing only the first word
    would quietly promote the assumed half.
    """
    lowered = text.lower()
    return [word for word in ("observed", "derived", "assumed") if word in lowered]


def _note(text: str) -> str:
    """The block's own sentence, with the bare label alone dropped as noise."""
    stripped = text.strip()
    if stripped.lower() in ("observed", "derived", "assumed"):
        return ""
    for separator in (" - ", " — "):
        if separator in stripped:
            return stripped.split(separator, 1)[1].strip()
    return stripped


# ----------------------------------------------------------------------- canal


def _canal(context: Context) -> QWidget:
    rows = context.archive.canal_rows()
    card = w.Card(
        context.pick(
            "Канал — ҳар бир пул бир қатор",
            "The canal — one row per reach",
        ),
        context.pick(
            "Ўлчам, тубнинг эни ва қиялиги, Маннинг коэффициенти, мақсадли сатҳ ва "
            "ўтказгич сарфи — эълон қилинган. Ўтказувчанлик, сатҳ йўлаги ва затвор "
            "ҳаракат тезлиги — булардан келтириб чиқарилган.",
            "Length, bed width and side slope, the Manning coefficient, the target "
            "level and the offtake are published. The conveyance capacity, the level "
            "band and the gate travel rate follow from them.",
        ),
    )
    if not rows:
        card.add(w.paragraph(context.ui("no_table")))
        return card

    # Two-line headings: the words stay whole. Squeezed onto one line the
    # column labels are the first thing the table elides, and a header
    # reading "казувчанлик, m" names nothing.
    columns = [
        ("label", context.pick("Пул", "Reach")),
        ("length_m", context.pick("Узунлик\nм", "Length\nm")),
        ("bed_width_m", context.pick("Туб эни\nм", "Bed width\nm")),
        ("canal_depth_m", context.pick("Чуқурлик\nм", "Depth\nm")),
        ("target_level_m", context.pick("Мақсадли сатҳ\nм", "Target level\nm")),
        ("offtake_m3_s", context.pick("Ўтказгич\nm³/s", "Offtake\nm³/s")),
        ("capacity_m3_s", context.pick("Ўтказувчанлик\nm³/s", "Capacity\nm³/s")),
        ("travel_rate_m3_s", context.pick("Затвор тезлиги\nm³/s", "Travel rate\nm³/s")),
    ]
    header = [title for _, title in columns]
    body = [
        [_cell(row.get(key, "")) for key, _ in columns]
        for row in rows
    ]
    card.add(w.grid_table(header, body, max_height=380, fit=True))
    card.add(w.mono(f"{context.ui('source_prefix')}: DATA/corning_canal.csv"))

    footer = QHBoxLayout()
    footer.addStretch(1)
    button = QPushButton(context.ui("open_csv"))
    button.setObjectName("Link")
    button.clicked.connect(
        lambda: w.open_file(context.paths.data / "corning_canal.csv")
    )
    footer.addWidget(button)
    card.add_layout(footer)
    return card


def _cell(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}" if abs(number) >= 0.001 or number == 0 else f"{number:.3g}"


# ------------------------------------------------------------------- citations


def _citations(context: Context) -> QWidget:
    canal = (context.archive.inputs().get("canal") or {})
    card = w.Card(
        context.pick(
            "Иқтибослар сонлар билан бирга юради",
            "The citations travel with the numbers",
        )
    )
    source = str(canal.get("geometry_source") or canal.get("source") or "")
    if not source:
        card.add(w.paragraph(context.ui("no_table")))
        return card
    card.add(w.paragraph(source))
    card.add(
        w.paragraph(
            context.pick(
                f"<span style='color:{COLORS['ink_faint']}'>Топология: "
                f"{canal.get('topology_rule', arc.DASH)}.</span>",
                f"<span style='color:{COLORS['ink_faint']}'>Topology: "
                f"{canal.get('topology_rule', arc.DASH)}.</span>",
            )
        )
    )
    row = QHBoxLayout()
    row.addStretch(1)
    button = QPushButton(context.ui("copy"))
    button.setObjectName("Link")
    button.clicked.connect(lambda: w.copy_to_clipboard(source))
    row.addWidget(button)
    card.add_layout(row)
    return card


# ---------------------------------------------------------------------- solver


def _solver(context: Context) -> QWidget:
    data = context.archive.inputs().get("solver") or {}
    environment = context.archive.environment()
    card = w.Card(
        context.pick(
            "Ечувчи ва унинг чегараси",
            "The solver, and the one limitation it carries",
        )
    )
    if data:
        options = data.get("options") or {}
        card.add(
            w.key_values(
                [
                    (context.pick("Усул", "Method"), str(data.get("method", arc.DASH))),
                    (
                        context.pick("Йўл қўйилувчанлик толеранси", "Feasibility tolerance"),
                        str(options.get("primal_feasibility_tolerance", arc.DASH)),
                    ),
                    (
                        context.pick("Тўйиниш толеранси", "Saturation tolerance"),
                        str(data.get("saturation_tolerance", arc.DASH)),
                    ),
                    (
                        context.pick("Оқимлар", "Threads"),
                        str((environment.get("solver") or {}).get("threads", arc.DASH)),
                    ),
                ],
                columns=2,
            )
        )
        note = str(data.get("note") or "")
        if note:
            card.add(w.quotation(note))
    if environment:
        packages = environment.get("packages") or {}
        python = environment.get("python") or {}
        platform = environment.get("platform") or {}
        card.add(
            w.mono(
                f"Python {python.get('version', arc.DASH)} · "
                f"{platform.get('system', arc.DASH)} {platform.get('release', '')} · "
                + " · ".join(f"{name} {value}" for name, value in sorted(packages.items()))
            )
        )
        card.add(w.mono(f"{context.ui('source_prefix')}: results/environment.json"))
    return card
