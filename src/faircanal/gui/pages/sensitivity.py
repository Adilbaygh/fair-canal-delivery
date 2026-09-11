"""6 · What survived a change of configuration, and what did not.

The scan was rerun three more times, changing one thing each time. The
comparison between criteria barely moved. Where the canal can be operated
at all moved a great deal. Both halves are on this page, because reporting
only the first would be advocacy and reporting only the second would bury
the result.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ... import archive as arc
from .. import i18n
from .. import widgets as w
from ..theme import COLORS
from . import Context

#: A short human sentence for each configuration, keyed by the label the
#: scan wrote. A label the archive has and this table does not still shows,
#: described from its own recorded settings.
DESCRIBED: dict[str, tuple[str, str]] = {
    "main": (
        "олдиндан эълон қилинган конфигурация",
        "the pre-registered configuration",
    ),
    "cut2e3": (
        "кесиш частотаси пасайтирилди",
        "the cut-off frequency lowered",
    ),
    "order4": (
        "фильтр тартиби 4 га кўтарилди",
        "the filter order raised to four",
    ),
    "budget": (
        "ҳажм бюджетига ортиқча берилиш рухсат этилди",
        "over-delivery allowed against the volume budget",
    ),
}


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Сезгирлик — нима ўзгарди, нима ўзгармади",
                "Sensitivity — what moved and what did not",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Ҳар сафар <b>биттагина</b> нарса ўзгартирилди ва бутун сканер қайта "
                "юритилди. Натижа иккига бўлинади, ва иккаласи ҳам шу ерда.",
                "Each rerun changed <b>one</b> thing and repeated the whole scan. The "
                "outcome splits in two, and both halves are here.",
            )
        )
    )

    layout.addWidget(_overview(context))
    layout.addWidget(_curves(context))
    layout.addWidget(_finding(context))
    return w.wrap_scroll(body)


# -------------------------------------------------------------------- overview


def _overview(context: Context) -> QWidget:
    archive = context.archive
    card = w.Card(
        context.pick(
            "Ҳар бир конфигурация нимани ўзгартиради",
            "What each configuration changes",
        ),
        context.pick(
            "«Ишлатиш оралиғи» — жадвал умуман мавжуд бўлган манба даражалари "
            "тўплами. Охирги устун — лексикографик мезоннинг ўзгартирилмаган "
            "буюртмадан устунлиги, ўша оралиқда.",
            "The operable range is the set of supply levels at which a schedule "
            "exists at all. The last column is the advantage of the lexicographic "
            "criterion over leaving the order unchanged, across that range.",
        ),
    )

    header = [
        context.ui("configuration"),
        context.pick("Ишлатиш оралиғи", "Operable range"),
        context.pick("Нуқталар", "Points"),
        context.pick("B4 энг ёмон улуш", "worst-off under B4"),
        context.pick("B4 − B1, танқисликда", "B4 − B1, under scarcity"),
    ]
    rows: list[list[str]] = []
    for label in archive.labels():
        solved = archive.solved_percents(label)
        worsts = [
            point.worst(arc.PROPOSED)
            for point in archive.points(label)
            if point.worst(arc.PROPOSED) is not None
        ]
        gains = [value for _, value in arc.advantage(archive, label, scarce_only=True)]
        rows.append([
            label,
            arc.operable_text(solved),
            str(len(solved)),
            arc.range_text([value for value in worsts if value is not None], digits=4),
            arc.range_text(gains, digits=3, sign=True) if gains else arc.DASH,
        ])
    card.add(w.grid_table(header, rows, max_height=220, fit=True))
    card.add(
        w.paragraph(
            context.pick(
                "Охирги устун тўлиқ таъминот нуқтасини ҳисобга олмайди: у ерда "
                "иккала мезон ҳам ҳар бир буюртмани тўлиқ бажаради, шунинг учун "
                "фарқ нолга тенг бўлади ва бу мезон ҳақида ҳеч нарса демайди. "
                "Тартиб 4 да танқисликда ечиладиган нуқта умуман йўқ, шунинг учун "
                "у ерда тире турибди.",
                "The last column leaves out full supply: there both criteria fill "
                "every order, so the difference is zero for a reason that says "
                "nothing about either. Order 4 has no solved point under scarcity at "
                "all, which is why it shows a dash.",
            )
        )
    )
    card.add(w.mono(f"{context.ui('source_prefix')}: results/scan/*/Q*.json"))

    main_cliff = archive.cliff(arc.MAIN_LABEL)
    lines: list[str] = []
    for label in archive.labels():
        described = context.pick(*_describe(archive, label, context))
        cliff = archive.cliff(label)
        if label == arc.MAIN_LABEL or cliff is None or main_cliff is None:
            moved = ""
        elif cliff == main_cliff:
            moved = context.pick(
                f" Жар қимирламади ({cliff}%).",
                f" The cliff did not move ({cliff}%).",
            )
        else:
            direction = (
                context.pick("пастга", "down") if cliff < main_cliff
                else context.pick("юқорига", "up")
            )
            moved = context.pick(
                f" Жар {main_cliff}% дан {cliff}% га — {direction} кўчди.",
                f" The cliff moved {direction}, from {main_cliff}% to {cliff}%.",
            )
        lines.append(f"<b>{label}</b> — {described}.{moved}")
    if lines:
        card.add(w.paragraph("<br>".join(lines)))
    return card


def _describe(archive: arc.Archive, label: str, context: Context) -> tuple[str, str]:
    if label in DESCRIBED:
        return DESCRIBED[label]
    settings = archive.settings(label)
    order = settings.get("filter_order")
    cutoff = settings.get("cutoff_rad_per_s")
    text = f"order {order}, {cutoff:g} rad/s" if cutoff is not None else str(order)
    return (text, text)


# ---------------------------------------------------------------------- curves


def _curves(context: Context) -> QWidget:
    archive = context.archive
    card = w.Card(
        context.pick(
            "Лексикографик жавоб, тўртта конфигурацияда",
            "The lexicographic answer under all four configurations",
        ),
        context.pick(
            "Ҳар бир чизиқнинг тўлдирилган маркери — жадвал ҳали мавжуд бўлган энг "
            "паст манба даражаси. Чизиқ ундан пастда давом этмайди, чунки у ерда "
            "чизиладиган сон йўқ.",
            "The filled marker on each line is the lowest supply at which a schedule "
            "still exists. The line does not continue below it, because there is no "
            "number there to draw.",
        ),
    )
    chart = w.Chart(height=320)
    ax = chart.axes()
    if ax is None:
        card.add(chart)
        return card

    palette = [COLORS["accent"], COLORS["short"], COLORS["filled"], "#7a5ea8"]
    drawn = False
    for index, label in enumerate(archive.labels()):
        pairs = [
            (point.percent, point.worst(arc.PROPOSED))
            for point in archive.points(label)
            if point.worst(arc.PROPOSED) is not None
        ]
        if not pairs:
            continue
        drawn = True
        pairs.sort()
        colour = palette[index % len(palette)]
        ax.plot(
            [pct for pct, _ in pairs],
            [value for _, value in pairs],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=colour,
            label=label,
        )
        ax.plot(
            [pairs[0][0]], [pairs[0][1]],
            marker="o", markersize=9, color=colour, zorder=5,
        )

    if not drawn:
        chart.say(context.ui("nothing_here"))
        card.add(chart)
        return card

    ax.set_xlabel(
        context.pick(
            "манба мавжудлиги, талабнинг % улуши",
            "source availability, % of aggregate demand",
        ),
        fontsize=9, color=COLORS["ink_soft"],
    )
    ax.set_ylabel(
        context.pick("B4 энг ёмон улуш", "worst-off share under B4"),
        fontsize=9, color=COLORS["ink_soft"],
    )
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    chart.draw()
    card.add(chart)
    return card


# --------------------------------------------------------------------- finding


def _finding(context: Context) -> QWidget:
    archive = context.archive
    cliffs = {label: archive.cliff(label) for label in archive.labels()}
    known = [value for value in cliffs.values() if value is not None]
    spread = (max(known) - min(known)) if known else 0

    everywhere_positive = True
    for label in archive.labels():
        for percent, gain in arc.advantage(archive, label):
            if percent < 100 and gain <= 0:
                everywhere_positive = False

    card = w.Card(
        context.pick(
            "Битта хулоса, иккита ярим",
            "One finding, in two halves",
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                f"<b>Фильтр параметрлари жадвал мавжудлигини ҳал қилади.</b> Ишлатиш "
                f"оралиғининг қуйи чегараси конфигурациялар бўйлаб {spread} фоиз "
                f"пунктга силжийди — фильтрнинг тартибини бир поғона кўтариш "
                f"каналнинг деярли бутун ишлаш оралиғини йўқ қилади.",
                f"<b>The filter parameters decide whether a schedule exists.</b> The "
                f"lower edge of the operable range moves by {spread} percentage "
                f"points across the configurations — raising the filter order by one "
                f"step removes almost the whole operating range of the canal.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Мезонлар орасидаги таққослашга эса деярли тегмайди.</b> "
                + (
                    "Танқислик бошланган ҳар бир нуқтада, тўртала конфигурацияда ҳам, "
                    "лексикографик мезон энг ёмон аҳволдаги фойдаланувчига "
                    "ўзгартирилмаган буюртмадан кўпроқ беради."
                    if everywhere_positive
                    else "Устунлик белгиси конфигурацияга қараб ўзгаради — қуйидаги "
                         "жадвал ҳар бирини алоҳида кўрсатади."
                ),
                "<b>It barely touches the comparison among criteria.</b> "
                + (
                    "At every point where scarcity has begun, in all four "
                    "configurations, the lexicographic criterion gives the worst-off "
                    "user more than leaving the order unchanged."
                    if everywhere_positive
                    else "The sign of the advantage changes with the configuration — "
                         "the table above shows each one separately."
                ),
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "Иккаласи ҳам мақолада шундай ёзилади. Биринчиси таклиф этилаётган "
                "усулнинг ютуғи эмас — у амалиётчи учун огоҳлантириш: фильтрни "
                "лойиҳалаш адолат мезонини танлашдан олдин келади. Иккинчиси эса "
                "асосий даъвонинг битта конфигурацияга боғлиқ эмаслигини кўрсатади.",
                "Both go into the article as they stand. The first is not a merit of "
                "the method proposed here — it is a warning to a practitioner: "
                "designing the filter comes before choosing a fairness criterion. The "
                "second shows that the central claim does not rest on one "
                "configuration.",
            )
        )
    )

    table = context.archive.table("main_summary.csv")
    if table is not None:
        card.add(
            w.paragraph(
                context.pick(
                    "Тўлиқ сонлар 8-саҳифадаги жадвалларда, ва улар шу ердаги "
                    "файллардан ҳисобланган.",
                    "The full numbers are in the tables on page 8, and they are "
                    "computed from the same files as this page.",
                )
            )
        )
    return card
