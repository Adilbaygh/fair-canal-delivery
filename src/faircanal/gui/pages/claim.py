"""1 · The claim, in the fewest numbers that carry it.

Every figure on this page is read from the archive at the moment the page
is built. Nothing here is typed in, so a rerun that changes a number
changes this page too, and a number that no longer exists becomes a dash
rather than a stale digit.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ... import archive as arc
from .. import widgets as w
from ..theme import COLORS
from . import Context

#: The verbatim sentence this study answers, and where it is printed. The
#: quotation was checked against the published PDF, not against a web page.
GAP_QUOTE = (
    "However, additional consideration might need to be taken to make sure "
    "that the farmers get the amount of water that they ordered and that the "
    "delivery time is not delayed too much by the low pass filter. This could "
    "be accomplished by, for example, modifying the farmer's order before "
    "applying the low-pass filter."
)
GAP_SOURCE = "Heyden, Pates, Rantzer, arXiv:2203.16575, §4.1"


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Адолат режалаштирилган жадвал бўйича эмас, етказилган ҳажм бўйича",
                "Fairness is measured on the volume delivered, not on the schedule",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Тўлқинларни сўндирувчи фильтр буюртманинг бир қисмини келишилган "
                "ойнадан ташқарига суради. Жадвал адолатли бўлиши мумкин, шунга "
                "қарамай фермер ойна ичида ҳақиқатда олган ҳажм адолатсиз бўлади. "
                "Биз буюртмани фильтрдан <b>олдин</b> қайта шакллантирамиз, ва "
                "мезонни фильтрдан <b>кейинги</b> етказишга қўямиз.",
                "The wave-damping filter pushes part of an order outside the agreed "
                "window. The schedule can be fair and the volume the farmer actually "
                "receives inside the window still not be. We reshape the order "
                "<b>before</b> the filter, and put the criterion on the delivery "
                "<b>after</b> it.",
            )
        )
    )

    layout.addWidget(_headline(context))
    layout.addWidget(_the_gap(context))
    layout.addWidget(_curve(context))
    layout.addWidget(_road(context))
    layout.addWidget(_not_claimed(context))
    return w.wrap_scroll(body)


# -------------------------------------------------------------------- headline


def _headline(context: Context) -> QWidget:
    archive = context.archive
    label = arc.MAIN_LABEL
    cliff = archive.cliff(label)
    card = w.Card(
        context.pick(
            "Энг танқис нуқтада энг ёмон аҳволдаги фойдаланувчи нима олади",
            "What the worst-off user receives at the tightest supply that works",
        )
    )

    if cliff is None:
        card.add(w.paragraph(context.ui("nothing_here")))
        return card

    point = archive.point(label, cliff)
    assert point is not None
    ours = point.worst(arc.PROPOSED)
    theirs = point.worst(arc.UNCHANGED)
    gain = None if (ours is None or theirs is None) else ours - theirs

    cards = [
        w.ValueCard(
            arc.shown(theirs, point.status(arc.UNCHANGED)),
            context.pick(
                "<b>B1</b> — буюртма ўзгартирилмайди. Бугунги амалиёт.",
                "<b>B1</b> — the order is left as placed. Today's practice.",
            ),
            point.key_source(f"variants.{arc.UNCHANGED}.worst"),
            tone="short",
        ),
        w.ValueCard(
            arc.shown(ours, point.status(arc.PROPOSED)),
            context.pick(
                "<b>B4</b> — лексикографик max-min. Таклиф этилаётган мезон.",
                "<b>B4</b> — lexicographic max-min. The criterion proposed here.",
            ),
            point.key_source(f"variants.{arc.PROPOSED}.worst"),
            tone="accent",
        ),
        w.ValueCard(
            arc.signed(gain),
            context.pick(
                "Энг ёмон аҳволдаги фойдаланувчи учун ютуқ — бир томчи ҳам кўп "
                "сув оқизмасдан.",
                "What the worst-off user gains, with not one extra drop released.",
            ),
            context.pick(
                f"ҳисобланди: {arc.PROPOSED}.worst − {arc.UNCHANGED}.worst, "
                f"{point.source}",
                f"computed: {arc.PROPOSED}.worst − {arc.UNCHANGED}.worst, "
                f"{point.source}",
            ),
            tone="filled",
            note=context.pick(
                f"{cliff}% манба мавжудлигида",
                f"at {cliff}% source availability",
            ),
        ),
    ]
    card.add(w.card_row(cards, columns=3))

    gains = [value for _, value in arc.advantage(archive, label)]
    scarce = [value for pct, value in arc.advantage(archive, label) if pct < 100]
    if scarce:
        card.add(
            w.paragraph(
                context.pick(
                    f"Танқислик бошланган ҳар бир нуқтада ютуқ мусбат: "
                    f"{arc.range_text(scarce, sign=True)}. Тўлиқ таъминотда "
                    f"иккала мезон ҳам ҳаммага бутун буюртмасини беради, шунинг "
                    f"учун у ерда фарқ йўқ ва бўлиши ҳам керак эмас.",
                    f"At every point where scarcity has begun the gain is positive: "
                    f"{arc.range_text(scarce, sign=True)}. At full supply both "
                    f"criteria fill every order, so there is no difference there, "
                    f"and there should not be.",
                )
            )
        )
    elif gains:
        card.add(w.paragraph(context.ui("nothing_here")))
    return card


# ------------------------------------------------------------------- the quote


def _the_gap(context: Context) -> QWidget:
    card = w.Card(
        context.pick(
            "Очиқ жой муаллифларнинг ўз сўзи билан",
            "The gap, in the authors' own words",
        )
    )
    card.add(w.quotation(f"“{GAP_QUOTE}”"))
    card.add(
        w.paragraph(
            context.pick(
                f"<span style='color:{COLORS['ink_faint']}'>{GAP_SOURCE}</span>",
                f"<span style='color:{COLORS['ink_faint']}'>{GAP_SOURCE}</span>",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "Бу жумла манба мақоласининг охирида турибди ва ечилмаган "
                "қолган. Биз мана шуни оламиз: буюртма фильтрдан олдин қайта "
                "шакллантирилади, бошқарувчига ва фильтрга умуман тегилмайди.",
                "That sentence sits at the end of the source paper and was left "
                "open. This is the study that takes it: the order is reshaped "
                "before the filter, and neither the controller nor the filter is "
                "touched.",
            )
        )
    )
    return card


# ----------------------------------------------------------------------- curve


def _curve(context: Context) -> QWidget:
    archive = context.archive
    label = arc.MAIN_LABEL
    card = w.Card(
        context.pick(
            "Бутун сканер бўйлаб — энг ёмон улуш, манба камайиб борганда",
            "The whole scan — the worst-off share as the source runs down",
        ),
        context.pick(
            "Ҳар бир нуқта битта ечилган жадвал. Ечим мавжуд бўлмаган "
            "даражаларда чизиқ узилади: у ерда чизиладиган сон йўқ.",
            "Every marker is one solved schedule. The lines stop where no "
            "schedule exists: there is no number there to draw.",
        ),
    )
    chart = w.Chart(height=300)
    ax = chart.axes()
    if ax is None:
        card.add(chart)
        return card

    drawn = False
    for code in (arc.UNCHANGED, "B3", arc.PROPOSED):
        xs: list[int] = []
        ys: list[float] = []
        for point in archive.points(label):
            value = point.worst(code)
            if value is not None:
                xs.append(point.percent)
                ys.append(value)
        if not xs:
            continue
        drawn = True
        order = sorted(range(len(xs)), key=lambda index: xs[index])
        ax.plot(
            [xs[i] for i in order],
            [ys[i] for i in order],
            marker="o",
            markersize=4,
            linewidth=1.9 if code == arc.PROPOSED else 1.3,
            color=w.criterion_color(code),
            label=context.pick(*_short_pair(code)),
        )

    if not drawn:
        chart.say(context.ui("nothing_here"))
        card.add(chart)
        return card

    cliff = archive.cliff(label)
    percents = [point.percent for point in archive.points(label)]
    if cliff is not None and percents and min(percents) < cliff:
        ax.axvspan(
            min(percents) - 2.5,
            cliff - 2.5,
            color=COLORS["fail_soft"],
            zorder=0,
        )
        ax.text(
            (min(percents) + cliff) / 2 - 2.5,
            0.52,
            context.pick("жадвал йўқ", "no schedule"),
            ha="center",
            fontsize=9,
            color=COLORS["fail"],
        )

    ax.set_xlabel(
        context.pick(
            "манба мавжудлиги, талабнинг % улуши",
            "source availability, % of aggregate demand",
        ),
        fontsize=9,
        color=COLORS["ink_soft"],
    )
    ax.set_ylabel(
        context.pick("энг ёмон улуш", "worst-off share"),
        fontsize=9,
        color=COLORS["ink_soft"],
    )
    ax.set_ylim(0.45, 1.03)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    chart.draw()
    card.add(chart)
    card.add(w.mono(f"{context.ui('source_prefix')}: results/scan/{label}/Q*.json"))
    return card


def _short_pair(code: str) -> tuple[str, str]:
    from .. import i18n  # noqa: PLC0415

    uzbek, english, _, _ = i18n.CRITERION[code]
    return uzbek, english


# ------------------------------------------------------------------------ road


def _road(context: Context) -> QWidget:
    card = w.Card(
        context.pick("Мақоланинг йўли — тўртта қадам", "The road the paper takes — four steps")
    )
    steps = (
        (
            "window",
            context.pick("1. Ўлчов", "1. The measure"),
            context.pick(
                "Нима ўлчанади: келишилган ойна ичида ҳақиқатда етказилган ҳажмнинг "
                "буюртмага нисбати. Фильтр нима қиладими — шу ерда кўринади.",
                "What is measured: the volume actually delivered inside the agreed "
                "window, over the volume ordered. What the filter does shows up here.",
            ),
        ),
        (
            "comparison",
            context.pick("2. Таққослаш", "2. The comparison"),
            context.pick(
                "Бир хил каналда, бир хил талабда бешта мезон. Ким нима олади — "
                "фойдаланувчи-фойдаланувчи.",
                "Five criteria on the same canal at the same demand. Who receives "
                "what, user by user.",
            ),
        ),
        (
            "verdicts",
            context.pick("3. Учта жавоб", "3. Three answers"),
            context.pick(
                "Ечилди, йўл қўйилмайди, ёки ҳукм йўқ. Бажарилмайдиган буюртма "
                "оптимистик жадвал эмас, сертификат олади.",
                "Solved, infeasible, or no verdict. An order that cannot be met "
                "gets a certificate, not an optimistic schedule.",
            ),
        ),
        (
            "theorems",
            context.pick("4. Нима исботланган", "4. What is proved"),
            context.pick(
                "Иккита теорема — маълумотдан мустақил. Ва улардан фарқли равишда, "
                "нима фақат ўлчанган.",
                "Two theorems, independent of any data — and, kept apart from them, "
                "what was only measured.",
            ),
        ),
    )
    for key, title, text in steps:
        row = QHBoxLayout()
        row.setSpacing(12)
        button = QPushButton(title)
        button.setMinimumWidth(150)
        button.clicked.connect(lambda _checked=False, target=key: context.goto(target))
        row.addWidget(button)
        row.addWidget(w.paragraph(text), 1)
        card.add_layout(row)
    return card


# ----------------------------------------------------------------- not claimed


def _not_claimed(context: Context) -> QWidget:
    return w.Card(
        context.pick("Нима даъво қилинмайди", "What is not claimed"),
        context.pick(
            "Бу иш янги бошқарувчи таклиф қилмайди ва мавжуд фильтрни рад этмайди — "
            "иккаласи ҳам ўз жойида қолади. Кўпроқ сув ҳам оқизмайди: ҳажм бюджети "
            "барча вариантларда бир хил. Ва бу ерда ўлчанган ҳар қандай сон "
            "<b>битта канал</b>да, битта талаб профилида олинган; қайси натижа "
            "теорема, қайси бири ўлчов эканини 5-саҳифа ажратиб кўрсатади.",
            "This work proposes no new controller and rejects no existing filter — "
            "both stay exactly where they are. It releases no extra water either: "
            "the volume budget is the same in every variant. And every number "
            "measured here comes from <b>one canal</b> at one demand profile; page "
            "5 keeps what is proved apart from what was measured.",
        ),
    )
