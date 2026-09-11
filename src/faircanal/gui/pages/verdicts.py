"""4 · Three answers, and what happens when the answer is "no".

A supply level was either solved, or proved infeasible, or the solver
returned no verdict at all. The third is not a rounding of the second.
Every number on this page comes with the verdict that licenses it, and a
verdict that licenses no number leaves a dash.

When an order cannot be met, the method returns a certificate rather than
an optimistic schedule: which users are short, by how many cubic metres,
and which constraint family would have to give - in its own units.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ... import archive as arc
from .. import i18n
from .. import widgets as w
from . import Context


def build(context: Context) -> QWidget:
    return VerdictsPage(context)


class VerdictsPage(QWidget):
    def __init__(self, context: Context) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.context = context
        self.archive = context.archive

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = w.column(18)
        self.body = w.body_of(body)

        self.body.addWidget(
            w.page_title(
                context.pick(
                    "Учта жавоб: ечилди, йўл қўйилмайди, ҳукм йўқ",
                    "Three answers: solved, infeasible, no verdict",
                )
            )
        )
        self.body.addWidget(
            w.page_lead(
                context.pick(
                    "Ечувчи «ечим йўқ» деганда — бу далил. Ечувчи ҳеч нарса "
                    "демаганда — бу далил эмас, ва уни ечим йўқлиги деб ёзиш хато "
                    "бўларди. Бу лойиҳада беш марта қулаган код айнан шу фарқни "
                    "ҳисобга олмаган эди, шунинг учун ажратиш ойнанинг ичига "
                    "қурилган: ҳукмсиз нуқтада сон умуман кўрсатилмайди.",
                    "When the solver says a point is infeasible, that is evidence. "
                    "When the solver says nothing at all, it is not, and recording it "
                    "as infeasible would be wrong. Five crashes in this project came "
                    "from code that missed that distinction, so the separation is "
                    "built into this window: a point with no verdict shows no number.",
                )
            )
        )

        self.labels = self.archive.labels() or (arc.MAIN_LABEL,)
        self.label = self.labels[0]

        self.body.addWidget(self._legend())
        self.body.addWidget(self._controls())

        self.holder = QVBoxLayout()
        self.holder.setSpacing(16)
        self.holder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.body.addLayout(self.holder)
        self.body.addStretch(1)

        outer.addWidget(w.wrap_scroll(body))
        self._reload_points()
        self._refresh()

    # -- chrome -------------------------------------------------------------

    def _legend(self) -> QWidget:
        context = self.context
        card = w.Card()
        for verdict, uzbek, english in (
            (
                arc.SOLVED,
                "жадвал мавжуд ва ҳисобланган; сонлар шу ердан келади",
                "a schedule exists and was computed; the numbers come from it",
            ),
            (
                arc.INFEASIBLE,
                "жадвал мавжуд эмаслиги исботланган — сертификат қайси чеклов "
                "оиласи бошқа нима беришини айтади",
                "no schedule exists, and it is proved; the certificate names the "
                "constraint family that would have to give",
            ),
            (
                arc.UNDECIDED,
                "ечувчи ҳукм қайтармади. Ҳеч ким ҳисобламаган — шунинг учун "
                "ҳеч қандай сон кўрсатилмайди",
                "the solver returned no verdict. Nobody computed it — so no number "
                "is shown",
            ),
            (
                arc.LEVEL_ONLY,
                "эркин затворли чегара: биринчи босқич ечилди, улушлар вектори "
                "эса бу дастурда умуман йўқ",
                "the free-gate bound: the first stage is solved, and this programme "
                "has no vector of shares at all",
            ),
        ):
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(w.verdict_chip(context.language, verdict))
            row.addWidget(w.paragraph(context.pick(uzbek, english)), 1)
            card.add_layout(row)
        return card

    def _controls(self) -> QWidget:
        context = self.context
        card = w.Card()
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(QLabel(context.ui("configuration")))
        self.label_box = w.combo([(name, name) for name in self.labels])
        self.label_box.currentIndexChanged.connect(self._on_label_changed)
        row.addWidget(self.label_box)

        row.addSpacing(18)
        row.addWidget(QLabel(context.pick("Сертификат нуқтаси", "Certificate at")))
        self.point_box = w.combo([])
        self.point_box.currentIndexChanged.connect(lambda _index: self._refresh())
        row.addWidget(self.point_box)
        row.addStretch(1)
        card.add_layout(row)
        return card

    def _on_label_changed(self, _index: int) -> None:
        self.label = self.label_box.currentData()
        self._reload_points()
        self._refresh()

    def _reload_points(self) -> None:
        self.point_box.blockSignals(True)
        self.point_box.clear()
        percents = self.archive.percents(self.label)
        for percent in percents:
            self.point_box.addItem(f"{percent}%", percent)
        cliff = self.archive.cliff(self.label)
        if cliff is not None and cliff in percents:
            self.point_box.setCurrentIndex(percents.index(cliff))
        self.point_box.blockSignals(False)

    # -- content ------------------------------------------------------------

    def _clear(self) -> None:
        while self.holder.count():
            item = self.holder.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _refresh(self) -> None:
        self._clear()
        self.holder.addWidget(self._matrix())
        percent = self.point_box.currentData()
        if percent is None:
            return
        point = self.archive.point(self.label, percent)
        if point is None:
            return
        self.holder.addWidget(self._certificate(point))

    def _matrix(self) -> QWidget:
        context = self.context
        points = self.archive.points(self.label)
        card = w.Card(
            context.pick(
                "Бутун сканернинг ҳукмлари",
                "Every verdict in the scan",
            ),
            context.pick(
                "Ҳар бир катакда бир сўз турибди. Ранг ёрдам беради, лекин маънони "
                "ранг ташимайди — қоғозга оқ-қора босилганда ҳам ўқилади.",
                "Every cell carries a word. Colour helps, but colour does not carry "
                "the meaning: the table still reads when it is printed in black and "
                "white.",
            ),
        )
        if not points:
            card.add(w.paragraph(context.ui("nothing_here")))
            return card

        # Bare codes in the header, and the names spelled out underneath: the
        # verdict words are long enough that the full names push the last
        # column off the screen, and a matrix nobody can see is no matrix.
        header = [context.ui("supply_level")] + list(arc.CODES)
        rows: list[list[str]] = []
        tones: list[list[str | None]] = []
        for point in points:
            rows.append(
                [f"{point.percent}%"]
                + [i18n.verdict_label(context.language, point.status(code))
                   for code in arc.CODES]
            )
            tones.append(
                [None]
                + [
                    {
                        arc.SOLVED: "filled",
                        arc.INFEASIBLE: "fail",
                        arc.UNDECIDED: "undecided",
                        arc.LEVEL_ONLY: "accent",
                    }.get(point.status(code), "ink_faint")
                    for code in arc.CODES
                ]
            )
        card.add(w.grid_table(header, rows, max_height=560, tones=tones, fit=True))
        card.add(
            w.mono(
                "   ·   ".join(
                    i18n.criterion_short(context.language, code) for code in arc.CODES
                )
            )
        )

        undecided = self.archive.undecided_points(self.label)
        solved = len(self.archive.solved_percents(self.label))
        card.add(
            w.paragraph(
                context.pick(
                    f"{len(points)} даражадан {solved} тасида лексикографик жадвал "
                    f"мавжуд. Ҳукмсиз ҳолат: {len(undecided)} та.",
                    f"Of {len(points)} supply levels, {solved} have a lexicographic "
                    f"schedule. Points with no verdict: {len(undecided)}.",
                )
            )
        )
        card.add(w.mono(f"{context.ui('source_prefix')}: results/scan/{self.label}/Q*.json"))
        return card

    # -- the certificate ----------------------------------------------------

    def _certificate(self, point: arc.Point) -> QWidget:
        context = self.context
        certificate = point.certificate
        card = w.Card(
            context.pick(
                f"{point.percent}% даги сертификат",
                f"The certificate at {point.percent}%",
            ),
            context.pick(
                "Бажарилмайдиган буюртма тақсимот қатламига оптимистик жадвал эмас, "
                "мана шуни қайтаради.",
                "This, and not an optimistic schedule, is what an order that cannot "
                "be met returns to the allocation layer.",
            ),
        )
        if not certificate:
            card.add(
                w.paragraph(
                    context.pick(
                        "Бу нуқтада сертификат ёзилмаган.",
                        "No certificate was written for this point.",
                    )
                )
            )
            return card

        ratios = certificate.get("ratios") or {}
        shortfalls = certificate.get("shortfalls_m3") or {}
        fulfilled = certificate.get("fulfilled")
        impossible = certificate.get("impossible_users") or []

        cards = [
            w.ValueCard(
                context.pick("ҲА", "YES") if fulfilled else context.pick("ЙЎҚ", "NO"),
                context.pick(
                    "Барча буюртмалар тўлиқ бажарилдими",
                    "Was every order filled in full",
                ),
                point.key_source("certificate.fulfilled"),
                tone="filled" if fulfilled else "short",
            ),
            w.ValueCard(
                str(len(shortfalls)),
                context.pick(
                    "камомад қолган фойдаланувчи",
                    "users left short of their order",
                ),
                point.key_source("certificate.shortfalls_m3"),
                tone="short",
            ),
            w.ValueCard(
                str(len(impossible)),
                context.pick(
                    "структуравий имконсиз буюртма — ҳеч қандай жадвал билан "
                    "бажарилмайди",
                    "structurally impossible orders — no schedule fills them",
                ),
                point.key_source("certificate.impossible_users"),
                tone="fail" if impossible else "ink_faint",
            ),
        ]
        card.add(w.card_row(cards, columns=3))

        binding = point.binding_families
        if binding:
            card.add(
                w.paragraph(
                    context.pick(
                        "Боғловчи чеклов оиласи: <b>"
                        + "</b>, <b>".join(
                            i18n.family_label(context.language, name) for name in binding
                        )
                        + "</b>",
                        "Binding constraint families: <b>"
                        + "</b>, <b>".join(binding)
                        + "</b>",
                    )
                )
            )

        slacks = [row for row in point.slacks if row[1] > 0.0]
        if slacks:
            card.add(
                w.section_title(
                    context.pick(
                        "Эластик релаксация — нима етишмаяпти, ўз бирлигида",
                        "The elastic relaxation — what is missing, in its own units",
                    )
                )
            )
            card.add(
                w.grid_table(
                    [
                        context.pick("Чеклов оиласи", "Constraint family"),
                        context.pick("Керакли қўшимча", "Additional amount required"),
                    ],
                    [
                        [i18n.family_label(context.language, name),
                         f"{value:.4g} {w.units(unit)}"]
                        for name, value, unit in slacks
                    ],
                    max_height=220,
                    fit=True,
                )
            )
            card.add(
                w.paragraph(
                    context.pick(
                        "Барча оилалар бирданига бўшатилади ва йиғинди слак "
                        "минималлаштирилади. Уларни бирма-бир синаш — алмаштирилган "
                        "савол: аудитдаги ҳақиқий сценарийда манба ва ўтказувчанлик "
                        "<i>биргаликда</i> бериши керак эди, ва биронтаси якка ўзи "
                        "етмасди.",
                        "Every family is relaxed at once and the total slack is "
                        "minimised. Testing them one at a time answers a different "
                        "question: in the case found during the audit the source and "
                        "the conveyance had to give <i>together</i>, and neither was "
                        "enough alone.",
                    )
                )
            )

        if shortfalls:
            card.add(
                w.section_title(
                    context.pick("Ким қанча олмади", "Who did not receive how much")
                )
            )
            users = sorted(shortfalls, key=w.short_name_sort_key)
            card.add(
                w.grid_table(
                    [
                        context.ui("user"),
                        context.ui("share"),
                        context.pick("камомад, m³", "shortfall, m³"),
                    ],
                    [
                        [
                            context.pick(
                                f"затвор {w.short_name(user)}", f"gate {w.short_name(user)}"
                            ),
                            arc.fraction(ratios.get(user)),
                            f"{shortfalls[user]:,.0f}".replace(",", " "),
                        ]
                        for user in users
                    ],
                    max_height=340,
                    fit=True,
                )
            )

        ceilings = certificate.get("ceilings") or {}
        if ceilings:
            below = [name for name, value in ceilings.items() if float(value) < 1.0 - 1e-9]
            card.add(
                w.paragraph(
                    context.pick(
                        f"Структуравий шифтлар: {len(ceilings)} фойдаланувчидан "
                        f"{len(below)} тасининг шифти 1 дан паст. Шифт — "
                        f"<b>зарурий, лекин етарли эмас</b> шарт: ҳамма шифтлар 1 "
                        f"бўлиши мумкин ва шунга қарамай ҳеч ким тўлиқ олмаслиги "
                        f"мумкин. Бу ўлчанган, ва сертификатнинг камчилиги эмас, "
                        f"чегараси.",
                        f"Structural ceilings: of {len(ceilings)} users, {len(below)} "
                        f"have a ceiling below one. A ceiling is a <b>necessary but "
                        f"not sufficient</b> condition: every ceiling can equal one "
                        f"and still nobody be filled. That was measured, and it is a "
                        f"limit of the certificate rather than a defect in it.",
                    )
                )
            )

        if point.digest:
            card.add(w.mono(f"digest  {point.digest}"))
        card.add(w.mono(f"{context.ui('source_prefix')}: {point.source} -> certificate"))
        return card
