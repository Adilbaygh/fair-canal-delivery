"""3 · Five criteria on the same canal at the same supply.

This is the page a reviewer will spend the longest on, so it is built to be
moved: pick a configuration, drag the supply level, and every number and
bar on the page is re-read from the archive for that point. Nothing is
re-solved and nothing is interpolated - a supply level the scan never ran
simply is not on the slider.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ... import archive as arc
from .. import i18n
from .. import widgets as w
from ..theme import COLORS
from . import Context

#: The four criteria drawn as bars. M1 answers the first stage only, so it
#: has no vector of shares and cannot appear here; it is shown as a line.
DRAWN: tuple[str, ...] = ("B1", "B3", "B5", "B4")


def build(context: Context) -> QWidget:
    return ComparisonPage(context)


class ComparisonPage(QWidget):
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
                    "Бир хил каналда бешта мезон — ким нима олади",
                    "Five criteria on one canal — who receives what",
                )
            )
        )
        self.body.addWidget(
            w.page_lead(
                context.pick(
                    "Манба даражасини суринг. Ҳар бир нуқта архивдан ўқилади: "
                    "сканер юритилмаган даража слайдерда умуман йўқ, ва ечими "
                    "бўлмаган нуқтада сон эмас, ҳукм кўрсатилади.",
                    "Drag the supply level. Every point is read from the archive: a "
                    "level the scan never ran is not on the slider at all, and a "
                    "point with no schedule shows a verdict rather than a number.",
                )
            )
        )

        self.labels = self.archive.labels() or (arc.MAIN_LABEL,)
        self.label = self.labels[0]
        self.percents: tuple[int, ...] = ()

        self.body.addWidget(self._controls())

        self.summary_holder = QVBoxLayout()
        self.summary_holder.setSpacing(16)
        self.summary_holder.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.body.addLayout(self.summary_holder)
        self.body.addStretch(1)

        outer.addWidget(w.wrap_scroll(body))
        self._reload_percents()
        self._refresh()

    # -- controls -----------------------------------------------------------

    def _controls(self) -> QWidget:
        context = self.context
        card = w.Card()
        row = QHBoxLayout()
        row.setSpacing(14)

        row.addWidget(QLabel(context.ui("configuration")))
        self.label_box = w.combo(
            [(_configuration_label(self.archive, name, context), name)
             for name in self.labels]
        )
        self.label_box.setMinimumWidth(330)
        self.label_box.currentIndexChanged.connect(self._on_label_changed)
        row.addWidget(self.label_box)

        row.addSpacing(18)
        row.addWidget(QLabel(context.ui("supply_level")))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumWidth(260)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(lambda _value: self._refresh())
        row.addWidget(self.slider, 1)

        self.readout = QLabel()
        self.readout.setObjectName("NumberValue")
        self.readout.setStyleSheet(f"color: {COLORS['accent']}; font-size: 20px;")
        self.readout.setMinimumWidth(78)
        self.readout.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.readout)

        card.add_layout(row)
        self.configuration_note = w.paragraph("")
        card.add(self.configuration_note)
        return card

    def _on_label_changed(self, _index: int) -> None:
        self.label = self.label_box.currentData()
        self._reload_percents()
        self._refresh()

    def _reload_percents(self) -> None:
        self.percents = tuple(sorted(self.archive.percents(self.label)))
        self.slider.blockSignals(True)
        if self.percents:
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.percents) - 1)
            cliff = self.archive.cliff(self.label)
            start = self.percents.index(cliff) if cliff in self.percents else len(self.percents) - 1
            self.slider.setValue(start)
            self.slider.setEnabled(True)
        else:
            self.slider.setMinimum(0)
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        self.slider.blockSignals(False)

    def current_percent(self) -> int | None:
        if not self.percents:
            return None
        index = min(max(self.slider.value(), 0), len(self.percents) - 1)
        return self.percents[index]

    # -- content ------------------------------------------------------------

    def _clear(self) -> None:
        while self.summary_holder.count():
            item = self.summary_holder.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _refresh(self) -> None:
        context = self.context
        self._clear()
        settings = self.archive.settings(self.label)
        self.configuration_note.setText(
            _settings_sentence(settings, context, self.label)
        )

        percent = self.current_percent()
        if percent is None:
            self.readout.setText(arc.DASH)
            self.summary_holder.addWidget(w.Card("", context.ui("nothing_here")))
            return
        self.readout.setText(f"{percent}%")

        point = self.archive.point(self.label, percent)
        if point is None:
            self.summary_holder.addWidget(
                w.Card("", context.pick(
                    "Бу нуқта учун файл топилмади.",
                    "No file was found for this point.",
                ))
            )
            return

        self.summary_holder.addWidget(self._verdict_card(point))
        self.summary_holder.addWidget(self._shares_card(point))
        self.summary_holder.addWidget(self._criteria_card())

    # -- the six criteria at this point -------------------------------------

    def _verdict_card(self, point: arc.Point) -> QWidget:
        context = self.context
        available = point.source_m3_s
        title = context.pick(
            f"{point.percent}% манба мавжудлигида — олтита мезоннинг жавоби",
            f"At {point.percent}% source availability — what the six criteria answer",
        )
        subtitle = ""
        if available is not None:
            subtitle = context.pick(
                f"Манба {available:g} m³/s. Ҳар бир қатор битта мезоннинг ўз "
                f"жавоби; ҳукм устуни ҳеч қачон бўш қолмайди.",
                f"The source runs at {available:g} m³/s. Each row is one criterion's "
                f"own answer; the verdict column is never left blank.",
            )
        card = w.Card(title, subtitle)

        header = [
            context.ui("criterion"),
            context.ui("verdict"),
            context.ui("worst_off"),
            context.ui("total"),
            context.ui("spread"),
            context.ui("programmes"),
            context.ui("seconds"),
        ]
        rows: list[list[str]] = []
        tones: list[list[str | None]] = []
        for code in arc.CODES:
            verdict = point.status(code)
            rows.append([
                i18n.criterion_short(context.language, code),
                i18n.verdict_label(context.language, verdict),
                arc.shown(point.worst(code), verdict),
                arc.shown(point.total(code), verdict),
                arc.shown(point.spread(code), verdict),
                str(point.programmes(code) or arc.DASH),
                arc.fraction(point.seconds(code), 3),
            ])
            tone = {
                arc.SOLVED: "filled",
                arc.INFEASIBLE: "fail",
                arc.UNDECIDED: "undecided",
                arc.LEVEL_ONLY: "accent",
            }.get(verdict, "ink_faint")
            tones.append([None, tone, None, None, None, None, None])

        card.add(w.grid_table(header, rows, max_height=250, tones=tones))
        card.add(w.mono(f"{context.ui('source_prefix')}: {point.source}"))

        undecided = [code for code in arc.CODES if point.status(code) == arc.UNDECIDED]
        if undecided:
            card.add(
                w.paragraph(
                    context.pick(
                        "Бу нуқтада ечувчи ҳукм қайтармаган. Устунлар бўш эмас — "
                        "уларда ҳисобланмаган сон йўқ, ва бу тахмин билан "
                        "тўлдирилмайди: " + "; ".join(point.why(code) for code in undecided),
                        "The solver returned no verdict here. The columns are not "
                        "empty by accident — there is no computed number, and none is "
                        "invented: " + "; ".join(point.why(code) for code in undecided),
                    )
                )
            )
        return card

    # -- who goes short -----------------------------------------------------

    def _shares_card(self, point: arc.Point) -> QWidget:
        context = self.context
        card = w.Card(
            context.pick("Ким қанча олади — фойдаланувчи бўйича",
                         "What each user receives, user by user"),
            context.pick(
                "Затвор рақами тартибида — эълон қилинган жадваллардаги билан бир хил. "
                "1,0 улуш буюртма тўлиқ бажарилганини билдиради.",
                "In gate-number order, the same order the published tables use. A "
                "share of 1.0 means the order was filled in full.",
            ),
        )

        series = [
            (code, point.ratios(code))
            for code in DRAWN
            if point.ratios(code)
        ]
        if not series:
            card.add(
                w.paragraph(
                    context.pick(
                        "Бу манба даражасида ҳеч бир мезон учун жадвал мавжуд эмас, "
                        "шунинг учун чизиладиган улуш ҳам йўқ.",
                        "No criterion has a schedule at this supply level, so there "
                        "are no shares to draw.",
                    )
                )
            )
            return card

        users = sorted(series[0][1], key=w.short_name_sort_key)
        chart = w.Chart(
            height=max(240, 34 * len(users)),
            # Wider on the left for the gate labels, and room above the axes
            # for the legend, which sits there rather than over the bars.
            margins={"left": 0.135, "top": 0.90, "bottom": 0.18},
        )
        ax = chart.axes()
        if ax is not None:
            count = len(series)
            height = 0.78 / count
            positions = list(range(len(users)))
            for index, (code, values) in enumerate(series):
                offsets = [
                    position + (index - (count - 1) / 2) * height
                    for position in positions
                ]
                ax.barh(
                    offsets,
                    [values.get(user, 0.0) for user in users],
                    height=height * 0.92,
                    color=w.criterion_color(code),
                    label=i18n.criterion_short(context.language, code),
                    edgecolor="none",
                )
            ax.set_yticks(positions)
            ax.set_yticklabels([
                context.pick(f"затвор {w.short_name(user)}", f"gate {w.short_name(user)}")
                for user in users
            ], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.06)
            ax.axvline(1.0, color=COLORS["ink_faint"], linewidth=0.9, linestyle=":")
            ax.set_xlabel(
                context.pick("ойна ичида етказилган улуш",
                             "share delivered inside the window"),
                fontsize=9, color=COLORS["ink_soft"],
            )
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.legend(
                fontsize=9, frameon=False, ncol=len(series),
                loc="lower center", bbox_to_anchor=(0.5, 1.0),
            )
            chart.draw()
        card.add(chart)

        header = [context.ui("user")] + [
            i18n.criterion_short(context.language, code) for code, _ in series
        ]
        rows = [
            [context.pick(f"затвор {w.short_name(user)}", f"gate {w.short_name(user)}")]
            + [arc.fraction(values.get(user)) for _, values in series]
            for user in users
        ]
        bound = point.worst(arc.BOUND)
        rows.append(
            [context.pick("энг ёмон", "worst-off")]
            + [arc.fraction(point.worst(code)) for code, _ in series]
        )
        card.add(w.grid_table(header, rows, max_height=420, fit=True))
        if bound is not None:
            card.add(
                w.paragraph(
                    context.pick(
                        f"Эркин затворли чегара M1 бу нуқтада {arc.fraction(bound)} "
                        f"беради — энг ёмон аҳволдаги фойдаланувчи учун умуман "
                        f"эришиш мумкин бўлган энг юқори улуш (5-саҳифа, Теорема 2).",
                        f"The free-gate bound M1 stands at {arc.fraction(bound)} here "
                        f"— the highest share the worst-off user could reach at all "
                        f"(page 5, Theorem 2).",
                    )
                )
            )
        card.add(w.mono(f"{context.ui('source_prefix')}: {point.source} -> variants.*.ratios"))
        return card

    # -- what each criterion asks for ---------------------------------------

    def _criteria_card(self) -> QWidget:
        context = self.context
        card = w.Card(
            context.pick("Ҳар бир мезон нимани сўрайди",
                         "What each criterion actually asks for")
        )
        for code in arc.CODES:
            row = QHBoxLayout()
            row.setSpacing(10)
            swatch = w.chip(
                i18n.criterion_short(context.language, code),
                "#ffffff",
                w.criterion_color(code),
            )
            row.addWidget(swatch)
            row.addWidget(w.paragraph(i18n.criterion_sentence(context.language, code)), 1)
            card.add_layout(row)
        return card


# --------------------------------------------------------------------- helpers


def _configuration_label(archive: arc.Archive, name: str, context: Context) -> str:
    settings = archive.settings(name)
    order = settings.get("filter_order")
    cutoff = settings.get("cutoff_rad_per_s")
    overshoot = settings.get("overshoot")
    parts: list[str] = []
    if order is not None:
        parts.append(context.pick(f"тартиб {order}", f"order {order}"))
    if cutoff is not None:
        parts.append(f"{cutoff * 1000:g} mrad/s")
    if overshoot:
        parts.append(context.pick(f"бюджет +{overshoot:.0%}", f"budget +{overshoot:.0%}"))
    # Whether this is the pre-registered configuration is said in the
    # sentence below the row, not here: a chooser that has to elide its own
    # entries tells the reader less than a shorter one that does not.
    return f"{name} — {', '.join(parts)}" if parts else name


def _settings_sentence(settings: dict, context: Context, label: str = "") -> str:
    if not settings:
        return ""
    tail = context.pick(
        "  Бу — олдиндан эълон қилинган конфигурация.",
        "  This is the pre-registered configuration.",
    ) if label == arc.MAIN_LABEL else ""
    order = settings.get("filter_order")
    cutoff = settings.get("cutoff_rad_per_s")
    overshoot = settings.get("overshoot") or 0.0
    horizon = settings.get("horizon_steps")
    return context.pick(
        f"Фильтр: {order}-тартибли Butterworth, кесиш {cutoff:g} рад/с. "
        f"Ҳажм бюджети: +{overshoot:.0%}. Горизонт: {horizon} қадам.{tail}",
        f"Filter: Butterworth of order {order}, cut-off {cutoff:g} rad/s. "
        f"Volume budget: +{overshoot:.0%}. Horizon: {horizon} steps.{tail}",
    )
