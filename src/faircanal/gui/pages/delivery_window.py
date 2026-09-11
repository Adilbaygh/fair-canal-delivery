"""2 · What is measured: the volume delivered inside the agreed window.

The page exists because the whole study turns on one substitution. The
canal-control literature scores a schedule; this study scores what arrived.
Between the two sits a low-pass filter that keeps the integral of a signal
over a long horizon and moves part of it outside the window the farmer was
promised.

The timeline is drawn from the exported inputs, so it is the horizon the
scan actually ran, not an illustration of one.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ... import archive as arc
from .. import widgets as w
from ..theme import COLORS
from . import Context


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Ойна ва ҳажм — нима ўлчанади",
                "The window and the volume — what is measured",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Фойдаланувчи <i>i</i> учун келишилган ойна ичида ўлчов битта нисбат: "
                "<b>r<sub>i</sub> = ойна ичида етказилган ҳажм ÷ буюртма қилинган ҳажм</b>. "
                "Асосий кўрсаткич — шу нисбатларнинг энг кичиги, кейин лексикографик "
                "тартибда навбатдагилари.",
                "For user <i>i</i> the measure inside the agreed window is one ratio: "
                "<b>r<sub>i</sub> = volume delivered in the window ÷ volume ordered</b>. "
                "The headline quantity is the smallest of those ratios, then the next "
                "smallest, and so on in lexicographic order.",
            )
        )
    )

    layout.addWidget(_why_it_differs(context))
    layout.addWidget(_timeline(context))
    layout.addWidget(_constants(context))
    return w.wrap_scroll(body)


# ------------------------------------------------------- why the two differ


def _why_it_differs(context: Context) -> QWidget:
    data = context.archive.inputs()
    filter_block = data.get("filter") or {}
    order = filter_block.get("order")
    cutoff = filter_block.get("cutoff_rad_per_s")
    memory = filter_block.get("memory_steps")

    card = w.Card(
        context.pick(
            "Нима учун жадвал ва етказилган ҳажм бир хил эмас",
            "Why the schedule and the delivered volume are not the same thing",
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "Каналда тўлқин жараёнларини сўндириш учун буюртмага паст частотали "
                "фильтр қўлланади. Фильтр сигналнинг интегралини узоқ горизонтда "
                "сақлайди — лекин унинг бир қисмини ойнадан ташқарига суради. Ҳажм "
                "йўқолмайди; у <i>кечикади</i>. Кечиккан сув эса буюртмани "
                "бажармайди.",
                "A low-pass filter is applied to the order to damp wave dynamics in "
                "the canal. The filter preserves the integral of the signal over a "
                "long horizon — but it moves part of it outside the window. The "
                "volume is not lost; it is <i>late</i>. Water that is late does not "
                "fill the order.",
            )
        )
    )
    if order is not None and cutoff is not None:
        card.add(
            w.key_values(
                [
                    (
                        context.pick("Фильтр", "Filter"),
                        context.pick(
                            f"{order}-тартибли Butterworth, кесиш частотаси "
                            f"{cutoff:g} рад/с",
                            f"Butterworth of order {order}, cut-off {cutoff:g} rad/s",
                        ),
                    ),
                    (
                        context.pick("Фильтр хотираси", "Filter memory"),
                        context.pick(
                            f"{memory} қадам — горизонт шунга мослаб узайтирилган",
                            f"{memory} steps — the horizon is lengthened to hold it",
                        ),
                    ),
                ],
                columns=1,
            )
        )
        card.add(w.mono(f"{context.ui('source_prefix')}: DATA/inputs.json -> filter"))
    return card


# -------------------------------------------------------------------- timeline


def _timeline(context: Context) -> QWidget:
    data = context.archive.inputs()
    discretisation = data.get("discretisation") or {}
    scenario = data.get("scenario") or {}
    filter_block = data.get("filter") or {}

    per_block = discretisation.get("steps_per_block")
    horizon = discretisation.get("horizon_steps")
    margin = discretisation.get("settle_margin_steps")
    blocks = scenario.get("blocks")
    lead = scenario.get("lead_blocks")
    memory = filter_block.get("memory_steps")

    card = w.Card(
        context.pick(
            "Битта юришнинг горизонти, ўлчов бирлиги — объект қадами",
            "The horizon of one run, measured in plant steps",
        )
    )
    if None in (per_block, horizon, margin, blocks, lead):
        card.add(w.paragraph(context.pick(
            "Кириш маълумотлари файли топилмади.",
            "The exported inputs file was not found.",
        )))
        return card

    # No y axis at all here, so the usual left margin would be empty space.
    chart = w.Chart(height=210, margins={"left": 0.03, "bottom": 0.26, "top": 0.97})
    ax = chart.axes()
    if ax is None:
        card.add(chart)
        return card

    window_start = lead * per_block
    window_end = horizon - 1
    order_end = blocks * per_block

    ax.grid(False)
    ax.set_ylim(0, 3)
    ax.set_xlim(-4, horizon + 6)
    ax.set_yticks([])
    for side in ("left",):
        ax.spines[side].set_visible(False)

    def band(y, start, end, color, text):
        ax.barh(
            y, end - start, left=start, height=0.62,
            color=color, edgecolor="none",
        )
        ax.text(
            (start + end) / 2, y, text, ha="center", va="center",
            fontsize=9, color="#ffffff" if color != COLORS["neutral_soft"] else COLORS["ink"],
        )

    band(2.2, 0, order_end, COLORS["accent"],
         context.pick(f"буюртма блоклари ({blocks} × {per_block} қадам)",
                      f"order blocks ({blocks} × {per_block} steps)"))
    band(2.2, order_end, horizon, COLORS["neutral_soft"],
         context.pick(f"тинчланиш ({margin})", f"settling ({margin})"))
    band(1.2, window_start, window_end + 1, COLORS["filled"],
         context.pick(f"етказиш ойнаси [{window_start}, {window_end}]",
                      f"delivery window [{window_start}, {window_end}]"))
    if memory:
        band(0.3, order_end, min(order_end + memory, horizon), COLORS["short"],
             context.pick(f"фильтр думи ({memory})", f"filter tail ({memory})"))

    ax.set_xlabel(
        context.pick("объект қадами (1 қадам = 60 с)", "plant step (1 step = 60 s)"),
        fontsize=9, color=COLORS["ink_soft"],
    )
    chart.draw()
    card.add(chart)
    card.add(
        w.paragraph(
            context.pick(
                f"Ойна буюртма берилгандан <b>{lead}</b> блок кейин очилади ва "
                f"горизонтнинг охиригача турибди, шунинг учун фильтрнинг думи ойна "
                f"ичида қолади. Тинчланиш чегараси ({margin} қадам) фильтр "
                f"хотирасидан узунроқ бўлиши шарт — акс ҳолда ҳар бир буюртманинг "
                f"бир қисми горизонт четидан тушиб кетади ва ҳажм баланси ҳеч нарса "
                f"демасдан бузилади.",
                f"The window opens <b>{lead}</b> blocks after the order is placed and "
                f"stays open to the end of the horizon, so the filter's tail lands "
                f"inside it. The settling margin ({margin} steps) has to exceed the "
                f"filter memory — otherwise part of every order falls off the end of "
                f"the horizon and the volume balance stops adding up without anything "
                f"complaining.",
            )
        )
    )
    card.add(w.mono(f"{context.ui('source_prefix')}: DATA/inputs.json"))
    return card


# ------------------------------------------------------------------ constants


def _constants(context: Context) -> QWidget:
    data = context.archive.inputs()
    canal = data.get("canal") or {}
    scenario = data.get("scenario") or {}
    discretisation = data.get("discretisation") or {}

    card = w.Card(
        context.pick(
            "Сценарий — ҳар бир юришда бир хил",
            "The scenario, identical in every run",
        ),
        context.pick(
            "Бешала мезон айнан шу шароитда таққосланади: бир хил канал, бир хил "
            "талаб, бир хил манба лимити, бир хил фильтр. Фарқланадиган ягона нарса "
            "— буюртмани қандай қайта шакллантириш.",
            "All five criteria are compared under exactly these conditions: the same "
            "canal, the same demand, the same source limit, the same filter. The one "
            "thing that differs is how the order is reshaped.",
        ),
    )
    pairs = [
        (
            context.pick("Канал", "Canal"),
            f"{canal.get('name', arc.DASH)} · "
            + context.pick(f"{canal.get('reaches', arc.DASH)} пул",
                           f"{canal.get('reaches', arc.DASH)} reaches"),
        ),
        (
            context.pick("Йиғма талаб", "Aggregate demand"),
            f"{canal.get('aggregate_demand_m3_s', arc.DASH)} m³/s",
        ),
        (
            context.pick("Блоклар", "Blocks"),
            f"{scenario.get('blocks', arc.DASH)} × "
            f"{discretisation.get('dt_block_s', arc.DASH):g} s"
            if discretisation.get("dt_block_s") is not None
            else str(scenario.get("blocks", arc.DASH)),
        ),
        (
            context.pick("Олдиндан хабар", "Lead time"),
            context.pick(f"{scenario.get('lead_blocks', arc.DASH)} блок",
                         f"{scenario.get('lead_blocks', arc.DASH)} blocks"),
        ),
        (
            context.pick("Улуш шифти", "Ratio cap"),
            str(scenario.get("ratio_cap", arc.DASH)),
        ),
        (
            context.pick("Манба сканери", "Source scan"),
            context.pick(
                f"{len(scenario.get('scan_fractions') or [])} нуқта, "
                f"100% дан 30% гача",
                f"{len(scenario.get('scan_fractions') or [])} points, "
                f"100% down to 30%",
            ),
        ),
    ]
    card.add(w.key_values(pairs, columns=2))
    card.add(
        w.paragraph(
            context.pick(
                "Улуш шифти 1 га тенг: ортиқча берилган сув адолат кўрсаткичини "
                "кўтара олмайди. Бу муҳим — акс ҳолда бир фойдаланувчига икки "
                "баравар бериб, «ўртача» яхшилаш мумкин бўларди.",
                "The ratio is capped at one, so over-delivery cannot inflate the "
                "fairness measure. That matters: without the cap, giving one user "
                "twice their order would improve the “average”.",
            )
        )
    )
    card.add(w.mono(f"{context.ui('source_prefix')}: DATA/inputs.json"))
    return card
