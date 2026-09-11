"""5 · What is proved, kept apart from what was measured.

Two theorems belong to the method and hold whatever the data say. Every
other finding in this study is an observation on one canal at one demand
profile, and the last card on this page names those explicitly rather than
letting them borrow the word "theorem".

The witnesses under each theorem are read live out of the archive. A
witness does not prove a theorem - the proof does that - but it does show
that the situation the theorem describes is not hypothetical here.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ... import archive as arc
from .. import i18n
from .. import widgets as w
from . import Context


def build(context: Context) -> QWidget:
    body = w.column(18)
    layout = w.body_of(body)

    layout.addWidget(
        w.page_title(
            context.pick(
                "Иккита теорема — ва улардан ажратилган ўлчовлар",
                "Two theorems — and the measurements kept apart from them",
            )
        )
    )
    layout.addWidget(
        w.page_lead(
            context.pick(
                "Бу ерда фақат <b>исботланадиган</b> нарса теорема деб аталади. "
                "Ўлчанган натижа — ўлчанган натижа, ва у шундай ёзилади.",
                "Only what can be <b>proved</b> is called a theorem here. A measured "
                "result is a measured result, and it is written as one.",
            )
        )
    )

    layout.addWidget(_theorem_one(context))
    layout.addWidget(_theorem_two(context))
    layout.addWidget(_classical(context))
    layout.addWidget(_not_theorems(context))
    return w.wrap_scroll(body)


# ----------------------------------------------------------------- theorem one


def _theorem_one(context: Context) -> QWidget:
    archive = context.archive
    card = w.Card(
        context.pick(
            "Теорема 1 — трансляцияга инвариант тарқоқлик мезони Парето-мос эмас",
            "Theorem 1 — a translation-invariant dispersion measure is "
            "Pareto-inconsistent",
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Таъриф.</b> <i>D</i> тарқоқлик мезони <i>трансляцияга инвариант</i> "
                "дейилади, агар ҳар қандай <i>r</i> ва ўзгармас <i>c</i> учун "
                "<i>D</i>(<i>r</i> + <i>c</i>·<b>1</b>) = <i>D</i>(<i>r</i>) бўлса. "
                "Стандарт четланиш, диапазон, ўртача абсолют четланиш ва Жини ўртача "
                "фарқи — ҳаммаси шу синфда.",
                "<b>Definition.</b> A dispersion measure <i>D</i> is "
                "<i>translation-invariant</i> if <i>D</i>(<i>r</i> + <i>c</i>·<b>1</b>) "
                "= <i>D</i>(<i>r</i>) for every <i>r</i> and every constant <i>c</i>. "
                "The standard deviation, the range, the mean absolute deviation and "
                "the Gini mean difference are all in this class.",
            )
        )
    )
    card.add(
        w.quotation(
            context.pick(
                "<b>Теорема.</b> Агар <i>r</i> ва <i>r</i> + <i>c</i>·<b>1</b> "
                "иккаласи ҳам эришиладиган бўлса ва <i>c</i> &gt; 0 бўлса, у ҳолда "
                "<i>D</i> уларни ажрата олмайди, ҳолбуки иккинчиси биринчисини "
                "<i>ҳар бир</i> координатада қатъий доминация қилади. Демак "
                "<i>D</i> ни минималлаштириш Парето-самарали нуқтани танламайди.",
                "<b>Theorem.</b> If both <i>r</i> and <i>r</i> + <i>c</i>·<b>1</b> are "
                "attainable and <i>c</i> &gt; 0, then <i>D</i> cannot tell them apart, "
                "although the second strictly dominates the first in <i>every</i> "
                "coordinate. Minimising <i>D</i> therefore does not select a "
                "Pareto-efficient point.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Исбот.</b> Инвариантликдан бевосита <i>D</i>(<i>r</i> + "
                "<i>c</i>·<b>1</b>) = <i>D</i>(<i>r</i>). <i>c</i> &gt; 0 бўлгани учун "
                "ҳар бир координатада қиймат ошган. Агар <i>r</i> минимум бўлса, "
                "<i>r</i> + <i>c</i>·<b>1</b> ҳам ўша қийматни беради, демак у ҳам "
                "минимум. ∎ Тескари томони: ҳар қандай лексимин-оптимал нуқта "
                "Парето-самарали.",
                "<b>Proof.</b> Invariance gives <i>D</i>(<i>r</i> + <i>c</i>·<b>1</b>) "
                "= <i>D</i>(<i>r</i>) at once. Since <i>c</i> &gt; 0 every coordinate "
                "has risen. If <i>r</i> is a minimiser then <i>r</i> + <i>c</i>·<b>1</b> "
                "attains the same value and is a minimiser too. ∎ The converse "
                "direction: every lexicographic-maximin point is Pareto-efficient.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Нима учун бу муҳим.</b> Теорема «σ ёмон мезон» демайди — у "
                "<b>σ бошқа саволни сўрайди</b> дейди. Тарқоқлик тенг улушлар юзининг "
                "қайси нуқтасида туришни аниқламайди, чунки бутун юз бўйлаб қиймати "
                "бир хил. Лексимин эса аниқлайди: у энг ёмон аҳволдагини биринчи "
                "кўтаради. Ва теорема бир вақтнинг ўзида диапазон, MAD ва Жини "
                "мезонларини ҳам қамрайди — яъни бу битта рақобатчи билан эмас, "
                "<b>бутун синф</b> билан.",
                "<b>Why this matters.</b> The theorem does not say the standard "
                "deviation is a bad measure — it says it <b>asks a different "
                "question</b>. Dispersion does not decide where on the equal-shares "
                "face to stand, because its value is the same everywhere on that "
                "face. Leximin does decide: it lifts the worst-off user first. And "
                "the theorem covers the range, the MAD and the Gini measure in the "
                "same breath — so this is an argument with a <b>whole class</b>, not "
                "with one rival.",
            )
        )
    )

    witnesses = arc.spread_witnesses(archive)
    dominated = [item for item in witnesses if item.kind == arc.DOMINATED]
    different = [item for item in witnesses if item.kind == arc.DIFFERENT_QUESTION]
    scarce = [
        point.percent
        for point in archive.points(arc.MAIN_LABEL)
        if point.percent < 100 and point.status(arc.PROPOSED) == arc.SOLVED
    ]

    card.add(w.section_title(context.pick("Гувоҳ — архивдан ўқилди", "Witness — read from the archive")))
    if not witnesses:
        card.add(w.paragraph(context.ui("nothing_here")))
        return card

    card.add(
        w.paragraph(
            context.pick(
                f"Танқислик бошланган {len(scarce)} нуқтадан <b>{len(dominated)}</b> "
                f"тасида B4 ва B5 иккаласининг тарқоқлиги усулнинг ўз аниқлигидан "
                f"фарқ қилмайди — иккаласи ҳам тенг улушли нуқта — ва шунга қарамай "
                f"B4 <b>ҳар бир фойдаланувчига</b> қатъий кўпроқ беради. Қолган "
                f"{len(different)} тасида B5 нинг тарқоқлиги қатъий кичикроқ, "
                f"лекин энг ёмон аҳволдаги фойдаланувчи B4 да юқорироқ туради.",
                f"Of the {len(scarce)} points where scarcity has begun, at "
                f"<b>{len(dominated)}</b> the spreads of B4 and B5 both sit inside "
                f"the method's own accuracy — both are equal-share points — and B4 "
                f"still gives <b>every user</b> strictly more. At the remaining "
                f"{len(different)}, B5 has the strictly smaller spread while the "
                f"worst-off user stands higher under B4.",
            )
        )
    )

    header = [
        context.ui("supply_level"),
        context.pick("B4\nтарқоқлик", "spread\nB4"),
        context.pick("B5\nтарқоқлик", "spread\nB5"),
        context.pick("B4\nэнг ёмон", "worst-off\nB4"),
        context.pick("B5\nэнг ёмон", "worst-off\nB5"),
        context.pick("B4 қатъий\nкўп берган", "users B4 gives\nstrictly more"),
    ]
    rows = [
        [
            f"{item.percent}%",
            f"{item.spread_proposed:.2e}",
            f"{item.spread_rival:.2e}",
            arc.fraction(item.worst_proposed),
            arc.fraction(item.worst_rival),
            f"{item.users_strictly_better} / {item.users_total}",
        ]
        for item in witnesses
    ]
    card.add(w.grid_table(header, rows, max_height=420, fit=True))
    card.add(
        w.mono(
            f"{context.ui('source_prefix')}: results/scan/{arc.MAIN_LABEL}/Q*.json "
            f"-> variants.B4, variants.B5"
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "«Нолдан фарқ қилмайди» бу ерда тўрт хона билан эмас, усулнинг ўз "
                "эълон қилган аниқлиги билан ҳал қилинади — у ҳам ўша файлдан "
                "ўқилади. Тарқоқлиги кўринишда нолга ўхшайдиган нуқтани шу тарзда "
                "санашгина ҳалол.",
                "“Indistinguishable from zero” is decided here by the accuracy the "
                "run states for itself — read from the same file — not by the four "
                "decimals a table happens to print. Counting a point whose spread "
                "merely looks like zero would not be honest.",
            )
        )
    )
    return card


# ----------------------------------------------------------------- theorem two


def _theorem_two(context: Context) -> QWidget:
    archive = context.archive
    card = w.Card(
        context.pick(
            "Теорема 2 — эркин затворли релаксация лексимин векторни чегаралайди",
            "Theorem 2 — the free-gate relaxation bounds the leximin vector",
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "Эркин затворли дастурда буюртмалар, командалар, оқимлар ва сатҳлар "
                "айни физик чекловларга бўйсунади, лекин командалар бошқарув қонуни "
                "билан белгиланмайди. Ҳар бир йўл қўйилувчи ёпиқ контур траекторияси "
                "у ерда ҳам йўл қўйилувчи бўлгани учун, эришиладиган улушлар тўплами "
                "кенгроқ.",
                "In the free-gate programme the orders, the commands, the flows and "
                "the levels obey exactly the same physical constraints, but the "
                "commands are not fixed by the control law. Every feasible "
                "closed-loop trajectory is feasible there too, so the attainable set "
                "of shares is larger.",
            )
        )
    )
    card.add(
        w.quotation(
            context.pick(
                "<b>Теорема.</b> Жойлаштириш фарази бажарилса, эркин затворли "
                "лексимин вектор ёпиқ контурникини лексикографик тарзда доминация "
                "қилади; хусусан, унинг энг ёмон улуши кичик эмас. Бирор нуқтада "
                "<b>тенглик</b> — бошқарув қатлами ўша ерда ҳеч нарса "
                "йўқотмаётганининг сертификати.",
                "<b>Theorem.</b> Under the embedding assumption the free-gate leximin "
                "vector dominates the closed-loop one lexicographically; in "
                "particular its worst-off share is no smaller. <b>Equality</b> at a "
                "point is a certificate that the control layer is losing nothing "
                "there.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Фараз текширилиши шарт, ва текширилган.</b> Жойлаштириш автоматик "
                "эмас: у эркин затворли дастурда ёпиқ контур траекторияси айнан "
                "такрорланишини талаб қилади. Лойиҳада бу фараз бир марта бузилган "
                "эди — сатҳ индексида бир қадамлик силжиш бор эди, фарқи 1,2 см, ва "
                "у жимча ўтарди. Тузатилгандан кейин иккала йўлнинг мослиги "
                "1,2·10⁻² дан 6,3·10⁻¹⁵ га тушди.",
                "<b>The assumption has to be checked, and it was.</b> The embedding "
                "is not automatic: it requires the closed-loop trajectory to be "
                "reproduced exactly inside the free-gate programme. The assumption "
                "was violated once in this project — a one-step offset in the level "
                "index, a difference of 1.2 cm, and it would have passed unnoticed. "
                "Once corrected, the agreement between the two paths fell from "
                "1.2·10⁻² to 6.3·10⁻¹⁵.",
            )
        )
    )

    gaps = arc.bound_gaps(archive)
    card.add(w.section_title(context.pick("Гувоҳ — тирқиш нуқта бўйича", "Witness — the gap point by point")))
    if not gaps:
        card.add(w.paragraph(context.ui("nothing_here")))
        return card

    attained = [item for item in gaps if item.attained]
    strict = [item for item in gaps if not item.attained]
    card.add(
        w.paragraph(
            context.pick(
                f"{len(gaps)} ечилган нуқтадан <b>{len(attained)}</b> тасида чегарага "
                f"эришилган: лексикографик тақсимот энг ёмон аҳволдагига берадиган "
                f"улуш — ўша шароитда умуман эришиш мумкин бўлган энг юқори улуш. "
                + (
                    f"Қолган {len(strict)} нуқтада тирқиш бор, ва у энг танқис "
                    f"нуқта: {', '.join(str(item.percent) + '%' for item in strict)}."
                    if strict else ""
                ),
                f"Of {len(gaps)} solved points the bound is attained at "
                f"<b>{len(attained)}</b>: the share the lexicographic allocation "
                f"gives the worst-off user is the highest share reachable at all. "
                + (
                    f"A gap remains at {len(strict)}, and it is the tightest point: "
                    f"{', '.join(str(item.percent) + '%' for item in strict)}."
                    if strict else ""
                ),
            )
        )
    )
    header = [
        context.ui("supply_level"),
        context.pick("M1\nчегара", "M1\nbound"),
        context.pick("B4\nэришилган", "B4\nachieved"),
        context.pick("тирқиш", "gap"),
        context.pick("усулнинг\nаниқлиги", "method\naccuracy"),
        context.pick("чегарага\nэришилди", "bound\nattained"),
    ]
    rows = [
        [
            f"{item.percent}%",
            arc.fraction(item.bound),
            arc.fraction(item.achieved),
            f"{item.gap:.2e}",
            f"{item.tolerance:.0e}",
            context.pick("ҳа", "yes") if item.attained else context.pick("йўқ", "no"),
        ]
        for item in gaps
    ]
    tones = [
        [None, None, None, None, None, "filled" if item.attained else "short"]
        for item in gaps
    ]
    card.add(w.grid_table(header, rows, max_height=420, tones=tones, fit=True))
    card.add(
        w.paragraph(
            context.pick(
                "Тенгликни нолга эмас, усулнинг ўзи эълон қилган аниқлигига қарши "
                "текширамиз. Босқичли процедура тўйинишни толерантлик билан "
                "аниқлайди, шунинг учун у ҳеч қачон бундан яқин тушишни даъво "
                "қилмаган; битта толерантлик катталигидаги фарқни «қатъий» деб "
                "аташ — усулда йўқ фарқни хабар қилиш бўларди.",
                "Equality is tested against the accuracy the method states for "
                "itself, not against zero. The staged procedure decides saturation "
                "with a tolerance and never claimed to land closer than that; calling "
                "a difference of one tolerance “strict” would report a distinction "
                "the method does not have.",
            )
        )
    )
    card.add(
        w.mono(
            f"{context.ui('source_prefix')}: results/scan/{arc.MAIN_LABEL}/Q*.json "
            f"-> variants.M1.worst, variants.B4.worst, variants.B4.detail.accuracy_bound"
        )
    )
    return card


# ------------------------------------------------------------------ classical


def _classical(context: Context) -> QWidget:
    card = w.Card(
        context.pick(
            "Классик натижалар — иқтибос қилинади, ўзимизники деб кўрсатилмайди",
            "Classical results — cited, not claimed",
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Лемма A.</b> Босқичли лексимин процедураси (ҳар босқичда "
                "max-min, тўйинганларни қотириб давом этиш) кўпи билан <i>n</i> "
                "босқичда лексимин-оптимал нуқтани қайтаради. Бу Ogryczak &amp; "
                "Śliwiński ва Luss ишларида исботланган; биз уни қўллаймиз, кашф "
                "этмаймиз. Бизнинг қўшимчамиз фақат амалиёт даражасида: тўйиниш "
                "тести толерантлик билан бажарилади, шунинг учун кафолат <i>аниқ "
                "тўйиниш аниқлаши остида</i> амал қилади, ва натижавий аниқлик ҳар "
                "бир юришнинг ўз файлида ёзилади.",
                "<b>Lemma A.</b> The staged leximin procedure — max-min at each "
                "stage, freeze the saturated users, continue — returns a "
                "leximin-optimal point in at most <i>n</i> stages. This is proved by "
                "Ogryczak &amp; Śliwiński and by Luss; we apply it, we did not "
                "discover it. Our addition is at the level of practice only: the "
                "saturation test uses a tolerance, so the guarantee holds <i>under "
                "exact detection of saturation</i>, and the resulting accuracy is "
                "written into every run's own file.",
            )
        )
    )
    card.add(
        w.paragraph(
            context.pick(
                "<b>Лемма B.</b> Frank–Wolfe нинг чизиқлаштириш тирқиши σ учун "
                "сертификатланган интервал беради. Бу стандарт хулоса, ва у бу ерда "
                "керак бўлди: тирқиқ бўйича тўхташ мезони σ* = 0 бўлган жойда "
                "фойдасиз — 200 итерация, 74 сония, яқинлашиш йўқ. Сертификатланган "
                "интервал билан 7 итерация, 2,4 сония.",
                "<b>Lemma B.</b> The Frank–Wolfe linearisation gap yields a certified "
                "interval for σ. That is the standard conclusion, and it was needed "
                "here: a gap-based stopping rule is useless where σ* = 0 — 200 "
                "iterations, 74 seconds, no convergence. With the certified interval: "
                "7 iterations, 2.4 seconds.",
            )
        )
    )
    return card


# -------------------------------------------------------------- not a theorem


def _not_theorems(context: Context) -> QWidget:
    archive = context.archive
    cliff = archive.cliff(arc.MAIN_LABEL)
    card = w.Card(
        context.pick(
            "Теоремага айланмайдиганлар — очиқ рўйхат",
            "What does not become a theorem — the open list",
        ),
        context.pick(
            "Булар мақолада сонли тажриба сифатида келтирилади, ва ҳар бирида "
            "инстанцияга хос экани айтилади. Уларни теорема деб эълон қилиш — бу "
            "лойиҳада беш марта тузатилган хатонинг айнан ўзи: ҳисобланмаган нарса "
            "ҳақида даъво.",
            "These appear in the article as numerical experiments, and each one says "
            "that it is specific to this instance. Declaring them theorems would be "
            "exactly the mistake corrected five times in this project: a claim about "
            "something nobody computed.",
        ),
    )
    rows = [
        [
            context.pick(
                f"{cliff}% даги жар" if cliff else context.pick("жар", "the cliff"),
                f"the cliff at {cliff}%" if cliff else "the cliff",
            ),
            context.pick(
                "битта канал, битта манба қиялиги, битта талаб профили",
                "one canal, one source ramp, one demand profile",
            ),
        ],
        [
            context.pick("B4 нинг B1 дан устунлиги", "B4's advantage over B1"),
            context.pick(
                "ўн бир нуқтада ўлчов; умумий кафолат эмас",
                "measured at eleven points; not a general guarantee",
            ),
        ],
        [
            context.pick("Фильтр тартиби жарни кўчириши", "The filter order moving the cliff"),
            context.pick("битта фильтр, битта канал", "one filter, one canal"),
        ],
        [
            context.pick("Структуравий шифтларнинг етарли эмаслиги",
                         "Ceilings being insufficient"),
            context.pick(
                "шу каналда барча шифтлар 1 бўлиб чиқди",
                "on this canal every ceiling came out equal to one",
            ),
        ],
        [
            context.pick("Детерминизм (иккита юриш айнан мос)",
                         "Determinism (two runs agree exactly)"),
            context.pick(
                "ўлчанган, исботланмаган: тенгликбузарнинг ягоналиги LP да "
                "кафолатланмайди",
                "measured, not proved: uniqueness of the tie-break is not guaranteed "
                "by the LP",
            ),
        ],
    ]
    card.add(
        w.grid_table(
            [
                context.pick("Натижа", "Result"),
                context.pick("Нима учун теорема эмас", "Why it is not a theorem"),
            ],
            rows,
            max_height=300,
            fit=True,
        )
    )
    return card
