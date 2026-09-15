"""Two languages, chosen once and applied everywhere.

Uzbek (Cyrillic) is the language this project is worked in; English is the
language the manuscript is written in. A reviewer reading the English side
should see the wording the manuscript uses, so the English strings here are
taken from the manuscript and the archive rather than translated freely.

Pages pass both strings at the point of use, which keeps a sentence and its
translation on adjacent lines where they can be compared. Only labels that
belong to the window frame - and the names of things the archive stores in
English - live in the tables below.

This module imports nothing but the standard library, so the page registry
and its labels can be read on a machine with no GUI toolkit installed.
"""

from __future__ import annotations

LANGUAGES: tuple[str, ...] = ("uz", "en")

DEFAULT_LANGUAGE = "uz"

#: What each language calls itself, for the switch in the header.
ENDONYM: dict[str, str] = {"uz": "Ўзбекча", "en": "English"}


def normalise(language: str | None) -> str:
    """Accept ``uz``/``uzbek``/``UZ``; anything else falls back to the default."""
    if not language:
        return DEFAULT_LANGUAGE
    lowered = language.strip().lower()
    if lowered.startswith("uz") or lowered.startswith("ўз"):
        return "uz"
    if lowered.startswith("en"):
        return "en"
    return DEFAULT_LANGUAGE


def pick(language: str, uzbek: str, english: str) -> str:
    """Return one of the two strings. Unknown languages get Uzbek."""
    return english if normalise(language) == "en" else uzbek


#: Labels that appear in the frame rather than inside a page.
UI: dict[str, tuple[str, str]] = {
    "app_title": (
        "Fair-Canal-Delivery — натижалар кўргичи",
        "Fair-Canal-Delivery — results viewer",
    ),
    "headline_strip": (
        "Адолат режалаштирилган эмас, етказилган ҳажм бўйича  ·  "
        "лексикографик max-min  ·  бажарилмаслик сертификати",
        "Fairness on delivered volume, not on the schedule  ·  "
        "lexicographic max-min  ·  an infeasibility certificate",
    ),
    "supply_level": ("Манба мавжудлиги", "Source availability"),
    "configuration": ("Конфигурация", "Configuration"),
    "of_demand": ("талабнинг % улуши", "% of aggregate demand"),
    "criterion": ("Мезон", "Criterion"),
    "verdict": ("Ҳукм", "Verdict"),
    "worst_off": ("Энг ёмон улуш", "Worst-off share"),
    "total": ("Йиғинди", "Total"),
    "spread": ("Тарқоқлик", "Spread"),
    "programmes": ("Дастурлар", "Programmes"),
    "seconds": ("Сония", "Seconds"),
    "user": ("Фойдаланувчи", "User"),
    "share": ("Улуш", "Share"),
    "source_prefix": ("Манба", "Source"),
    "open_folder": ("Папкани очиш", "Open folder"),
    "open_file": ("Файлни очиш", "Open the file"),
    "open_csv": ("CSV ни очиш", "Open the CSV"),
    "open_full_size": ("Тўлиқ ҳажмда очиш", "Open full size"),
    "copy": ("Нусха олиш", "Copy"),
    "copied": ("Нусха олинди", "Copied"),
    "run": ("Юритиш", "Run"),
    "stop": ("Тўхтатиш", "Stop"),
    "running": ("Юритилмоқда…", "Running…"),
    "done": ("Тайёр", "Done"),
    "failed": ("Хато", "Failed"),
    "not_run": ("Юритилмаган", "Not run"),
    "no_figure": (
        "Расм ҳали чизилмаган. «Қайта юритиш» саҳифасидан расмларни чизинг.",
        "This figure has not been drawn yet. Draw the figures from the "
        "“Reproduce” page.",
    ),
    "no_table": (
        "Бу жадвал файли топилмади.",
        "This table file was not found.",
    ),
    "archive_complete": ("Архив тўлиқ", "Archive complete"),
    "archive_missing": ("файл етишмайди", "files missing"),
    "rows_shown": ("сатр кўрсатилди", "rows shown"),
    "nothing_here": (
        "Бу конфигурацияда ечиладиган нуқта йўқ.",
        "This configuration has no point with a schedule.",
    ),
}


def ui(language: str, key: str) -> str:
    uzbek, english = UI[key]
    return pick(language, uzbek, english)


# ---------------------------------------------------------------------------
# The vocabulary of the study
# ---------------------------------------------------------------------------

#: The three verdicts, spelled out. Colour never carries this on its own: a
#: reader who prints the window in black and white, or who does not separate
#: the two hues, still reads the word.
VERDICT: dict[str, tuple[str, str]] = {
    "solved": ("ечилди", "solved"),
    "infeasible": ("йўл қўйилмайди", "infeasible"),
    "undecided": ("ҳукм йўқ", "no verdict"),
    "level only": ("фақат даража", "level only"),
    "missing": ("ёзилмаган", "not recorded"),
}


def verdict_label(language: str, verdict: str) -> str:
    uzbek, english = VERDICT.get(verdict, VERDICT["missing"])
    return pick(language, uzbek, english)


#: The criteria compared at every supply level. The short name is what a
#: table column says; the sentence is what the criterion actually asks for.
CRITERION: dict[str, tuple[str, str, str, str]] = {
    # code: (uzbek short, english short, uzbek sentence, english sentence)
    "B1": (
        "B1 ўзгартирилмаган",
        "B1 unchanged",
        "Буюртма қандай берилган бўлса, шундай юборилади. Фильтр нима "
        "қолдирса, ўша етказилади.",
        "The order is sent exactly as placed. Whatever the filter leaves is "
        "what gets delivered.",
    ),
    "B2": (
        "B2 бир блок эрта",
        "B2 one block early",
        "Ўша буюртма, бир блок олдин. Фильтр кечикишини вақт суриш билан "
        "қоплашга уриниш.",
        "The same order, one block sooner: an attempt to pay for the "
        "filter's delay with a time shift.",
    ),
    "B3": (
        "B3 утилитар",
        "B3 utilitarian",
        "Етказилган ҳажмлар йиғиндиси максималлаштирилади. Кимга тегиши "
        "ҳисобга олинмайди.",
        "Maximise the sum of delivered volumes. Who receives them is not "
        "part of the question.",
    ),
    "B4": (
        "B4 лексикографик",
        "B4 lexicographic",
        "Аввал энг ёмон аҳволдаги фойдаланувчи кўтарилади, кейин ундан "
        "кейингиси — бу мақоланинг таклифи.",
        "Raise the worst-off user first, then the next, and so on. This is "
        "the criterion the paper proposes.",
    ),
    "B5": (
        "B5 энг кам тарқоқлик",
        "B5 least spread",
        "Улушларнинг стандарт четланиши минималлаштирилади — канал "
        "адабиётидаги одатий мезон.",
        "Minimise the standard deviation of the shares: the criterion the "
        "canal literature usually reaches for.",
    ),
    "M1": (
        "M1 эркин затвор",
        "M1 free gates",
        "Затворлар бошқарув қонунидан озод қилинган ҳолдаги юқори чегара. "
        "Векторни эмас, фақат биринчи босқични беради.",
        "The upper bound with the gates freed from the control law. It "
        "answers the first stage only, not the whole vector.",
    ),
}


def criterion_short(language: str, code: str) -> str:
    if code not in CRITERION:
        return code
    uzbek, english, _, _ = CRITERION[code]
    return pick(language, uzbek, english)


def criterion_sentence(language: str, code: str) -> str:
    if code not in CRITERION:
        return ""
    _, _, uzbek, english = CRITERION[code]
    return pick(language, uzbek, english)


#: Constraint families as the certificate names them. The archive stores the
#: English string, so the Uzbek is a lookup and an unknown family falls back
#: to whatever the file said rather than to a guess.
FAMILY: dict[str, str] = {
    "C2 delivered flow non-negative": "C2 етказилган оқим манфий эмас",
    "C3 volume budget": "C3 ҳажм бюджети",
    "C5 gate flow inside the reach's conveyance": "C5 затвор оқими ўтказувчанлик ичида",
    "C5' command inside the linear model's range": (
        "C5′ буйруқ чизиқли моделнинг оралиғида"
    ),
    "C6 gate travel rate": "C6 затвор ҳаракат тезлиги",
    "C7 level inside its band": "C7 сатҳ ўз йўлагида",
    "C8 source availability": "C8 манба мавжудлиги",
    "C9 pool storage between empty and full": "C9 ҳовуз захираси бўш ва тўла орасида",
    "C9' storage agrees with the level": "C9′ захира сатҳ билан келишади",
}


def family_label(language: str, family: str) -> str:
    if normalise(language) == "en":
        return family
    return FAMILY.get(family, family)


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
    "kappa125": (
        "фойдаланувчининг ўз қувури номиналдан 1,25 баравар тез тортади",
        "each user's own outlet draws 1.25 times its nominal rate",
    ),
    "kappa200": (
        "фойдаланувчининг ўз қувури номиналдан 2 баравар тез тортади",
        "each user's own outlet draws twice its nominal rate",
    ),
    "head008": (
        "затворлар 0,08 м босим фарқида ўқилди — рухсат этилган энг тори",
        "the gates read at a head of 0.08 m, the narrowest the canal admits",
    ),
    "head020": (
        "затворлар 0,20 м босим фарқида ўқилди",
        "the gates read at a head of 0.20 m",
    ),
    "band010": (
        "захира ва сатҳ ҳисоблари 0,10 м гача келиштирилди",
        "the two storage accounts reconciled to within 0.10 m",
    ),
    "band025": (
        "захира ва сатҳ ҳисоблари 0,25 м гача келиштирилди",
        "the two storage accounts reconciled to within 0.25 m",
    ),
}

#: What the pre-registered run used, so that a label this page has never
#: heard of can still be described by whichever setting differs from it.
FROZEN: dict[str, tuple[float, str, str]] = {
    "filter_order": (3, "фильтр тартиби", "filter order"),
    "cutoff_rad_per_s": (3.0e-3, "кесиш частотаси", "cut-off"),
    "overshoot": (0.0, "ҳажм устамаси", "volume allowance"),
    "gate_head_m": (0.10, "затвор боши", "gate head"),
    "outlet_headroom": (1.5, "қувур заҳираси", "outlet headroom"),
    "band_tolerance_m": (0.15, "захира йўлаги", "storage band"),
    "announce_block": (1, "эълон блоки", "announcement block"),
    "cap_scale": (1.0, "ўтказувчанлик заҳираси", "conveyance headroom"),
    "demand_scale": (1.0, "талаб", "demand"),
}


def describe_configuration(settings: dict, label: str) -> "tuple[str, str]":
    """One line about a configuration: written above, or read off the run.

    The fallback used to name the filter and nothing else, which was
    enough while every sensitivity run moved the filter. It is not enough
    now: six of them move the gate head, the outlet headroom or the
    storage band instead, and described by filter alone all six would read
    as the pre-registered configuration - the one thing they are not. So
    it names whichever settings differ from the frozen ones, and says
    plainly when the run recorded none of them, because "this run used the
    frozen value" and "this run did not write the value down" are
    different statements and only one of them is a fact about the canal.
    """
    if label in DESCRIBED:
        return DESCRIBED[label]
    uzbek, english = [], []
    for key, (frozen, name_uz, name_en) in FROZEN.items():
        value = settings.get(key)
        if value is None or value == frozen:
            continue
        uzbek.append(f"{name_uz} {value:g}")
        english.append(f"{name_en} {value:g}")
    if uzbek:
        return (", ".join(uzbek), ", ".join(english))
    if any(settings.get(key) is not None for key in FROZEN):
        return ("қотирилган созламалар билан", "the frozen settings throughout")
    return ("созламалари ёзилмаган юриш", "a run that recorded no settings")


#: Provenance, in four words that are not interchangeable. The fourth was
#: added when the storage band stopped being chosen and started being
#: measured: a number this study obtained by running its own model and
#: reading the answer off it is neither observed, nor derived by a stated
#: formula, nor assumed, and calling it any of those three would misstate
#: where it came from.
PROVENANCE: dict[str, tuple[str, str]] = {
    "observed": ("кузатилган", "observed"),
    "derived": ("келтириб чиқарилган", "derived"),
    "assumed": ("фараз қилинган", "assumed"),
    "measured": ("ўлчанган", "measured"),
}


def provenance_label(language: str, kind: str) -> str:
    uzbek, english = PROVENANCE.get(kind, (kind, kind))
    return pick(language, uzbek, english)
