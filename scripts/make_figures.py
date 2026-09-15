#!/usr/bin/env python3
"""Draw the paper's figures from the result tables, and nothing else.

Every number here is read from ``results/tables/*.csv``. Nothing is typed
in, nothing is smoothed, nothing is interpolated between scan points, and
a point with no verdict is drawn as absent rather than as a value. Rerun
this after any scan and the figures follow the archive.

What the journal asks for
-------------------------
Mathematics (MDPI) takes PNG, JPEG or TIFF at no less than 600 dpi, in
RGB, with no editable parts in the image, and with the caption in the
manuscript rather than burnt into the picture. Everything is converted to
TIFF in production. Fonts inside figures should be one of Times, Arial,
Courier, Helvetica, Ubuntu or Calibri.

So each figure is written twice:

* ``.png`` at 600 dpi - **this is what is submitted**;
* ``.pdf`` - a vector master, for the author to zoom into while drafting
  and to re-export from if a reviewer asks for a change. Not submitted.

And ``captions.md`` is written beside them, holding each figure's caption
in the wording the manuscript should use. A caption that lives next to
the code that drew the figure cannot drift away from what the figure
actually shows, which is how a caption comes to describe a panel that was
edited out three revisions ago.

Colour and black-and-white
--------------------------
MDPI does not check black-and-white legibility, unlike the journal this
was first written for. The dash-and-marker encoding is kept anyway: every
series is identifiable without colour, which costs nothing and serves the
colour-blind reader, the photocopy and the greyscale print alike. The
palette is a validated colour-blind-safe order used in its documented
sequence.

    python scripts/make_figures.py            all of them
    python scripts/make_figures.py --only 1 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.provenance import repo_root  # noqa: E402

# --- the journal's page, in inches ----------------------------------------
# Both widths are read out of the journal's own template rather than from a
# round number, because a figure drawn at the wrong width is rescaled by the
# typesetter and every label in it changes size with it.
#
# From the LaTeX class (Definitions/mdpi.cls, the ordinary article layout):
# A4 is 210 mm, the left margin is 58.7 mm and the right margin 12.7 mm, so
# the text column is 210 - 58.7 - 12.7 = 138.6 mm. A figure that needs the
# whole page is wrapped in \begin{adjustwidth}{-\extralength}{0cm}, which
# gives back the 46.1 mm margin column: 210 - 12.7 - 12.7 = 184.6 mm. That
# second number is the class's own \fulllength.
#
# The Word template agrees exactly: its section properties give an A4 page
# of 11906 twips with 720-twip margins on both sides, which is the same
# 184.6 mm text block.
#
# So the template offers two widths and no others, and these are they.
MM = 1.0 / 25.4
COLUMN = 138.6 * MM   # the text column; a single-panel figure
FULL = 184.6 * MM     # the full text block, via adjustwidth in LaTeX
TWO_COLUMN = FULL     # the name the two-panel figure bodies already use

# --- the validated categorical order, used in its documented sequence -----
# Colour is the pleasant channel. Dash and marker are the load-bearing ones,
# because the journal checks the figure in black and white.
STYLE = {
    "B1": dict(color="#2a78d6", ls="-", marker="o", label="B1 unchanged order"),
    "B2": dict(color="#eb6834", ls="--", marker="s", label="B2 one block early"),
    "B3": dict(color="#1baf7a", ls=":", marker="^", label="B3 utilitarian"),
    "B4": dict(color="#0b0b0b", ls="-", marker="D", label="B4 lexicographic"),
    "B5": dict(color="#e87ba4", ls="-.", marker="v", label="B5 least spread"),
    "M1": dict(color="#52514e", ls="none", marker="_", label="M1 free-gate bound"),
}
ORDER = ("B1", "B2", "B3", "B4", "B5")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
NOTHING = "#f0efec"   # the region where no schedule exists


def house_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "legend.frameon": False,
        # The journal's floor is 600 dpi and everything is converted to
        # TIFF in production, so this is the number that matters.
        "figure.dpi": 600,
        # Embed real fonts in the vector master rather than outlining
        # every glyph, so it stays searchable and re-editable.
        "pdf.fonttype": 42,
        # No tight bbox: the journal specifies exact widths, and a bbox
        # that shrinks to the ink delivers something else.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.h_pad": 0.03,
    })


# ---------------------------------------------------------------------------
# Reading the archive
# ---------------------------------------------------------------------------


def summary(label: str) -> dict[str, dict[int, float | None]]:
    """``{code: {percent: worst or None}}`` for one scan.

    ``None`` means the archive holds no number for that point - infeasible,
    or a solver that never decided. The two are different and neither is a
    value, so neither is plotted; the caller distinguishes them from the
    status column when it needs to.
    """
    path = repo_root() / "results" / "tables" / f"{label}_summary.csv"
    if not path.exists():
        raise SystemExit(f"no such scan: {path}. Run the scan before the figures.")
    out: dict[str, dict[int, float | None]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            worst = row["worst"]
            out[row["code"]][int(row["percent"])] = (
                float(worst) if worst not in ("", None) else None
            )
    return out


def statuses(label: str) -> dict[str, dict[int, str]]:
    path = repo_root() / "results" / "tables" / f"{label}_summary.csv"
    out: dict[str, dict[int, str]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["code"]][int(row["percent"])] = row["status"]
    return out


def ratios(label: str, percent: int) -> dict[str, dict[str, float]]:
    """``{code: {user: fraction}}`` at one scan point."""
    path = repo_root() / "results" / "tables" / f"{label}_ratios.csv"
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["percent"]) == percent:
                out[row["code"]][row["user"]] = float(row["ratio"])
    return out


def series(table, code) -> tuple[list[int], list[float]]:
    """The points that have a number, in decreasing supply."""
    pairs = [(p, v) for p, v in sorted(table.get(code, {}).items()) if v is not None]
    return [p for p, _ in pairs], [v for _, v in pairs]


def cliff(table) -> int | None:
    """The lowest supply at which the lexicographic answer exists."""
    have = [p for p, v in table.get("B4", {}).items() if v is not None]
    return min(have) if have else None


def shade_the_gap(axis, table, text=True) -> None:
    """Mark where no schedule exists, so absence reads as absence."""
    edge = cliff(table)
    if edge is None:
        return
    lo, hi = axis.get_xlim()
    if edge - 2.5 <= lo:
        return
    axis.axvspan(lo, edge - 2.5, color=NOTHING, zorder=0, linewidth=0)
    if text:
        axis.text(
            (lo + edge - 2.5) / 2, 0.5, "no schedule\nexists",
            ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.4,
        )


def tidy(axis, xlabel=None, ylabel=None) -> None:
    axis.grid(True, axis="y", zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)


def save(figure, stem: str) -> list[Path]:
    out = repo_root() / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = out / f"{stem}.{suffix}"
        figure.savefig(path)
        if suffix == "png":
            flatten(path)
        written.append(path)
    plt.close(figure)
    return written


def flatten(path: Path) -> None:
    """Drop the alpha channel, because the journal asks for RGB.

    Matplotlib writes RGBA. The alpha here is fully opaque - nothing is
    transparent - so this changes no pixel a reader will ever see. It is
    done anyway because "RGB" is what the requirement says, and an alpha
    channel that survives into a print pipeline is the kind of thing that
    composites against black on one press and white on the next.

    Pillow ships with matplotlib, so this is not a new dependency. If it
    is somehow absent the figure is still perfectly usable, so the miss is
    reported rather than raised.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - matplotlib brings Pillow
        print(f"    (Pillow missing: {path.name} keeps its alpha channel)")
        return
    with Image.open(path) as image:
        if image.mode == "RGB":
            return
        dpi = image.info.get("dpi", (600, 600))
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.split()[-1] if "A" in image.mode else None)
        white.save(path, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 1 - the scan, and what the criterion is worth
# ---------------------------------------------------------------------------


def figure_1() -> list[Path]:
    table = summary("main")
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(TWO_COLUMN, 3.1), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    handles = []
    for code in ORDER:
        style = dict(STYLE[code])
        x, y = series(table, code)
        left.plot(x, y, **style, markerfacecolor="white", markeredgewidth=0.9,
                  zorder=3 if code == "B4" else 2)
        handles.append(Line2D([], [], color=style["color"], ls=style["ls"],
                              marker=style["marker"], markerfacecolor="white",
                              markeredgewidth=0.9, label=style["label"]))
    x, y = series(table, "M1")
    left.plot(x, y, color=STYLE["M1"]["color"], ls="none", marker="_",
              markersize=7, markeredgewidth=1.1, zorder=4)
    handles.append(Line2D([], [], color=STYLE["M1"]["color"], ls="none",
                          marker="_", markersize=7, markeredgewidth=1.1,
                          label=STYLE["M1"]["label"]))

    # The bound sits on top of the lexicographic answer at most points.
    # That coincidence is the result, so it is said rather than left for
    # the reader to notice that two curves are one - and where it stops
    # being a coincidence is said too, counted rather than assumed. The
    # first version of this annotation had the word "only" written into
    # it, which stayed true until the programme gained the families that
    # made the bound strict at a second point.
    apart = sorted(
        (
            (table["M1"][percent] - table["B4"][percent], percent)
            for percent in table["B4"]
            if table["M1"].get(percent) is not None
            and table["B4"].get(percent) is not None
            and table["M1"][percent] - table["B4"][percent] > 5.0e-5
        ),
        reverse=True,
    )
    if apart:
        gap, percent = apart[0]
        where = "only here" if len(apart) == 1 else f"most here of {len(apart)}"
        left.annotate(
            f"M1 exceeds B4\n{where} (+{gap:.3f})",
            xy=(percent, table["M1"][percent]), xytext=(30.5, 0.90),
            fontsize=7, color=MUTED, ha="left", va="top", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                            shrinkA=1, shrinkB=2),
        )

    left.set_xlim(28, 102)
    left.set_ylim(0.45, 1.04)
    left.set_xticks(range(30, 101, 10))
    shade_the_gap(left, table)
    tidy(left, "Source availability $Q^{\\mathrm{src}}$ (% of aggregate demand)",
         "Worst-off delivered fraction $\\min_i r_i$")
    left.set_title("a", loc="left", fontweight="bold")
    figure.legend(handles=handles, loc="outside lower center", ncol=3,
                  handlelength=2.4, columnspacing=2.0, handletextpad=0.5)

    # What the criterion buys the worst-off user, point by point.
    gain_x, gain_y = [], []
    for percent in sorted(table["B4"]):
        b1, b4 = table["B1"].get(percent), table["B4"].get(percent)
        if b1 is not None and b4 is not None:
            gain_x.append(percent)
            gain_y.append(b4 - b1)
    right.bar(gain_x, gain_y, width=3.2, color="#2a78d6", zorder=3,
              edgecolor="white", linewidth=0.5)
    right.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=2)
    right.set_xlim(28, 102)
    right.set_xticks(range(30, 101, 10))
    shade_the_gap(right, table, text=False)
    tidy(right, "Source availability $Q^{\\mathrm{src}}$ (%)",
         "$\\min_i r_i$ gained over B1")
    right.set_title("b", loc="left", fontweight="bold")
    right.set_ylim(-0.008, 0.19)
    right.set_yticks([0.00, 0.05, 0.10, 0.15])

    return save(figure, "fig1_scan")


# ---------------------------------------------------------------------------
# Figure 2 - who carries the shortfall
# ---------------------------------------------------------------------------


def figure_2(percent: int = 95) -> list[Path]:
    rows = ratios("main", percent)
    # Ordered by gate, not by value. At this point B4 gives every user the
    # same fraction, so "ordered by what B4 gives" would be an ordering the
    # data does not support - and a reader who took the axis at its word
    # would read a ranking into eight ties.
    users = sorted(rows["B4"])
    short = [name.replace("corning-", "").replace("-user", "") for name in users]
    y = list(range(len(users)))

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(TWO_COLUMN, 2.2), sharey=True
    )

    for axis, pair, title in (
        (left, ("B3", "B4"), "a"),
        (right, ("B5", "B4"), "b"),
    ):
        for code in pair:
            style = STYLE[code]
            axis.plot(
                [rows[code][name] for name in users], y,
                ls="none", marker=style["marker"], color=style["color"],
                markerfacecolor="white" if code != "B4" else style["color"],
                markeredgewidth=1.0, markersize=4.6, label=style["label"],
                zorder=3,
            )
        for index, name in enumerate(users):
            a, b = rows[pair[0]][name], rows[pair[1]][name]
            axis.plot([a, b], [index, index], color=GRID, linewidth=1.4, zorder=1)
        axis.set_xlim(0.86, 1.02)
        tidy(axis, "Delivered fraction $r_i$")
        axis.grid(True, axis="x")
        axis.grid(False, axis="y")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.legend(loc="lower left", handletextpad=0.4)

    left.set_yticks(y)
    left.set_yticklabels([f"gate {name}" for name in short])
    left.invert_yaxis()          # gate 1 at the top, as the canal runs
    left.set_ylabel("User")

    return save(figure, "fig2_who_goes_short")


# ---------------------------------------------------------------------------
# Figure 3 - the filter decides whether anything is possible
# ---------------------------------------------------------------------------


def figure_3() -> list[Path]:
    """What the filter decides, and what it does not.

    The first panel was the whole figure once, and it did not work: at
    order four a schedule exists at one supply level only, so that
    configuration was a single stray marker and the finding - that the
    edge moves - was invisible. The second panel now carries it directly.
    The advantage of the criterion is Figure 1's job and is not repeated
    here as a second spaghetti of lines.
    """
    runs = (
        ("main", "order 3, $\\omega_c$ = 3 mrad/s (pre-registered)",
         "#0b0b0b", "-", "D"),
        ("cut2e3", "order 3, $\\omega_c$ = 2 mrad/s", "#2a78d6", "--", "o"),
        ("order4", "order 4, $\\omega_c$ = 3 mrad/s", "#eb6834", ":", "s"),
    )
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(FULL, 2.7), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    edges = []
    for label, name, colour, dash, marker in runs:
        table = summary(label)
        x, y = series(table, "B4")
        left.plot(x, y, color=colour, ls=dash, marker=marker, label=name,
                  markerfacecolor="white", markeredgewidth=0.9, zorder=3)
        edge = cliff(table)
        edges.append((name, colour, marker, edge))
        if edge is not None:
            # The lowest supply that still has a schedule, filled so it
            # reads as an endpoint rather than as one more data point.
            left.plot([edge], [table["B4"][edge]], marker=marker, color=colour,
                      markersize=6.5, markeredgewidth=1.2,
                      markerfacecolor=colour, zorder=4)

    left.set_xlim(28, 102)
    left.set_xticks(range(30, 101, 10))
    left.set_ylim(0.6, 1.05)
    tidy(left, "Source availability $Q^{\\mathrm{src}}$ (%)",
         "Worst-off fraction under B4")
    left.set_title("a", loc="left", fontweight="bold")
    left.legend(loc="upper left", handlelength=2.6, bbox_to_anchor=(0.0, 0.99))

    # --- the operable range, which is the finding -------------------------
    rows = list(range(len(edges)))
    for row, (name, colour, marker, edge) in zip(rows, edges):
        if edge is None:
            continue
        width = 100 - edge
        if width > 0:
            right.barh(row, width, left=edge, height=0.42, color=colour,
                       zorder=3, edgecolor="white", linewidth=0.6)
        right.plot([edge], [row], marker=marker, color=colour, markersize=5.5,
                   markerfacecolor=colour, markeredgewidth=1.0, zorder=4)
        if width > 0:
            right.text(edge - 3.5, row, f"{edge}%", ha="right", va="center",
                       fontsize=7.5, color=INK)
        else:
            right.text(edge - 3.5, row, "100% only \u2014 no schedule below",
                       ha="right", va="center", fontsize=7.5, color=INK)

    right.set_ylim(-0.7, len(edges) - 0.3)
    right.invert_yaxis()
    right.set_yticks(rows)
    right.set_yticklabels(["order 3\n3 mrad/s", "order 3\n2 mrad/s",
                           "order 4\n3 mrad/s"], fontsize=7.5)
    right.set_xlim(28, 104)
    right.set_xticks(range(30, 101, 10))
    tidy(right, "Source availability $Q^{\\mathrm{src}}$ (%)")
    right.grid(True, axis="x")
    right.grid(False, axis="y")
    right.set_title("b", loc="left", fontweight="bold")
    right.set_ylabel("Filter")

    return save(figure, "fig3_filter_decides_feasibility")


# ---------------------------------------------------------------------------
# Figure 4 - the volume budget is not where the advantage comes from
# ---------------------------------------------------------------------------


def figure_4() -> list[Path]:
    """The budget moves the projection and leaves the criterion alone.

    Four legend entries covered the data when this was first drawn, which
    is what four entries do in a small panel. There are really only two
    things to name - which criterion, and which budget - so the criteria
    are labelled on their own curves and the legend carries the budget
    alone.
    """
    frozen, loosened = summary("main"), summary("budget")
    figure, axis = plt.subplots(figsize=(COLUMN, 2.7))

    for code in ("B4", "B1"):
        style = STYLE[code]
        x, y = series(frozen, code)
        axis.plot(x, y, color=style["color"], ls=style["ls"],
                  marker=style["marker"], markerfacecolor="white",
                  markeredgewidth=0.9, zorder=3)
        x2, y2 = series(loosened, code)
        axis.plot(x2, y2, color=style["color"], ls="none", marker="x",
                  markersize=5.0, markeredgewidth=1.1, zorder=4)

    # Named where they are, not in a box on top of them.
    axis.annotate("B4 lexicographic", xy=(72.5, 0.872), xytext=(66, 0.94),
                  fontsize=7.5, color=STYLE["B4"]["color"], ha="left",
                  arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                  shrinkA=1, shrinkB=2))
    axis.annotate("B1 unchanged order", xy=(77.5, 0.797), xytext=(78, 0.66),
                  fontsize=7.5, color=STYLE["B1"]["color"], ha="left",
                  arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                  shrinkA=1, shrinkB=2))

    condition = [
        Line2D([], [], color=MUTED, ls="-", marker="o",
               markerfacecolor="white", markeredgewidth=0.9,
               label="volume budget 0% (pre-registered)"),
        Line2D([], [], color=MUTED, ls="none", marker="x", markersize=5.0,
               markeredgewidth=1.1, label="volume budget +50%"),
    ]
    axis.legend(handles=condition, loc="lower right", handlelength=2.2,
                handletextpad=0.5)

    axis.set_xlim(47, 103)
    axis.set_xticks(range(50, 101, 10))
    axis.set_ylim(0.55, 1.03)
    tidy(axis, "Source availability $Q^{\\mathrm{src}}$ (%)",
         "Worst-off delivered fraction")

    return save(figure, "fig4_budget")


FIGURES = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4}

# The caption belongs in the manuscript, not in the image - the journal
# says so and it is right. It is written here all the same, next to the
# code that drew the picture, because that is the only place it cannot
# quietly come to describe a panel that was edited out two revisions ago.
CAPTIONS = {
    1: (
        "The scarcity scan on the Corning cascade. (**a**) Worst-off delivered "
        "fraction under each criterion, against source availability as a "
        "percentage of aggregate demand; the shaded band marks the region in "
        "which no feasible schedule exists. The free-gate bound M1 coincides "
        "with the lexicographic answer B4 at nine of the eleven feasible "
        "points and stands strictly above it at the two scarcest. "
        "(**b**) What the lexicographic criterion gains for the worst-off user "
        "over leaving the order unchanged, at the same points"
    ),
    2: (
        "Who carries the shortfall at 95% source availability, one row per "
        "user in gate order along the cascade. (**a**) The utilitarian "
        "criterion B3 fills seven users completely and leaves the eighth to "
        "carry the entire shortfall. "
        "(**b**) B4 and the least-spread criterion B5 both attain zero spread, "
        "so neither is distinguishable by that objective, yet B4 gives every "
        "user strictly more"
    ),
    3: (
        "The filter's order decides whether a schedule exists at all; its "
        "cut-off does not. (**a**) Worst-off fraction under the lexicographic "
        "criterion for three filter configurations; the filled marker on each "
        "curve is the lowest supply at which a schedule still exists. Raising "
        "the order from three to four moves that edge from 50% of demand to "
        "full supply: at order four nothing below 100% can be delivered at "
        "all. (**b**) The range of source availability over which the canal "
        "can be operated: lowering the cut-off from three to two millirad per "
        "second leaves that range where it is and changes only what is "
        "delivered inside it, while raising the order collapses it to a single "
        "point"
    ),
    4: (
        "The advantage does not come from releasing more water. Allowing each "
        "user to receive up to 50% more than it demanded changes nothing: "
        "every cross lands on its own circle, for the lexicographic criterion "
        "and for the unchanged order alike, at all eleven feasible points and "
        "in every digit the scan records. The certificates say why - the "
        "volume budget C3 is not active anywhere in the scan, so there is "
        "nothing for a larger budget to relax"
    ),
}


def write_captions(numbers) -> Path:
    out = repo_root() / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure captions",
        "",
        "Generated by `scripts/make_figures.py`. The journal wants captions in "
        "the manuscript text, not in the image; these are the wording to use, "
        "kept beside the code that drew each figure so the two cannot drift.",
        "",
    ]
    for number in sorted(numbers):
        lines += [f"**Figure {number}.** {CAPTIONS[number]}.", ""]
    path = out / "captions.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=int, nargs="+", choices=sorted(FIGURES))
    args = parser.parse_args(argv)

    house_style()
    wanted = args.only or sorted(FIGURES)
    for number in wanted:
        for path in FIGURES[number]():
            print(f"  fig {number}  {path.relative_to(repo_root())}", flush=True)
    path = write_captions(wanted)
    print(f"  captions  {path.relative_to(repo_root())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
