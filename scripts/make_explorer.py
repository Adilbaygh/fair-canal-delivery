#!/usr/bin/env python3
"""Build the interactive appendix the paper points at.

One HTML file, everything inside it. It has to open from a downloaded copy
of the archive with no network at all - a reader who fetches the
supplementary material from the repository or from Zenodo, opens it from a
folder on a laptop with the aeroplane wifi off, and expects it to work. So
there is no CDN, no font download, no fetch: the result tables are
embedded as JSON at build time and the page draws its own charts in SVG.

What it shows, and what it refuses to show
------------------------------------------
The reader picks a configuration and a supply level and sees what every
user actually received under each criterion, the summary figures beside
it, and what the certificate said. The point of the appendix is that the
claims in the paper can be checked one point at a time.

It refuses to show a number that was never computed. A supply level with
no feasible schedule says so; a variant whose solver never reached a
verdict says that, and says it differently, because those two are not the
same thing and conflating them is the mistake this project spent a week
correcting.

    python scripts/make_explorer.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.provenance import repo_root, write_text  # noqa: E402

SCANS = {
    "main": "order 3, cut-off 3 mrad/s, budget 0% (pre-registered)",
    "cut2e3": "order 3, cut-off 2 mrad/s",
    "order4": "order 4, cut-off 3 mrad/s",
    "budget": "order 3, cut-off 3 mrad/s, volume budget +50%",
    # The assumptions the eleven-family programme rests on, moved one at a
    # time away from their frozen values (kappa 1.5, head 0.10 m, band
    # 0.15 m). The head cannot go below 0.076 m: there the narrowest gate
    # stops passing the flow its own reach already carries.
    "kappa125": "outlet headroom 1.25x nominal",
    "kappa200": "outlet headroom 2.00x nominal",
    "head008": "gate rating head 0.08 m",
    "head020": "gate rating head 0.20 m",
    "band010": "storage reconciliation band 0.10 m",
    "band025": "storage reconciliation band 0.25 m",
}
CODES = ("B1", "B2", "B3", "B4", "B5", "M1")
NAMES = {
    "B1": "unchanged order",
    "B2": "one block early",
    "B3": "utilitarian",
    "B4": "lexicographic",
    "B5": "least spread",
    "M1": "free-gate bound",
}
COLOURS = {
    "B1": "#2a78d6", "B2": "#eb6834", "B3": "#1baf7a",
    "B4": "#0b0b0b", "B5": "#e87ba4", "M1": "#52514e",
}


def read(label: str, kind: str) -> list[dict]:
    path = repo_root() / "results" / "tables" / f"{label}_{kind}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe(value):
    return None if value in ("", None) else float(value)


def collect() -> dict:
    data = {}
    for label in SCANS:
        summary: dict = defaultdict(dict)
        for row in read(label, "summary"):
            summary[row["percent"]][row["code"]] = {
                "status": row["status"],
                "worst": maybe(row["worst"]),
                "total": maybe(row["total"]),
                "spread": maybe(row["spread"]),
                "seconds": maybe(row["seconds"]),
            }
        ratios: dict = defaultdict(lambda: defaultdict(dict))
        users: list[str] = []
        for row in read(label, "ratios"):
            ratios[row["percent"]][row["code"]][row["user"]] = float(row["ratio"])
            if row["user"] not in users:
                users.append(row["user"])
        certs = {}
        for row in read(label, "certificates"):
            certs[row["percent"]] = {
                "fulfilled": row["fulfilled"] == "1",
                "short": int(row["short"] or 0),
                "worst_user": row["worst_user"],
                "worst_ratio": maybe(row["worst_ratio"]),
                "impossible": row["impossible"],
                "binding": row["binding"],
                "digest": row["digest"][:16],
            }
        if not summary:
            continue
        data[label] = {
            "note": SCANS[label],
            "levels": sorted(summary, key=lambda p: -int(p)),
            "summary": {k: dict(v) for k, v in summary.items()},
            "ratios": {k: {c: dict(u) for c, u in v.items()} for k, v in ratios.items()},
            "certificates": certs,
            "users": sorted(users),
        }
    return data


def environment() -> dict:
    path = repo_root() / "results" / "environment.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "commit": payload.get("git_commit", "")[:12],
        "python": payload.get("python", {}).get("version", ""),
        "scipy": payload.get("packages", {}).get("scipy", ""),
        "numpy": payload.get("packages", {}).get("numpy", ""),
        "system": payload.get("platform", {}).get("system", ""),
        "recorded": payload.get("recorded_utc", ""),
    }


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive appendix - lexicographic water delivery</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --soft: #52514e;
  --muted: #898781; --grid: #e1e0d9; --line: #c3c2b7; --nothing: #f0efec;
  --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #212120; --ink: #ffffff; --soft: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --line: #383835; --nothing: #262624;
    --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Arial, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 1.45rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: .95rem; margin: 0 0 .7rem; letter-spacing: .02em;
     text-transform: uppercase; color: var(--soft); font-weight: 600; }
p.lede { margin: 0 0 1.6rem; color: var(--soft); max-width: 68ch; }
.panel { background: var(--panel); border: 1px solid var(--grid);
         border-radius: 10px; padding: 18px; margin-bottom: 18px; }
.controls { display: flex; flex-wrap: wrap; gap: 18px 28px; align-items: flex-end; }
label { display: block; font-size: .78rem; text-transform: uppercase;
        letter-spacing: .04em; color: var(--muted); margin-bottom: .35rem; }
select, input[type=range] { font: inherit; }
select { padding: .35rem .5rem; border: 1px solid var(--line);
         border-radius: 6px; background: var(--panel); color: var(--ink); }
input[type=range] { width: 260px; accent-color: var(--accent); }
.reading { font-variant-numeric: tabular-nums; font-size: 1.5rem;
           font-weight: 600; line-height: 1; }
.reading small { display: block; font-size: .72rem; font-weight: 400;
                 text-transform: uppercase; letter-spacing: .04em;
                 color: var(--muted); margin-top: .35rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .38rem .5rem; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
thead th { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em;
           color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--line); }
tbody tr:last-child td { border-bottom: none; }
.swatch { display: inline-block; width: .62rem; height: .62rem; border-radius: 2px;
          margin-right: .45rem; vertical-align: baseline; }
.absent { color: var(--muted); font-style: italic; }
.flag { display: inline-block; padding: .12rem .45rem; border-radius: 999px;
        font-size: .72rem; letter-spacing: .02em; border: 1px solid var(--line);
        color: var(--soft); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.foot { color: var(--muted); font-size: .8rem; line-height: 1.6; }
.foot code { font-size: .95em; }
@media (max-width: 760px) { .grid2 { grid-template-columns: 1fr; }
  input[type=range] { width: 100%; } }
</style>
</head>
<body>
<div class="wrap">
<h1>Lexicographic water delivery - interactive appendix</h1>
<p class="lede">Every number on this page is read from the study's result
archive. Pick a configuration and a supply level to see what each user
actually received under each criterion. Where a schedule does not exist, or
where the solver never reached a verdict, the page says so instead of
showing a number.</p>

<div class="panel controls">
  <div>
    <label for="scan">Configuration</label>
    <select id="scan"></select>
  </div>
  <div>
    <label for="level">Source availability <span id="levelText"></span></label>
    <input type="range" id="level" min="0" max="0" step="1">
  </div>
  <div>
    <div class="reading" id="worstReading">-</div>
    <small style="display:block;font-size:.72rem;color:var(--muted);
      text-transform:uppercase;letter-spacing:.04em;margin-top:.35rem">
      worst-off fraction, lexicographic</small>
  </div>
</div>

<div class="panel">
  <h2>What each user received</h2>
  <div id="chart"></div>
</div>

<div class="grid2">
  <div class="panel">
    <h2>Criteria at this point</h2>
    <table id="summary"></table>
  </div>
  <div class="panel">
    <h2>Certificate</h2>
    <div id="certificate"></div>
  </div>
</div>

<div class="panel foot" id="provenance"></div>
</div>

<script>
const DATA = __DATA__;
/* Explicit, because the payload is written with sorted keys so that two
   builds of the same archive are byte-identical - and sorted keys are not
   the order a reader should meet these in. */
const ORDER = __ORDER__;
const ENV = __ENV__;
const CODES = __CODES__;
const NAMES = __NAMES__;
const COLOURS = __COLOURS__;

const scanSel = document.getElementById("scan");
const levelInput = document.getElementById("level");
const levelText = document.getElementById("levelText");

ORDER.forEach(key => {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = DATA[key].note;
  scanSel.appendChild(option);
});

function shortUser(name) {
  return name.replace("corning-", "gate ").replace("-user", "");
}
function fmt(value, places) {
  return value === null || value === undefined ? null : value.toFixed(places || 4);
}

/* One horizontal bar per user per criterion. SVG by hand: a chart library
   would be a download, and this file has to open with the wifi off. */
function drawChart(scan, level) {
  const box = document.getElementById("chart");
  const ratios = scan.ratios[level] || {};
  const shown = CODES.filter(c => ratios[c] && Object.keys(ratios[c]).length);
  if (!shown.length) {
    box.innerHTML = '<p class="absent">No schedule exists at this supply ' +
      'level, so there is nothing any user received.</p>';
    return;
  }
  const users = scan.users;
  // One arithmetic for the layout and the height, so the last user cannot
  // fall off the bottom of the box - which is what happened when the two
  // were computed separately.
  const barH = 6, gap = 2, groupGap = 12;
  const padL = 74, padR = 58, padT = 26, padB = 10;
  const groupH = shown.length * (barH + gap) + groupGap;
  const height = padT + users.length * groupH + padB;
  const width = 760;
  const x = v => padL + v * (width - padL - padR);

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="100%" ` +
    `height="${height}" role="img" aria-label="Delivered fraction per user">`;
  for (let t = 0; t <= 1.0001; t += 0.25) {
    svg += `<line x1="${x(t)}" y1="${padT - 8}" x2="${x(t)}" y2="${height - 6}" ` +
      `stroke="var(--grid)" stroke-width="1"/>` +
      `<text x="${x(t)}" y="${padT - 13}" font-size="10" fill="var(--muted)" ` +
      `text-anchor="middle">${t.toFixed(2)}</text>`;
  }
  let y = padT;
  users.forEach(user => {
    svg += `<text x="${padL - 8}" y="${y + (shown.length * (barH + gap)) / 2 + 3}" ` +
      `font-size="11" fill="var(--soft)" text-anchor="end">${shortUser(user)}</text>`;
    shown.forEach(code => {
      const value = ratios[code][user];
      if (value === undefined) { y += barH + gap; return; }
      svg += `<rect x="${padL}" y="${y}" width="${Math.max(0, x(value) - padL)}" ` +
        `height="${barH}" fill="${COLOURS[code]}" rx="2"><title>${code} ` +
        `${NAMES[code]} - ${shortUser(user)}: ${value.toFixed(4)}</title></rect>`;
      y += barH + gap;
    });
    y += groupGap;
  });
  svg += "</svg>";

  const key = shown.map(c =>
    `<span style="margin-right:14px;white-space:nowrap"><span class="swatch" ` +
    `style="background:${COLOURS[c]}"></span>${c} ${NAMES[c]}</span>`).join("");
  box.innerHTML = svg + `<div style="margin-top:8px;font-size:.8rem;` +
    `color:var(--soft)">${key}</div>`;
}

function drawSummary(scan, level) {
  const rows = scan.summary[level] || {};
  let html = "<thead><tr><th>Criterion</th><th>Worst-off</th><th>Total</th>" +
    "<th>Spread</th><th>Status</th></tr></thead><tbody>";
  CODES.forEach(code => {
    const row = rows[code];
    if (!row) return;
    const solved = row.worst !== null;
    const status = row.status === "undecided"
      ? '<span class="flag">no verdict</span>'
      : (solved ? row.status : '<span class="flag">' + row.status + "</span>");
    html += `<tr><td><span class="swatch" style="background:${COLOURS[code]}">` +
      `</span>${code} ${NAMES[code]}</td>` +
      `<td>${solved ? fmt(row.worst) : '<span class="absent">-</span>'}</td>` +
      `<td>${row.total !== null ? fmt(row.total, 3) : '<span class="absent">-</span>'}</td>` +
      `<td>${row.spread !== null ? fmt(row.spread) : '<span class="absent">-</span>'}</td>` +
      `<td>${status}</td></tr>`;
  });
  document.getElementById("summary").innerHTML = html + "</tbody>";
}

function drawCertificate(scan, level) {
  const box = document.getElementById("certificate");
  const cert = scan.certificates[level];
  if (!cert) {
    box.innerHTML = '<p class="absent">No certificate was written at this ' +
      'point: the solver did not reach a verdict on the programmes the ' +
      'certificate is built from. That is not the same as an empty ' +
      'certificate.</p>';
    return;
  }
  const rows = [
    ["Every order filled", cert.fulfilled ? "yes" : "no"],
    ["Users short", String(cert.short)],
    ["Worst-off user", cert.worst_user
      ? shortUser(cert.worst_user) + " at " + fmt(cert.worst_ratio) : "-"],
    ["Structurally impossible", cert.impossible || "none"],
    ["Binding families", cert.binding || "none"],
    ["Digest", "<code>" + cert.digest + "</code>"],
  ];
  box.innerHTML = "<table><tbody>" + rows.map(
    ([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") + "</tbody></table>";
}

function render() {
  const scan = DATA[scanSel.value];
  const level = scan.levels[Number(levelInput.value)];
  levelText.textContent = "- " + level + "% of aggregate demand";
  const b4 = (scan.summary[level] || {}).B4;
  document.getElementById("worstReading").textContent =
    b4 && b4.worst !== null ? b4.worst.toFixed(4) : "no schedule";
  drawChart(scan, level);
  drawSummary(scan, level);
  drawCertificate(scan, level);
}

function reset() {
  const scan = DATA[scanSel.value];
  levelInput.max = String(scan.levels.length - 1);
  levelInput.value = "0";
  render();
}

scanSel.addEventListener("change", reset);
levelInput.addEventListener("input", render);

document.getElementById("provenance").innerHTML =
  "Built from the result archive at commit <code>" + (ENV.commit || "unknown") +
  "</code>, recorded " + (ENV.recorded || "unknown") + " on " + (ENV.system || "?") +
  " with Python " + (ENV.python || "?") + ", SciPy " + (ENV.scipy || "?") +
  ", NumPy " + (ENV.numpy || "?") + ". Regenerate with " +
  "<code>python scripts/make_explorer.py</code>. This page contains no " +
  "network requests and works from a downloaded copy.";

reset();
</script>
</body>
</html>
"""


def main() -> int:
    data = collect()
    if not data:
        raise SystemExit("no result tables found. Run the scans first.")
    page = (
        PAGE.replace("__ORDER__", json.dumps([k for k in SCANS if k in data]))
        .replace("__DATA__", json.dumps(data, sort_keys=True))
        .replace("__ENV__", json.dumps(environment(), sort_keys=True))
        .replace("__CODES__", json.dumps(list(CODES)))
        .replace("__NAMES__", json.dumps(NAMES, sort_keys=True))
        .replace("__COLOURS__", json.dumps(COLOURS, sort_keys=True))
    )
    path = write_text(repo_root() / "results" / "explorer.html", page)
    size = path.stat().st_size / 1024
    print(f"  {path.relative_to(repo_root())}  ({size:.0f} kB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
