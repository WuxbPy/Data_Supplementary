# -*- coding: utf-8 -*-
"""Build the four submission figures for the Data manuscript.

Inputs (all inside the material package):
  ../../Software/code/data/scorecard.npz        cross-section coords/types (4,785 institutions)
  ../../results/scorecard_b3_spectrum.csv       bandwidth scan
  ../../results/scorecard_b2_clq_timeseries.csv thirty-wave trajectory
  ../../results/scorecard_b2_event_study.csv    event study
  ../../results/scorecard_b2_summary.json       DiD / placebo estimates
  data/us-states.json                           state outlines (public GeoJSON, for orientation)

Outputs: ../fig1.png ... ../fig4.png at 600 dpi, plus vector PDFs in ../pdf/.
Style: 7 pt minimum type, no top/right spines, one colour palette across figures.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.dirname(HERE)                 # .../Data/figures
PKG = os.path.dirname(FIG)                  # .../Data
NPZ = os.path.join(PKG, "code", "data", "scorecard.npz")
RES = os.path.join(PKG, "results")

TYPE_COL = {"Public": "#1F4E79", "Private nonprofit": "#B03A2E",
            "Private for-profit": "#E08214"}
GREY = "#8C949C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    # 720 dpi on the natural canvas keeps every figure above 600 dpi at the
    # insertion widths used in the manuscript (5.5 in for Figure 1, 5.9 in for
    # Figures 2-4), including where bbox_inches='tight' trims the canvas.
    out = os.path.join(FIG, name + ".png")
    fig.savefig(out, dpi=720)
    os.makedirs(os.path.join(FIG, "pdf"), exist_ok=True)
    fig.savefig(os.path.join(FIG, "pdf", name + ".pdf"))
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- Albers equal-area
def albers(lon, lat, lon0=-96.0, lat1=29.5, lat2=45.5, lat0=37.5):
    """Spherical Albers equal-area conic; x, y in units of one Earth radius."""
    lam = np.radians(lon - lon0)
    phi = np.radians(lat)
    p1, p2, p0 = np.radians(lat1), np.radians(lat2), np.radians(lat0)
    n = (np.sin(p1) + np.sin(p2)) / 2.0
    cc = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho = np.sqrt(cc - 2 * n * np.sin(phi)) / n
    rho0 = np.sqrt(cc - 2 * n * np.sin(p0)) / n
    return rho * np.sin(n * lam), rho0 - rho * np.cos(n * lam)


def fig1():
    z = np.load(NPZ, allow_pickle=True)
    coords, types = z["coords"], z["types"]
    names = ["Public", "Private nonprofit", "Private for-profit"]

    states = json.load(open(os.path.join(HERE, "data", "us-states.json")))
    skip = {"Alaska", "Hawaii", "Puerto Rico"}

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for f in states["features"]:
        if f["properties"]["name"] in skip:
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            x, y = albers(ring[:, 0], ring[:, 1])
            ax.plot(x, y, color="#CBD0D6", lw=0.35, zorder=1)

    for i, nm in enumerate(names):
        m = types == i
        x, y = albers(coords[m, 0], coords[m, 1])
        ax.scatter(x, y, s=1.8, c=TYPE_COL[nm], alpha=0.70, linewidths=0, zorder=2 + i,
                   label="%s (%s)" % (nm, format(int(m.sum()), ",")))

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    km_per_unit = 6371.0
    L = 500.0 / km_per_unit
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    xb, yb = x0 + 0.70 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.plot([xb, xb + L], [yb, yb], color="k", lw=1.2, solid_capstyle="butt", zorder=6)
    ax.text(xb + L / 2, yb + 0.012 * (y1 - y0), "500 km", ha="center", va="bottom", fontsize=6.5)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=0.93, borderpad=0.35,
                    handletextpad=0.4, markerscale=4.5, fontsize=6.3,
                    bbox_to_anchor=(0.005, 0.02))
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor("#B8BEC5")

    save(fig, "fig1")


def fig2():
    sp = pd.read_csv(os.path.join(RES, "scorecard_b3_spectrum.csv"))
    pair = {"营利→营利": "Private for-profit", "非营利→非营利": "Private nonprofit",
            "公立→公立": "Public"}
    col = {"营利→营利": TYPE_COL["Private for-profit"],
           "非营利→非营利": TYPE_COL["Private nonprofit"],
           "公立→公立": TYPE_COL["Public"]}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax = axes[0]
    for p, lab in pair.items():
        d = sp[sp.pair == p].sort_values("b_km")
        ax.plot(d.b_km, d.CLQ, "-o", ms=2.4, lw=1.1, color=col[p], label=lab)
        ax.plot(d.b_km, d.CLQ1, "--", lw=0.9, color=col[p], alpha=0.8)
    ax.axvline(150, color="k", ls=":", lw=0.8)
    ax.text(162, 0.60, "b = 150 km", fontsize=6.4, rotation=90, va="bottom")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(4, 700)
    ax.set_xlabel("Bandwidth b (km)")
    ax.set_ylabel("Co-location value C, first-order C$^{(1)}$")
    ax.set_title("(a)  C (solid) and C$^{(1)}$ (dashed) separate only where $\\kappa$ is small", loc="left", fontsize=7.6)
    tidy(ax)

    ax = axes[1]
    for p, lab in pair.items():
        d = sp[sp.pair == p].sort_values("b_km")
        ax.plot(d.b_km, d.CLQ2, "-o", ms=2.4, lw=1.1, color=col[p], label=lab)
    ax.axhline(1, color="k", lw=0.6)
    # Admissibility bands: light fill for colour, hatch overlay for greyscale and
    # colour-vision deficiency (the two bands must remain distinguishable in print).
    for x0, x1, col, ht in ((5, 100, "#2E7D32", "//"), (200, 500, "#B03A2E", "\\\\")):
        ax.axvspan(x0, x1, facecolor=col, alpha=0.07, lw=0, zorder=0)
        ax.axvspan(x0, x1, facecolor="none", hatch=ht, edgecolor=col,
                   lw=0, alpha=0.18, zorder=0)
    ax.text(9, 1.115, "$\\rho \\leq 0.01$", fontsize=7, color="#2E7D32")
    ax.text(360, 1.115, "$\\kappa \\geq 30$", fontsize=7, color="#B03A2E")
    ax.annotate("", xy=(100, 1.10), xytext=(200, 1.10),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="k"))
    ax.text(141, 1.107, "empty", fontsize=6.3, ha="center")
    ax.set_xscale("log")
    ax.set_xlim(4, 700)
    ax.set_ylim(0.66, 1.16)
    ax.set_xlabel("Bandwidth b (km)")
    ax.set_ylabel("Second-order term C$^{(2)}$")
    ax.set_title("(b)  No b satisfies both admissibility criteria", loc="left")
    tidy(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig2")


def fig3():
    ts = pd.read_csv(os.path.join(RES, "scorecard_b2_clq_timeseries.csv"))
    names = ["Public", "Private nonprofit", "Private for-profit"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    for nm in names:
        d = ts[(ts.source == nm) & (ts.target == nm)].sort_values("year")
        ax.plot(d.year, d.n_A, "-o", ms=2.2, lw=1.2, color=TYPE_COL[nm], label=nm)
    ytop = ax.get_ylim()[1]
    for y, lab in [(2008, "2008"), (2012, "2012"), (2020, "2020")]:
        ax.axvline(y, color="k", ls=":", lw=0.7)
        ax.text(y + 0.6, ytop * 0.995, lab, fontsize=6.2, rotation=90, va="top")
    ax.set_xlabel("Academic year")
    ax.set_ylabel("Institutions in the panel")
    ax.set_title("(a)  Counts by control", loc="left")
    tidy(ax)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    order = [("Private for-profit", "Private for-profit", "-"),
             ("Private nonprofit", "Private nonprofit", "-"),
             ("Public", "Public", "-"),
             ("Private for-profit", "Private nonprofit", "--"),
             ("Private for-profit", "Public", ":"),
             ("Private nonprofit", "Public", "--")]
    ab = {"Private for-profit": "FP", "Private nonprofit": "NP", "Public": "PU"}
    for s, t, ls in order:
        d = ts[(ts.source == s) & (ts.target == t)].sort_values("year")
        ax.plot(d.year, d.CLQ2, ls, color=TYPE_COL[s], lw=1.1,
                label="%s→%s" % (ab[s], ab[t]))
    for y in (2008, 2012, 2020):
        ax.axvline(y, color="k", ls=":", lw=0.7)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xlabel("Academic year")
    ax.set_ylabel("Second-order term C$^{(2)}$")
    ax.set_title("(b)  Six ordered pairs, thirty waves", loc="left")
    tidy(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.9, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig3")


def fig4():
    es = pd.read_csv(os.path.join(RES, "scorecard_b2_event_study.csv"))
    d = es[es.outcome == "log1p(n_forprofit)"].sort_values("year")
    summ = json.load(open(os.path.join(RES, "scorecard_b2_summary.json")))
    did = {r["outcome"] + "|" + r["design"]: r for r in summ["did"]}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(2011, color="k", ls=":", lw=0.7)
    ax.errorbar(d.year, d.coef, yerr=1.96 * d.se, fmt="o", ms=2.6, lw=0.8,
                color=TYPE_COL["Private for-profit"], capsize=1.2, elinewidth=0.7)
    ax.set_xlabel("Year (reference: 2011)")
    ax.set_ylabel("Event-study coefficient")
    ax.set_title("(a)  All 15 pre-period coefficients differ from zero", loc="left", fontsize=7.4)
    tidy(ax)
    ax.text(2011.9, -0.235, "2012 rule", fontsize=6.4, rotation=90, va="bottom", ha="left")

    ax = axes[1]
    items = [("True event 2012", did["log1p(n_forprofit)|TWFE"]),
             ("Placebo 1999", did["placebo_log1p(n_fp)@1999|placebo"]),
             ("Placebo 2002", did["placebo_log1p(n_fp)@2002|placebo"]),
             ("Placebo 2005", did["placebo_log1p(n_fp)@2005|placebo"]),
             ("Cell trends", did["log1p(n_forprofit)|TWFE_cell_trends"])]
    ypos = np.arange(len(items))[::-1]
    for y, (lab, r) in zip(ypos, items):
        c = TYPE_COL["Public"] if lab.startswith("True") else (
            GREY if "Cell" in lab else TYPE_COL["Private nonprofit"])
        ax.errorbar(r["coef"], y, xerr=1.96 * r["se"], fmt="o", ms=3.4, color=c,
                    capsize=1.8, elinewidth=0.9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in items], fontsize=6.6)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("Interaction coefficient (95% CI)")
    ax.set_title("(b)  Placebos exceed the estimate", loc="left", fontsize=7.4)
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig4")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4}[w]()
