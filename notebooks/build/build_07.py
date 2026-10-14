"""Build and execute notebooks/07_operational_model.ipynb (figures fig07_* of the operational model).

Run from the repository root:  .venv/Scripts/python.exe notebooks/build/build_07.py
The notebook is constructed with nbformat, executed headless with nbclient (kernel cwd = notebooks/), and written with
its outputs. Figures land in docs/figures/fig07_*.png through notebooks/paper_style.py.

Scope: docs/operational_model.md (the operational deliverable of findings 10.35). No model is trained and no data
frame is read. The notebook reads the files `lab/op_ladder_report.py` writes:
- docs/operational_ladder.csv, every row's metrics per population, airport and month;
- docs/operational_ladder_calibration.csv, rule C's coverage per deliverable and airport.
Run `python lab/op_ladder_report.py` first.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "07_operational_model.ipynb"


def md(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


cells = []

cells.append(md(r"""
# 07 The operational model: accuracy by decision moment

Figures for `docs/operational_model.md`. The operational deliverable of findings 10.35 is one global CatBoost per
decision moment:
- 24 h and 3 h before the scheduled off-block;
- H1, 15 min before the last filed off-block;
- H0, at the off-block, a 2025-only comparison.

Every input exists at that moment (`docs/operational_features.md`).

**How the figures are read.**
- **Population and holdout:** the OP population (NM-matched departures with a target below one hour) of the Jan+Jul
  2025 holdout. Each model is fitted on the other ten months.
- **Deliverable rows:** the seed means (two seeds at 24 h and 3 h, three at H1). Their calibrated medians come from
  tables fitted on the 2025 month pairs only. At H1 the file ships rule M's median instead: a model fitted on the
  absolute error, clamped into the calibrated P10-P90 band (row `D-H1-MAE`).
- **Inputs:** `docs/operational_ladder.csv` and `docs/operational_ladder_calibration.csv`, both written by
  `lab/op_ladder_report.py`. No model is trained here.

Each figure's cell prints the numbers it draws.
"""))

cells.append(code(r"""
import sys; sys.path.insert(0, ".")
import paper_style as ps; ps.setup()
from prc import paths
import pandas as pd, numpy as np, matplotlib.pyplot as plt

T = pd.read_csv(paths.ROOT / "docs" / "operational_ladder.csv")
CAL = pd.read_csv(paths.ROOT / "docs" / "operational_ladder_calibration.csv")
MOMENTS = [("D-24H", "24 h ahead"), ("D-3H", "3 h ahead"), ("D-H1", "H1\n(15 min)"),
           ("op_h0_r4_nodoy", "H0\n(push)")]
OP = T[(T["population"] == "OP") & (T["slice"] == "all")].set_index("row")
present = [(k, n) for k, n in MOMENTS if k in OP.index]
print("rows found:", [k for k, _ in present])
"""))

cells.append(md(r"""
## Accuracy against the distance to the movement

The closer the prediction to the movement, the more it knows: the weather in force, the traffic of the last hour,
NM's readings of the flights that just departed, and finally the push itself. The left panel shows the OP MAE of the
mean output, of the calibrated median and, at H1, of the median the file ships (rule M). The right panel shows the
share within ±2 min, the A-CDM manual's
(tentative) accuracy target for taxi times. The stand × runway table (B1), the airports' variable-taxi-time practice,
is the reference line.
"""))

cells.append(code(r"""
rows = []
for key, name in present:
    r = {"moment": name.replace("\n", " "), "mae_mean": OP.loc[key, "mae"], "w120_mean": OP.loc[key, "within_120"]}
    med = f"{key}-MED"
    if med in OP.index:
        r.update(mae_med=OP.loc[med, "mae"], w120_med=OP.loc[med, "within_120"])
    ship = f"{key}-MAE"
    if ship in OP.index:
        r.update(mae_ship=OP.loc[ship, "mae"], w120_ship=OP.loc[ship, "within_120"])
    rows.append(r)
H = pd.DataFrame(rows)
print(H.round(3).to_string(index=False))
b1 = OP.loc["B1"] if "B1" in OP.index else None
if b1 is not None:
    print(f"B1 stand x runway table: MAE {b1['mae']:.1f} s, within 2 min {100 * b1['within_120']:.1f} %")

fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.9))
x = np.arange(len(H))
for ax, (m, mm, ms, ylab, scale) in zip(axes, (("mae_mean", "mae_med", "mae_ship", "OP MAE (s)", 1.0),
                                               ("w120_mean", "w120_med", "w120_ship", "within ±2 min (%)", 100.0))):
    ax.plot(x, scale * H[m], "o-", color="#4D4D4D", label="mean output")
    if mm in H:
        ax.plot(x, scale * H[mm], "s--", color="#4D4D4D", mfc="white", label="calibrated median (rule C)")
    if ms in H:
        ax.plot(x, scale * H[ms], "D", color="#000000", ms=5, label="median shipped at H1 (MAE loss, rule M)")
    if b1 is not None:
        ax.axhline(scale * (b1["mae"] if m == "mae_mean" else b1["within_120"]), color="#969696", lw=1, ls=":",
                   label="stand × runway table")
    ax.set_xticks(x, [n for _, n in present], fontsize=7)
    ax.set_ylabel(ylab)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.13), frameon=False)
fig.tight_layout()
ps.save(fig, "fig07_01_moments")
plt.show()
"""))

cells.append(md(r"""
## The same per airport

Airports differ more than the moments do: their layout, runway system and traffic set the level. The figure shows the
OP MAE of each deliverable's mean output per airport.
"""))

cells.append(code(r"""
A = T[(T["population"] == "OP") & (T["slice"] == "apt")]
piv = A[A["row"].isin([k for k, _ in present])].pivot_table(index="level", columns="row", values="mae", aggfunc="first")
piv = piv[[k for k, _ in present if k in piv.columns]].reindex([a for a in ps.AIRPORTS if a in piv.index])
print(piv.round(1).to_string())
fig, ax = plt.subplots(figsize=(ps.WIDTH, 3.0))
w = 0.8 / max(len(piv.columns), 1)
greys = ["#D9D9D9", "#A6A6A6", "#636363", "#252525"]
for i, col in enumerate(piv.columns):
    ax.bar(np.arange(len(piv)) + (i - (len(piv.columns) - 1) / 2) * w, piv[col], width=w, color=greys[i % 4],
           label=dict(present)[col].replace("\n", " "))
ax.set_xticks(np.arange(len(piv)), [ps.label(a, "code") for a in piv.index])
ax.set_ylabel("OP MAE (s)")
ax.set_ylim(0, 1.3 * float(piv.max().max()))
ax.legend(ncol=4, loc="upper center", fontsize=7)
fig.tight_layout()
ps.save(fig, "fig07_02_airports")
plt.show()
"""))

cells.append(md(r"""
## How honest the 80 % band is

The P10-P90 band comes from the residuals of the 2025 month pairs, not from the January and July holdout it is
checked on. The figure shows the share of OP departures inside the band per airport and deliverable. 80 % is the
nominal value, and rule C of findings 10.35 accepts 75-85 % overall.
"""))

cells.append(code(r"""
if CAL.empty:
    print("no calibration file rows: rule C was not evaluated")
else:
    piv = CAL[CAL["apt"] != "all"].pivot_table(index="apt", columns="deliverable", values="coverage", aggfunc="first")
    order = [k for k, _ in present if k in piv.columns]
    piv = piv[order].reindex([a for a in ps.AIRPORTS if a in piv.index])
    print((100 * piv).round(1).to_string())
    print("overall:", {r.deliverable: round(100 * r.coverage, 1) for r in CAL[CAL["apt"] == "all"].itertuples()})
    fig, ax = plt.subplots(figsize=(ps.WIDTH, 2.8))
    w = 0.8 / max(len(order), 1)
    greys = ["#D9D9D9", "#A6A6A6", "#636363"]
    for i, col in enumerate(order):
        ax.bar(np.arange(len(piv)) + (i - (len(order) - 1) / 2) * w, 100 * piv[col], width=w, color=greys[i % 3],
               label=dict(present)[col].replace("\n", " "))
    ax.axhspan(75, 85, color="#EBEBEB", zorder=0)
    ax.axhline(80, color="#636363", lw=0.8, ls=":")
    ax.set_ylim(50, 100)
    ax.set_xticks(np.arange(len(piv)), [ps.label(a, "code") for a in piv.index])
    ax.set_ylabel("inside P10-P90 (%)")
    ax.legend(ncol=3, loc="upper left", fontsize=7)
    fig.tight_layout()
    ps.save(fig, "fig07_03_band_coverage")
"""))

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.cells = cells

if __name__ == "__main__":
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    nbformat.write(nb, OUT)
    print("wrote", OUT)
