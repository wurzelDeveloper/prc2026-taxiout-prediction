"""Shared style for the figure notebooks (04, 07, 08): airport order, labels and colours, semantic colours, the
coverage vocabulary, and figure export.

Every figure is drawn at a text width of 16 cm (WIDTH = 6.3 in), so the font sizes set here are the
printed sizes: nothing is scaled down when the figure is placed at 16 cm. No figure uses a font below 7 pt.

Every figure is saved to docs/figures/<name>.png (200 dpi) and shown inline. Notebooks import this module from the
notebooks/ folder and import data access from `prc`; nothing in `prc` imports from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 6.3          # inches; a 16 cm text column. Every figure uses figsize=(WIDTH, height).
MIN_FONT = 7         # smallest font size any figure uses (pt, at print size)

# Airports in region order, with a fixed colour each (tab10). Airport colours mean
# "this airport" and nothing else; every chart also labels its airports, so colour is never the only key.
AIRPORTS = ["EHAM", "LFPG", "EGLL", "EDDF", "EDDM", "LSZH", "LTFM", "LIRF", "LEMD", "LEBL"]
AIRPORT_NAMES = {"EHAM": "Amsterdam", "LFPG": "Paris", "EGLL": "Heathrow", "EDDF": "Frankfurt",
                 "EDDM": "Munich", "LSZH": "Zürich", "LTFM": "Istanbul", "LIRF": "Rome",
                 "LEMD": "Madrid", "LEBL": "Barcelona"}
_tab10 = plt.get_cmap("tab10").colors
AIRPORT_COLORS = {a: _tab10[i] for i, a in enumerate(AIRPORTS)}


def label(a: str, style: str = "full") -> str:
    """One airport label format for every figure: 'Rome (LIRF)' (style='full') or 'LIRF' (style='code', used only in
    dense small multiples; the caption gives the key). Anything that is not an airport code is returned unchanged."""
    if a not in AIRPORT_NAMES:
        return a
    return a if style == "code" else f"{AIRPORT_NAMES[a]} ({a})"


CODE_KEY = ", ".join(f"{a} {AIRPORT_NAMES[a]}" for a in AIRPORTS)   # caption key for code-labelled figures

# Year colours: reserved for 2025 (training year, holdout) and 2026 (scored year) and used for nothing else.
C_2025, C_2026 = "#4C72B0", "#DD8452"
# Before / after a step (base model vs final submission).
C_BEFORE, C_AFTER = "#8C8C8C", "#2E7D32"
# Record kinds: drawn in black, grey and by style (hatch, hollow marker), never in an airport hue.
C_STALE = "#252525"          # stale record: filled black
C_WRONG = "#252525"          # wrong-date record: hollow / hatched black (WRONG_HATCH, WRONG_MARKER)
WRONG_HATCH, WRONG_MARKER = "////", "D"
C_OTHER = "#BDBDBD"          # other records: neutral grey
# The two second-stage formulations (residual and direct), outside the tab10 hues.
C_RESID, C_DIRECT = "#00897B", "#B8860B"
# Event days: white face, black edge, on top of any airport colour.
C_EVENT_FACE, C_EVENT_EDGE = "white", "black"
# A step that made things worse (sign colour, used only for signed changes).
C_WORSE = "#B2182B"
# Availability tiers (docs/feature_table.md), most restrictive first: a grey ramp (dark = further from operational).
TIER_ORDER = ["M", "C", "A", "P", "T", "O"]
TIER_NAMES = {"M": "M  own movement (ADS-B)", "C": "C  challenge artefact", "A": "A  after off-block",
              "P": "P  published later", "T": "T  takeoff-anchored", "O": "O  operational"}
TIER_COLORS = {"M": "#111111", "C": "#3D3D3D", "A": "#6B6B6B", "P": "#969696", "T": "#C2C2C2", "O": "#EBEBEB"}
TIER_TEXT = {"M": "white", "C": "white", "A": "white", "P": "black", "T": "black", "O": "black"}
# Kinds of 2025 event day: marker shapes, not colours.
KIND_MARKERS = {"snow": "o", "freezing fog / LVP": "s", "dry frost": "D", "thunderstorm": "^", "wind": ">",
                "runway closure": "X"}

# ADS-B coverage vocabulary. One word per definition, used in every axis label and caption.
COV_LINKED = "linked"            # adsb_linked: an ADS-B track was linked to the departure (56.6 % / 70.4 %)
COV_CORRECTED = "corrected"      # linked with a first ground sighting: the residual stage's gate (56.5 % / 70.4 %)
COV_BOTH = "corrected by both stages"   # corrected and stale probability <= 0.05 (55.0 % / 69.2 %)
COVERAGE_TERMS = {
    COV_LINKED: "an ADS-B track is linked to the departure",
    COV_CORRECTED: "linked and with a first ground sighting (the residual stage's gate, hedged rows included)",
    COV_BOTH: "corrected and with a stale probability of at most 0.05 (rows the direct stage also corrects)",
}


def setup() -> None:
    """Matplotlib defaults for all figures (sizes are print sizes at WIDTH)."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.titlesize": 9, "legend.title_fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "figure.figsize": (WIDTH, 3.6), "hatch.linewidth": 0.6,
    })


def save(fig, name: str) -> Path:
    """Save to docs/figures/<name>.png and return the path; the notebook still shows the figure inline."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def drop(name: str) -> None:
    """Remove a figure that a notebook no longer makes (e.g. a duplicate dropped in review)."""
    (FIG_DIR / f"{name}.png").unlink(missing_ok=True)


def minutes(seconds):
    """Seconds to minutes, for axis values (the documents report seconds; axes in minutes read more easily)."""
    return seconds / 60.0
