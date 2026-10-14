"""Hand-placed stands: file validation, the stands they become, and how they override the match. No data."""

from __future__ import annotations

import pandas as pd
import pytest

from prc.config import load_config
from prc.geometry.manual import MANUAL_ID_BASE, apply_manual, load_manual_stands, manual_stands


def _write(tmp_path, text: str):
    p = tmp_path / "stands.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_accepts_the_report_layout_and_blank_rows(tmp_path):
    p = _write(tmp_path, (
        "apt,stand_mvt,n_dep,match_source,lat,lon,ref,note\n"
        "lszh, GA6 ,2485,none,47.4600,8.5500,GA 6,general aviation apron\n"
        "EDDM,356,1652,none,,,,not mapped -- leave unknown\n"
        "EDDF,A14,3532,exact,50.047776,8.567961,,\n"
    ))
    rows = load_manual_stands(p, ["EDDF", "EDDM", "LSZH"])
    assert rows["apt"].tolist() == ["LSZH", "EDDM", "EDDF"]
    assert rows["stand_mvt"].tolist() == ["GA6", "356", "A14"]
    assert rows["has_coords"].tolist() == [True, False, True]
    assert rows.loc[0, "lat"] == pytest.approx(47.46) and rows.loc[0, "ref"] == "GA 6"
    assert pd.isna(rows.loc[1, "lat"]) and rows.loc[1, "note"].startswith("not mapped")


@pytest.mark.parametrize(
    "text, match",
    [
        ("apt,stand_mvt,lat\nEDDF,A1,50.0\n", "missing column"),
        ("apt,stand_mvt,lat,lon\nEDDF,A1,50.0,\n", "half-filled"),
        ("apt,stand_mvt,lat,lon\nEDDF,A1,abc,8.5\n", "half-filled"),
        ("apt,stand_mvt,lat,lon\nEDDF,A1,95.0,8.5\n", "out of range"),
        ("apt,stand_mvt,lat,lon\nEDDF,A1,50.0,8.5\nEDDF,A1,50.1,8.6\n", "duplicate"),
        ("apt,stand_mvt,lat,lon\nEDDF,,50.0,8.5\n", "empty stand_mvt"),
        ("apt,stand_mvt,lat,lon\nXXXX,A1,50.0,8.5\n", "not in the data"),
    ],
)
def test_load_rejects_what_would_be_silently_wrong(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_manual_stands(_write(tmp_path, text), ["EDDF"])


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manual_stands(tmp_path / "nope.csv")


def _rows():
    return pd.DataFrame({
        "apt": ["LSZH", "LSZH", "EDDF"],
        "stand_mvt": ["GA6", "MFG", "A14"],
        "lat": [47.46, None, 50.0],
        "lon": [8.55, None, 8.5],
        "ref": ["GA 6", "", ""],
        "note": ["", "", ""],
        "has_coords": [True, False, True],
    })


def test_manual_stands_have_their_own_id_space_and_no_entry_node():
    placed = manual_stands(_rows(), "LSZH")
    assert placed["stand_id"].tolist() == [MANUAL_ID_BASE]
    assert placed["kind"].tolist() == ["manual"] and placed["entry_node"].tolist() == [-1]
    assert placed["ref_raw"].tolist() == ["GA 6"] and placed["ref_norm"].tolist() == ["GA6"]
    assert placed["stand_mvt"].tolist() == ["GA6"]
    assert manual_stands(_rows(), "EDDM").empty
    assert manual_stands(_rows(), "EDDF")["ref_raw"].tolist() == ["A14"]  # falls back to the stand id


def test_apply_manual_overrides_and_blanks():
    matches = pd.DataFrame({
        "apt": "LSZH", "stand_mvt": ["GA6", "MFG", "A11"],
        "stand_id": [-1, 4242, -777], "match_source": ["none", "gate", "exact"],
        "ambiguous": [0, 1, 0], "n_dep_train": 1, "n_dep_rank": 1,
    })
    out = apply_manual(matches, manual_stands(_rows(), "LSZH"), blank={"MFG"}).set_index("stand_mvt")
    assert out.loc["GA6", ["stand_id", "match_source", "ambiguous"]].tolist() == [MANUAL_ID_BASE, "manual", 0]
    assert out.loc["MFG", ["stand_id", "match_source"]].tolist() == [-1, "none"]   # a wrong gate match, dropped
    assert out.loc["A11", ["stand_id", "match_source"]].tolist() == [-777, "exact"]  # untouched


def test_config_carries_the_stands_file():
    assert load_config("base.yaml").geometry.stands_file is None
    cfg = load_config("base.yaml", ["geometry.stands_file=data/additional_data/airport_geometry/stands_manual.csv"])
    assert cfg.geometry.stands_file.endswith("stands_manual.csv")
    with pytest.raises(ValueError, match="unknown key"):
        load_config("base.yaml", ["geometry.stand_file=x.csv"])


def test_hand_placed_stands_do_not_move_the_zones_of_the_others():
    """The apron zones are fitted on the OSM parking positions only; a stand placed by hand (and a gate) takes the zone
    of its neighbourhood, so adding one never relabels the existing stands (geo_stand_zone is a model categorical)."""
    import numpy as np

    from prc.geometry.normalise import _zones

    rng = np.random.default_rng(0)
    centres = [(-3000, 0), (0, 2500), (2800, -1500)]
    pts = np.vstack([rng.normal(c, 150, size=(20, 2)) for c in centres])
    osm = pd.DataFrame({"east_m": pts[:, 0], "north_m": pts[:, 1], "kind": ["pp_node"] * 30 + ["pp_way"] * 30})
    before = _zones(osm, k=3)
    extra = pd.DataFrame({"east_m": [-2900.0, 2700.0, 6000.0, 0.0], "north_m": [50.0, -1400.0, 6000.0, 2400.0],
                          "kind": ["manual", "manual", "manual", "gate"]})
    after = _zones(pd.concat([osm, extra], ignore_index=True), k=3)
    assert np.array_equal(after[: len(osm)], before)
    assert after[len(osm)] == before[0] and after[len(osm) + 1] == before[-1]   # the zone of their neighbourhood
