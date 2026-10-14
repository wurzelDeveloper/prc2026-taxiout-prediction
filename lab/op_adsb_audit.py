"""Own-airframe contamination and coverage of the pre-movement ADS-B surface picture (``op_h1_adsb`` / ``op_h0_adsb``).

    python lab/op_adsb_audit.py [--snapshot s1] [--links-dir <main checkout>/cache/adsb]
                                [--months 2025-01 2025-07] [--coverage-months 2025-01 ... | --no-coverage]
                                [--out docs/op_adsb_audit.md]

A gate before any fit with ADS-B (findings 10.31; critique F2). Exit code 1 when a gate fails. ``--snapshot`` selects
the snapshot the way the blocks do (``PRC_ADSB_SNAPSHOT``, set before `prc` is imported), so the audit reads exactly the
tables a training run with that environment would hash and read.

(1) OWN CONTAMINATION, at H1 and H0. The blocks remove the flight's own airframe by rules known before the movement
    (its callsign, or an event origin within 100 m of its own stand; `prc.features._adsblib`). Here the airframe is
    identified by the challenge link instead (``cache/adsb/links_<day>.parquet``: the ADS-B takeoff matched to the
    recorded takeoff; tier T, used ONLY in this audit, never an input). A departure is contaminated at a horizon when
    an event of its own departure (the link's icao; PUSH, TAXI_OUT, RWY_HOLD, REMOTE_STOP, TAKEOFF within 3 h before
    its ADS-B takeoff) passes the rules and enters one of the block's columns the model sees at ``q = anchor - 30 s``
    (`relevant` with ``model``: the taxi, hold and stand families' windows); one that enters only a coverage statistic
    (the 2-hour ratios, the 6-hour push share, the activity heartbeat; dropped from every matrix, they move a gate only
    at its threshold) is reported beside it (``coverage``). Contamination is observable on linked rows only, and
    matters only where a column family of the block is gated in at that horizon (the row is "exposed"), so the gates
    are on the linked exposed rows (spec C5: below 0.2 % at H1 and 1 % at H0 overall, below 1 % and 3 % at every
    airport), and an airport whose exposed rows are linked on fewer than ``MIN_LINK_SHARE`` cannot pass (its
    contamination is not measured, not low). The share of all OP rows is reported beside them. Also reported, never
    gated: any event of the icao, its inbound included (an inbound still taxiing in at H1 is another flight's movement
    and legitimate pre-departure information).
(2) RECEPTION per airport and month (the drift table): departures; the shares whose base gate (``cov_ok``) and the
    taxi, hold and stand family gates hold at H1; the medians of the movement ratio, the taxi-seen share, the push
    share, taxi-outs and runway holds per ADS-B takeoff and the activity ratio; the share of departures the challenge
    link found (the readers' coverage measure); and the snapshot's training support of each family at that airport
    (its 2025 gated share and ratio range). 2025 and the 2026 months both the movement frame and the snapshot hold
    (Jan/Jul; on the final set's tree also Feb and Jun, findings 10.33): a family open in 2026 where 2025 had no
    support, or a 2026 ratio median outside the 2025 range, is a regime the model never saw (the blocks NaN such rows;
    the table shows how many there would have been).
(3) PLACEBO: LTFM (no ADS-B) must be NaN in every surface and stand column.

OP = target below 3,600 s and NM-matched (the operational ladder's population). Writes the report and three csv files
next to it. Lab: the population reads the target through `prc.data.load_target`, the H0 anchor is the recorded
off-block (`_oplib.load_offblocks`, the blocks' own loader).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _snapshot_arg(argv: list[str]) -> str | None:
    """``--snapshot`` of the command line, read before `prc` is imported (it fixes the blocks' external paths)."""
    for i, a in enumerate(argv):
        if a == "--snapshot" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--snapshot="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__" and _snapshot_arg(sys.argv[1:]):
    os.environ["PRC_ADSB_SNAPSHOT"] = _snapshot_arg(sys.argv[1:])

from prc import paths  # noqa: E402
from prc.features import _adsblib, _oplib  # noqa: E402

DEPARTURE_KINDS = ("PUSH", "TAXI_OUT", "RWY_HOLD", "REMOTE_STOP", "TAKEOFF")
OWN_LOOKBACK_S = 3 * 3600
GATES = {"h1": (0.002, 0.01), "h0": (0.01, 0.03)}
"""(overall, per airport) maximal contaminated share of the linked exposed OP rows."""
MIN_LINK_SHARE = 0.2
"""An airport (or the whole set) with exposed rows passes only when at least this share of them is linked."""
FAMILY_GATES = ("cov_taxi_ok", "cov_hold_ok", "cov_stand_ok")


def relevant(kind: np.ndarray, ts, tks, tke, q, dur=None, start=None, model: bool = False) -> np.ndarray:
    """Whether an event enters one of the block's statistics at ``q``: every window of `_adsblib.surface_at` (the
    surface counts, the coverage counts and ratios, the activity heartbeat), or with ``model`` only the windows of the
    columns the model sees (the taxi, hold and stand families; the coverage columns are dropped from every matrix, so an
    event that enters only them can reach the model only by moving a gate across its threshold). ``dur``: the events'
    durations (a takeoff enters the family columns only with its taxi-out's duration); ``start``: whether a taxi-out
    counts as a new start (`taxi_out_starts`); None counts every takeoff and every taxi-out."""
    ts, tks, tke, q = (np.asarray(x, dtype="float64") for x in (ts, tks, tke, q))
    kind = np.asarray(kind, dtype=object)
    has_dur = np.ones(len(kind), dtype=bool) if dur is None else np.isfinite(np.asarray(dur, dtype="float64"))
    is_start = np.ones(len(kind), dtype=bool) if start is None else np.asarray(start, dtype=bool)
    with np.errstate(invalid="ignore"):
        def win(w):
            return (tks >= q - w) & (tks <= q)

        def prog(cap):
            stop = np.maximum(tks, np.fmin(tke, ts + cap))
            return (tks <= q) & (q < stop)

        def ended(w):
            return (tke >= q - w) & (tke <= q)

        if model:
            m = (kind == "PUSH") & win(900)
            m |= (kind == "TAXI_OUT") & ((win(900) & is_start) | prog(_adsblib.CAP_TAXI_OUT_S))
            m |= (kind == "RWY_HOLD") & (prog(_adsblib.CAP_HOLD_S) | ended(1800))
            m |= (kind == "REMOTE_STOP") & (prog(_adsblib.CAP_REMOTE_S) | win(3600) | ended(3600))
            m |= (kind == "TAKEOFF") & win(3600) & has_dur
            m |= (kind == "TAXI_IN") & prog(_adsblib.CAP_TAXI_IN_S)
            return m
        heartbeat = np.isin(kind, _adsblib.ACT_DEP_KINDS + ("LANDING",)) & win(_adsblib.ACT_BASE_S)
        r = heartbeat
        r |= (kind == "PUSH") & win(900)
        r |= (kind == "TAXI_OUT") & (win(7200) | prog(_adsblib.CAP_TAXI_OUT_S))
        r |= (kind == "RWY_HOLD") & (win(7200) | prog(_adsblib.CAP_HOLD_S) | ended(1800))
        r |= (kind == "REMOTE_STOP") & (prog(_adsblib.CAP_REMOTE_S) | win(3600) | ended(3600))
        r |= (kind == "TAKEOFF") & win(max(7200, _adsblib.PUSH_SHARE_S))
        r |= (kind == "LANDING") & win(7200)
        r |= (kind == "TAXI_IN") & prog(_adsblib.CAP_TAXI_IN_S)
    return r


def taxi_out_starts(events: pd.DataFrame) -> np.ndarray:
    """Taxi-outs `_adsblib.surface_at` counts as new starts (``taxi_out_start_15``): from a push or a stop, or first
    seen near a stand."""
    origin = events["origin_kind"].astype(str).to_numpy(dtype=object)
    near = np.isfinite(events["origin_stand"].to_numpy(dtype="float64"))
    return np.isin(origin, ("push", "stop")) | ((origin == "first_seen") & near)


def own_contamination(events: pd.DataFrame, deps: pd.DataFrame, links: pd.DataFrame, anchors: dict[str, pd.Series],
                      stands: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per audited departure (index mvt_id) and horizon: ``<h>_own`` (an own-departure event passes the identity rules
    and enters a column the model sees at q), ``<h>_cov`` (... or any statistic, the coverage ones included) and
    ``<h>_any`` (any event of the link's icao enters any statistic, the inbound included).

    ``deps``: apt, STAND_mvt, CALLSIGN_flt of the departures; ``links``: icao and t_takeoff per mvt_id (tier T);
    ``anchors``: the H1 / H0 anchor per mvt_id (NaN where none)."""
    rows = deps.index.intersection(links.index)
    out = pd.DataFrame(index=rows)
    if not len(rows):
        return out
    lk = links.loc[rows]
    q = {h: np.floor(a.reindex(rows).to_numpy(dtype="float64")) - _adsblib.ADSB_LAG_S for h, a in anchors.items()}
    q_all = np.column_stack(list(q.values()))
    lo = np.nanmin(q_all, axis=1) - max(7200, _adsblib.PUSH_SHARE_S)
    hi = np.nanmax(q_all, axis=1)
    ev = events.loc[events["icao"].astype(str).isin(set(lk["icao"].astype(str)))]
    ev = ev.assign(icao=ev["icao"].astype(str), kind=ev["kind"].astype(str)).sort_values(["icao", "t_known_start"], kind="stable")
    icaos = pd.Index(pd.unique(ev["icao"]))
    ev_code = icaos.get_indexer(ev["icao"]).astype("int64")
    t0 = float(np.nanmin([ev["t_known_start"].min(), np.nanmin(lo)])) - 1 if len(ev) else 0.0
    span = int(max(float(ev["t_known_start"].max()) if len(ev) else 0.0, float(np.nanmax(hi))) - t0) + 2
    key = ev_code * span + (ev["t_known_start"].to_numpy() - t0).astype("int64")
    r_code = icaos.get_indexer(lk["icao"].astype(str)).astype("int64")
    ok = (r_code >= 0) & np.isfinite(lo) & np.isfinite(hi)
    lo_pos = np.where(ok, np.searchsorted(key, r_code * span + (np.nan_to_num(lo) - t0).astype("int64"), "left"), 0)
    hi_pos = np.where(ok, np.searchsorted(key, r_code * span + (np.nan_to_num(hi) - t0).astype("int64"), "right"), 0)
    counts = np.maximum(hi_pos - lo_pos, 0)
    row_pos = np.repeat(np.arange(len(rows)), counts)
    ev_pos = np.repeat(lo_pos, counts) + (np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts))
    for h in anchors:
        out[f"{h}_own"] = False
        out[f"{h}_cov"] = False
        out[f"{h}_any"] = False
    if not len(row_pos):
        return out
    p = ev.iloc[ev_pos].reset_index(drop=True)
    p["row"] = row_pos
    kind = p["kind"].to_numpy(dtype=object)
    t_tko = lk["t_takeoff"].to_numpy(dtype="float64")[row_pos]
    # the own departure: after the airframe's last landing, or its last earlier takeoff (when the return landing was
    # not received), at the airport before its takeoff -- an earlier rotation of the same airframe is another flight --
    # and at most OWN_LOOKBACK_S before the takeoff
    prev = ev["kind"].eq("LANDING") | ev["kind"].eq("TAKEOFF")
    ldg = ev.loc[prev, ["icao", "t_start"]]
    last_ldg = np.full(len(rows), -np.inf)
    if len(ldg):
        lk_t = pd.DataFrame({"icao": lk["icao"].astype(str).to_numpy(), "t": lk["t_takeoff"].to_numpy(dtype="float64") - 600,
                             "row": np.arange(len(rows))}).sort_values("t")
        lk_t = lk_t.loc[np.isfinite(lk_t["t"])]
        m = pd.merge_asof(lk_t, ldg.sort_values("t_start").rename(columns={"t_start": "t_ldg"}),
                          left_on="t", right_on="t_ldg", by="icao", direction="backward")
        last_ldg[m["row"].to_numpy()] = m["t_ldg"].fillna(-np.inf).to_numpy()
    start = np.maximum(t_tko - OWN_LOOKBACK_S, last_ldg[row_pos])
    own_dep = np.isin(kind, DEPARTURE_KINDS) & (p["t_start"].to_numpy() > start) & (p["t_start"].to_numpy() <= t_tko + 120)
    # the block's identity rules, per pair: the event's callsign is the flight's, or its origin is near the own stand
    own_cs = _adsblib.norm_callsign(deps.loc[rows, "CALLSIGN_flt"].to_numpy(dtype=object))[row_pos]
    ev_cs = _adsblib.norm_callsign(p["cs"].to_numpy(dtype=object))
    by_cs = pd.notna(own_cs) & (own_cs == ev_cs)
    stand_cats = pd.Index(np.unique(np.concatenate([stands["stand_id"].to_numpy(dtype="float64"),
                                                    p["origin_stand"].dropna().to_numpy(dtype="float64")])))
    nb_rows, nb_codes = _adsblib.neighbour_pairs(deps.loc[rows, "apt"].astype(str).to_numpy(dtype=object),
                                                 deps.loc[rows, "STAND_mvt"].astype("string").to_numpy(dtype=object),
                                                 stands, matches, stand_cats)
    nb = set(zip(nb_rows.tolist(), nb_codes.tolist()))
    os_code = _adsblib._codes(p["origin_stand"].to_numpy(dtype="float64"), stand_cats)
    by_stand = np.array([(r, c) in nb for r, c in zip(row_pos.tolist(), os_code.tolist())], dtype=bool)
    passes = ~(by_cs | by_stand)
    for h, qh in q.items():
        times = (kind, p["t_start"], p["t_known_start"], p["t_known_end"], qh[row_pos])
        rel = relevant(*times) & passes
        rel_model = relevant(*times, dur=p["dur"], start=taxi_out_starts(p), model=True) & passes
        for col, mask in ((f"{h}_own", rel_model & own_dep), (f"{h}_cov", rel & own_dep), (f"{h}_any", rel)):
            hit = np.zeros(len(rows), dtype=bool)
            np.logical_or.at(hit, row_pos[mask], True)
            out[col] = hit
    return out


def exposure(mvt: pd.DataFrame, op: pd.DataFrame, anchors: dict[str, pd.Series], params: dict | None = None,
             events: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Per horizon, whether each OP row has any column family of the block gated in (`FAMILY_GATES`): only there can
    an own event change a value the model sees."""
    out = {}
    for h, a in anchors.items():
        X = _adsblib.surface_at(mvt, op.index, a.reindex(op.index).to_numpy(dtype="float64"), params, events=events)
        out[h] = (X[list(FAMILY_GATES)] == 1).any(axis=1)
    return out


def contamination_table(flags: pd.DataFrame, op: pd.DataFrame, exposed: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Per horizon and airport: OP rows, exposed rows (a family gated in; every row without ``exposed``), the linked
    exposed rows and their share of the exposed, the contaminated ones (an own event in a column the model sees),
    their share of the linked exposed rows (the gated share) and of all OP rows (the narrative), the rows where an own
    event enters a coverage statistic too (``coverage``, ``share_cov``: reported, not gated), and rows where any event
    of the icao enters."""
    recs = []
    for h in [c[:-4] for c in flags.columns if c.endswith("_own")]:
        f = flags.reindex(op.index)
        ex = (exposed or {}).get(h)
        ex = pd.Series(True, index=op.index) if ex is None else ex.reindex(op.index).fillna(False).astype(bool)
        for apt, g in [("ALL", op)] + list(op.groupby(op["apt"].astype(str))):
            fg = f.loc[g.index]
            e = ex.loc[g.index].to_numpy()
            linked = fg[f"{h}_own"].notna().to_numpy() & e
            own = fg[f"{h}_own"].fillna(False).astype(bool).to_numpy() & linked
            cov = fg[f"{h}_cov"].fillna(False).astype(bool).to_numpy() & linked if f"{h}_cov" in fg else own
            n, n_exp, n_link, n_own = len(g), int(e.sum()), int(linked.sum()), int(own.sum())
            recs.append({"horizon": h, "apt": apt, "op_rows": n, "exposed": n_exp, "linked": n_link,
                         "link_share": n_link / n_exp if n_exp else np.nan, "contaminated": n_own,
                         "share": n_own / n_link if n_link else np.nan, "share_op": n_own / n if n else np.nan,
                         "coverage": int(cov.sum()), "share_cov": int(cov.sum()) / n_link if n_link else np.nan,
                         "any_icao_event": int((fg[f"{h}_any"].fillna(False).astype(bool).to_numpy() & e).sum())})
    return pd.DataFrame(recs)


def gate_failures(table: pd.DataFrame) -> list[str]:
    fails = []
    for _, r in table.iterrows():
        overall, per_apt = GATES.get(r["horizon"], (np.inf, np.inf))
        limit = overall if r["apt"] == "ALL" else per_apt
        if r["exposed"] and not r["link_share"] >= MIN_LINK_SHARE:
            fails.append(f"{r['horizon']} {r['apt']}: {r['exposed']:,} exposed OP rows, only {100 * r['link_share']:.1f} % "
                         f"linked (< {100 * MIN_LINK_SHARE:.0f} %): contamination not measured")
        elif r["linked"] and r["share"] >= limit:
            fails.append(f"{r['horizon']} {r['apt']}: {100 * r['share']:.3f} % of the linked exposed OP rows contaminated "
                         f"(gate < {100 * limit:.1f} %)")
    return fails


def _support_columns(support: pd.DataFrame | None, apt: str) -> dict:
    out = {}
    if support is None or not len(support):
        return out
    s = support.loc[support["apt"].astype(str).eq(apt)].set_index("family")
    for fam in ("taxi", "hold", "stand"):
        if fam in s.index:
            out[f"sup_{fam}"] = float(s.loc[fam, "share"])
            out[f"sup_{fam}_range"] = f"{s.loc[fam, 'lo']:.2f}-{s.loc[fam, 'hi']:.2f}" if np.isfinite(s.loc[fam, "lo"]) else "-"
    if "own_hold" in s.index:
        out["sup_hold_agree"] = float(s.loc["own_hold", "share"])
    return out


def coverage_table(mvt: pd.DataFrame, months: list[str], links: pd.DataFrame | None = None,
                   days: set[str] | None = None, support: pd.DataFrame | None = None,
                   events: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The reception (drift) table per airport and month of the H1 anchor (only ``days``, the snapshot's built days,
    when given) and the LTFM placebo rows (surface columns not all NaN). The family gates are the block's with its
    support table; ``support`` adds the training support columns."""
    dep = _oplib.departures(mvt)
    a = pd.Series(_oplib.anchor_h1(dep), index=dep.index)
    stamp = pd.to_datetime(a, unit="s", utc=True)
    month = pd.Series(stamp.dt.strftime("%Y-%m").to_numpy(), index=dep.index)
    on_day = np.ones(len(dep), dtype=bool) if days is None else stamp.dt.strftime("%Y-%m-%d").isin(days).to_numpy()
    recs, placebo = [], []
    for m in months:
        idx = dep.index[month.eq(m).to_numpy() & on_day]
        if not len(idx):
            continue
        X = _adsblib.surface_at(mvt, idx, a.loc[idx].to_numpy(), events=events)
        X["apt"] = dep.loc[idx, "apt"].astype(str).to_numpy()
        if links is not None:
            X["linked"] = idx.isin(links.index)
        for apt, g in X.groupby("apt"):
            rec = {"month": m, "apt": apt, "departures": len(g),
                   "days": int(stamp.loc[idx[X["apt"].eq(apt).to_numpy()]].dt.strftime("%d").nunique()),
                   "cov_ok": g["cov_ok"].mean(), "taxi_ok": g["cov_taxi_ok"].mean(), "hold_ok": g["cov_hold_ok"].mean(),
                   "stand_ok": g["cov_stand_ok"].mean(), "mov_ratio_med": g["cov_mov_ratio_120"].median(),
                   "taxi_share_med": g["cov_taxi_share_120"].median(), "push_share_med": g["cov_push_share_360"].median(),
                   "taxi_ratio_med": g["cov_taxi_ratio_120"].median(), "hold_ratio_med": g["cov_hold_ratio_120"].median(),
                   "act_ratio_med": g["cov_act_ratio_5"].median()}
            if links is not None:
                rec["link_share"] = g["linked"].mean()
            rec.update(_support_columns(support, apt))
            recs.append(rec)
        lt = X.loc[X["apt"].eq("LTFM"), list(_adsblib.SURFACE + _adsblib.STAND)]
        placebo.append({"month": m, "ltfm_rows": len(lt), "ltfm_not_nan": int(lt.notna().any(axis=1).sum())})
    return pd.DataFrame(recs), pd.DataFrame(placebo)


def load_links(links_dir: Path, days: list[str]) -> pd.DataFrame:
    parts = []
    for d in days:
        f = links_dir / f"links_{d}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["icao", "t_takeoff"]))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["icao", "t_takeoff"])


def _md(frame: pd.DataFrame) -> str:
    """A plain markdown table (no tabulate dependency)."""
    if frame is None or not len(frame):
        return "(none)"
    cols = list(frame.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("-" if isinstance(v, (float, np.floating)) else str(v))  # noqa: E731
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(out)


def markdown(cont: pd.DataFrame, cov: pd.DataFrame, placebo: pd.DataFrame, fails: list[str], meta: dict) -> str:
    share = cont
    if len(cont):
        share = cont.assign(share_pct=100 * cont["share"], share_op_pct=100 * cont["share_op"],
                            share_cov_pct=100 * cont["share_cov"]).drop(columns=["share", "share_op", "share_cov"])
    lines = ["# ADS-B surface picture: own-airframe contamination and coverage", "",
             f"Snapshot `{meta['snapshot']}`; challenge links from `{meta['links_dir']}` (tier T, used only here); audited "
             f"months {', '.join(meta['months'])}, on the snapshot's {meta['built']} built days only. Written by "
             "`lab/op_adsb_audit.py`.", "",
             "## Own contamination (gates on the linked exposed OP rows: H1 < 0.2 % overall and < 1 % per airport, "
             f"H0 < 1 % and < 3 %; at least {100 * MIN_LINK_SHARE:.0f} % of the exposed rows linked)", "",
             "`contaminated`: an own-departure event passes the identity rules into a column the model sees (gated); "
             "`coverage`: into any statistic, the coverage ratios and the heartbeat included (reported: those columns are "
             "dropped from the matrix and move a gate only at its threshold; mostly the airframe's repositioning or its "
             "inbound leg when the landing was not received).", "",
             _md(share), "", "Verdict: " + ("PASS" if not fails else "FAIL"), ""]
    lines += [f"- {f}" for f in fails]
    lines += ["", "## Reception at H1 per airport and month (the drift table; sup_* = the snapshot's 2025 training support)",
              "", _md(cov), "", "## Placebo: LTFM rows with any surface or stand column", "", _md(placebo)]
    return "\n".join(lines) + "\n"


def audit(events: pd.DataFrame, mvt: pd.DataFrame, y: pd.Series, offblocks: pd.Series, links: pd.DataFrame,
          months: list[str], coverage_months: list[str] | None, stands: pd.DataFrame, matches: pd.DataFrame,
          days: set[str] | None = None, support: pd.DataFrame | None = None):
    """(contamination table, coverage table, placebo table, gate failures) of one snapshot; ``days``: the snapshot's
    built (non-warm-up) days -- departures of other days have no events to be contaminated by and are left out.
    ``y``: the training target (the OP population only); ``offblocks``: the recorded off-blocks (the H0 anchor)."""
    dep = _oplib.departures(mvt)
    month = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    on_day = np.ones(len(dep), dtype=bool) if days is None else dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").isin(days).to_numpy()
    audited = dep.loc[month.isin(months).to_numpy() & dep["split"].astype(str).eq("train").to_numpy() & on_day]
    yy = y.reindex(audited.index)
    op = audited.loc[(yy < 3600).to_numpy() & audited["FLIGHT_ID_mvt"].notna().to_numpy()]
    anchors = {"h1": pd.Series(_oplib.anchor_h1(op), index=op.index),
               "h0": pd.Series(_oplib.anchor_h0(op, offblocks), index=op.index)}
    flags = own_contamination(events, op, links, anchors, stands, matches)
    cont = contamination_table(flags, op, exposure(mvt, op, anchors, events=events))
    fails = gate_failures(cont)
    cov, placebo = (coverage_table(mvt, coverage_months, links=links, days=days, support=support, events=events)
                    if coverage_months else (pd.DataFrame(), pd.DataFrame()))
    if len(placebo) and placebo["ltfm_not_nan"].sum():
        fails.append(f"placebo: {int(placebo['ltfm_not_nan'].sum())} LTFM rows carry surface or stand values")
    return cont, cov, placebo, fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default=None, help="default: $PRC_ADSB_SNAPSHOT or s1 (selected before prc is imported)")
    ap.add_argument("--links-dir", default=str(paths.CACHE / "adsb"))
    ap.add_argument("--months", nargs="*", default=["2025-01", "2025-07"])
    ap.add_argument("--coverage-months", nargs="*", default=None, help="default: every month of the snapshot's day set")
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--out", default=str(paths.ROOT / "docs" / "op_adsb_audit.md"))
    args = ap.parse_args(argv)

    from prc import data
    from prc.adsb import surface

    snapshot = args.snapshot or paths.ADSB_SURFACE_SNAPSHOT
    if snapshot != paths.ADSB_SURFACE_SNAPSHOT:
        raise SystemExit(f"--snapshot {snapshot} but the blocks' snapshot is {paths.ADSB_SURFACE_SNAPSHOT}: run this script "
                         f"as a program (it sets PRC_ADSB_SNAPSHOT before importing prc), or set the variable yourself")
    events = pd.read_parquet(paths.ADSB_SURFACE_EVENTS_FILE, columns=list(_adsblib.READ_COLUMNS))
    cols = ["apt", "is_dep", "split", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "IOBT_flt", "LOBT_flt",
            "STAND_mvt", "RUNWAY_mvt", "CALLSIGN_flt"]
    mvt = data.load_movements(columns=cols)
    y = data.load_target()
    offblocks = _oplib.load_offblocks()
    man = surface.read_manifest(snapshot)
    coverage_months = None if args.no_coverage else (args.coverage_months or sorted({d[:7] for d in man.get("day_set", [])}))
    audit_days = {d.strftime("%Y-%m-%d") for m in args.months
                  for d in pd.date_range(f"{m}-01", pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0), freq="D")}
    links = load_links(Path(args.links_dir), sorted(set(man.get("day_set", [])) | audit_days))
    stands, matches = _adsblib.load_stand_tables()
    built = {d for d, e in man.get("days", {}).items() if not e.get("key", {}).get("warmup")}
    support = _adsblib.load_support()
    cont, cov, placebo, fails = audit(events, mvt, y, offblocks, links, args.months, coverage_months, stands, matches,
                                      built, support)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"snapshot": snapshot, "links_dir": args.links_dir, "months": args.months, "built": len(built)}
    out.write_text(markdown(cont, cov, placebo, fails, meta), encoding="utf-8")
    cont.to_csv(out.with_name(out.stem + "_contamination.csv"), index=False)
    if len(cov):
        cov.to_csv(out.with_name(out.stem + "_coverage.csv"), index=False)
    if len(placebo):
        placebo.to_csv(out.with_name(out.stem + "_placebo.csv"), index=False)
    print(out.read_text(encoding="utf-8"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
