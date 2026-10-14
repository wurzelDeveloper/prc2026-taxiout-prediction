"""`prc audit`: what the training data contains, before any model touches it.

Answered from the interim tables, with no model and no network:

* per airport -- how many departures have impossible or extreme taxi times
  (non-positive, under a minute, over an hour, over two hours, a whole day
  off), no aircraft type, no NM match, or a helicopter pad as runway;
* the untyped segment -- where it is, how it behaves, and how many of its
  2026 twins the ranking set holds (the reason it cannot just be dropped);
* the helicopter rows -- why the minimum-taxi rule exempts them;
* with ``-c``: what that config's ``data.clean`` rules would flag, per rule
  and airport, and how much of that sits in the holdout months;
* with ``--run``: how concentrated that run's squared error is, and what its
  RMSE would be without the rows above two hours or without the flagged rows,
  which scores any past run under a new rule set without retraining;
* the disrupted airport-days (`prc.disruption`): the days whose arrivals
  taxied in far longer than usual, in 2025 and in the 2026 ranking months,
  and, with ``--run``, that run's RMSE on their departures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc import cleaning, data, disruption, paths
from prc.config import ExperimentConfig

COLUMNS = sorted(
    {"apt", "split", "is_dep", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"}
    | set(disruption.MVT_COLUMNS)
)
DAY_TOL_SEC = cleaning.DAY_OFFSET_REPORT_TOL_SEC
LONG_SEC = 7200.0


def report(cfg: ExperimentConfig | None = None, run_id: str | None = None) -> str:
    mvt = data.load_movements(columns=COLUMNS)
    y = data.load_target()
    dep = mvt.loc[mvt["is_dep"]]
    tr = dep.loc[dep["split"].eq("train")].copy()
    tr["y"] = y.reindex(tr.index)
    rk = dep.loc[dep["split"].eq("rank")]
    apt_tr = tr["apt"].astype(str).to_numpy()
    apt_rk = rk["apt"].astype(str).to_numpy()
    yv = tr["y"]
    heli = cleaning.is_heli_runway(tr["RUNWAY_mvt"]).to_numpy()
    untyped = cleaning.is_untyped(tr["AIRCRAFT_TYPE_mvt"]).to_numpy()
    k = np.round(yv / cleaning.DAY_SEC)

    # --- 1. per-airport counts ------------------------------------------------
    flags = pd.DataFrame({
        "le0": yv <= 0, "lt30": yv < 30, "lt60": yv < 60, "gt3600": yv > 3600, "gt7200": yv > LONG_SEC,
        "day_mult": (k >= 1) & ((yv - k * cleaning.DAY_SEC).abs() <= DAY_TOL_SEC),
        "untyped": untyped, "no_nm": tr["AOBT_3_flt"].isna().to_numpy(), "heli_rwy": heli,
    }, index=tr.index)
    counts = flags.groupby(apt_tr).sum().astype(int)
    counts["n"] = pd.Series(apt_tr).value_counts().reindex(counts.index).to_numpy()
    counts.loc["total"] = counts.sum()
    shares = (100 * counts.drop(columns="n").div(counts["n"], axis=0)).round(3)

    # --- 2. the untyped segment -------------------------------------------------
    u = tr.loc[untyped]
    u_apt = u["apt"].astype(str).to_numpy()
    seg = pd.DataFrame({
        "n": pd.Series(u_apt).value_counts(),
        "mean": u.groupby(u_apt)["y"].mean().round(0),
        "median": u.groupby(u_apt)["y"].median().round(0),
        "p90": u.groupby(u_apt)["y"].quantile(0.9).round(0),
        "gt2h_pct": (100 * u.groupby(u_apt)["y"].apply(lambda s: (s > LONG_SEC).mean())).round(1),
        "no_nm_pct": (100 * u.groupby(u_apt)["AOBT_3_flt"].apply(lambda s: s.isna().mean())).round(1),
        "n_rank": pd.Series(apt_rk[cleaning.is_untyped(rk["AIRCRAFT_TYPE_mvt"]).to_numpy()]).value_counts(),
    }).fillna({"n": 0, "n_rank": 0})
    seg["share_pct"] = (100 * seg["n"] / counts["n"].reindex(seg.index)).round(3)
    seg = seg[["n", "share_pct", "mean", "median", "p90", "gt2h_pct", "no_nm_pct", "n_rank"]].sort_values("n", ascending=False)

    # --- 3. helicopters --------------------------------------------------------------
    h = tr.loc[heli]
    h_apt = h["apt"].astype(str).to_numpy()
    heli_rk = cleaning.is_heli_runway(rk["RUNWAY_mvt"]).to_numpy()
    heli_table = pd.DataFrame({
        "n_train": pd.Series(h_apt).value_counts(),
        "n_rank": pd.Series(apt_rk[heli_rk]).value_counts(),
        "median_taxi": h.groupby(h_apt)["y"].median().round(0),
        "neg_pct": (100 * h.groupby(h_apt)["y"].apply(lambda s: (s <= 0).mean())).round(1),
        "top_types": h.groupby(h_apt)["AIRCRAFT_TYPE_mvt"].apply(
            lambda s: ", ".join(f"{t}:{n}" for t, n in s.astype(str).value_counts().head(3).items())
        ),
    }).fillna({"n_train": 0, "n_rank": 0})

    out = [
        "=== training departures: counts per airport ===",
        counts.to_string(),
        "\n=== the same as % of the airport's departures ===",
        shares.to_string(),
        "\n=== the untyped segment (no AIRCRAFT_TYPE_mvt): where the tail lives; n_rank = its 2026 twins ===",
        seg.to_string() if len(seg) else "  none",
        "\n=== helicopter pads (RUNWAY_mvt H/HEL): tiny and negative taxi times that are real ===",
        heli_table.to_string() if len(heli_table) else "  none",
    ]

    # --- 4. what a config's rules would flag ------------------------------------------
    flagged = None
    if cfg is not None and cfg.data.clean.is_active:
        table = cleaning.flag_table(mvt, y, cfg.data.clean)
        flagged = table[list(cleaning.RULES)].any(axis=1)
        month = tr["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
        in_holdout = flagged & month.isin(cfg.validation.holdout_months).to_numpy()
        out += [
            f"\n=== rules of '{cfg.name}' ({cfg.data.clean.summary()}): rows flagged per rule and airport ===",
            cleaning.describe_flags(table, tr["apt"]).to_string(),
            f"\n{'the fit would drop' if cfg.data.clean.drop_from_fit else 'flag only, the fit keeps'} "
            f"{int(flagged.sum()):,} of {len(tr):,} training departures ({flagged.mean():.3%}); "
            f"{int(in_holdout.sum()):,} of them fall in the holdout months {cfg.validation.holdout_months} "
            "and stay in the validation score",
        ]
    elif cfg is not None:
        out.append(f"\n=== '{cfg.name}' sets no data.clean rules ===")

    # --- 5. a run's error concentration ---------------------------------------------
    if run_id:
        oof_path = paths.run_dir(run_id) / "oof_predictions.parquet"
        if not oof_path.exists():
            out.append(f"\n=== run {run_id}: no oof_predictions.parquet ===")
        else:
            oof = pd.read_parquet(oof_path)
            se = (oof["y_pred"] - oof["y_true"]) ** 2
            tot = float(se.sum())
            order = se.sort_values(ascending=False)
            n = len(oof)
            share = {
                "top 10 rows": order.head(10).sum() / tot,
                "top 100 rows": order.head(100).sum() / tot,
                "top 0.1 %": order.head(max(int(0.001 * n), 1)).sum() / tot,
                "top 1 %": order.head(max(int(0.01 * n), 1)).sum() / tot,
            }
            by_apt = (100 * se.groupby(mvt.loc[oof.index, "apt"].astype(str).to_numpy()).sum() / tot).round(1)
            lines = [f"  {k:<14} {v:6.1%} of the squared error" for k, v in share.items()]
            rmse_all = float(np.sqrt(se.mean()))
            long_mask = oof["y_true"] > LONG_SEC
            rmse_short = float(np.sqrt(se[~long_mask].mean()))
            lines.append(f"  rmse all rows {rmse_all:8.1f}   without the {int(long_mask.sum())} rows above 2 h {rmse_short:8.1f}")
            if "flagged" in oof.columns:
                lines.append(f"  rmse_clean as stored by the run (its own rules): {float(np.sqrt(se[~oof['flagged']].mean())):8.1f}"
                             f"   ({int(oof['flagged'].sum())} flagged)")
            if flagged is not None:
                f = flagged.reindex(oof.index, fill_value=False).to_numpy(dtype=bool)
                lines.append(f"  rmse under '{cfg.name}' rules without the {int(f.sum())} flagged rows: {float(np.sqrt(se[~f].mean())):8.1f}")
            out += [
                f"\n=== run {run_id}: where its squared error lives ({n:,} holdout rows) ===",
                "\n".join(lines),
                "  share of squared error by airport (%):",
                by_apt.sort_values(ascending=False).to_string(),
            ]

    # --- 6. disrupted airport-days ---------------------------------------------------
    days = disruption.disrupted_days(mvt)
    days["day"] = pd.to_datetime(days["day"]).dt.strftime("%Y-%m-%d")
    days["mean_anomaly"] = days["mean_anomaly"].round(0)
    out.append(
        f"\n=== disrupted airport-days: arrivals' mean taxi-in anomaly above {disruption.DISRUPTION_THRESHOLD_S:.0f} s "
        f"({len(days)} days; a report mask, `rmse_disrupted` scores their departures) ==="
    )
    for year in ("2025", "2026"):
        block = days.loc[days["day"].str.startswith(year)].head(15)
        out.append(f"  {year} ({int(days['day'].str.startswith(year).sum())} days, worst 15):")
        out.append(block.to_string(index=False) if len(block) else "    none")
    if run_id and (paths.run_dir(run_id) / "oof_predictions.parquet").exists():
        oof = pd.read_parquet(paths.run_dir(run_id) / "oof_predictions.parquet")
        hit = disruption.disrupted_rows(mvt, oof.index).to_numpy(dtype=bool)
        se = (oof["y_pred"] - oof["y_true"]) ** 2
        if hit.any():
            out.append(
                f"  run {run_id}: rmse on the {int(hit.sum()):,} holdout rows of disrupted days "
                f"{float(np.sqrt(se[hit].mean())):.1f}, on the rest {float(np.sqrt(se[~hit].mean())):.1f}"
            )
        else:
            out.append(f"  run {run_id}: no holdout row falls on a disrupted day")
    return "\n".join(out)
