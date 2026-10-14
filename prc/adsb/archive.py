"""Consolidate the lab's per-flight stand-archive observations into the input file of the block `op_adsb_archive`.

    python -m prc.adsb.archive [--obs DIR] [--src DIR] [--out FILE] [--all-splits]

Reads every day's ``<day>.parquet`` of ``--obs`` (default `paths.ADSB_STAND_OBS`: lab/ideas/stand_archive.py pass 1,
one row per linked departure with the off-block kind and its timings where the aircraft's own track shows the
off-block fully) and the pushback detector's ``features3.parquet`` of ``--src`` (default `paths.ADSB_TABLES`): only the
columns `prc.features.adsb_archive.INPUT_COLUMNS` names. Writes one row per departure (index ``mvt_id``, plus the
``day`` of its file) to ``--out`` (default `paths.ADSB_ARCHIVE_FILE`) and a manifest next to it
(`paths.ADSB_ARCHIVE_MANIFEST` for the default output): the sources with their sha1 digests (the lab rewrites its
tables in place, so a rebuild can differ silently; the digests show it), the days read and kept, and the guard.

The archive is history. By default only training-split departures are kept (the block takes each fold's fit rows,
and the final model's fit rows are the 2025 departures), and the build fails unless every kept departure took off
before the first ranking-split departure (the guard: no record of a scored period, while the crawl of 2026 ADS-B
writes into the same tree). ``--all-splits`` keeps the ranking rows too and skips the guard.

cache/ follows the checkout: build once (from a worktree pass the main checkout's ``cache/lab/ideas/stand_archive_obs``
and ``cache/adsb``) and copy the file and its manifest to the checkout that trains, rather than rebuilding there. Pure
consolidation: every rule that turns these columns into archive values lives in the feature module, so it is part of
the block's cache key; this file's digest is too (external input).

Light: a few columns of ~400 small parquet files and one column of a large one; single process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prc import paths
from prc.features.adsb_archive import INPUT_COLUMNS

INDEX = "mvt_id"
F3_FILE = "features3.parquet"


def archive_days(obs: Path) -> list[str]:
    """Every day with an observation table in ``obs``, in date order."""
    return sorted(p.stem for p in Path(obs).glob("*.parquet"))


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(columns))
    if frame.index.name != INDEX:
        raise ValueError(f"{path}: expected the index {INDEX!r}, found {frame.index.name!r}")
    return frame


def consolidate(obs: Path, src: Path, keep: pd.Index | None = None, days: list[str] | None = None) -> pd.DataFrame:
    """One row per observed departure of ``days`` (default: every day in ``obs``) with the `INPUT_COLUMNS`.

    ``features3.parquet`` of ``src`` is required: without the push cluster's stand distance no off-block can be
    classified. ``keep`` restricts the rows to those departures.
    """
    obs, src = Path(obs), Path(src)
    parts = []
    for day in days if days is not None else archive_days(obs):
        part = _read(obs / f"{day}.parquet", INPUT_COLUMNS["obs"])
        part["day"] = day
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no <day>.parquet in {obs}")
    out = pd.concat(parts)
    if out.index.has_duplicates:
        raise ValueError(f"departures observed on two days: {out.index[out.index.duplicated()][:5].tolist()}")
    f3 = src / F3_FILE
    if not f3.exists():
        raise FileNotFoundError(f"{f3} is missing: the pushback detector's table is required")
    out = out.join(_read(f3, INPUT_COLUMNS["features3"]).loc[lambda f: ~f.index.duplicated(keep="last")])
    if keep is not None:
        out = out.loc[out.index.isin(keep)]
    for col in ("apt", "kind"):
        out[col] = out[col].astype("string")
    out.index.name = INDEX
    return out.sort_index()


def movement_splits() -> pd.DataFrame:
    """Every departure's split and recorded takeoff (the movement frame's ``split`` / ``MVT_TIME_UTC_mvt``)."""
    m = pd.read_parquet(paths.MOVEMENTS_FILE, columns=["is_dep", "split", "MVT_TIME_UTC_mvt"])
    return m.loc[m["is_dep"].to_numpy(dtype=bool), ["split", "MVT_TIME_UTC_mvt"]]


def training_departures(splits: pd.DataFrame) -> pd.Index:
    return splits.index[splits["split"].astype("string").eq("train").fillna(False).to_numpy(dtype=bool)]


def guard(kept: pd.Index, splits: pd.DataFrame) -> dict:
    """Whether every kept departure took off before the first ranking-split departure (raises otherwise)."""
    t = splits["MVT_TIME_UTC_mvt"]
    rank = splits["split"].astype("string").eq("rank").fillna(False).to_numpy(dtype=bool)
    first_rank = t[rank].min() if rank.any() else pd.NaT
    last_kept = t.reindex(kept).max() if len(kept) else pd.NaT
    unknown = int(t.reindex(kept).isna().sum())
    ok = unknown == 0 and (pd.isna(first_rank) or pd.isna(last_kept) or last_kept < first_rank)
    out = {"kept_last_takeoff": str(last_kept), "first_ranking_takeoff": str(first_rank),
           "kept_without_takeoff": unknown, "ok": bool(ok)}
    if not ok:
        raise ValueError(f"archive guard failed: a kept departure is not before the first scored takeoff ({out})")
    return out


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digests(obs: Path, src: Path, days: list[str]) -> dict:
    """sha1 of ``features3.parquet`` and one digest over the (name, sha1) of every observation file read."""
    lines = "\n".join(f"{d}.parquet {_sha1(Path(obs) / f'{d}.parquet')}" for d in days)
    return {"obs_files_sha1": hashlib.sha1(lines.encode("utf-8")).hexdigest(), "features3_sha1": _sha1(Path(src) / F3_FILE)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m prc.adsb.archive", description=__doc__.split("\n\n")[0])
    ap.add_argument("--obs", type=Path, default=paths.ADSB_STAND_OBS, help="directory of the lab's pass-1 observations per day")
    ap.add_argument("--src", type=Path, default=paths.ADSB_TABLES, help="directory holding features3.parquet")
    ap.add_argument("--out", type=Path, default=paths.ADSB_ARCHIVE_FILE, help="the archive input file to write")
    ap.add_argument("--all-splits", action="store_true", help="keep ranking-split departures too (no guard)")
    args = ap.parse_args(argv)
    splits = None if args.all_splits else movement_splits()
    keep = None if splits is None else training_departures(splits)
    days = archive_days(args.obs)
    frame = consolidate(args.obs, args.src, keep, days)
    checked = None if splits is None else guard(frame.index, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    kinds = frame["kind"].value_counts(dropna=True).astype(int).to_dict()
    per_apt = frame.groupby(frame["apt"], observed=True).size().astype(int).to_dict()
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "obs": str(args.obs),
        "src": str(args.src),
        "out": str(args.out),
        "out_sha1": _sha1(args.out),
        **source_digests(args.obs, args.src, days),
        "files_days": len(days),
        "files_first_day": days[0],
        "files_last_day": days[-1],
        "rows": int(len(frame)),
        "rows_fully_observed": int(frame["kind"].isin(["push", "power"]).fillna(False).sum()),
        "rows_by_kind": kinds,
        "rows_days": int(frame["day"].nunique()),
        "rows_first_day": str(frame["day"].min()) if len(frame) else None,
        "rows_last_day": str(frame["day"].max()) if len(frame) else None,
        "splits": "all" if args.all_splits else "train",
        "guard": checked if checked is not None else "skipped (--all-splits)",
        "rows_per_airport": per_apt,
        "columns": list(frame.columns),
    }
    manifest_path = paths.ADSB_ARCHIVE_MANIFEST if args.out == paths.ADSB_ARCHIVE_FILE else args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(frame):,} departures ({manifest['rows_fully_observed']:,} fully observed) on "
          f"{manifest['rows_days']} days ({manifest['rows_first_day']} .. {manifest['rows_last_day']}; {len(days)} day "
          f"files read); guard {manifest['guard']}; per airport {per_apt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
