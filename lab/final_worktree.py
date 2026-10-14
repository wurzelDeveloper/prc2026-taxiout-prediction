"""The isolated checkout for running the submitted recipe on another ranking set (findings 10.32).

The ADS-B stage reads single files under cache/: the assembled per-flight tables (cache/adsb/features3.parquet and
features4.parquet, the archives' and the map's assembled tables under cache/lab/ideas/) and the lab matrix
(cache/lab/lean2/). Built for the final set (January, February, June and July 2026; findings 10.33) in the main
checkout they would replace the tables of the delivered Jan+Jul set. This script makes a second checkout of the same
commit (git worktree) that shares what can be shared and copies what must not be:

  runs/, cache/features/, .venv/, cache/adsb_surface/, cache/adsb_archive/   junctions to the main checkout's
      (trained runs and new runs side by side; feature caches are per data.version and the new tree's is new)
  cache/adsb/, cache/lab/ideas/                                              private copies
      (lab/adsb_new_days.sh adds the new days' tables and rewrites the assembled ones over the tree's days)
  cache/lab/                                                                 a real directory (lean2__<version> goes there)

and gives a data tree its ADS-B points as hard links to the delivered tree's day files, for exactly the days to serve:
every 2025 day and the ranking months (a day with points but without movements would get empty tables, which every
later start skips as done).

Usage (from the main checkout):
  python lab/final_worktree.py create --worktree ../taxiout-finalrun
  python lab/final_worktree.py points --tree <tree> --months 2026-01 2026-02 2026-06 2026-07
  python lab/final_worktree.py strip --worktree ../taxiout-finalrun --months 2026-01 2026-02 2026-06 2026-07   # forget
      the months' per-day tables, so that lab/adsb_new_days.sh rebuilds them: on the final set all four (the delivered
      Jan+Jul tables were linked to the unjittered times); in the rehearsal --months 2026-01 2026-07
  python lab/final_worktree.py compare --worktree ../taxiout-finalrun --months 2026-01 2026-07  # rehearsal: rebuilt = main
  python lab/final_worktree.py unlink --worktree ../taxiout-finalrun     # BEFORE `git worktree remove`: removes the
      junctions themselves (os.rmdir), never what they point to; `git worktree remove --force` on a checkout that
      still holds them could delete the main checkout's runs/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import paths  # noqa: E402

JUNCTIONS = ("runs", "cache/features", ".venv", "cache/adsb_surface", "cache/adsb_archive")
PRIVATE = ("cache/adsb", "cache/lab/ideas")
PRIVATE_FILES = ("cache/lab/adsb_resid.parquet",)
"""Single files the chain reads (lab/idea_harness.py's holdout frame, which first_reception.py assemble reindexes to)."""
MARKER = "cache/final_worktree.json"
DAY_DIRS = ("first_reception", "pushback_days", "stand_archive_days", "stand_archive_obs", "stop_class",
            "timemap_arrvisits", "timemap_days", "timemap_visits", "track_quality_days")
"""The per-day directories under cache/lab/ideas/ that lab/adsb_new_days.sh writes (one <day>.parquet each)."""
ADSB_DAY_PREFIXES = ("links_", "features_", "features2_")
"""The per-day tables under cache/adsb/ (<prefix><day>.parquet)."""


def _arg(flag: str):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def _args(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    out = []
    for a in sys.argv[sys.argv.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(a)
    return out


def is_junction(p: Path) -> bool:
    try:
        return bool(os.readlink(p))
    except (OSError, ValueError):
        return False


def junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or is_junction(link):
        print(f"  exists  {link}")
        return
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    print(f"  junction {link} -> {target}")


def create(worktree: Path) -> None:
    if worktree.exists():
        raise SystemExit(f"{worktree} exists: remove it first (unlink, then git worktree remove) or use another path")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "worktree", "add", "--detach", str(worktree), head], cwd=ROOT, check=True)
    for rel in JUNCTIONS:
        if (ROOT / rel).exists():
            junction(worktree / rel, ROOT / rel)
    for rel in PRIVATE:
        print(f"  copying {rel} ...", flush=True)
        shutil.copytree(ROOT / rel, worktree / rel)
    (worktree / "cache" / "lab").mkdir(parents=True, exist_ok=True)
    for rel in PRIVATE_FILES:
        shutil.copy2(ROOT / rel, worktree / rel)
    (worktree / MARKER).write_text(json.dumps({"main": str(ROOT), "commit": head, "junctions": list(JUNCTIONS),
                                               "private": list(PRIVATE) + list(PRIVATE_FILES),
                                               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                              indent=1))
    print(f"isolated checkout {worktree} at {head[:10]}: junctions {list(JUNCTIONS)}, private copies {list(PRIVATE)}")


def months_days(months: list[str]) -> list[str]:
    out = []
    for m in months:
        start = pd.Timestamp(f"{m}-01")
        out += [d.strftime("%Y-%m-%d") for d in pd.date_range(start, start + pd.offsets.MonthEnd(0))]
    return out


def points(tree: Path, months: list[str]) -> None:
    src = paths.ADSB_POINTS
    if tree.resolve() == paths.DATA.resolve():
        raise SystemExit("--tree is the delivered data tree itself: give the final (or rehearsal) tree")
    dst = tree / src.relative_to(paths.DATA)
    dst.mkdir(parents=True, exist_ok=True)
    want = set(months_days([f"2025-{m:02d}" for m in range(1, 13)]) + months_days(months))
    have = {p.stem for p in src.glob("*.parquet")}
    linked = existing = 0
    for d in sorted(want & have):
        target = dst / f"{d}.parquet"
        if target.exists():
            existing += 1
            continue
        os.link(src / f"{d}.parquet", target)
        linked += 1
    extra = sorted(p.stem for p in dst.glob("*.parquet") if p.stem not in want)
    if extra:
        raise SystemExit(f"{dst} holds days outside 2025 and {months}: {extra[:5]}... (remove them)")
    print(f"{dst}: {linked} hard links made, {existing} present; {len(want & have)} days "
          f"({len(want - have)} of the wanted days have no points: {sorted(want - have)[:6]})")


def _check_worktree(worktree: Path) -> None:
    if not (worktree / MARKER).exists():
        raise SystemExit(f"{worktree} is not an isolated checkout of this script ({MARKER} missing)")
    for rel in PRIVATE:
        if is_junction(worktree / rel):
            raise SystemExit(f"{worktree / rel} is a junction, not a private copy: refusing")


def _day_files(cache: Path, days: set[str]) -> list[Path]:
    out = []
    for p in (cache / "adsb").glob("*.parquet"):
        for pre in ADSB_DAY_PREFIXES:
            if p.name.startswith(pre) and p.stem[len(pre):] in days:
                out.append(p)
    for sub in DAY_DIRS:
        out += [p for p in (cache / "lab" / "ideas" / sub).glob("*.parquet") if p.stem in days]
    return sorted(out)


def strip(worktree: Path, months: list[str]) -> None:
    _check_worktree(worktree)
    files = _day_files(worktree / "cache", set(months_days(months)))
    for f in files:
        f.unlink()
    print(f"removed {len(files)} per-day tables of {months} from the private copies in {worktree / 'cache'}")


def _same(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, float]:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, float("inf")
    a, b = a.sort_index(), b.sort_index()
    if not a.index.equals(b.index):
        return False, float("inf")
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xv, yv = x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
            nan = np.isnan(xv) | np.isnan(yv)
            if (np.isnan(xv) != np.isnan(yv)).any():
                return False, float("inf")
            if (~nan).any():
                worst = max(worst, float(np.abs(xv[~nan] - yv[~nan]).max()))
        elif not x.astype("string").fillna("<NA>").equals(y.astype("string").fillna("<NA>")):
            return False, float("inf")
    return worst == 0.0, worst


def compare(worktree: Path, months: list[str]) -> None:
    _check_worktree(worktree)
    days = set(months_days(months))
    main_files = _day_files(ROOT / "cache", days)
    bad, missing, stale = [], [], []
    for f in main_files:
        g = worktree / f.relative_to(ROOT)
        if not g.exists():
            missing.append(str(f.relative_to(ROOT)))
            continue
        ok, worst = _same(pd.read_parquet(f), pd.read_parquet(g))
        if not ok:
            day = next(d for d in days if d in f.stem)
            pts = paths.ADSB_POINTS / f"{day}.parquet"
            # a main-checkout table older than its day's points (the points were rewritten after it, e.g. the staging
            # merge of 2 Oct) is stale there; the rebuild from the current points is the right one
            (stale if pts.exists() and f.stat().st_mtime < pts.stat().st_mtime else bad).append((str(f.relative_to(ROOT)), worst))
    print(f"per-day tables of {months}: {len(main_files)} in the main checkout, {len(missing)} not rebuilt, "
          f"{len(bad)} different, {len(stale)} different where the main checkout's table predates its day's points")
    for name, worst in bad[:12]:
        print(f"  DIFFERENT {name}: max |diff| {worst}")
    if stale:
        kinds = sorted({Path(n).parent.name for n, _ in stale})
        print(f"  stale in the main checkout (computed before the points were rewritten): {len(stale)} in {kinds}")
    for name in missing[:12]:
        print(f"  MISSING   {name}")
    mv = pd.read_parquet(paths.MOVEMENTS_FILE, columns=["MVT_TIME_UTC_mvt"])
    ids = mv.index[mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").isin(days)]
    for rel in ("cache/adsb/features3.parquet", "cache/adsb/features4.parquet"):
        a, b = pd.read_parquet(ROOT / rel), pd.read_parquet(worktree / rel)
        ok, worst = _same(a.loc[a.index.intersection(ids)], b.loc[b.index.intersection(ids)])
        print(f"{rel} rows of {months}: {'identical' if ok else f'DIFFERENT (max |diff| {worst})'} "
              f"({len(a.index.intersection(ids)):,} rows)")
        bad += [] if ok else [(rel, worst)]
    if bad or missing:
        raise SystemExit("REHEARSAL FAILED: the rebuilt tables differ from the main checkout's")
    print("REHEARSAL PASSED: the new-days chain rebuilds the stage's tables exactly"
          + (f" (and {len(stale)} intermediate tables that were stale in the main checkout)" if stale else ""))


def unlink(worktree: Path) -> None:
    for rel in JUNCTIONS:
        p = worktree / rel
        if is_junction(p):
            os.rmdir(p)   # removes the junction only
            print(f"  removed junction {p}")
    print(f"no junction left in {worktree}: `git worktree remove --force {worktree}` is now safe")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "create":
        create(Path(_arg("--worktree")).resolve())
    elif cmd == "points":
        points(Path(_arg("--tree")).resolve(), _args("--months"))
    elif cmd == "strip":
        strip(Path(_arg("--worktree")).resolve(), _args("--months"))
    elif cmd == "compare":
        compare(Path(_arg("--worktree")).resolve(), _args("--months"))
    elif cmd == "unlink":
        unlink(Path(_arg("--worktree")).resolve())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
