#!/usr/bin/env bash
# The ADS-B stage's per-day tables for new days (the final set: January, February, June and July 2026, the Jan+Jul
# days' tables stripped first with lab/final_worktree.py strip; findings 10.32, 10.33) with the pooled archives
# FROZEN: the per-day steps of lab/rebuild_adsb_extra.sh for every day that has ADS-B points in the tree and no table
# yet (links, feature sets 1-2, pushback, first reception, track quality, stop classes, the stand archive's and the
# time-to-location map's per-day estimates), then the assembled tables (features3, features4). Neither the stand
# archive nor the map is rebuilt (no `archive`, no `map`/`arrmap`, no --force), so every existing day keeps the values
# the stage models were trained on, and the new days are read against the archives of the training period (2025 and
# Jan+Jul 2026: the 412 days of the submitted stage).
#
# Run it in the isolated checkout of lab/final_worktree.py (its cache/adsb and cache/lab/ideas are private copies:
# the assembled tables are rewritten over the tree's days), with PRC_DATA_DIR set to the tree whose movements the new
# days link to. The tree's additional_data/adsb/points must hold exactly the days to serve (training days and the
# ranking months): a day with points but without movements would get empty tables, which every later start would
# skip as done. Logs: runs/adsb_new_days/<step>.log.
# Usage: bash lab/adsb_new_days.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=${LOG:-runs/adsb_new_days}
mkdir -p "$LOG"
step() { local name=$1; shift; echo "$(date -u +%H:%M:%S) start $name"; "$@" > "$LOG/$name.log" 2>&1; echo "$(date -u +%H:%M:%S) done  $name"; }

[ -n "${PRC_DATA_DIR:-}" ] || { echo "PRC_DATA_DIR is not set: point it at the tree whose movements the days link to" >&2; exit 2; }
$PY - <<'EOF'
from prc import paths
days = sorted(p.stem for p in paths.ADSB_POINTS.glob("*.parquet"))
missing = [d for d in days if not (paths.CACHE / "adsb" / f"links_{d}.parquet").exists()]
print(f"tree {paths.DATA}: {len(days)} days with ADS-B points, {len(missing)} without tables: "
      f"{missing[:3]}{' ...' if len(missing) > 3 else ''}{missing[-1:] if len(missing) > 3 else ''}")
if not days:
    raise SystemExit(f"no ADS-B points in {paths.ADSB_POINTS}")
if not (paths.CACHE / "final_worktree.json").exists():
    raise SystemExit(f"{paths.CACHE} is not the cache of an isolated checkout (lab/final_worktree.py create): the "
                     "assembled tables would replace the delivered set's")
EOF
step links           $PY lab/adsb_links.py --workers 4
step set1            $PY lab/adsb_features.py --workers 6
step set2            $PY lab/adsb_features2.py --workers 6
NEW_PUSH=$($PY -c "
from prc import paths
d = paths.CACHE / 'lab' / 'ideas' / 'pushback_days'
print(' '.join(p.stem for p in sorted(paths.ADSB_POINTS.glob('*.parquet')) if not (d / (p.stem + '.parquet')).exists()))")
if [ -n "$NEW_PUSH" ]; then step pushback $PY lab/ideas/pushback.py --workers 2 --days $NEW_PUSH; else echo "pushback: every day done"; fi
step first_reception $PY lab/ideas/first_reception.py days --workers 2
step first_rec_asm   $PY lab/ideas/first_reception.py assemble
step track_quality   $PY lab/ideas/track_quality.py --workers 2
step stop_class      $PY lab/ideas/stop_class.py --workers 2
step features3       $PY lab/adsb_features3.py
step stand_obs_days  $PY lab/ideas/stand_archive.py obs days --workers 2
step timemap_extract $PY lab/ideas/timemap.py extract --workers 2
step timemap_arr     $PY lab/ideas/timemap.py arr --workers 2
step timemap_days    $PY lab/ideas/timemap.py days --workers 2
step timemap_asm     $PY lab/ideas/timemap.py assemble
step features4       $PY lab/adsb_features4.py
echo "$(date -u +%H:%M:%S) per-day tables complete for every day with points in $PRC_DATA_DIR"
