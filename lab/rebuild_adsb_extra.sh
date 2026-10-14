#!/usr/bin/env bash
# Extend every ADS-B feature to the Feb+Dec 2025 training days (findings 10.13), rebuild the pooled archives over all
# days, build the combined base, and retrain the ADS-B stage on 121 days. Run from the repo root after
# `python -m prc fetch-adsb --days <Feb+Dec 2025>` has finished. Logs in runs/rebuild_extra/<step>.log.
# Usage: bash lab/rebuild_adsb_extra.sh <winter_run_1> [<winter_run_2> ...]
set -euo pipefail
PY=.venv/Scripts/python.exe
LOG=${LOG:-runs/rebuild_extra}          # log directory; override per rebuild (LOG=runs/rebuild_240)
NAME=${NAME:-v46a_resid_121d}           # name of the refitted residual member
STACK=2026-10-01T04-48-58__blend11_stack__b942ea3d
mkdir -p "$LOG"
step() { local name=$1; shift; echo "$(date -u +%H:%M:%S) start $name"; "$@" > "$LOG/$name.log" 2>&1; echo "$(date -u +%H:%M:%S) done  $name"; }

# the extra months' days that were actually crawled (31 Dec 2025 failed twice: no readable assets in its release);
# MONTHS lists the extra months (default Feb+Dec 2025), e.g. MONTHS="2025-02 2025-12 2025-03 2025-11"
MONTHS=${MONTHS:-"2025-02 2025-12"}
NEW_DAYS=$($PY -c "from prc import paths; import sys; print(' '.join(sorted(p.stem for m in sys.argv[1:] for p in paths.ADSB_POINTS.glob(m + '-*.parquet'))))" $MONTHS)
echo "months: $MONTHS; new days: $(echo $NEW_DAYS | wc -w)"
ALL_DAYS=$($PY -c "from prc import paths; print(' '.join(sorted(p.stem for p in paths.ADSB_POINTS.glob('*.parquet'))))")

# per-day features: the scripts skip days that already exist, so only the new days are computed
step links           $PY lab/adsb_links.py --workers 4
step set1            $PY lab/adsb_features.py --workers 6
step set2            $PY lab/adsb_features2.py --workers 6
step pushback        $PY lab/ideas/pushback.py --workers 2 --days $NEW_DAYS
step first_reception $PY lab/ideas/first_reception.py days --workers 2
step first_rec_asm   $PY lab/ideas/first_reception.py assemble
step track_quality   $PY lab/ideas/track_quality.py --workers 2
step stop_class      $PY lab/ideas/stop_class.py --workers 2
step features3       $PY lab/adsb_features3.py
# pooled archives change with the new days: rebuild them and every day's estimates
step stand_archive   $PY lab/ideas/stand_archive.py obs archive days --force --workers 2
step timemap_extract $PY lab/ideas/timemap.py extract --workers 2
step timemap_arr     $PY lab/ideas/timemap.py arr --workers 2
step timemap_map     $PY lab/ideas/timemap.py map
step timemap_arrmap  $PY lab/ideas/timemap.py arrmap
step timemap_days    $PY lab/ideas/timemap.py days --force --workers 2
step timemap_asm     $PY lab/ideas/timemap.py assemble
step features4       $PY lab/adsb_features4.py
# the combined base and the ADS-B stage on 121 days (CPU; the direct model on the GPU)
step winter_base     $PY lab/build_winter_base.py $STACK "$@"
BASE=$(ls -td runs/*__base_plus_winter__* | head -1 | xargs basename)
echo "combined base: $BASE"
if [ -n "${SKIP_REFIT:-}" ]; then echo "SKIP_REFIT set: base built, no refit"; exit 0; fi
step refit_resid     $PY lab/adsb_apply.py $BASE --depth 8 --iters 3000 --lr 0.03 --regime --v5 --drop-prefix tarr_,it_ --write $NAME
grep -E "days (base|corrected x1.0)|projected|wrote" "$LOG/refit_resid.log"
