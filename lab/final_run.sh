#!/usr/bin/env bash
# The final files (findings 10.32, 10.33): the challenge model (v70's recipe) on the final ranking set (January,
# February, June and July 2026; the recorded times jittered by up to 12 s) -- the scored file -- and the two
# operational approaches beside it (not scored). Every step was rehearsed on the delivered Jan+Jul set. Start it when
# the machine is free; it resumes: every step re-uses what an earlier start made, and a second start never runs a
# second GPU phase beside a live one.
#   G  (GPU, a background job with a lock): the five stage members of v70 refitted once on every covered 2025 day and
#      saved (lab/adsb_apply.py --no-cv --save-models; v55c on the CPU beside the GPU members), then the long-horizon
#      deliverables refitted (approach 1: planning data and history, 3 h and 24 h ahead); with FINAL_R6=1 afterwards
#      the ADS-B surface rung R6 (approach 2's variant; nothing waits for it)
#   C1 (CPU, this checkout): the final data tree, the release validated and installed, prepare, the 27 base members
#      of blend11_window_lobt predicted with their saved refit models, the blends, the stack (lab/final_inputs.py runbook)
#   C2 (CPU, the isolated checkout of lab/final_worktree.py): ADS-B points for 2025 and the four months, the four
#      months' tables stripped whenever the release's data.version changes (the Jan+Jul tables of the delivered set
#      were linked to the unjittered times), the per-day tables of the 120 days against the frozen pools
#   F  (CPU, isolated checkout, once G's stage members exist): lab/final_challenge.py --stage-models -> the final run,
#      submitted at once (the scored file)
#   O  (CPU, once G's long refits exist): the operational approaches predicted on the final tree (approach 2: op_h1_r4,
#      15 min before the planned off-block; approach 1: op_hl3_r4_rwyblind, op_hl24_r4_rwyblind), written as files
#      beside the scored one; the scored file is submitted again last, so submissions/submitting.parquet is its file
# Needs: the release in data_release/2026-10-09/, the isolated checkout ../taxiout-finalrun (python lab/final_worktree.py
# create --worktree ../taxiout-finalrun), and no uncommitted change under prc/, lab/ or configs/ (the isolated checkout
# runs the committed code). Windows with Git Bash as tested; on Linux the interpreter is .venv/bin/python.
# Launch (PowerShell, detached; CHAIN_START delays the start, e.g. "2026-10-11 19:00"):
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "lab/final_run.sh" -WorkingDirectory <checkout>
#     -WindowStyle Hidden -RedirectStandardOutput runs\final_run.out -RedirectStandardError runs\final_run.err
set -u
cd "$(dirname "$0")/.."
wpwd() { pwd -W 2>/dev/null || pwd; }   # C:/... in Git Bash: Python and PRC_DATA_DIR get Windows paths
MAIN=$(wpwd)
unset PRC_DATA_DIR PRC_EXPECTED_RANK_ROWS   # G trains on the delivered tree; every later step names its tree itself
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
LOG="$MAIN/runs/final_run.log"
RELEASE="$MAIN/data_release/2026-10-09"
TREE="$MAIN/data_final"
WT="$(cd "$MAIN/.." && wpwd)/taxiout-finalrun"
MONTHS="2026-01 2026-02 2026-06 2026-07"
BLEND11=2026-10-01T04-48-58__blend11_window_lobt__f13b7e97
BASE=2026-10-01T04-48-58__base_plus_winter__80cbb34d
OP_H1_R4=2026-10-08T02-37-29__op_h1_r4__3fcdc779__s42   # the refit behind v68 (approach 2)
G_PIDFILE="$MAIN/runs/.final_gpu.pid"
say() { echo "$(date -u +%Y-%m-%d\ %H:%M) [final] $*" | tee -a "$LOG"; }
winpid() { cat "/proc/$1/winpid" 2>/dev/null || echo "$1"; }
alive() { [ -n "$1" ] && { tasklist //FI "PID eq $1" 2>/dev/null | grep -q " $1 " || kill -0 "$1" 2>/dev/null; }; }
stage_run() {   # the complete stage run named $1 (metrics.json and the saved models written last); moves a partial aside
  local d
  for d in $(ls -d "$MAIN"/runs/*__"$1"__* 2>/dev/null | sort -r); do
    if [ -f "$d/metrics.json" ] && [ -f "$d/stage_models/stage_models.json" ]; then basename "$d"; return; fi
    mkdir -p "$MAIN/runs/_partial" && mv "$d" "$MAIN/runs/_partial/" && say "moved the partial run $(basename "$d") to runs/_partial/"
  done
}
refit_run() {   # the newest refitted run of config $1 under its current hash (empty if none)
  local key d
  key=$($PY -c "from prc.config import load_config; c = load_config('$1.yaml'); print(f'{c.name}__{c.hash()}')")
  for d in $(ls -d "$MAIN"/runs/*__"$key"__s* 2>/dev/null | sort -r); do
    [ -f "$d/test_predictions.parquet" ] && [ -f "$d/refit.json" ] && { basename "$d"; return; }
  done
}

[ -f "$RELEASE/final_ranking.parquet" ] && [ -f "$RELEASE/final_submitting.parquet" ] || { say "the release is not in $RELEASE"; exit 1; }
[ -f "$WT/cache/final_worktree.json" ] || { say "no isolated checkout at $WT (python lab/final_worktree.py create --worktree $WT)"; exit 1; }
[ -z "$(git status --porcelain -- prc lab configs)" ] || { say "uncommitted changes under prc/, lab/ or configs/: commit first (the isolated checkout runs the committed code)"; exit 1; }
START=$(date -u -d "${CHAIN_START:-now}" +%s)
say "armed (bash $$, windows pid $(winpid $$)); start $(date -u -d @$START +%H:%M) UTC"
while [ "$(date -u +%s)" -lt "$START" ]; do sleep 30; done
git -C "$WT" checkout -q --detach "$(git rev-parse HEAD)" && say "isolated checkout at $(git -C "$WT" rev-parse --short HEAD)"

# --- G: GPU (a background job, one at a time) ------------------------------------------------------------------------
gpu_phase() {
  local RES="--regime --v5 --drop-prefix tarr_,it_" DIR="--direct --rich --v5 --drop-prefix tarr_,it_" cpu cfg
  stage() {   # name v70-member args...: one stage member refitted on every covered 2025 day and saved
    local name=$1 member=$2; shift 2
    if [ -n "$(stage_run "$name")" ]; then say "G have  $name"; return; fi
    say "G start $name"
    $PY lab/adsb_apply.py "$BASE" "$@" --no-cv --save-models --oof-from "$member" --write "$name" > "runs/$name.out" 2>&1
    say "G done  $name: $(grep -E 'wrote|ranking rows|Traceback|Error' "runs/$name.out" | tail -2 | tr '\n' ' ' | cut -c1-220)"
  }
  stage v70s_v55c 2026-10-01T04-48-58__v55c_resid_360d_d8_6000__59e75cb5 --depth 8 --iters 6000 --lr 0.02 $RES &
  cpu=$!
  stage v70s_v55g 2026-10-01T04-48-58__v55g_resid_360d_d10__f7e949fa --depth 10 --iters 5000 --lr 0.02 $RES --gpu
  stage v70s_v58g 2026-10-01T04-48-58__v58g_resid_360d_d10_s142__e6924d0b --depth 10 --iters 5000 --lr 0.02 $RES --gpu --seed-offset 100
  stage v70s_v55d 2026-10-01T04-48-58__v55d_direct_360d_d10_8000__3ca2df95 $DIR --depth 10 --iters 8000 --lr 0.02 --gpu
  stage v70s_v58d 2026-10-01T04-48-58__v58d_direct_360d_d10_8000_s142__15a9c9a9 $DIR --depth 10 --iters 8000 --lr 0.02 --gpu --seed-offset 100
  wait "$cpu"
  touch "$MAIN/runs/.final_stage_done"
  say "G stage members done"
  for cfg in op_hl3_r4_rwyblind op_hl24_r4_rwyblind; do   # approach 1: refit_on_all, so no --no-test
    if [ -n "$(refit_run "$cfg")" ]; then say "G have  $cfg refit"; continue; fi
    say "G start $cfg (holdout + refit)"
    $PY -m prc train -c "$cfg.yaml" > "runs/$cfg.final.out" 2>&1
    say "G done  $cfg: $(grep -E 'artifacts|Traceback|Error' "runs/$cfg.final.out" | tail -1 | cut -c1-200)"
  done
  touch "$MAIN/runs/.final_long_done"
  if [ -n "${FINAL_R6:-}" ]; then
    say "G start the R6 rungs (lab/op_ladder_run.sh)"
    bash lab/op_ladder_run.sh "$(date -u -d '+12 hours' '+%Y-%m-%d %H:%M')" op_h1_r6 op_h1_r6_s43 op_h1_r6_shift7 2>&1 | sed 's/^/    /' >> "$LOG"
  fi
  say "G finished"
}
GWIN=$(cat "$G_PIDFILE" 2>/dev/null || true)
if alive "$GWIN"; then
  say "G is already running (windows pid $GWIN): not started again"
else
  rm -f "$MAIN/runs/.final_stage_done" "$MAIN/runs/.final_long_done"
  gpu_phase &
  GWIN=$(winpid $!)
  echo "$GWIN" > "$G_PIDFILE"
  say "G started (windows pid $GWIN)"
fi
wait_for() {   # marker: wait until G wrote it; fail if G is gone without it
  while [ ! -f "$MAIN/runs/$1" ]; do
    alive "$GWIN" || { [ -f "$MAIN/runs/$1" ] || { say "G ended without $1"; return 1; }; }
    sleep 60
  done
}

# --- C1: the final tree and the base (runbook) -----------------------------------------------------------------------
say "C1 start: runbook for $MONTHS"
$PY lab/final_inputs.py runbook --tree "$TREE" --from-blend "$BLEND11" --months $MONTHS --out runs/final_runbook.sh \
  --shell sh --threads 4 >> "$LOG" 2>&1 || { say "C1 FAILED to write the runbook"; exit 1; }
bash runs/final_runbook.sh "$RELEASE/final_ranking.parquet" "$RELEASE/final_submitting.parquet" > runs/final_runbook.out 2>&1 \
  || { say "C1 FAILED: $(tail -3 runs/final_runbook.out | tr '\n' ' ' | cut -c1-300)"; exit 1; }
VERSION=$($PY lab/final_inputs.py derive version --tree "$TREE" | tail -1)
ROWS=$($PY lab/final_inputs.py derive rank-rows --tree "$TREE" | tail -1)
B02=$(grep -E '^  final_b02 = ' runs/final_runbook.out | tail -1 | awk '{print $3}')
STACK=$(grep -E '^  final_stack = ' runs/final_runbook.out | tail -1 | awk '{print $3}')
[ -n "$VERSION" ] && [ -n "$B02" ] && [ -n "$STACK" ] || { say "C1 FAILED: no version, blend or stack (runs/final_runbook.out)"; exit 1; }
say "C1 done: data.version $VERSION, $ROWS ranking rows, blend $B02, stack $STACK"

# --- C2: the ADS-B tables of the four months (isolated checkout) -----------------------------------------------------
$PY lab/final_worktree.py points --tree "$TREE" --months $MONTHS >> "$LOG" 2>&1 || { say "C2 FAILED: points"; exit 1; }
if [ "$(cat "$WT/cache/.final_stripped" 2>/dev/null)" != "$VERSION" ]; then   # tables of another release (or none)
  $PY lab/final_worktree.py strip --worktree "$WT" --months $MONTHS >> "$LOG" 2>&1 || { say "C2 FAILED: strip"; exit 1; }
  echo "$VERSION" > "$WT/cache/.final_stripped"
  say "C2 stripped the four months' per-day tables (release $VERSION)"
fi
say "C2 start: per-day ADS-B tables (frozen pools)"
( cd "$WT" && PRC_DATA_DIR="$TREE" LOG=runs/adsb_new_days_final bash lab/adsb_new_days.sh ) > runs/adsb_new_days_final.out 2>&1 \
  || { say "C2 FAILED: $(tail -3 runs/adsb_new_days_final.out | tr '\n' ' ' | cut -c1-300)"; exit 1; }
say "C2 done: $(tail -1 runs/adsb_new_days_final.out | cut -c1-200)"

# --- F: the challenge model on the final set (the scored file) -------------------------------------------------------
wait_for .final_stage_done || exit 1
IDS=()
for k in v55g v55c v58g v55d v58d; do
  id=$(stage_run "v70s_$k"); [ -n "$id" ] || { say "F FAILED: v70s_$k missing or incomplete"; exit 1; }; IDS+=("$k=$id")
done
say "F start: lab/final_challenge.py with ${IDS[*]}"
( cd "$WT" && $PY lab/final_challenge.py --tree "$TREE" --data-version "$VERSION" --blend "$B02" \
    --stack "$STACK" --name final_v70 --stage-models "${IDS[@]}" --threads 4 ) > runs/final_challenge_final.out 2>&1 \
  || { say "F FAILED: $(tail -3 runs/final_challenge_final.out | tr '\n' ' ' | cut -c1-300)"; exit 1; }
FINAL=$(grep -E '^final ' runs/final_challenge_final.out | tail -1 | awk '{print $2}' | tr -d ':')
[ -n "$FINAL" ] || { say "F FAILED: no final run in runs/final_challenge_final.out"; exit 1; }
PRC_DATA_DIR="$TREE" $PY -m prc submit --run "$FINAL" >> "$LOG" 2>&1 || { say "SUBMIT FAILED for $FINAL"; exit 1; }
say "F done: $FINAL, submitted (submissions/${FINAL}__submitting.parquet)"

# --- O: the operational approaches (files beside the scored one) -----------------------------------------------------
if wait_for .final_long_done; then
  for src in "$OP_H1_R4" "$(refit_run op_hl3_r4_rwyblind)" "$(refit_run op_hl24_r4_rwyblind)"; do
    [ -n "$src" ] || { say "O skip: a deliverable has no refit"; continue; }
    name="final_$(echo "$src" | awk -F'__' '{print $2}')"
    run=$($PY lab/final_inputs.py latest "$name" --source "$src" --data-version "$VERSION" --missing-ok | tail -1)
    if [ -z "$run" ]; then
      PRC_DATA_DIR="$TREE" $PY -m prc predict --run "$src" -o "data.version=$VERSION" --tree "$TREE" --threads 4 \
        --name "$name" > "runs/$name.out" 2>&1 || { say "O FAILED $name: $(tail -2 "runs/$name.out" | tr '\n' ' ' | cut -c1-250)"; continue; }
      run=$($PY lab/final_inputs.py latest "$name" --source "$src" --data-version "$VERSION" | tail -1)
    fi
    PRC_DATA_DIR="$TREE" $PY -m prc submit --run "$run" >> "$LOG" 2>&1 && say "O file submissions/${run}__submitting.parquet"
  done
else
  say "O skipped: the long-horizon refits are missing"
fi

# --- the scored file, last (prc submit also copies to submissions/submitting.parquet) -------------------------------
PRC_DATA_DIR="$TREE" $PY -m prc submit --run "$FINAL" >> "$LOG" 2>&1 || { say "SUBMIT FAILED for $FINAL"; exit 1; }
a=$(sha1sum "submissions/${FINAL}__submitting.parquet" | cut -c1-40); b=$(sha1sum submissions/submitting.parquet | cut -c1-40)
[ "$a" = "$b" ] && say "the scored file submissions/submitting.parquet = $FINAL (sha1 ${a:0:12})" || say "WARNING: submitting.parquet is not $FINAL's file"
PRC_DATA_DIR="$TREE" $PY -m prc levelcheck --run "$FINAL" >> "$LOG" 2>&1
say "finished"
