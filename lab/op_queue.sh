#!/usr/bin/env bash
# The GPU queue of the operational deliverable (findings 10.35, 11-13 Oct 2026). It runs the steps of runs/op_queue.txt
# one at a time, in file order, each once, and re-reads the file after every step: steps are appended, inserted or
# reordered without touching this script (write the file whole: a temporary file, then mv). With no step pending it
# waits for new ones; it ends when runs/op_queue.stop exists (checked between steps) or after IDLE_H hours idle.
# Restart-safe: a step listed in runs/op_queue.done never runs again (delete its line there to rerun it), and a second
# start while one is alive exits at once (runs/.op_queue.pid).
#   bash lab/op_queue.sh        (detached, from the repository root, e.g. via Git\bin\bash.exe from PowerShell)
# A step line:   ID | BUDGET_MIN | KIND | COMMAND
#   KIND gate   a failure stops the queue (nothing trains on a feature set that failed a leakage gate);
#   KIND run    a failure is logged and the queue goes on.
# COMMAND runs under `bash -c` from the repository root with $PY set; its output goes to runs/opq_<ID>.out. A step is
# killed after 2.5 x BUDGET + 10 min (a hung fit must not hold the queue). Blank lines and lines starting with # are
# ignored. Never run a fit, `prc features` or `prc predict` beside the queue: they share the CPU with its steps.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python
export PY PYTHONUNBUFFERED=1 PYTHONUTF8=1
unset PRC_DATA_DIR PRC_EXPECTED_RANK_ROWS   # every step names its tree itself (the final-set steps set PRC_DATA_DIR)
Q=runs/op_queue.txt
DONE=runs/op_queue.done
LOG=runs/op_queue.log
PIDFILE=runs/.op_queue.pid
STOP=runs/op_queue.stop
IDLE_H=${IDLE_H:-14}
say() { echo "$(date -u +%m-%dT%H:%M) [queue] $*" | tee -a "$LOG"; }
winpid() { cat "/proc/$1/winpid" 2>/dev/null || echo "$1"; }
alive() { [ -n "$1" ] && { tasklist //FI "PID eq $1" 2>/dev/null | grep -q " $1 " || kill -0 "$1" 2>/dev/null; }; }

if [ -f "$PIDFILE" ] && alive "$(cat "$PIDFILE")"; then
  echo "queue already running (pid $(cat "$PIDFILE"))"; exit 0
fi
winpid $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
touch "$DONE" "$Q"
rm -f "$STOP"
say "started (bash $$, windows pid $(winpid $$)); queue $Q"

trim() { local s=$1; s=${s#"${s%%[![:space:]]*}"}; echo "${s%"${s##*[![:space:]]}"}"; }
next_step() {   # prints the first pending line of the queue file, or nothing
  local line id
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    [ -z "$(trim "$line")" ] && continue
    case $(trim "$line") in \#*) continue ;; esac
    id=$(trim "${line%%|*}")
    grep -q "^$id " "$DONE" && continue
    echo "$line"; return
  done < "$Q"
}

idle_since=$(date -u +%s)
while :; do
  if [ -f "$STOP" ]; then say "stop file found: queue ends"; rm -f "$STOP"; break; fi
  line=$(next_step)
  if [ -z "$line" ]; then
    if [ $(( $(date -u +%s) - idle_since )) -ge $(( IDLE_H * 3600 )) ]; then say "idle for ${IDLE_H} h: queue ends"; break; fi
    sleep 60; continue
  fi
  IFS='|' read -r id budget kind cmd <<< "$line"
  id=$(trim "$id"); budget=$(trim "$budget"); kind=$(trim "$kind"); cmd=$(trim "$cmd")
  case $budget in ''|*[!0-9]*) say "bad budget in line for $id: skipped"; echo "$id BAD $(date -u +%FT%H:%M) -" >> "$DONE"; continue ;; esac
  limit=$(( budget * 5 / 2 + 10 ))
  say "start $id ($kind, budget ${budget} min, killed after ${limit} min): $cmd"
  start=$(date -u +%s)
  timeout --kill-after=120 "${limit}m" bash -c "$cmd" > "runs/opq_$id.out" 2>&1
  rc=$?
  mins=$(( ($(date -u +%s) - start + 59) / 60 ))
  echo "$id $rc $(date -u -d @"$start" +%FT%H:%M) $(date -u +%FT%H:%M)" >> "$DONE"
  tail_txt=$(grep -E 'overall |overall_bulk|artifacts|"choice"|"tripped"|Traceback|Error|passed|failed' "runs/opq_$id.out" | tail -4 | tr -d '\r' | tr '\n' ' ' | cut -c1-300)
  if [ $rc -eq 0 ]; then
    say "done  $id in ${mins} min: $tail_txt"
  else
    [ $rc -eq 124 ] && say "KILLED $id after ${limit} min (budget ${budget})"
    say "FAIL  $id (exit $rc) in ${mins} min: $(tail -3 "runs/opq_$id.out" | tr -d '\r' | tr '\n' ' ' | cut -c1-300)"
    if [ "$kind" = gate ]; then say "gate $id failed: queue stops"; break; fi
  fi
  idle_since=$(date -u +%s)
done
say "ended"
