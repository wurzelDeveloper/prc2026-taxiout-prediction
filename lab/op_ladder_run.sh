#!/usr/bin/env bash
# The operational ladder on real data (findings 10.25; spec in configs/op_base.yaml): the gates, the cache warm, a
# smoke fit, then the rungs on the GPU, each started only if it is expected to end before DEADLINE.
# Run from the main checkout after operational_ladder is merged, with no other GPU job and no heavy CPU job:
#   bash lab/op_ladder_run.sh "2026-10-08 05:00" [rung ...]
# Without rung names it runs the default order below. Every gate that fails stops the script (nothing is trained
# on a feature set that failed a leakage gate). Logs: runs/op_ladder.log, one runs/op_*.out per step.
# The ADS-B rungs (op_h1_r6*, op_h1_r2_adsb, op_h0_*_adsb) also need the ADS-B gates of the snapshot they read
# ($PRC_ADSB_SNAPSHOT, default s1; built, supported and frozen): the three ADS-B test files, the real slice
# (OPLAD_REAL=1, must pass, not skip) and lab/op_adsb_audit.py exiting 0. They run once per snapshot content
# (runs/op_gates_adsb_<snapshot>_<events sha1>_<support sha1>.passed); the general gates' marker does not cover them.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
DEADLINE=$(date -u -d "$1" +%s); shift
LOG=runs/op_ladder.log
say() { echo "$(date -u +%H:%M) [ladder] $*" | tee -a "$LOG"; }
fits() { local minutes=$1; [ $(( $(date -u +%s) + minutes * 60 )) -le "$DEADLINE" ]; }
gate() {    # name command...
  local name=$1; shift
  say "gate  $name"
  if "$@" > "runs/op_gate_$name.out" 2>&1; then
    say "pass  $name: $(tail -1 runs/op_gate_$name.out | cut -c1-160)"
  else
    say "FAIL  $name: $(tail -3 runs/op_gate_$name.out | tr '\n' ' ' | cut -c1-300)"
    say "ladder stopped at gate $name"
    exit 1
  fi
}
train() {   # minutes config extra-args...
  local minutes=$1 cfg=$2; shift 2
  if fits "$minutes"; then
    say "start $cfg $*"
    $PY -m prc train -c "$cfg.yaml" "$@" > "runs/op_$cfg.out" 2>&1
    say "done  $cfg: $(grep -E 'overall |overall_bulk|artifacts|Traceback|Error' runs/op_$cfg.out | tr '\n' ' ' | cut -c1-260)"
  else
    say "skip  $cfg (would end after the deadline)"
  fi
}
passed_pytest() {   # pytest args...: fails unless the tests ran and passed (a skipped real slice is no gate)
  local out rc
  out=$($PY -m pytest -q "$@" 2>&1); rc=$?
  echo "$out"
  [ $rc -eq 0 ] && echo "$out" | grep -qE '[0-9]+ passed'
}
adsb_real_slice() { OPLAD_REAL=1 passed_pytest tests/test_operational_adsb.py -k real_slice; }
adsb_gates() {  # once per frozen snapshot content: the ADS-B tests, the real slice, the own-contamination audit
  local snap=${PRC_ADSB_SNAPSHOT:-s1} digest marker
  digest=$($PY -c "
from prc.adsb import surface
m = surface.read_manifest('$snap')
c, s = m.get('consolidated') or {}, m.get('support') or {}
print(f\"{c.get('sha1')}_{s.get('sha1')}\" if m.get('frozen') and c.get('sha1') and s.get('sha1') else 'unfrozen')")
  if [ "$digest" = unfrozen ] || [ -z "$digest" ]; then
    say "ADS-B snapshot $snap is not built, supported and frozen (prc adsb-surface: build, --check, --support, --freeze)"
    say "ladder stopped before the ADS-B rungs"
    exit 1
  fi
  marker="runs/op_gates_adsb_${snap}_${digest}.passed"
  if [ -f "$marker" ]; then say "ADS-B gates passed earlier for $snap ($digest)"; return; fi
  gate adsb_tests passed_pytest tests/test_adsb_surface.py tests/test_operational_adsb.py tests/test_op_adsb_audit.py
  gate adsb_real adsb_real_slice
  gate adsb_audit $PY lab/op_adsb_audit.py --snapshot "$snap"
  date -u > "$marker"
}

say "ladder run started (deadline $(date -u -d @$DEADLINE +%H:%M) UTC)"
if [ ! -f runs/op_gates.passed ]; then
  gate tests $PY -m pytest -q tests/test_operational.py tests/test_operational_hind.py tests/test_op_ladder_report.py tests/test_pipeline.py
  gate anchor env OPLAD_FULL=1 $PY -m pytest -q tests/test_operational.py::test_anchor_sanity_on_2025
  gate realslice env OPLAD_REAL=1 OPLAD_REAL_T2_ROWS=40 $PY -m pytest -q tests/test_operational.py::test_real_slice_t1_t1b_t2
  gate warm_h1 $PY -m prc features -c op_h1_r5.yaml -o "features.exclude=[op_h1_nm]"
  gate warm_h0 $PY -m prc features -c op_h0_r5.yaml -o "features.exclude=[op_h0_nm]"
  gate smoke $PY -m prc train -c op_h1_r4.yaml -o data.sample_frac=0.05 model.params.iterations=300 evaluation.permutation.enabled=false --no-test
  # the MAE rung is optional: a GPU that rejects the MAE loss drops it instead of stopping the ladder
  if $PY -m prc train -c op_h1_r4_mae.yaml -o data.sample_frac=0.05 model.params.iterations=300 --no-test > runs/op_gate_smoke_mae.out 2>&1; then
    say "pass  smoke_mae"
  else
    say "FAIL  smoke_mae (op_h1_r4_mae dropped): $(tail -2 runs/op_gate_smoke_mae.out | tr -d '\r' | paste -sd ' ' | cut -c1-200)"
    touch runs/op_skip_mae
  fi
  date -u > runs/op_gates.passed
else
  say "gates passed earlier ($(cat runs/op_gates.passed)); skipping them"
fi

RUNGS=("$@")
[ ${#RUNGS[@]} -eq 0 ] && RUNGS=(op_h1_r0 op_h1_r1 op_h1_r2 op_h1_r3 op_h1_r4 op_h1_r5 op_h0_r5 op_h1_r4_rwyblind op_h1_r2_winter op_h1_r4_mae op_h0_r5_hind)
for cfg in "${RUNGS[@]}"; do
  if [ "$cfg" = op_h1_r4_mae ] && [ -f runs/op_skip_mae ]; then say "skip  op_h1_r4_mae (smoke failed)"; continue; fi
  key=$($PY -c "from prc.config import load_config; c = load_config('$cfg.yaml'); print(f'{c.name}__{c.hash()}')")
  if ls runs/*__"$key"__s*/metrics.json > /dev/null 2>&1 && [ -z "${FORCE:-}" ]; then
    say "have  $cfg ($key; FORCE=1 reruns)"
    continue
  fi
  case $cfg in   # the ADS-B rungs only after the ADS-B gates of the snapshot they read (critique F2; review of 9 Oct)
    op_h1_r6*|op_h1_r2_adsb|op_h0_*_adsb) adsb_gates ;;
  esac
  case $cfg in
    op_h1_r4) train 110 op_h1_r4 ;;                 # the board rung: refit on all twelve months, permutation on
    op_h1_r6) train 130 op_h1_r6 ;;                 # R4 + ADS-B: refit on all twelve months, permutation on
    op_h0_r4_adsb) train 60 op_h0_r4_adsb --no-test ;;    # LAB, permutation on
    op_h0_r5) train 60 op_h0_r5 --no-test ;;        # permutation on
    op_h1_r4_mae) train 45 op_h1_r4_mae --no-test ;;      # MAE loss and early stopping (metric sensitivity), no refit
    op_h0_r5_hind) train 60 op_h0_r5_hind --no-test ;;    # LAB, HINDSIGHT (own-taxi counts), permutation on, never a board run
    *)        train 45 "$cfg" --no-test ;;
  esac
done
say "ladder run finished"
