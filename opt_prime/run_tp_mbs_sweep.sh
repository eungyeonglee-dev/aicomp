#!/usr/bin/env bash
set -euo pipefail

# Sweep TP and MBS for layer-time profiling runs.
# - Runs inside the "optimus-prime" container (same style as run_layer_prof.sh)
# - Ensures PP*DP*TP == NPROC_PER_NODE by computing PP automatically.
# - Writes one stdout log + one gpustat log per (TP, MBS) combo.
#
# Usage (host):
#   export HF_ACCESS_TOKEN=...
#   ./run_tp_mbs_sweep.sh
#
# Optional overrides (host env):
#   NPROC_PER_NODE=8
#   DP=1
#   GBS=32
#   PROFILE_STEP=20
#   TP_LIST="1 2 4"
#   MBS_LIST="1 2 4 8"
#   SCHEDULE="1f1b"   # or gpipe
#   LOG_SUBDIR="_logs/tp_mbs_sweep"

docker exec optimus-prime bash -lc '
set -euo pipefail
cd /workspace/aicomp/opt_prime

: "${HF_ACCESS_TOKEN:?HF_ACCESS_TOKEN must be set in the container env (export it on host and pass through).}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

DP="${DP:-1}"
GBS="${GBS:-32}"
PROFILE_STEP="${PROFILE_STEP:-20}"
SCHEDULE="${SCHEDULE:-1f1b}"
LOG_SUBDIR="${LOG_SUBDIR:-_logs/tp_mbs_sweep}"

# Space-separated lists (strings) -> arrays
TP_LIST_STR="${TP_LIST:-2}"
MBS_LIST_STR="${MBS_LIST:-1}"

mkdir -p "$LOG_SUBDIR"

echo "[sweep] NPROC_PER_NODE=$NPROC_PER_NODE NNODES=$NNODES MASTER_PORT=$MASTER_PORT"
echo "[sweep] DP=$DP GBS=$GBS PROFILE_STEP=$PROFILE_STEP SCHEDULE=$SCHEDULE"
echo "[sweep] TP_LIST=($TP_LIST_STR)"
echo "[sweep] MBS_LIST=($MBS_LIST_STR)"

for TP in $TP_LIST_STR; do
  denom=$((DP * TP))
  if (( denom <= 0 )); then
    echo "[sweep] skip TP=$TP (invalid denom=$denom)"
    continue
  fi
  if (( NPROC_PER_NODE % denom != 0 )); then
    echo "[sweep] skip TP=$TP (NPROC_PER_NODE=$NPROC_PER_NODE not divisible by DP*TP=$denom)"
    continue
  fi
  PP=$((NPROC_PER_NODE / denom))
  if (( PP <= 0 )); then
    echo "[sweep] skip TP=$TP (computed PP=$PP)"
    continue
  fi

  for MBS in $MBS_LIST_STR; do
    DATE=$(date +%Y%m%d_%H%M%S)
    EXT="_TP${TP}_DP${DP}_PP${PP}_MBS${MBS}_GBS${GBS}"
    OUT_LOG="${LOG_SUBDIR}/${DATE}${EXT}_log.txt"
    GPU_LOG="${LOG_SUBDIR}/${DATE}_gpustat${EXT}.log"

    echo "[sweep] run: TP=$TP DP=$DP PP=$PP MBS=$MBS GBS=$GBS -> $OUT_LOG"

    (while true; do
        echo "===== $(date "+%F %T") ====="
        gpustat --no-color || true
        echo
        sleep 1
      done) >> "$GPU_LOG" 2>&1 &
    GPUSTAT_PID=$!

    # Run a short profiling session (prints layer timing when enabled in code)
    set +e
    torchrun --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      --nnodes="$NNODES" \
      --master_port="$MASTER_PORT" \
      examples/pp_train_llama4.py \
        --access-token "${HF_ACCESS_TOKEN}" \
        --pp-degree "$PP" --tp-degree "$TP" --dp-degree "$DP" \
        --micro-batch-size "$MBS" --batch-size "$GBS" \
        --pipeline-parallel-schedule "$SCHEDULE" \
        --profile-cut True --profile-step "$PROFILE_STEP" \
      > "$OUT_LOG" 2>&1
    RC=$?
    set -e

    kill "$GPUSTAT_PID" 2>/dev/null || true
    wait "$GPUSTAT_PID" 2>/dev/null || true

    if (( RC != 0 )); then
      echo "[sweep] WARN: run failed (rc=$RC): TP=$TP DP=$DP PP=$PP MBS=$MBS (see $OUT_LOG)"
    fi
  done
done

echo "[sweep] done. logs in: $LOG_SUBDIR"
'


