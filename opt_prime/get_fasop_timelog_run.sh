#!/usr/bin/env bash
set -euo pipefail

# Run OptPrime llama training for configs listed in FASOP csv (rank 1~5),
# collect per-step component/layer timing logs (FX instrumentation),
# and write a results csv with log pointers.
#
# This script is derived from get_fasop_run.sh but is specialized for:
#   - profile-mode=0
#   - log-level=1
#   - FX layer/component timing window (default steps 51~60)
#
# IMPORTANT:
# - Uses examples/pp_train_llama.py because it prints FX layer/component timing summaries.
# - For FX instrumentation, we set:
#     OPTPRIME_FX_LAYER_TIMING=1
#     OPTPRIME_FX_LAYER_START_STEP / OPTPRIME_FX_LAYER_END_STEP
#
# Usage:
#   bash get_fasop_timelog_run.sh <DATE> [INPUT_CSV] [OUT_CSV]
#
# Example:
#   DATE=$(date +%Y%m%d_%H%M%S)
#   bash get_fasop_timelog_run.sh "$DATE" /home/ieg95/workspace/FASOP/main_logs/llama_evenly.csv ./_logs/${DATE}_fasop_rank1-5_timelog.csv

DATE="${1:-$(date +%Y%m%d_%H%M%S)}"
INPUT_CSV="${2:-/workspace/FASOP/main_logs/llama_evenly_mbs_test.csv}"
OUT_CSV="${3:-./_logs/${DATE}_fasop_rank1-5_timelog.csv}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRYNUMBER="${TRYNUMBER:-1}"
GBS_DEFAULT="${GBS_DEFAULT:-32}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29500}"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-{your_huggingface_token_here}}"

# Fixed per request
PROFILE_MODE="0"
LOG_LEVEL="1"

# Keep training short but ensure we include end_step+1 to stop capture windows if enabled.
PROFILE_CUT="${PROFILE_CUT:-True}"
PROFILE_STEP="${PROFILE_STEP:-65}"

# FX layer/component timing window (1-based inclusive)
LAYER_START_STEP="${LAYER_START_STEP:-51}"
LAYER_END_STEP="${LAYER_END_STEP:-60}"
LAYER_TIMING_MODE="${LAYER_TIMING_MODE:-fx}"   # fx|wrap

LOG_DIR="${LOG_DIR:-./_logs}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[error] input csv not found: $INPUT_CSV" >&2
  echo "        If you're running inside docker, bind-mount /home/ieg95/workspace/FASOP into the container," >&2
  echo "        or copy the csv under /workspace/aicomp and pass that path." >&2
  exit 1
fi

echo "[info] DATE=$DATE"
echo "[info] INPUT_CSV=$INPUT_CSV"
echo "[info] OUT_CSV=$OUT_CSV"
echo "[info] NPROC_PER_NODE=$NPROC_PER_NODE TRYNUMBER=$TRYNUMBER GBS_DEFAULT=$GBS_DEFAULT"
echo "[info] profile-mode=$PROFILE_MODE log-level=$LOG_LEVEL"
echo "[info] layer window: ${LAYER_START_STEP}..${LAYER_END_STEP} (mode=${LAYER_TIMING_MODE})"

# Extract configs (rank 1..5) from csv robustly (handles quoted fields w/ commas),
# keep only configs with PP>1, and cap to at most 5 configs.
# Output format (tab-separated):
#   rank \t mbs \t tp \t dp \t pp \t fasop_step_time_s
mapfile -t CONFIG_LINES < <(
python3 - <<'PY' "$INPUT_CSV"
import csv, sys
from collections import OrderedDict

path = sys.argv[1]
out = OrderedDict()  # (rank,mbs,tp,dp,pp)->fasop_step_time_s (keep first)
limit = 10

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rank = int(float((row.get("rank", "") or "").strip()))
        except Exception:
            continue

        try:
            mbs = int(float(row["mbs"]))
            tp = int(float(row["tp"]))
            dp = int(float(row["dp"]))
            pp = int(float(row["pp"]))
        except Exception:
            continue
        if pp <= 1:
            continue

        step_time_s = ""
        if "step_time(s)" in row and row["step_time(s)"] not in (None, ""):
            try:
                step_time_s = float(row["step_time(s)"])
            except Exception:
                step_time_s = ""

        key = (rank, mbs, tp, dp, pp)
        if key not in out:
            out[key] = step_time_s
            if len(out) >= limit:
                break

for (rank, mbs, tp, dp, pp), st in out.items():
    st_s = "" if st == "" else f"{st:.6f}"
    print(f"{rank}\t{mbs}\t{tp}\t{dp}\t{pp}\t{st_s}")
PY
)

if [[ "${#CONFIG_LINES[@]}" -eq 0 ]]; then
  echo "[error] no configs found for rank 1..5 in $INPUT_CSV" >&2
  exit 2
fi

echo "[info] extracted ${#CONFIG_LINES[@]} unique configs (rank 1..5)"

echo "rank,mbs,tp,dp,pp,gbs,tries,layer_start,layer_end,profile_step,status,log_prefix,stdout_log,fx_extract_log,fasop_step_time_s" > "$OUT_CSV"

cfg_idx=0
for line in "${CONFIG_LINES[@]}"; do
  cfg_idx=$((cfg_idx + 1))
  IFS=$'\t' read -r RANK MBS TP DP PP FASOP_STEP_S <<< "$line"

  # Basic sanity check: dp*tp*pp should equal world size
  world=$((DP * TP * PP))
  if [[ "$world" -ne "$NPROC_PER_NODE" ]]; then
    echo "[warn] skip rank=$RANK mbs=$MBS tp=$TP dp=$DP pp=$PP (dp*tp*pp=$world != $NPROC_PER_NODE)"
    echo "$RANK,$MBS,$TP,$DP,$PP,$GBS_DEFAULT,0,$LAYER_START_STEP,$LAYER_END_STEP,$PROFILE_STEP,skip_world_mismatch,,,,${FASOP_STEP_S}" >> "$OUT_CSV"
    continue
  fi

  GBS="$GBS_DEFAULT"
  config_name="rank${RANK}_MBS${MBS}_TP${TP}_DP${DP}_PP${PP}_GBS${GBS}"
  log_prefix="${LOG_DIR}/${DATE}_${config_name}"
  stdout_log="${log_prefix}_log_llama_fx_${LOG_LEVEL}.txt"
  fx_extract_log="${log_prefix}_fx_timing_extract.txt"

  echo "=========================================="
  echo "[run] ($cfg_idx/${#CONFIG_LINES[@]}) $config_name"
  echo "=========================================="

  TRAIN_ARGS="--access-token ${HF_ACCESS_TOKEN}
              --pp-degree ${PP}
              --tp-degree ${TP}
              --dp-degree ${DP}
              --micro-batch-size ${MBS}
              --batch-size ${GBS}
              --profile-mode ${PROFILE_MODE}
              --profile-cut ${PROFILE_CUT}
              --profile-step ${PROFILE_STEP}
              --log-level ${LOG_LEVEL}
              --layer-timing True
              --layer-timing-mode ${LAYER_TIMING_MODE}
              --layer-timing-start-step ${LAYER_START_STEP}
              --layer-timing-end-step ${LAYER_END_STEP}
              --emit-nvtx False
              --nsys-capture False
  "

  declare -i tries_ok=0
  status="ok"

  # Enable FX instrumentation inside Optimus_p init (IR instrumentation)
  export OPTPRIME_FX_LAYER_TIMING=1
  export OPTPRIME_FX_LAYER_START_STEP="${LAYER_START_STEP}"
  export OPTPRIME_FX_LAYER_END_STEP="${LAYER_END_STEP}"

  : > "$fx_extract_log"
  for i in $(seq 1 "$TRYNUMBER"); do
    # Per-run port to reduce collision risk if any process lingers.
    export MASTER_PORT=$((BASE_MASTER_PORT + cfg_idx * 10 + i))

    echo "[try] ${i}/${TRYNUMBER} MASTER_PORT=${MASTER_PORT}"
    torchrun --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1 --node_rank=0 \
      --master_port="${MASTER_PORT}" \
      examples/pp_train_llama.py ${TRAIN_ARGS} > "${stdout_log%.*}_${i}.txt" 2>&1 || {
        echo "[warn] torchrun failed (try=${i}) for $config_name"
        status="torchrun_failed"
        break
      }

    tries_ok=$((tries_ok + 1))

    # Extract the most relevant timing summaries for quick inspection.
    {
      echo "=== TRY ${i} : FX timing extracts (${config_name}) ==="
      echo "[from] ${stdout_log%.*}_${i}.txt"
      echo ""
      grep -n "=== FX Component fwd timing" -A10 "${stdout_log%.*}_${i}.txt" || true
      echo ""
      grep -n "=== FX Layer timing" -A200 "${stdout_log%.*}_${i}.txt" | head -n 260 || true
      echo ""
      grep -n "=== FX Layer fwd totals" -A120 "${stdout_log%.*}_${i}.txt" | head -n 160 || true
      echo ""
    } >> "$fx_extract_log"
  done

  if [[ "$tries_ok" -eq 0 ]]; then
    echo "[warn] no successful tries for $config_name"
    echo "$RANK,$MBS,$TP,$DP,$PP,$GBS,0,$LAYER_START_STEP,$LAYER_END_STEP,$PROFILE_STEP,${status},${log_prefix},,${fx_extract_log},${FASOP_STEP_S}" >> "$OUT_CSV"
    continue
  fi

  echo "[done] $config_name tries_ok=$tries_ok"
  # For convenience, store the last try's stdout path in the csv.
  echo "$RANK,$MBS,$TP,$DP,$PP,$GBS,$tries_ok,$LAYER_START_STEP,$LAYER_END_STEP,$PROFILE_STEP,ok,${log_prefix},${stdout_log%.*}_${tries_ok}.txt,${fx_extract_log},${FASOP_STEP_S}" >> "$OUT_CSV"
done

echo ""
echo "[done] wrote: $OUT_CSV"

