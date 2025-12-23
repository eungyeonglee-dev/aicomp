#!/usr/bin/env bash
set -euo pipefail

# Run OptPrime llama training for configs listed in FASOP csv (rank 1~5),
# measure steptime (ms/batch for steps 51~60), and write a results csv.
#
# Intended to be run from: /workspace/aicomp/opt_prime (inside container) OR host.
# If running inside container, make sure the input csv is visible inside the container
# (e.g., bind-mount /home/ieg95/workspace/FASOP or copy the csv into /workspace/aicomp).
#
# Usage:
#   bash get_fasop_run.sh <DATE> [INPUT_CSV] [OUT_CSV]
#
# Example:
#   DATE=$(date +%Y%m%d_%H%M%S)
#   bash get_fasop_run.sh "$DATE" /home/ieg95/workspace/FASOP/main_logs/llama_evenly.csv ./_logs/${DATE}_fasop_rank1-5.csv

DATE="${1:-$(date +%Y%m%d_%H%M%S)}"
INPUT_CSV="${2:-/workspace/FASOP/main_logs/llama_evenly_mbs_test.csv}"
OUT_CSV="${3:-./_logs/${DATE}_fasop_rank1-5_steptime.csv}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRYNUMBER="${TRYNUMBER:-3}"
GBS_DEFAULT="${GBS_DEFAULT:-32}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29500}"

HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-{your_huggingface_token_here}}"
PROFILE_MODE="${PROFILE_MODE:-0}"
PROFILE_CUT="${PROFILE_CUT:-True}"
PROFILE_STEP="${PROFILE_STEP:-60}"

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

# Extract configs (rank 1..5) from csv robustly (handles quoted fields w/ commas),
# keep only configs with PP>1, and cap to at most 5 configs.
# Output format (tab-separated):
#   rank \t mbs \t tp \t dp \t pp \t fasop_step_time_s
mapfile -t CONFIG_LINES < <(
python3 - <<'PY' "$INPUT_CSV"
import csv, sys
from collections import OrderedDict

path = sys.argv[1]


out = OrderedDict()  # (mbs,tp,dp,pp)->fasop_step_time_s (keep first)
limit = 10

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            rank = int(float(row.get("rank", "").strip()))
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

echo "rank,mbs,tp,dp,pp,gbs,tries,avg_ms_batch,trial_std_ms_batch,avg_s_batch,fasop_step_time_s,status,log_prefix" > "$OUT_CSV"

cfg_idx=0
for line in "${CONFIG_LINES[@]}"; do
  cfg_idx=$((cfg_idx + 1))
  IFS=$'\t' read -r RANK MBS TP DP PP FASOP_STEP_S <<< "$line"

  # Basic sanity check: dp*tp*pp should equal world size
  world=$((DP * TP * PP))
  if [[ "$world" -ne "$NPROC_PER_NODE" ]]; then
    echo "[warn] skip rank=$RANK mbs=$MBS tp=$TP dp=$DP pp=$PP (dp*tp*pp=$world != $NPROC_PER_NODE)"
    echo "$RANK,$MBS,$TP,$DP,$PP,$GBS_DEFAULT,0,,,,${FASOP_STEP_S},skip_world_mismatch," >> "$OUT_CSV"
    continue
  fi

  GBS="$GBS_DEFAULT"
  config_name="rank${RANK}_MBS${MBS}_TP${TP}_DP${DP}_PP${PP}_GBS${GBS}"
  log_prefix="${LOG_DIR}/${DATE}_${config_name}"

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
  "

  declare -a AVGS_MS=()
  status="ok"

  for i in $(seq 1 "$TRYNUMBER"); do
    # Per-run port to reduce collision risk if any process lingers.
    export MASTER_PORT=$((BASE_MASTER_PORT + cfg_idx * 10 + i))

    echo "[try] ${i}/${TRYNUMBER} MASTER_PORT=${MASTER_PORT}"
    torchrun --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" --nnodes=1 --node_rank=0 \
      --master_port="${MASTER_PORT}" \
      examples/pp_train_llama4.py ${TRAIN_ARGS} > "${log_prefix}_log_llama4_${i}.txt" 2>&1 || {
        echo "[warn] torchrun failed (try=${i}) for $config_name"
        status="torchrun_failed"
        break
      }

    ./get_time.sh "${log_prefix}_log_llama4_${i}.txt" > "${log_prefix}_steptime_log_llama4_${i}.txt" 2>&1 || {
      echo "[warn] get_time.sh failed (try=${i}) for $config_name"
      status="time_parse_failed"
      continue
    }

    avg_file="${log_prefix}_log_llama4_${i}_avg.txt"
    if [[ ! -f "$avg_file" ]]; then
      echo "[warn] missing avg file: $avg_file"
      status="avg_missing"
      continue
    fi

    avg_ms="$(head -n 1 "$avg_file" | tr -d '\r' | xargs)"
    if [[ -z "$avg_ms" ]]; then
      echo "[warn] empty avg value in $avg_file"
      status="avg_empty"
      continue
    fi
    AVGS_MS+=("$avg_ms")
    echo "[ok] try=${i} avg_ms_batch(51~60)=${avg_ms}"
  done

  tries_ok="${#AVGS_MS[@]}"
  if [[ "$tries_ok" -eq 0 ]]; then
    echo "[warn] no successful tries for $config_name"
    echo "$RANK,$MBS,$TP,$DP,$PP,$GBS,0,,,,${FASOP_STEP_S},${status},${log_prefix}" >> "$OUT_CSV"
    continue
  fi

  # Compute mean and std across successful tries (ms/batch).
  read -r mean_ms std_ms <<< "$(
    python3 - <<'PY' "${AVGS_MS[@]}"
import sys, math
vals = [float(x) for x in sys.argv[1:]]
mean = sum(vals)/len(vals)
var = sum((x-mean)**2 for x in vals)/len(vals)
std = math.sqrt(var)
print(f'{mean:.3f} {std:.3f}')
PY
  )"

  mean_s="$(python3 - <<PY "$mean_ms"
import sys
print(f'{float(sys.argv[1])/1000.0:.6f}')
PY
)"

  echo "[done] $config_name tries_ok=$tries_ok mean_ms=${mean_ms} std_ms=${std_ms}"
  echo "$RANK,$MBS,$TP,$DP,$PP,$GBS,$tries_ok,$mean_ms,$std_ms,$mean_s,${FASOP_STEP_S},ok,${log_prefix}" >> "$OUT_CSV"
done

echo ""
echo "[done] wrote: $OUT_CSV"


