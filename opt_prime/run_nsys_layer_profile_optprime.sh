#!/usr/bin/env bash
set -euo pipefail

# Capture OptPrime (pp_train_llama.py) forward layer-block NVTX with nsys and save artifacts.
#
# Config requested:
#   pp=2, tp=1, dp=4, mbs=1  (world_size=8)
#
# Output:
#   opt_prime/_logs/<DATE>_optprime_pp2_tp1_dp4_mbs1_layerblocks.nsys-rep
#   opt_prime/_logs/<DATE>_optprime_pp2_tp1_dp4_mbs1_nvtxsum.csv
#
# Run from host or inside container, but paths assume repo at /workspace/aicomp in container.
#
# Requirements:
# - `nsys` available in PATH
# - `LLAMA_ACCESS_TOKEN` set (or pass via env)

DATE="${1:-$(date +%Y%m%d_%H%M%S)}"

PP=2
TP=1
DP=4
MBS=1
WORLD=8
GBS=$((MBS * DP))   # GAS=1 baseline

START_STEP="${START_STEP:-51}"
END_STEP="${END_STEP:-60}"

OUT_DIR="/workspace/aicomp/opt_prime/_logs"
mkdir -p "$OUT_DIR"

if [[ -z "${LLAMA_ACCESS_TOKEN:-}" ]]; then
  echo "[error] LLAMA_ACCESS_TOKEN is not set" >&2
  exit 2
fi

export OPTPRIME_FX_LAYER_TIMING=1
export OPTPRIME_FX_LAYER_START_STEP="$START_STEP"
export OPTPRIME_FX_LAYER_END_STEP="$END_STEP"

# IMPORTANT: keep --emit-nvtx False so we don't get per-op NVTX spam (only FX-inserted block ranges).
OUT_PREFIX="${OUT_DIR}/${DATE}_optprime_pp${PP}_tp${TP}_dp${DP}_mbs${MBS}_layerblocks"

echo "[run] writing to ${OUT_PREFIX}.nsys-rep"
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o "${OUT_PREFIX}" \
  torchrun --standalone --nproc_per_node="${WORLD}" --nnodes=1 --master_port=29677 \
    /workspace/aicomp/opt_prime/examples/pp_train_llama.py \
      --access-token "${LLAMA_ACCESS_TOKEN}" \
      --pp-degree "${PP}" --tp-degree "${TP}" --dp-degree "${DP}" \
      --micro-batch-size "${MBS}" --batch-size "${GBS}" \
      --profile-mode 0 --profile-cut True --profile-step "${END_STEP}" --log-level 0 \
      --layer-timing True --layer-timing-mode fx \
      --layer-timing-start-step "${START_STEP}" --layer-timing-end-step "${END_STEP}" \
      --emit-nvtx False \
      --nsys-capture True \
  > "${OUT_PREFIX}_stdout.txt" 2>&1

if [[ ! -f "${OUT_PREFIX}.nsys-rep" ]]; then
  echo "[error] nsys report not found: ${OUT_PREFIX}.nsys-rep" >&2
  exit 3
fi

echo "[stats] nvtxsum -> ${OUT_PREFIX}_nvtxsum.csv"
nsys stats --report nvtxsum --format csv "${OUT_PREFIX}.nsys-rep" > "${OUT_PREFIX}_nvtxsum.csv"

echo "[done] ${OUT_PREFIX}.nsys-rep"
echo "[done] ${OUT_PREFIX}_nvtxsum.csv"


