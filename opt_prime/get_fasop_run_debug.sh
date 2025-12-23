trynumber=3
log_prefix="/home/ieg95/workspace/aicomp/opt_prime/_logs"
log_prefix="${log_prefix}/20251219_074639_rank2_MBS8_TP1_DP2_PP4_GBS32"
for i in $(seq 1 $trynumber); do
    avg_file="${log_prefix}_log_llama4_${i}_avg.txt"
    avg_ms="$(head -n 1 "$avg_file" | tr -d '\r' | xargs)"
    if [[ -z "$avg_ms" ]]; then
      echo "[warn] empty avg value in $avg_file"
      status="avg_empty"
      continue
    fi
    AVGS_MS+=("$avg_ms")
    echo "[ok] try=${i} avg_ms_batch(51~60)=${avg_ms}"
done
  read -r mean_ms std_ms <<< "$(
    python3 - <<'PY' "${AVGS_MS[@]}"
import sys, math
vals = [float(x) for x in sys.argv[1:]]
mean = sum(vals)/len(vals)
var = sum((x-mean)**2 for x in vals)/len(vals)
std = math.sqrt(var)
print(f"{mean:.3f} {std:.3f}")
PY
  )"

  mean_s="$(python3 - <<PY "$mean_ms"
import sys
print(f"{float(sys.argv[1])/1000.0:.6f}")
PY
)"  
echo "[done] $config_name tries_ok=$tries_ok mean_ms=${mean_ms} std_ms=${std_ms}"