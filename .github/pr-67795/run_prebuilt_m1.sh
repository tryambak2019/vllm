#!/usr/bin/env bash
# Benchmark the prebuilt PR #67795 baseline and candidate at M=1.

set -euo pipefail

readonly baseline_sha='5e51c0ca1041168de7aa9e096982aebaeebabd38'
readonly candidate_sha='2c94a97dede4fef539e9bba8ddd9e5def7d845bf'
readonly benchmark='/opt/bench/benchmark_gptq_m1.py'
readonly result_root="$RESULTS_DIR/m1-prebuilt-$(date -u +%Y%m%dT%H%M%SZ)"
readonly variants='/opt/vllm-variants'
readonly extension_relpath="$(cat "$variants/extension-path")"
readonly extension_name="$(basename "$extension_relpath")"
readonly active_extension="/workspace/vllm/$extension_relpath"
readonly rate_per_second='0.00119368'

mkdir -p "$result_root"
run_start=$(date +%s)

cleanup() {
  local exit_code=$?
  if [[ -n ${monitor_pid:-} ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  sync
  exit "$exit_code"
}
trap cleanup EXIT

while true; do
  printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi \
    --query-gpu=power.draw,power.limit,pstate,clocks.current.sm,clocks.max.sm,temperature.gpu \
    --format=csv,noheader,nounits
  sleep 1
done >"$result_root/gpu-telemetry.csv" &
monitor_pid=$!

run_variant() {
  local label=$1
  local variant=$2
  local sha=$3
  cp "$variants/$variant/$extension_name" "$active_extension"
  .venv/bin/python -P "$benchmark" \
    --label "$label" \
    --commit "$sha" \
    --output "$result_root/$label.json"
}

cd /workspace/vllm
run_variant baseline-a baseline "$baseline_sha"
run_variant candidate candidate "$candidate_sha"
run_variant baseline-b baseline "$baseline_sha"

run_end=$(date +%s)
elapsed=$((run_end - run_start))
estimated_cost=$(awk -v seconds="$elapsed" -v rate="$rate_per_second" \
  'BEGIN { printf "%.4f", seconds * rate }')
printf '{"elapsed_seconds":%d,"estimated_modal_cost_usd":%s,"rate_usd_per_second":%s}\n' \
  "$elapsed" "$estimated_cost" "$rate_per_second" \
  >"$result_root/cost.json"
printf 'Results: %s\nElapsed: %s seconds\nEstimated cost: $%s\n' \
  "$result_root" "$elapsed" "$estimated_cost"
