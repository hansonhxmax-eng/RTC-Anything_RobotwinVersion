#!/usr/bin/env bash
# Collect 50 successful RoboTwin expert demonstrations for each RTC proxy.
set -Eeuo pipefail

ROOT=/workspace/RoboTwin
BASE=/workspace/rtc_three_suite_collection
PY=/opt/conda/envs/RoboTwin/bin/python
TASKS=(rtc_table_cleaning rtc_object_sweeping rtc_clothes_folding)
mkdir -p "$BASE"
exec 9>"$BASE/lock"
flock -n 9 || { echo 'RTC data collection already running' >&2; exit 75; }
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN"
ln -sfn "$RUN" "$BASE/latest"
trap 'echo "$?" > "$RUN/exit"' EXIT

if ! nvidia-smi -L > "$RUN/gpu_preflight.log" 2>&1; then
  echo 'NVIDIA GPU unavailable' >> "$RUN/gpu_preflight.log"
  exit 70
fi
if ! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    vulkaninfo --summary > "$RUN/vulkan_preflight.log" 2>&1 \
    || ! grep -q 'deviceName.*NVIDIA' "$RUN/vulkan_preflight.log"; then
  echo 'NVIDIA Vulkan unavailable' >> "$RUN/vulkan_preflight.log"
  exit 71
fi

cd "$ROOT"
for task in "${TASKS[@]}"; do
  log="$RUN/$task.log"
  data="$ROOT/data/demo_clean/$task/aloha_agilex/data"
  mkdir -p "$data"
  printf 'Started %s at %s\n' "$task" "$(date -u +%FT%TZ)" | tee "$log"
  set +e
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=60s 36h \
    "$PY" scripts/collect_data.py "$task" demo_clean >> "$log" 2>&1
  result=$?
  set -e
  count=$(find "$data" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)
  printf 'Exit=%s HDF5=%s/50 Finished=%s\n' \
    "$result" "$count" "$(date -u +%FT%TZ)" | tee -a "$log"
  printf '%s\n' "$result" > "$RUN/$task.exit"
  if (( result != 0 || count < 50 )); then
    echo "Stopping: $task did not produce 50 demonstrations" | tee -a "$RUN/status.log"
    exit 1
  fi
done
echo 'All three tasks have 50 HDF5 demonstrations' | tee "$RUN/status.log"
