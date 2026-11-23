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

# RoboTwin's instruction generator requires one template JSON per task. These
# three proxy tasks are new to this fork, so install their prompt templates
# before collection. Existing user-edited templates are preserved.
mkdir -p "$ROOT/description/task_instruction"
if [[ ! -f "$ROOT/description/task_instruction/rtc_table_cleaning.json" ]]; then
  cat > "$ROOT/description/task_instruction/rtc_table_cleaning.json" <<'JSON'
{"full_description":"collect three blocks and wipe six marked table patches with a rigid towel proxy","schema":"{A} are the blocks; {B} is the rigid towel proxy","preference":"Describe collecting the blocks and wiping the table patches.","seen":["Collect {A}, then wipe the marked table patches using {B}.","Move {A} to the collection area and clean the patches with {B}.","Use {B} to wipe all marked patches after collecting {A}.","Clear {A} and wipe the table with {B}."],"unseen":["Gather {A} and wipe each marked patch using {B}.","Put away {A}, then clean the marked areas with {B}."]}
JSON
fi
if [[ ! -f "$ROOT/description/task_instruction/rtc_object_sweeping.json" ]]; then
  cat > "$ROOT/description/task_instruction/rtc_object_sweeping.json" <<'JSON'
{"full_description":"sweep five rigid blocks into the marked collection zone with a brush proxy","schema":"{A} are the blocks; {B} is the brush proxy","preference":"Describe sweeping all blocks into the marked zone.","seen":["Sweep {A} into the marked collection zone using {B}.","Use {B} to push all five blocks into the collection area.","Gather {A} in the green marked zone with {B}.","Sweep the blocks into the target area with {B}."],"unseen":["Push each block into the collection zone using {B}.","Use {B} to sweep {A} together inside the marked area."]}
JSON
fi
if [[ ! -f "$ROOT/description/task_instruction/rtc_clothes_folding.json" ]]; then
  cat > "$ROOT/description/task_instruction/rtc_clothes_folding.json" <<'JSON'
{"full_description":"fold one rigid padded panel onto another as a clothes-folding proxy","schema":"{A} is the rigid garment panel; {B} is the base panel","preference":"Describe folding the movable panel onto the base panel.","seen":["Fold {A} onto {B}.","Place {A} on top of {B} to fold the garment.","Use the right arm to fold {A} onto {B}.","Fold the movable panel over the base panel."],"unseen":["Lay {A} over {B} to complete the fold.","Stack the two garment panels by folding {A} onto {B}."]}
JSON
fi

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
