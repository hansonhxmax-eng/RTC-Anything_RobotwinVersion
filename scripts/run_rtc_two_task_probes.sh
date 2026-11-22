#!/usr/bin/env bash
# Detached single-seed probes for the two experimental RTC task proxies.
set -Eeuo pipefail

BASE=/workspace/rtc_other_task_probes
RTC=/workspace/RTC-Anything_RobotwinVersion
ROOT=/workspace/RoboTwin
mkdir -p "$BASE"
exec 9>"$BASE/lock"
flock -n 9 || { echo 'Probes already running' >&2; exit 75; }
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN"
ln -sfn "$RUN" "$BASE/latest"

if ! nvidia-smi -L > "$RUN/gpu_preflight.log" 2>&1; then
    echo 'NVIDIA GPU unavailable; no task probe started' >> "$RUN/gpu_preflight.log"
    exit 70
fi
if ! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    vulkaninfo --summary > "$RUN/vulkan_preflight.log" 2>&1; then
    echo 'NVIDIA Vulkan unavailable; no task probe started' >> "$RUN/vulkan_preflight.log"
    exit 71
fi
if ! grep -q 'deviceName.*NVIDIA' "$RUN/vulkan_preflight.log"; then
    echo 'Vulkan did not select NVIDIA; no task probe started' >> "$RUN/vulkan_preflight.log"
    exit 71
fi

cd "$ROOT"
for task in rtc_object_sweeping rtc_clothes_folding; do
    test -s "envs/$task.py"
done
/opt/conda/envs/RoboTwin/bin/python -m py_compile \
    envs/rtc_object_sweeping.py envs/rtc_clothes_folding.py \
    "$RTC/scripts/diagnose_rtc_expert.py"

# The existing table-cleaning probe may still be running on this GPU.
while pgrep -af '^/opt/conda/envs/RoboTwin/bin/python -u /workspace/RTC-Anything_RobotwinVersion/scripts/diagnose_table_cleaning.py' >/dev/null; do
    echo "Waiting for table probe: $(date -u +%FT%TZ)" >> "$RUN/wait.log"
    sleep 15
done

for task in rtc_object_sweeping rtc_clothes_folding; do
    log="$RUN/$task.log"
    printf 'Started: %s\n' "$(date -u +%FT%TZ)" > "$log"
    set +e
    PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 1200 \
        /opt/conda/envs/RoboTwin/bin/python -u \
        "$RTC/scripts/diagnose_rtc_expert.py" \
        --task-name "$task" --seed 0 >> "$log" 2>&1
    result=$?
    set -e
    printf '%s\n' "$result" > "$RUN/$task.exit"
    printf 'Finished: %s; exit=%s\n' "$(date -u +%FT%TZ)" "$result" >> "$log"
done
