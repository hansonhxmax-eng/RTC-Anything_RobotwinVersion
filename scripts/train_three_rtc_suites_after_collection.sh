#!/usr/bin/env bash
# Wait for the detached 3-task collector, then convert, normalize, and train.
set -Eeuo pipefail

ROOT=/workspace/RoboTwin
RTC=/workspace/RTC-Anything_RobotwinVersion
POLICY="$ROOT/XPolicyLab/policy/Pi_0"
OPENPI="$POLICY/openpi"
PY=/opt/conda/envs/RoboTwin/bin/python
REPO_ID=rtc_three_suites_clean_v1
RUN_NAME=rtc_three_suites_pi0_30k
BASE=/workspace/rtc_three_suite_training
mkdir -p "$BASE"
exec 9>"$BASE/lock"
flock -n 9 || { echo 'Post-collection training pipeline already running' >&2; exit 75; }
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN"
ln -sfn "$RUN" "$BASE/latest"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
trap 'echo "$?" > "$RUN/exit"' EXIT

COLLECTION=/workspace/rtc_three_suite_collection/latest
echo "Waiting for data collection: $COLLECTION"
while [[ ! -f "$COLLECTION/exit" ]]; do sleep 60; done
collector_status=$(cat "$COLLECTION/exit")
if [[ "$collector_status" != 0 ]]; then
  echo "Data collection failed with exit=$collector_status"
  exit 1
fi

cd "$ROOT"
available_gb=$(df -BG "$ROOT" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')
echo "Disk free: ${available_gb}G"
if (( available_gb < 50 )); then
  echo 'Less than 50 GB free; stopping before dataset conversion/training.'
  exit 72
fi

"$PY" "$RTC/scripts/prepare_rtc_multitask.py" \
  --robotwin-root "$ROOT" --repo-id "$REPO_ID" --episodes-per-task 50 \
  2>&1 | tee "$RUN/convert.log"

"$PY" "$RTC/scripts/prepare_robotwin_pi0.py" \
  --robotwin-root "$ROOT" --batch-size 32 --fsdp-devices 1 \
  --steps 30000 --save-interval 5000 \
  2>&1 | tee "$RUN/prepare_model.log"

export HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot
export OPENPI_DATA_REPO_ID="$REPO_ID"
cd "$OPENPI"
uv run scripts/compute_norm_stats.py \
  --config-name rtc_robotwin_pi0_lora 2>&1 | tee "$RUN/norm_stats.log"
test -f "assets/rtc_robotwin_pi0_lora/$REPO_ID/norm_stats.json"

cd "$RTC"
"$PY" scripts/train_robotwin_pi0.py \
  --robotwin-root "$ROOT" --repo-id "$REPO_ID" \
  --run-name "$RUN_NAME" --gpus 0 --seed 0 \
  2>&1 | tee "$RUN/train.log"

echo 'Dataset conversion, normalization, and 30k-step π0 training completed.'
