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
export PATH="/opt/conda/bin:${PATH}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$ROOT/envs/curobo/curobo/src:${PYTHONPATH:-}"
mkdir -p "$BASE"
exec 9>"$BASE/lock"
flock -n 9 || { echo 'Post-collection training pipeline already running' >&2; exit 75; }
RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN"
ln -sfn "$RUN" "$BASE/latest"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
server_pid=
finish() {
  result=$?
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  echo "$result" > "$RUN/exit"
}
trap finish EXIT

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
converted_info="/root/.cache/huggingface/lerobot/$REPO_ID/meta/info.json"
if (( available_gb < 50 )) && [[ ! -f "$converted_info" ]]; then
  echo 'Less than 50 GB free for dataset conversion.'
  exit 72
fi
if [[ -f "$converted_info" ]]; then
  "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["total_episodes"] == 150' "$converted_info"
  echo "Using existing converted dataset: $converted_info"
else
  "$PY" "$RTC/scripts/prepare_rtc_multitask.py" \
    --robotwin-root "$ROOT" --repo-id "$REPO_ID" --episodes-per-task 50 \
    2>&1 | tee "$RUN/convert.log"
fi

"$PY" "$RTC/scripts/prepare_robotwin_pi0.py" \
  --robotwin-root "$ROOT" --batch-size 32 --fsdp-devices 1 \
  --steps 30000 --save-interval 5000 \
  2>&1 | tee "$RUN/prepare_model.log"

export HF_LEROBOT_HOME=/root/.cache/huggingface/lerobot
export OPENPI_DATA_REPO_ID="$REPO_ID"
cd "$OPENPI"
norm_stats="assets/rtc_robotwin_pi0_lora/$REPO_ID/norm_stats.json"
if [[ -f "$norm_stats" ]]; then
  echo "Using existing normalization: $norm_stats"
else
  "$OPENPI/.venv/bin/python" scripts/compute_norm_stats.py \
    --config-name rtc_robotwin_pi0_lora 2>&1 | tee "$RUN/norm_stats.log"
fi
test -f "$norm_stats"

checkpoint_dir="$POLICY/checkpoints/$RUN_NAME"
checkpoint_step=
if [[ -d "$checkpoint_dir" ]]; then
  checkpoint_step=$(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | awk '/^[0-9]+$/ {print}' | sort -n | tail -n 1)
fi
if [[ -n "$checkpoint_step" && "$checkpoint_step" -ge 29999 ]]; then
  echo "Using completed checkpoint: $checkpoint_step"
else
  resume_arg=()
  if [[ -n "$checkpoint_step" ]]; then resume_arg=(--resume); fi
  cd "$RTC"
  "$PY" scripts/train_robotwin_pi0.py \
    --robotwin-root "$ROOT" --repo-id "$REPO_ID" \
    --run-name "$RUN_NAME" --gpus 0 --seed 0 "${resume_arg[@]}" \
    2>&1 | tee "$RUN/train.log"
fi
checkpoint_step=$(find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d \
  -printf '%f\n' | awk '/^[0-9]+$/ {print}' | sort -n | tail -n 1)
if [[ -z "$checkpoint_step" ]]; then
  echo "No numeric checkpoint found in $checkpoint_dir"
  exit 1
fi

"$PY" "$RTC/scripts/configure_robotwin_deploy.py" \
  --robotwin-root "$ROOT" --run-name "$RUN_NAME" \
  --repo-id "$REPO_ID" --checkpoint-step "$checkpoint_step"

server_log="$RUN/policy_server.log"
# Keep the submodule launcher unchanged. Its script directory must remain Pi_0.
server_script="$POLICY/.rtc_setup_eval_policy_server.sh"
sed 's|^YAML_PYTHON=.*|YAML_PYTHON="/opt/conda/envs/RoboTwin/bin/python"|' \
  "$POLICY/setup_eval_policy_server.sh" > "$server_script"
grep -q 'YAML_PYTHON="/opt/conda/envs/RoboTwin/bin/python"' "$server_script"
cd "$POLICY"
nohup bash "$server_script" \
  RoboTwin rtc_table_cleaning "$RUN_NAME" arx_x5 joint 0 0 uv 6000 localhost \
  > "$server_log" 2>&1 < /dev/null &
server_pid=$!
echo "Policy server PID=$server_pid; checkpoint=$checkpoint_step"

server_ready=false
for _ in $(seq 1 600); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$server_log"
    echo 'Policy server exited before opening port 6000'
    exit 1
  fi
  if (echo > /dev/tcp/127.0.0.1/6000) >/dev/null 2>&1; then
    server_ready=true
    break
  fi
  sleep 2
done
if [[ "$server_ready" != true ]]; then
  cat "$server_log"
  echo 'Policy server did not become ready within 20 minutes'
  exit 1
fi

train_log="$RUN/train.log"
if [[ ! -s "$train_log" ]]; then
  train_log=$(find "$BASE" -mindepth 2 -maxdepth 2 -type f -name train.log \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$train_log" && -s "$train_log" ]] || { echo "Training log missing for plots"; exit 1; }
  echo "Using previous training log: $train_log"
fi

for task in rtc_table_cleaning rtc_object_sweeping rtc_clothes_folding; do
  eval_log="$RUN/${task}_eval100.log"
  echo "Starting async evaluation: $task (100 episodes)"
  timeout --signal=TERM --kill-after=60s 12h \
    "$PY" "$RTC/scripts/eval_robotwin_rtc.py" \
    --robotwin-root "$ROOT" --task-name "$task" --ckpt-setting "$RUN_NAME" \
    --port 6000 --test-num 100 --request-interval 8 \
    > "$eval_log" 2>&1
  "$PY" "$RTC/scripts/plot_robotwin_metrics.py" \
    --train-log "$train_log" --eval-log "$eval_log" \
    --output-dir "$RUN/metrics/$task"
done

echo 'Collection, conversion, normalization, 30k-step training, three async evaluations, and plots completed.'
