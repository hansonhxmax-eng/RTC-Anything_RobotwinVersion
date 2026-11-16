#!/usr/bin/env bash
# Download a new RoboTwin task dataset, convert it to LeRobot v3, compute
# normalization statistics, and start JAX pi0 LoRA training. No old run is used.
set -Eeuo pipefail
trap 'printf "[training setup failed] line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/workspace/RoboTwin}"
RTC_ROOT="${RTC_ROOT:-/workspace/RTC-Anything_RobotwinVersion}"
OPENPI_ROOT="$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/openpi"
TASK_NAME="${TASK_NAME:-place_a2b_left}"
REPO_ID="${REPO_ID:-rtc_${TASK_NAME}_clean}"
RUN_NAME="${RUN_NAME:-rtc_${TASK_NAME}_lora_seed0}"
CONFIG_NAME=rtc_robotwin_pi0_lora
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/.cache/huggingface/lerobot}"
export OPENPI_DATA_REPO_ID="$REPO_ID"

if [[ ! "$TASK_NAME" =~ ^[a-zA-Z0-9_]+$ \
   || ! "$REPO_ID" =~ ^[a-zA-Z0-9_]+$ \
   || ! "$RUN_NAME" =~ ^[a-zA-Z0-9_]+$ ]]; then
  echo 'Task, dataset and run names must contain only letters, digits and underscores' >&2
  exit 1
fi
test -f "$ROBOTWIN_ROOT/scripts/download_xpolicylab_data.sh"
test -x "$OPENPI_ROOT/.venv/bin/python"
test -f "$RTC_ROOT/scripts/train_robotwin_pi0.py"

exec 9>"/workspace/${RUN_NAME}.setup.lock"
flock -n 9 || { echo "Setup or training for $RUN_NAME is already running" >&2; exit 1; }

source /opt/conda/etc/profile.d/conda.sh
conda activate RoboTwin
cd "$ROBOTWIN_ROOT"
input_dir="$ROBOTWIN_ROOT/data/demo_clean/$TASK_NAME/aloha_agilex/data"

echo "[1/4] Downloading new demonstrations for $TASK_NAME"
if [[ -d "$input_dir" ]] && find "$input_dir" -type f -name 'episode_*.hdf5' -print -quit | grep -q .; then
  echo "Using existing source episodes in $input_dir"
else
  HF_KEEP_ARCHIVES=0 bash scripts/download_xpolicylab_data.sh "$TASK_NAME"
fi
source_count=$(find "$input_dir" -type f -name 'episode_*.hdf5' | wc -l)
if (( source_count < 1 )); then
  echo "No source episodes found in $input_dir" >&2
  exit 1
fi
echo "Source episodes: $source_count"

echo "[2/4] Converting to a new local LeRobot v3 dataset: $REPO_ID"
dataset_dir="$HF_LEROBOT_HOME/$REPO_ID"
if [[ ! -e "$dataset_dir" ]]; then
  "$OPENPI_ROOT/.venv/bin/python" \
    "$ROBOTWIN_ROOT/XPolicyLab/scripts/transform_lerobot_v30_format.py" \
    "demo_clean.$TASK_NAME.aloha_agilex" \
    --repo_id "$REPO_ID" --max_episode "$source_count"
else
  echo "Dataset already exists; checking completeness: $dataset_dir"
fi

SOURCE_COUNT="$source_count" DATASET_DIR="$dataset_dir" \
  "$OPENPI_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ['DATASET_DIR']) / 'meta/info.json'
if not path.is_file():
    raise SystemExit(f'Dataset metadata missing: {path}')
info = json.loads(path.read_text(encoding='utf-8'))
actual = int(info.get('total_episodes', 0))
expected = int(os.environ['SOURCE_COUNT'])
print(f'Converted episodes: {actual}/{expected}; frames: {info.get("total_frames")}')
if actual != expected or int(info.get('total_frames', 0)) < actual:
    raise SystemExit('Incomplete dataset; inspect conversion errors before training')
PY

echo '[3/4] Computing new state and action normalization statistics'
norm_stats="$OPENPI_ROOT/assets/$CONFIG_NAME/$REPO_ID/norm_stats.json"
if [[ -f "$norm_stats" ]]; then
  echo "Using existing stats: $norm_stats"
else
  cd "$OPENPI_ROOT"
  .venv/bin/python scripts/compute_norm_stats.py "$CONFIG_NAME"
fi
test -s "$norm_stats"
echo "Normalization stats: $norm_stats"

echo "[4/4] Launching a new JAX pi0 LoRA training run: $RUN_NAME"
checkpoint_dir="$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_0/checkpoints/$RUN_NAME"
resume=()
if [[ -e "$checkpoint_dir" ]]; then
  if find "$checkpoint_dir" -mindepth 1 -maxdepth 1 -type d \
      -regex '.*/[0-9]+' -print -quit | grep -q .; then
    echo "Found a finalized checkpoint: resuming $RUN_NAME"
    resume=(--resume)
  else
    echo "Checkpoint directory exists without a finalized step: $checkpoint_dir" >&2
    echo 'Inspect it before retrying; no files will be overwritten.' >&2
    exit 1
  fi
fi
cd "$RTC_ROOT"
python scripts/train_robotwin_pi0.py \
  --robotwin-root "$ROBOTWIN_ROOT" --repo-id "$REPO_ID" \
  --run-name "$RUN_NAME" --gpus 0 "${resume[@]}"
