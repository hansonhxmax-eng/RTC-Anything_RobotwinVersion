#!/usr/bin/env bash
set -Eeuo pipefail

root=/workspace/RoboTwin
rtc=/workspace/RTC-Anything_RobotwinVersion
base=/workspace/rtc_three_suite_training
policy="$root/XPolicyLab/policy/Pi_0"
py=/opt/conda/envs/RoboTwin/bin/python
run_name=rtc_three_suites_pi0_30k
port=6001

original=$(readlink -f "$base/latest")
[[ -s "$original/rtc_table_cleaning_eval100.log" && -s "$original/rtc_object_sweeping_eval100.log" ]]
[[ -d "$policy/checkpoints/$run_name/29999/params" ]]
[[ -x "$py" && -r "$policy/.rtc_setup_eval_policy_server.sh" ]]
train_log="$base/20261120T111850Z-67644/train.log"
[[ -s "$train_log" ]]
export PATH="/opt/conda/bin:$PATH"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$root/envs/curobo/curobo/src:${PYTHONPATH:-}"
nvidia-smi -L

out=$(mktemp -d "$base/sweep-diagnostic-folding.XXXXXX")
ln -sfn "$out" "$base/sweep-diagnostic-folding-latest"
exec >> "$out/run.log" 2>&1
server_pid=
stop_server() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
finish() {
  local status=$?
  stop_server
  printf '%s\n' "$status" > "$out/exit"
}
trap finish EXIT

exec 9>"$base/lock"
if ! flock -n 9; then
  pattern='^timeout .*eval_robotwin_rtc[.]py .*--task-name rtc_object_sweeping --ckpt-setting rtc_three_suites_pi0_30k --port 6000 --test-num 100( |$)'
  sweep_pid=$(pgrep -f "$pattern" || true)
  if [[ ! "$sweep_pid" =~ ^[0-9]+$ ]]; then
    echo "Original pipeline is active, but no unique sweeping process matched. Leaving it running."
    exit 1
  fi
  echo "Stopping only the sweeping process (PID $sweep_pid); previous results: $original"
  kill -TERM "$sweep_pid"
  flock -w 90 9 || { echo "Original pipeline has not released the lock"; exit 1; }
fi

start_server() {
  local task=$1 log="$out/$1.server.log"
  cd "$policy"
  bash .rtc_setup_eval_policy_server.sh \
    RoboTwin "$task" "$run_name" arx_x5 joint 0 0 uv "$port" localhost \
    > "$log" 2>&1 < /dev/null 9>&- &
  server_pid=$!
  for ((i=0; i<120; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then return 0; fi
    sleep 2
  done
  tail -n 25 "$log"
  return 1
}

echo "Starting one sweeping diagnostic"
if start_server rtc_object_sweeping; then
  cd "$root"
  if ! timeout --kill-after=30s 15m "$py" -u - "$rtc/scripts/eval_robotwin_rtc.py" \
      --robotwin-root "$root" --task-name rtc_object_sweeping --ckpt-setting "$run_name" \
      --port "$port" --seed 0 --test-num 1 --request-interval 8 \
      > "$out/sweep_diagnostic.log" 2>&1 <<'PY'
import runpy
import sys

sys.path.insert(0, ".")
from envs.rtc_object_sweeping import rtc_object_sweeping

original_close = rtc_object_sweeping.close_env

def close_with_diagnostics(self, *args, **kwargs):
    if getattr(self, "eval_mode", False) and getattr(self, "take_action_cnt", 0):
        print(f"[SWEEP] steps={self.take_action_cnt} right_gripper_open={self.is_right_gripper_open()}")
        for i, obj in enumerate(self.objects, 1):
            x, y, z = obj.get_pose().p
            inside = 0.04 < x < 0.40 and 0.04 < y < 0.18 and 0.74 < z < 0.84
            print(f"[SWEEP] block={i} x={x:.5f} y={y:.5f} z={z:.5f} inside={inside}")
    return original_close(self, *args, **kwargs)

rtc_object_sweeping.close_env = close_with_diagnostics
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
  then
    echo "Sweeping diagnostic stopped with an error; see $out/sweep_diagnostic.log"
  fi
else
  echo "Sweeping diagnostic server failed to start"
fi
stop_server

if [[ -f "$original/exit" && $(cat "$original/exit") == 0 ]]; then
  echo "The original pipeline already completed folding."
  exit 0
fi
echo "Starting clothes folding: 100 episodes"
start_server rtc_clothes_folding
timeout --signal=TERM --kill-after=60s 12h "$py" -u "$rtc/scripts/eval_robotwin_rtc.py" \
  --robotwin-root "$root" --task-name rtc_clothes_folding --ckpt-setting "$run_name" \
  --port "$port" --test-num 100 --request-interval 8 > "$out/rtc_clothes_folding_eval100.log" 2>&1
"$py" "$rtc/scripts/plot_robotwin_metrics.py" \
  --train-log "$train_log" --eval-log "$out/rtc_clothes_folding_eval100.log" \
  --output-dir "$out/metrics/rtc_clothes_folding"
echo "Folding evaluation and plots completed"
