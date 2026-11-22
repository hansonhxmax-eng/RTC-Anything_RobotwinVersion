#!/usr/bin/env bash
set -Eeuo pipefail

root=/workspace/RoboTwin
rtc=/workspace/RTC-Anything_RobotwinVersion
base=/workspace/rtc_three_suite_training
policy="$root/XPolicyLab/policy/Pi_0"
py=/opt/conda/envs/RoboTwin/bin/python
run_name=rtc_three_suites_pi0_30k
port=6001

[[ -d "$policy/checkpoints/$run_name/29999/params" ]]
[[ -x "$py" && -r "$policy/.rtc_setup_eval_policy_server.sh" ]]
export PATH="/opt/conda/bin:$PATH"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export PYTHONPATH="$root/envs/curobo/curobo/src:${PYTHONPATH:-}"

out=$(mktemp -d "$base/sweep-gripper.XXXXXX")
ln -sfn "$out" "$base/sweep-gripper-latest"
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
echo "Waiting for the running evaluation to release the GPU"
flock 9
nvidia-smi -L

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

start_server rtc_object_sweeping
cd "$root"
timeout --kill-after=30s 15m "$py" -u - "$rtc/scripts/eval_robotwin_rtc.py" \
  --robotwin-root "$root" --task-name rtc_object_sweeping --ckpt-setting "$run_name" \
  --port "$port" --seed 0 --test-num 1 --request-interval 8 \
  > "$out/sweep_gripper.log" 2>&1 <<'PY'
import runpy
import sys

import numpy as np

sys.path.insert(0, ".")
from envs.rtc_object_sweeping import rtc_object_sweeping

original_take_action = rtc_object_sweeping.take_action
original_close = rtc_object_sweeping.close_env

def trace_action(self, action, action_type="qpos"):
    previous_step = self.take_action_cnt
    result = original_take_action(self, action, action_type=action_type)
    if self.eval_mode and self.take_action_cnt > previous_step:
        trace = self.__dict__.setdefault("_sweep_gripper_trace", [])
        trace.append((
            self.take_action_cnt,
            float(np.asarray(action).reshape(-1)[-1]),
            float(self.robot.get_right_gripper_val()),
            float(self.robot.get_normal_real_gripper_val()[1]),
        ))
    return result

def trace_close(self, *args, **kwargs):
    trace = self.__dict__.pop("_sweep_gripper_trace", [])
    if trace:
        print(f"\n[GRIP] steps={len(trace)} commanded_above_0.8={sum(row[1] > 0.8 for row in trace)}")
        print("[GRIP] step requested normalized_target measured_opening")
        for step, requested, target, measured in trace[-100:]:
            print(f"[GRIP] {step} {requested:.5f} {target:.5f} {measured:.5f}")
        print(f"[GRIP] final_judge_open={self.is_right_gripper_open()}")
    return original_close(self, *args, **kwargs)

rtc_object_sweeping.take_action = trace_action
rtc_object_sweeping.close_env = trace_close
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
echo "Gripper diagnostic finished"
