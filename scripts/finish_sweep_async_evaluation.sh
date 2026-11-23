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

log_prefix=sweep-async-completion
out=$(mktemp -d "$base/$log_prefix.XXXXXX")
ln -sfn "$out" "$base/$log_prefix-latest"
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

cd "$root"
original=$(readlink -f "$base/latest")
folding=$(readlink -f "$base/sweep-diagnostic-folding-latest")
train_log="$base/20261120T111850Z-67644/train.log"
read -r completed next_seed < <("$py" - "$original" "$folding" <<'PY'
from pathlib import Path
import re
import sys

original, folding = map(Path, sys.argv[1:])
def latest_rate(path):
    log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", path.read_text(errors="replace"))
    rates = re.findall(r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)", log)
    if not rates:
        raise SystemExit(f"No completed episodes in {path}")
    return tuple(map(int, rates[-1]))

for directory, task in [(original, "rtc_table_cleaning"), (folding, "rtc_clothes_folding")]:
    successes, completed, _ = latest_rate(directory / f"{task}_eval100.log")
    if completed != 100:
        raise SystemExit(f"{task}: only {completed} completed")
    print(f"{task}: {successes}/{completed}", file=sys.stderr)
_, completed, seed = latest_rate(original / "rtc_object_sweeping_eval100.log")
if not 1 <= completed <= 100:
    raise SystemExit(f"Invalid completed sweep count: {completed}")
print(completed, seed + 1)
PY
)
remaining=$((100 - completed))
echo "Asynchronous sweep baseline: $completed/100 already saved, $remaining remaining, next seed $next_seed"
old_log="$original/rtc_object_sweeping_eval100.log"
new_log="$out/remaining_sweeps.log"
if (( remaining > 0 )); then
  start_server rtc_object_sweeping
  cd "$root"
  timeout --signal=TERM --kill-after=60s 12h "$py" -u - "$next_seed" "$rtc/scripts/eval_robotwin_rtc.py" \
    --robotwin-root "$root" --task-name rtc_object_sweeping --ckpt-setting "$run_name" \
    --port "$port" --seed 0 --test-num "$remaining" --request-interval 8 \
    > "$new_log" 2>&1 <<'PY'
from pathlib import Path
import runpy
import sys

start_seed = int(sys.argv[1])
sys.argv = sys.argv[2:]
root = Path.cwd()
sys.path[:0] = [str(root / "scripts"), str(root)]
import eval_policy_xpolicylab as evaluator

evaluate = evaluator.eval_remote_policy
build_instruction = evaluator.build_instruction

def resume_evaluation(task_name, task_env, args, model_client, st_seed, **kwargs):
    return evaluate(task_name, task_env, args, model_client, start_seed, **kwargs)

def use_original_instruction_budget(args, episode_info, instruction_type, test_num, seed):
    return build_instruction(args, episode_info, instruction_type, 100, seed)

evaluator.eval_remote_policy = resume_evaluation
evaluator.build_instruction = use_original_instruction_budget
runpy.run_path(sys.argv[0], run_name="__main__")
PY
else
  touch "$new_log"
fi

merged="$out/rtc_object_sweeping_eval100.log"
"$py" - "$old_log" "$new_log" "$completed" "$merged" <<'PY'
from pathlib import Path
import re
import sys

old_path, new_path, completed, output = sys.argv[1:]
completed = int(completed)
def rates(path):
    log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", Path(path).read_text(errors="replace"))
    return [tuple(map(int, row)) for row in re.findall(
        r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)", log)]

old = rates(old_path)
new = rates(new_path)
if [row[1] for row in old] != list(range(1, completed + 1)):
    raise SystemExit("Original episode counters are inconsistent")
if [row[1] for row in new] != list(range(1, 101 - completed)):
    raise SystemExit("Continuation did not finish the expected episodes")
combined = old + [(s + old[-1][0], n + completed, seed) for s, n, seed in new]
seeds = [seed for _, _, seed in combined]
if any(b <= a for a, b in zip(seeds, seeds[1:])):
    raise SystemExit("Episode seeds overlap or are out of order")
Path(output).write_text("".join(
    f"Success rate: {s}/{n} => {100*s/n:.1f}%, current seed: {seed}\n"
    for s, n, seed in combined))
print(f"Final asynchronous sweeping success: {combined[-1][0]}/100")
PY
"$py" "$rtc/scripts/plot_robotwin_metrics.py" \
  --train-log "$train_log" --eval-log "$merged" --output-dir "$out/metrics/rtc_object_sweeping"
echo "Three asynchronous task evaluations are complete"
