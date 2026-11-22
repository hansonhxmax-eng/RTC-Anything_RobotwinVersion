#!/usr/bin/env bash
# One isolated expert-demo smoke test. Run inside robotwin-dev (detached).
set -Eeuo pipefail

ROOT=/workspace/RoboTwin
RUN=/workspace/rtc_table_smoke_v2
TASK_CONFIG=rtc_table_smoke_v2
exec 9>/workspace/rtc_table_smoke.lock
flock -n 9 || { echo 'Another table-cleaning smoke test holds the lock'; exit 73; }
if [[ -e "$RUN" ]]; then
  echo "Refusing to reuse $RUN; inspect the previous attempt first" >&2
  exit 2
fi
mkdir -p "$RUN"
cd "$ROOT"
test -s envs/rtc_table_cleaning.py
"/opt/conda/envs/RoboTwin/bin/python" -m py_compile envs/rtc_table_cleaning.py
cp env_cfg/task_config/demo_clean.yml "env_cfg/task_config/${TASK_CONFIG}.yml"
sed -i -e 's/^episode_num:.*/episode_num: 1/' \
  -e "s#^save_path:.*#save_path: $RUN/data#" \
  "env_cfg/task_config/${TASK_CONFIG}.yml"
printf 'Starting one episode; log: %s\n' "$RUN/run.log" > "$RUN/run.log"
set +e
PYTHONUNBUFFERED=1 timeout 600 \
  /opt/conda/envs/RoboTwin/bin/python -u scripts/collect_data.py \
  rtc_table_cleaning "$TASK_CONFIG" >> "$RUN/run.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$RUN/exit"
printf 'Smoke test exit: %s\n' "$status" >> "$RUN/run.log"
exit "$status"
