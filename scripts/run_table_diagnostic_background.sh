#!/usr/bin/env bash
# Run exactly one RoboTwin table-cleaning expert diagnosis in the background.
# Invoked with docker exec -d; logs and status live on the mounted /workspace.
set -Eeuo pipefail

BASE=/workspace/rtc_table_diag_bg
ROOT=/workspace/RoboTwin
RTC=/workspace/RTC-Anything_RobotwinVersion
mkdir -p "$BASE"
exec 9>"$BASE/lock"
flock -n 9 || { echo "A table diagnosis is already running" >>"$BASE/launcher.log"; exit 75; }

RUN="$BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$RUN"
ln -sfn "$RUN" "$BASE/latest"
LOG="$RUN/run.log"
printf 'Started: %s\n' "$(date -u +%FT%TZ)" > "$LOG"
printf '%s\n' "$$" > "$RUN/pid"
finish() {
    local result=$?
    printf '%s\n' "$result" > "$RUN/exit"
    printf 'Finished: %s; exit=%s\n' "$(date -u +%FT%TZ)" "$result" >> "$LOG"
}
trap finish EXIT

# Avoid another, manually launched diagnosis running concurrently.
if pgrep -af '^/opt/conda/envs/RoboTwin/bin/python -u /workspace/RTC-Anything_RobotwinVersion/scripts/diagnose_table_cleaning.py' >> "$LOG"; then
    echo 'Another manually launched diagnosis is still running' >> "$LOG"
    exit 75
fi

cd "$ROOT"
/opt/conda/envs/RoboTwin/bin/python -m py_compile \
    envs/rtc_table_cleaning.py "$RTC/scripts/diagnose_table_cleaning.py" >> "$LOG" 2>&1

set +e
PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 1200 \
    /opt/conda/envs/RoboTwin/bin/python -u \
    "$RTC/scripts/diagnose_table_cleaning.py" --seed 0 >> "$LOG" 2>&1
status=$?
set -e
if (( status != 0 )); then
    printf 'Diagnosis process exited with %s\n' "$status" >> "$LOG"
    exit "$status"
fi

if grep -Fq '[FINAL] check_success=True' "$LOG" \
    && grep -Fq '[FINAL] planned_joints_legal=(True' "$LOG"; then
    echo 'Expert demonstration passed both checks' >> "$LOG"
else
    echo 'Expert demonstration failed or did not reach final checks' >> "$LOG"
    exit 1
fi
