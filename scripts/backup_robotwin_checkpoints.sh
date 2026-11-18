#!/usr/bin/env bash
# Archive an OpenPI run's complete checkpoint directory for migration.
# Usage: bash backup_robotwin_checkpoints.sh RUN_NAME
set -Eeuo pipefail

RUN_NAME=${1:?Usage: backup_robotwin_checkpoints.sh RUN_NAME}
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Invalid run name: $RUN_NAME" >&2
    exit 2
fi

BASE=/workspace
SOURCE="RoboTwin/XPolicyLab/policy/Pi_0/checkpoints/$RUN_NAME"
OUT="$BASE/rtc_backups"
mkdir -p "$OUT"
exec 9>"$OUT/$RUN_NAME.lock"
flock -n 9 || { echo "Backup already running: $RUN_NAME" >&2; exit 75; }
if [[ ! -d "$BASE/$SOURCE" ]]; then
    echo "No checkpoint directory: $BASE/$SOURCE" >&2
    exit 2
fi
if ! find "$BASE/$SOURCE" -type f -name norm_stats.json -print -quit | grep -q .; then
    echo "No normalization statistics in $BASE/$SOURCE" >&2
    exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="$OUT/${RUN_NAME}_${STAMP}.tar"
TMP="$ARCHIVE.partial"
STATUS="$ARCHIVE.exit"
LOG="$ARCHIVE.log"
trap 'result=$?; printf "%s\n" "$result" > "$STATUS"; if (( result != 0 )); then rm -f "$TMP"; fi' EXIT

SOURCE_BYTES=$(du -sb "$BASE/$SOURCE" | cut -f1)
FREE_BYTES=$(df -B1 --output=avail "$OUT" | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < SOURCE_BYTES + 1073741824 )); then
    echo "Insufficient free space: checkpoint=$SOURCE_BYTES, free=$FREE_BYTES" | tee "$LOG"
    exit 3
fi

{
    printf 'Start: %s\n' "$(date -u +%FT%TZ)"
    printf 'Source: %s\n' "$BASE/$SOURCE"
    printf 'Size: %s bytes\n' "$SOURCE_BYTES"
    printf 'Archive: %s\n' "$ARCHIVE"
} > "$LOG"

# Tar relative to /workspace, so a new machine can restore with tar -C /workspace.
tar -C "$BASE" -cf "$TMP" "$SOURCE" >> "$LOG" 2>&1
mv -f "$TMP" "$ARCHIVE"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
tar -tf "$ARCHIVE" | grep -E '/(params|train_state|norm_stats.json)' | head -n 20 >> "$LOG" || true
printf 'Complete: %s\n' "$(date -u +%FT%TZ)" >> "$LOG"
