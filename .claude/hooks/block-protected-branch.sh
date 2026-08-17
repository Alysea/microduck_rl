#!/usr/bin/env bash
# PreToolUse guard: refuse Write/Edit while HEAD is on a protected branch.
# Project rule: all work happens on a feature branch, never develop or main.
set -u

repo="/home/steve/Project/Repo/mjlab_microduck"
branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null)"

case "$branch" in
  develop | main)
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"On protected branch %s. Project rule: never edit files on develop or main -- create or switch to a feature branch first."}}\n' "$branch"
    ;;
esac

exit 0
