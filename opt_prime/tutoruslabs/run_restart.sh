#!/bin/bash

set -euo pipefail

CONTAINER_NAME="${1:-etri_test_container}"

usage() {
  cat <<EOF
Usage:
  bash run_restart.sh [CONTAINER_NAME]

Default:
  CONTAINER_NAME=etri_test_container

Behavior:
  - If container exists: docker kill (ignore if already stopped) -> docker start
  - If container does not exist: exit with error
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if ! sudo docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "ERROR: container '$CONTAINER_NAME' not found." >&2
  exit 1
fi

sudo docker kill "$CONTAINER_NAME" >/dev/null 2>&1 || true
sudo docker start "$CONTAINER_NAME" >/dev/null

echo "OK: restarted '$CONTAINER_NAME'"


