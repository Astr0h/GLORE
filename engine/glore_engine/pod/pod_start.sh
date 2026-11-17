#!/usr/bin/env bash
# GLORE pod start-up. GLORE uploads this to the network volume and the pod runs it
# as its start command. Whatever happens, the pod terminates itself at the end so
# it never keeps billing.
set -u
export GLORE_ROOT=/workspace
export HF_HOME=/workspace/.cache/hf
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
export UV_LINK_MODE=copy
export PATH=/workspace/.cache/bin:$PATH
export PYTHONUNBUFFERED=1
LOG=/workspace/glore/pod-${RUNPOD_POD_ID:-local}.log
mkdir -p /workspace/glore /workspace/.cache/bin

terminate() {
  echo "[glore] pod_start.sh exiting - terminating pod" >> "$LOG"
  KEY="${GLORE_RUNPOD_KEY:-${RUNPOD_API_KEY:-}}"
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "$KEY" ]; then
    curl -s -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"action":"terminate"}' >> "$LOG" 2>&1 \
    || curl -s -X DELETE "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer $KEY" >> "$LOG" 2>&1
  fi
  sleep 300
}
trap terminate EXIT

{
  echo "[glore] pod ${RUNPOD_POD_ID:-?} starting $(date -u)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  NEED=""
  for b in curl git ffmpeg; do command -v $b >/dev/null 2>&1 || NEED="$NEED $b"; done
  if [ -n "$NEED" ] || ! ldconfig -p | grep -q libsndfile; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      curl ca-certificates git ffmpeg libsndfile1 build-essential >/dev/null
  fi
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/.cache/bin UV_NO_MODIFY_PATH=1 sh
  fi
} >> "$LOG" 2>&1

# The trainer itself is standard-library Python; uv provides the interpreter.
uv run --no-project --python 3.11 /workspace/glore/glore_pod.py >> "$LOG" 2>&1
