#!/usr/bin/env bash
set -euo pipefail

cd /cc/openpi_wam

OPENPI_PYTHON="${OPENPI_PYTHON:-/cc/openpi_wam/.venv/bin/python}"

CHECKPOINT="${CHECKPOINT:-/cc/openpi_wam/checkpoints/pi05_cosmos_libero_all/libero_all_wam_full_4gpu_bs4_worldpilot_dropout/37500}"
WAIT_FOR_PID="${WAIT_FOR_PID:-551613}"
SERVER_GPU="${SERVER_GPU:-4}"
WAM_GPU="${WAM_GPU:-0}"
PORT="${PORT:-8020}"
WAM_OFFLOAD="${WAM_OFFLOAD:-0}"
WAM_NUM_STEPS="${WAM_NUM_STEPS:-5}"
WAM_GUIDANCE="${WAM_GUIDANCE:-7.0}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_START="${TASK_START:-0}"
MAX_TASKS="${MAX_TASKS:-10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
RUN_NAME="${RUN_NAME:-libero_standard_37500_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/cc/openpi_wam/outputs/eval/${RUN_NAME}}"

mkdir -p "${OUT_DIR}/results" "${OUT_DIR}/logs"

echo "[eval] checkpoint: ${CHECKPOINT}"
echo "[eval] output dir: ${OUT_DIR}"
echo "[eval] server gpu: ${SERVER_GPU}"
echo "[eval] wam gpu: ${WAM_GPU}"
echo "[eval] wam offload: ${WAM_OFFLOAD}"
echo "[eval] wam num steps: ${WAM_NUM_STEPS}"
echo "[eval] wam guidance: ${WAM_GUIDANCE}"
echo "[eval] port: ${PORT}"
echo "[eval] suites: ${SUITES}"
echo "[eval] task start: ${TASK_START}"
echo "[eval] max tasks per suite: ${MAX_TASKS}"
echo "[eval] trials per task: ${NUM_TRIALS_PER_TASK}"

if ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; then
  echo "[eval] waiting for training pid ${WAIT_FOR_PID} to finish..."
  while ps -p "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[eval] training pid ${WAIT_FOR_PID} finished."
fi

SERVER_LOG="${OUT_DIR}/logs/policy_server.log"
COSMOS_LOG="${OUT_DIR}/logs/cosmos_worker.log"
COSMOS_OFFLOAD_ARGS=()
if [[ "${WAM_OFFLOAD}" == "1" || "${WAM_OFFLOAD}" == "true" || "${WAM_OFFLOAD}" == "TRUE" ]]; then
  COSMOS_OFFLOAD_ARGS=(
    --cosmos-offload-diffusion-model
    --cosmos-offload-text-encoder
    --cosmos-offload-tokenizer
  )
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && ps -p "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[eval] stopping policy server pid ${SERVER_PID}"
    kill "${SERVER_PID}" || true
    wait "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT

echo "[eval] starting policy server..."
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${OPENPI_PYTHON}" scripts/serve_policy.py \
  --port="${PORT}" \
  --cosmos-cache-mode=realtime \
  --cosmos-python=/cc/openpi_wam/cosmos-predict2.5/.venv/bin/python \
  --cosmos-worker-cuda-visible-devices="${WAM_GPU}" \
  --cosmos-worker-log-path="${COSMOS_LOG}" \
  --cosmos-resolution=224,224 \
  --cosmos-num-steps="${WAM_NUM_STEPS}" \
  --cosmos-guidance="${WAM_GUIDANCE}" \
  --cosmos-latent-dim=12544 \
  "${COSMOS_OFFLOAD_ARGS[@]}" \
  policy:checkpoint \
  --policy.config=pi05_cosmos_libero_all \
  --policy.dir="${CHECKPOINT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[eval] policy server pid: ${SERVER_PID}"
"${OPENPI_PYTHON}" - "${PORT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
for _ in range(240):
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect(("127.0.0.1", port))
    except OSError:
        time.sleep(5)
    else:
        sock.close()
        sys.exit(0)
raise SystemExit(f"Timed out waiting for port {port}")
PY
echo "[eval] policy server is accepting connections."

for suite in ${SUITES}; do
  echo "[eval] running ${suite}: ${MAX_TASKS} tasks x ${NUM_TRIALS_PER_TASK} init states"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${OPENPI_PYTHON}" examples/libero/main.py \
    --args.host=127.0.0.1 \
    --args.port="${PORT}" \
    --args.libero-mode=standard \
    --args.task-suite-name="${suite}" \
    --args.task-start="${TASK_START}" \
    --args.max-tasks="${MAX_TASKS}" \
    --args.num-trials-per-task="${NUM_TRIALS_PER_TASK}" \
    --args.replan-steps=16 \
    --args.no-save-videos \
    --args.abort-on-error \
    --args.results-path="${OUT_DIR}/results/${suite}.jsonl" \
    --args.video-out-path="${OUT_DIR}/videos/${suite}" \
    2>&1 | tee "${OUT_DIR}/logs/${suite}.log"
done

echo "[eval] finished all standard LIBERO suites."
echo "[eval] results: ${OUT_DIR}/results"
"${OPENPI_PYTHON}" - "${OUT_DIR}/results" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
total_episodes = 0
total_successes = 0
print("[eval] success-rate summary")
for path in sorted(results_dir.glob("*.jsonl")):
    suite_summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "suite_summary":
            suite_summary = record
    if suite_summary is None:
        print(f"[eval] {path.stem}: missing suite_summary")
        continue
    episodes = int(suite_summary["episodes"])
    successes = int(suite_summary["successes"])
    total_episodes += episodes
    total_successes += successes
    rate = successes / episodes if episodes else 0.0
    print(f"[eval] {path.stem}: {successes}/{episodes} = {rate:.4%}")

overall = total_successes / total_episodes if total_episodes else 0.0
print(f"[eval] overall: {total_successes}/{total_episodes} = {overall:.4%}")
PY
