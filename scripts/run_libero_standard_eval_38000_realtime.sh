#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OPENPI_PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
COSMOS_REPO="${COSMOS_REPO:-${REPO_ROOT}/cosmos-predict2.5}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_REPO}/.venv/bin/python}"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/pi05_cosmos_libero_all/libero_all_wam_train/38000}"
COSMOS_POLICY_ROOT="${COSMOS_POLICY_ROOT:-${REPO_ROOT}/cosmos_checkpoints/Cosmos-Policy-LIBERO-Predict2-2B}"
COSMOS_POLICY_CHECKPOINT="${COSMOS_POLICY_CHECKPOINT:-${COSMOS_POLICY_ROOT}/Cosmos-Policy-LIBERO-Predict2-2B.pt}"
COSMOS_POLICY_STATS="${COSMOS_POLICY_STATS:-${COSMOS_POLICY_ROOT}/libero_dataset_statistics.json}"
COSMOS_POLICY_TEXT_EMBEDDINGS="${COSMOS_POLICY_TEXT_EMBEDDINGS:-${COSMOS_POLICY_ROOT}/libero_t5_embeddings.pkl}"
COSMOS_POLICY_VAE_PATH="${COSMOS_POLICY_VAE_PATH:-/root/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth}"
COSMOS_POLICY_CONFIG="${COSMOS_POLICY_CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
COSMOS_POLICY_CONFIG_FILE="${COSMOS_POLICY_CONFIG_FILE:-cosmos_predict2/_src/predict2/cosmos_policy/config/config.py}"

# Use separate GPUs for the JAX OpenPI server and the PyTorch Cosmos worker.
SERVER_GPU="${SERVER_GPU:-2}"
WAM_GPU="${WAM_GPU:-0}"
PORT="${PORT:-8020}"
COSMOS_POLICY_NUM_STEPS="${COSMOS_POLICY_NUM_STEPS:-5}"
COSMOS_POLICY_CHUNK_SIZE="${COSMOS_POLICY_CHUNK_SIZE:-16}"
RUN_NAME="${RUN_NAME:-libero_standard_38000_realtime_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/eval/${RUN_NAME}}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:-10}"

mkdir -p "${OUT_DIR}/results" "${OUT_DIR}/logs" "${OUT_DIR}/videos"

for required_path in "${CHECKPOINT}" "${COSMOS_POLICY_CHECKPOINT}" "${COSMOS_POLICY_STATS}" "${COSMOS_POLICY_TEXT_EMBEDDINGS}" "${COSMOS_POLICY_VAE_PATH}"; do
  [[ -e "${required_path}" ]] || { echo "[eval] missing required path: ${required_path}" >&2; exit 1; }
done

echo "[eval] OpenPI checkpoint: ${CHECKPOINT}"
echo "[eval] Cosmos Policy checkpoint: ${COSMOS_POLICY_CHECKPOINT}"
echo "[eval] OpenPI GPU: ${SERVER_GPU}; Cosmos Policy GPU: ${WAM_GPU}"
echo "[eval] Cosmos Policy denoising steps: ${COSMOS_POLICY_NUM_STEPS}"
echo "[eval] action chunk: ${COSMOS_POLICY_CHUNK_SIZE}"
echo "[eval] suites: ${SUITES}; ${MAX_TASKS} tasks x ${NUM_TRIALS_PER_TASK} init states"
echo "[eval] output: ${OUT_DIR}"

SERVER_LOG="${OUT_DIR}/logs/policy_server.log"
COSMOS_LOG="${OUT_DIR}/logs/cosmos_policy_worker.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] stopping policy server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[eval] starting OpenPI policy server with realtime Cosmos Policy..."
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${OPENPI_PYTHON}" scripts/serve_policy.py \
  --port="${PORT}" \
  --cosmos-cache-mode=realtime \
  --cosmos-python="${COSMOS_PYTHON}" \
  --cosmos-repo="${COSMOS_REPO}" \
  --cosmos-worker-cuda-visible-devices="${WAM_GPU}" \
  --cosmos-worker-log-path="${COSMOS_LOG}" \
  --cosmos-resolution=224,224 \
  --cosmos-num-steps="${COSMOS_POLICY_NUM_STEPS}" \
  --cosmos-guidance=7.0 \
  --cosmos-vae-path="${COSMOS_POLICY_VAE_PATH}" \
  --cosmos-latent-dim=12544 \
  --cosmos-policy-checkpoint="${COSMOS_POLICY_CHECKPOINT}" \
  --cosmos-policy-config="${COSMOS_POLICY_CONFIG}" \
  --cosmos-policy-config-file="${COSMOS_POLICY_CONFIG_FILE}" \
  --cosmos-policy-dataset-stats="${COSMOS_POLICY_STATS}" \
  --cosmos-policy-text-embeddings="${COSMOS_POLICY_TEXT_EMBEDDINGS}" \
  --cosmos-policy-num-steps="${COSMOS_POLICY_NUM_STEPS}" \
  --cosmos-policy-chunk-size="${COSMOS_POLICY_CHUNK_SIZE}" \
  policy:checkpoint \
  --policy.config=pi05_cosmos_libero_all \
  --policy.dir="${CHECKPOINT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

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
raise SystemExit(f"Timed out waiting for policy server port {port}")
PY

for suite in ${SUITES}; do
  echo "[eval] running ${suite}: ${MAX_TASKS} tasks x ${NUM_TRIALS_PER_TASK} initial states"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${OPENPI_PYTHON}" examples/libero/main.py \
    --args.host=127.0.0.1 \
    --args.port="${PORT}" \
    --args.libero-mode=standard \
    --args.task-suite-name="${suite}" \
    --args.max-tasks="${MAX_TASKS}" \
    --args.num-trials-per-task="${NUM_TRIALS_PER_TASK}" \
    --args.replan-steps=16 \
    --args.no-save-videos \
    --args.abort-on-error \
    --args.results-path="${OUT_DIR}/results/${suite}.jsonl" \
    --args.video-out-path="${OUT_DIR}/videos/${suite}" \
    2>&1 | tee "${OUT_DIR}/logs/${suite}.log"
done

"${OPENPI_PYTHON}" - "${OUT_DIR}/results" <<'PY'
import json
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
total_episodes = 0
total_successes = 0
print("[eval] success-rate summary")
for path in sorted(results_dir.glob("*.jsonl")):
    summary = None
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "suite_summary":
            summary = record
    if summary is None:
        print(f"[eval] {path.stem}: missing suite_summary")
        continue
    episodes = int(summary["episodes"])
    successes = int(summary["successes"])
    total_episodes += episodes
    total_successes += successes
    print(f"[eval] {path.stem}: {successes}/{episodes} = {successes / episodes:.4%}")

print(f"[eval] overall: {total_successes}/{total_episodes} = {total_successes / total_episodes:.4%}")
PY

echo "[eval] completed; results are in ${OUT_DIR}/results"
