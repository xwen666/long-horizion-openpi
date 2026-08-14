#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/cc/openpi}"
ROOT_DIR="${ROOT_DIR:-${OPENPI_DIR}}"
DATASET_ROOT="${OPENPI_GRASP_DATASET_ROOT:-${OPENPI_DIR}/grasp}"
SPLITS_PATH="${OPENPI_GRASP_SPLITS_PATH:-${OPENPI_DIR}/grasp_splits/splits.json}"

CONFIG_NAME="${CONFIG_NAME:-pi05_grasp_low_mem_finetune}"
EXP_NAME="${EXP_NAME:-grasp_jax_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-16}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_DIR}/outputs/openpi_checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${OPENPI_DIR}/outputs/openpi_assets}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
UV_RUN_ARGS=(${UV_RUN_ARGS:---no-sync})
JAX_CUDA_PACKAGE="${JAX_CUDA_PACKAGE:-jax[cuda13]}"

export OPENPI_GRASP_REPO_ID="${OPENPI_GRASP_REPO_ID:-grasp}"
export OPENPI_GRASP_DATASET_ROOT="${DATASET_ROOT}"
export OPENPI_GRASP_SPLITS_PATH="${SPLITS_PATH}"
export OPENPI_GRASP_PROMPT="${OPENPI_GRASP_PROMPT:-grasp the object}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

cd "${OPENPI_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[train] uv is missing. Run ${OPENPI_DIR}/install_openpi_jax_deps.sh first." >&2
  exit 1
fi

echo "[train] config=${CONFIG_NAME} exp=${EXP_NAME}"
echo "[train] dataset=${OPENPI_GRASP_DATASET_ROOT}"
echo "[train] splits=${OPENPI_GRASP_SPLITS_PATH}"

if ! uv run "${UV_RUN_ARGS[@]}" python - <<'PY'
import jax
import ml_dtypes
import orbax.checkpoint as ocp
from packaging.version import Version

if Version(ml_dtypes.__version__) < Version("0.5.0"):
    raise SystemExit(f"ml_dtypes too old: {ml_dtypes.__version__}")
print("jax", jax.__version__)
print("ml_dtypes", ml_dtypes.__version__)
print("orbax.checkpoint", getattr(ocp, "__version__", "unknown"))
print("devices", jax.devices())
PY
then
  echo "[train] repairing JAX deps: ${JAX_CUDA_PACKAGE} + ml-dtypes>=0.5.0 + orbax-checkpoint>=0.11.32"
  if [[ "${JAX_CUDA_PACKAGE}" == *cuda13* ]]; then
    uv --no-config pip uninstall jax-cuda12-plugin jax-cuda12-pjrt || true
  fi
  uv --no-config pip install --upgrade "${JAX_CUDA_PACKAGE}" "ml-dtypes>=0.5.0" "orbax-checkpoint>=0.11.32"
  uv run "${UV_RUN_ARGS[@]}" python - <<'PY'
import jax
import ml_dtypes
import orbax.checkpoint as ocp
print("jax", jax.__version__)
print("ml_dtypes", ml_dtypes.__version__)
print("orbax.checkpoint", getattr(ocp, "__version__", "unknown"))
print("devices", jax.devices())
PY
fi
if [ -f "${ASSETS_BASE_DIR}/${CONFIG_NAME}/${OPENPI_GRASP_REPO_ID:-grasp}/norm_stats.json" ]; then
  echo "[train] norm stats already exist at ${ASSETS_BASE_DIR}/${CONFIG_NAME}/${OPENPI_GRASP_REPO_ID:-grasp}/norm_stats.json, skipping recompute"
else
  uv run "${UV_RUN_ARGS[@]}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--batch-size=${BATCH_SIZE}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--save-interval=${SAVE_INTERVAL}"
  "--eval-interval=${EVAL_INTERVAL}"
  "--eval-num-batches=${EVAL_NUM_BATCHES}"
  "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
)

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  TRAIN_ARGS+=("--overwrite")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--resume")
fi
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--wandb-enabled")
fi

uv run "${UV_RUN_ARGS[@]}" scripts/train.py "${TRAIN_ARGS[@]}"
