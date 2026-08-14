#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/cc/Evo-RL}"
OPENPI_DIR="${OPENPI_DIR:-${ROOT_DIR}/openpi}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
# RTX 5090 / Blackwell is compute capability 12.0. Prefer current CUDA 13 JAX wheels.
# Override examples:
#   JAX_CUDA_PACKAGE='jax[cuda12]' bash scripts/install_openpi_jax_deps.sh
#   JAX_CUDA_PACKAGE='jax[cuda13-local]' bash scripts/install_openpi_jax_deps.sh
#   JAX_CUDA_PACKAGE='' bash scripts/install_openpi_jax_deps.sh   # keep OpenPI lock as-is
JAX_CUDA_PACKAGE="${JAX_CUDA_PACKAGE:-jax[cuda13]}"

cd "${OPENPI_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] uv not found, installing uv for the current user..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "[install] uv: $(uv --version)"
echo "[install] creating OpenPI Python ${PYTHON_VERSION} environment in ${OPENPI_DIR}/.venv"
uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" .venv

echo "[install] syncing OpenPI locked dependencies (JAX/CUDA12 path, no PyTorch training script needed)"
uv sync --frozen

if [[ -n "${JAX_CUDA_PACKAGE}" ]]; then
  if [[ "${JAX_CUDA_PACKAGE}" == *cuda13* ]]; then
    uv --no-config pip uninstall jax-cuda12-plugin jax-cuda12-pjrt || true
  fi
  echo "[install] overriding JAX wheel for this machine: ${JAX_CUDA_PACKAGE}"
  uv --no-config pip install --upgrade "${JAX_CUDA_PACKAGE}" "ml-dtypes>=0.5.0" "orbax-checkpoint>=0.11.32"
fi

echo "[install] checking JAX devices"
uv run --no-sync python - <<'PY'
import jax
import ml_dtypes
import orbax.checkpoint as ocp
print("jax", jax.__version__)
print("ml_dtypes", ml_dtypes.__version__)
print("orbax.checkpoint", getattr(ocp, "__version__", "unknown"))
print("devices", jax.devices())
PY

cat <<'EOF'
[install] done.

Before training on a new shell:
  cd /home/xwen/pi0.5/openpi
  source .venv/bin/activate

If your server uses a custom PyPI mirror, set UV_INDEX_URL before running this script.
EOF
