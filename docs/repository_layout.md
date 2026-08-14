# Repository Layout

The repository is split into source code, reproducibility metadata, and local
runtime resources.

```text
src/openpi/                 Core OpenPI package and Cosmos/WAM model code
scripts/                    Training, serving, setup, and evaluation entrypoints
tools/                      Cache builders, diagnostics, workers, and grasp tools
examples/                   Upstream OpenPI examples and LIBERO client
configs/                    Environment templates and experiment records
docs/                       Project documentation
docs/assets/                Small project diagrams and documentation assets
packages/                   OpenPI workspace packages
third_party/                Git submodules (LIBERO and ALOHA)

assets/                     Local norm stats and small runtime assets
checkpoints/                Local training checkpoints
datasets/                   Local LeRobot datasets
cosmos_cache/               Local WorldPilot/Cosmos latent cache
cosmos_checkpoints/         Local Cosmos model checkpoints
cosmos-predict2.5/          External Cosmos Predict checkout
outputs/                    Local evaluation outputs and logs
```

The bottom section is deliberately ignored by Git. It is not part of the
source release and must be restored separately on another machine.

The architecture sketch is available at
[`assets/worldpilot_architecture.png`](assets/worldpilot_architecture.png).

## Environments

OpenPI and Cosmos use separate virtual environments because their dependency
stacks and Python versions differ:

```text
${OPENPI_WAM_ROOT}/.venv
${OPENPI_WAM_ROOT}/cosmos-predict2.5/.venv
```

The OpenPI server launches the Cosmos worker as a subprocess. A user running
LIBERO should activate only the OpenPI environment; the worker receives the
Cosmos interpreter path through `COSMOS_PYTHON` or the serving arguments.

## GitHub Upload Boundary

Commit source, tests, documentation, submodule metadata, and configuration
templates. Do not commit model weights, datasets, latent caches, videos,
virtual environments, or absolute-path logs. The root `.gitignore` encodes
this boundary.
