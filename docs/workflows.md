# Workflows

## WorldPilot-style LIBERO training

1. Prepare the four LeRobot LIBERO suites locally.
2. Download or generate the matching Cosmos/WAM cache.
3. Export the paths in `configs/env.example`.
4. Compute or verify the combined LIBERO normalization statistics.
5. Train with `scripts/train.py pi05_cosmos_libero_all`.

The cache must use the same episode/frame numbering as the LeRobot dataset.
The combined config validates cache shapes and alignment before training.

## Offline cache training

Training reads `.npz` files through `AttachCosmosLatent`; Cosmos is frozen and
is not started during the data-loader path. The cache builder is:

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/build_cosmos_latent_cache.py --help
```

## Realtime evaluation

The standard realtime evaluator starts one OpenPI server and one persistent
Cosmos/WAM worker. It calls WAM once per 16-step action chunk:

```bash
CHECKPOINT=/path/to/checkpoint \
OPENPI_PYTHON=/path/to/openpi-wam/.venv/bin/python \
COSMOS_PYTHON=/path/to/cosmos-predict2.5/.venv/bin/python \
SERVER_GPU=0 WAM_GPU=1 PORT=8020 \
bash scripts/run_libero_standard_eval_38000_realtime.sh
```

For long runs, launch the command inside `tmux` and store results under
`outputs/`; that directory is intentionally ignored.
