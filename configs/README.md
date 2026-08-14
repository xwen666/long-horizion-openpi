# Configurations

This directory contains reproducibility-oriented configuration files.

- `env.example`: local path and environment variable template.
- `legacy/`: older machine-specific experiment records. These are kept for
  reference and should be edited before reuse.

The model and data config registry remains in
`src/openpi/training/config.py` so the existing OpenPI CLI continues to work:

```bash
python scripts/train.py pi05_cosmos_libero_all --exp-name=run_name
```

Cosmos/WAM defaults can be overridden without editing source code by exporting
the variables listed in `env.example`.
