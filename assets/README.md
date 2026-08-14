# Local Assets

This directory is for local normalization statistics and other small runtime
assets. Large checkpoints and datasets are intentionally ignored by Git.

For a new machine, generate or download the required assets into a separate
local resource directory and point the training config at it with
`AssetsConfig` or the relevant `OPENPI_*` environment variable.
