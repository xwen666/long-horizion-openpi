"""Portable paths for the OpenPI/WAM workspace.

The training and serving code can run from any checkout location. Set
``OPENPI_WAM_ROOT`` when the package is installed outside the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the workspace root containing ``src/``, ``scripts/`` and ``tools/``."""
    configured = os.environ.get("OPENPI_WAM_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def local_path(relative: str) -> str:
    """Resolve a workspace-relative path, unless an absolute path was supplied."""
    path = Path(relative).expanduser()
    return str(path if path.is_absolute() else repo_root() / path)


def configured_path(env_name: str, relative_default: str) -> str:
    """Read a path from an environment variable with a workspace-relative default."""
    return os.environ.get(env_name, local_path(relative_default))
