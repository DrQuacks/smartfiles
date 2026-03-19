from __future__ import annotations

import os
from pathlib import Path


ENV_DATA_DIR = "SMARTFILES_DATA_DIR"


def get_data_dir() -> Path:
    """Return the base data directory for SmartFiles.

    If the environment variable SMARTFILES_DATA_DIR is set, that value is
    used (expanded and resolved). Otherwise, we default to a `.smartfiles`
    directory in the user's home directory.
    """

    env_value = os.getenv(ENV_DATA_DIR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path.home() / ".smartfiles"
