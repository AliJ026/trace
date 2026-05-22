"""ForensicLens package bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
_LOCAL_DEPENDENCIES = _PROJECT_ROOT / ".python_packages"

if _LOCAL_DEPENDENCIES.exists():
    _deps_path = str(_LOCAL_DEPENDENCIES)
    if _deps_path not in sys.path:
        sys.path.insert(0, _deps_path)
