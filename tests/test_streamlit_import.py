"""Smoke test: Streamlit app module imports without raising."""

import sys
from pathlib import Path

import pytest

# Repo root on path so apps.* resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def test_apps_streamlit_app_imports():
    """Importing apps.streamlit_app must not raise."""
    import apps.streamlit_app  # noqa: F401
