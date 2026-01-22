"""
Root-level Streamlit app wrapper for Cloud deployment.

This wrapper ensures Streamlit Cloud can find the entry point at the repo root.
It imports and runs the main Streamlit app from apps/streamlit_app.py.
"""

import sys
from pathlib import Path

# Add src to path for imports
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import the main Streamlit app
# Since apps/streamlit_app.py runs on import, we just need to import it
try:
    import apps.streamlit_app
except ImportError as e:
    # Fallback: try to run the app file directly
    import streamlit as st
    st.error(f"Failed to import Streamlit app: {e}")
    st.info("Please ensure apps/streamlit_app.py exists and is properly configured.")
    st.stop()
