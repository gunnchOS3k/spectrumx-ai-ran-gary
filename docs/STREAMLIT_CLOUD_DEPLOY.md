# Streamlit Cloud Deploy (Python + Dependencies)

This doc explains how Streamlit Cloud uses `requirements.txt` and how to set the **Python version** so the app starts without the protobuf/Streamlit import crash.

## Root cause of the crash

- **Symptom:** `TypeError: Descriptors cannot be created directly` in `streamlit/proto/..._pb2.py`
- **Cause:** Mismatch between Streamlit’s protobuf-generated code and the installed `protobuf` (e.g. Python 3.13 + streamlit 1.9.0 + protobuf 6.x).
- **Fix:** Use a **supported Python version** (3.11 or 3.12) and **pinned dependency versions** so Streamlit and protobuf are compatible.

## How Streamlit Cloud uses requirements

- Streamlit Cloud installs dependencies from **`requirements.txt`** at the repo root.
- It does **not** install from `apps/requirements.txt` unless that file is the one you point to (we use root `requirements.txt`).
- Pins in this repo:
  - `streamlit>=1.40.0,<1.45.0` — modern Streamlit compatible with current protobuf.
  - `protobuf>=5.28.0,<6` — keeps ML stacks (e.g. torch) happy while avoiding protobuf 6.x edge cases that can break Streamlit.

## Setting Python version on Streamlit Cloud

- **Option 1 — `runtime.txt` (if supported):**  
  We provide `runtime.txt` at repo root with:
  ```text
  python-3.11.9
  ```
  If your Cloud stack respects it, this will force Python 3.11.

- **Option 2 — Cloud UI (recommended):**  
  If `runtime.txt` is ignored:
  1. Open [share.streamlit.io](https://share.streamlit.io) → your app.
  2. **Manage app** → **Settings** (or **Advanced settings**).
  3. Find **Python version** (or “Python”).
  4. Set to **3.11** (or **3.12**), then save and **Reboot app**.

## Entry point

- **Main file path:** `streamlit_app.py` (repo root).
- Root `streamlit_app.py` imports and runs the app from `apps/streamlit_app.py`.  
  So the **effective** app file is `apps/streamlit_app.py`.

## After changing Python or requirements

1. Commit `requirements.txt` and, if used, `runtime.txt`.
2. Push to the branch that Streamlit Cloud deploys from.
3. In Cloud: set Python to 3.11 (or 3.12) if needed, then **Reboot app**.
4. Check **Logs** for import errors; they should disappear once the version combo is correct.

## Local run (same stack as Cloud)

```bash
# Python 3.11 recommended
pip install -r requirements.txt
streamlit run streamlit_app.py
# or
streamlit run apps/streamlit_app.py
```

This keeps local behavior aligned with Streamlit Cloud and avoids the protobuf/Streamlit crash.
