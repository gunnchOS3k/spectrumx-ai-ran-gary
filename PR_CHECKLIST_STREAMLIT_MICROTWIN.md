# PR checklist: Streamlit Cloud fix + Gary Micro-Twin

Copy this into your PR description.

## Checklist

- [ ] Streamlit runs locally: `streamlit run apps/streamlit_app.py` (or `streamlit run streamlit_app.py`)
- [ ] Generator CLI runs: `python -m edge_ran_gary.digital_twin.cli_generate --config configs/gary_micro_twin.yaml --n 50 --out data/synthetic/gary_micro_twin`
- [ ] No secrets committed; `.env` and `.streamlit/secrets.toml` are in `.gitignore`
- [ ] Streamlit Cloud deploy steps documented in `docs/STREAMLIT_CLOUD_DEPLOY.md` and `docs/STREAMLIT_DEPLOY.md`

## How to run locally

```bash
# From repo root
pip install -r requirements.txt
streamlit run apps/streamlit_app.py
# or
streamlit run streamlit_app.py
```

Micro-Twin generator:

```bash
python -m edge_ran_gary.digital_twin.cli_generate --config configs/gary_micro_twin.yaml --n 50 --out data/synthetic/gary_micro_twin
```

## How to deploy on Streamlit Cloud

1. Connect repo `gunnchOS3k/spectrumx-ai-ran-gary`; main file path: `streamlit_app.py`.
2. In **Settings**, set **Python version** to **3.11** (or 3.12) so the protobuf/Streamlit crash is avoided.
3. Deploy; if `runtime.txt` is honored, Python 3.11 will be used automatically.
4. See `docs/STREAMLIT_CLOUD_DEPLOY.md` for details.
