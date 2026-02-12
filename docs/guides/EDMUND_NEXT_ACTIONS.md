# Edmund — Next actions

## What changed

- **Gary Micro-Twin v1 (competition-safe):**
  - `configs/gary_micro_twin.yaml`: 3 zones — `city_hall`, `high_school`, `library` — with `landmark_name`, `center_lat`, `center_lon`, `radius_m`, `snr_db_range`, `occupancy_prior`, `waveform_mix` (all editable in YAML).
  - `src/edge_ran_gary/digital_twin/contracts.py`: `DigitalTwinSample` dataclass (iq, label, metadata with required keys).
  - `src/edge_ran_gary/digital_twin/samples.py`: `generate_sample(seed, zone_id, label=None)` → `DigitalTwinSample`; invalid `zone_id` raises `ValueError`.
  - `src/edge_ran_gary/digital_twin/generate.py`: CLI that writes `iq/*.npy`, `metadata.csv`, `manifest.json`.
  - `docs/DIGITAL_TWIN_OUTPUT_SPEC.md`: Output layout and contract for Ananya.
- **Streamlit:**
  - Safety banner at top: do not upload competition data to Cloud.
  - Demo buttons set all needed session state before `st.rerun()` so visualizations appear on the same rerun.
- **Tests:**
  - `tests/test_streamlit_import.py`: imports `apps.streamlit_app` without raising.
  - `tests/test_digital_twin_contract.py`: invalid zone_id raises; label=1 snr in range; QPSK has non-zero imaginary; contract keys present.
- **Repo:** `.gitignore` already includes `.env`, `competition_dataset/`, `*.npy`, `data/synthetic/`, `.streamlit/secrets.toml`.

## What’s left

- Run tests and generator locally; fix any failures.
- Optionally create/update GitHub issues from `docs/ISSUE_UPDATES_PROPOSAL.md`.
- Streamlit Cloud: ensure Main file path is set (e.g. `apps/streamlit_app.py` or `streamlit_app.py`) and Python version 3.11 if needed.

## Commands to run

```bash
# From repo root
pytest

# Generate 200 synthetic samples
python -m edge_ran_gary.digital_twin.generate --config configs/gary_micro_twin.yaml --n 200 --out data/synthetic/gary_micro_twin

# Run Streamlit locally
streamlit run apps/streamlit_app.py
```
