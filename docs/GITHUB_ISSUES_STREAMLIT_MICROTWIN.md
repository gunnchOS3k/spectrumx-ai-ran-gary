# GitHub issues: Streamlit Cloud fix + Gary Micro-Twin

Use these to create or adjust issues (light touch; do not mass-delete existing issues).

---

## 1. Fix Streamlit Cloud protobuf crash

**Title:** Fix Streamlit Cloud startup/import crash (protobuf + Python version)

**Body:**
- **Root cause:** Python 3.13 + streamlit 1.9.0 + protobuf 6.x can trigger `TypeError: Descriptors cannot be created directly` in `streamlit/proto/*_pb2.py`.
- **Acceptance criteria:**
  - [ ] Pin Streamlit to a version compatible with protobuf 5.x (e.g. `streamlit>=1.40.0,<1.45.0`).
  - [ ] Pin `protobuf>=5.28,<6` in `requirements.txt`.
  - [ ] Add `runtime.txt` with `python-3.11.9` (or 3.12).
  - [ ] Document in `docs/STREAMLIT_CLOUD_DEPLOY.md` how to set Python version in Streamlit Cloud if `runtime.txt` is ignored.
  - [ ] App imports and runs on Streamlit Cloud without crash.
- **How to test:** Deploy on Streamlit Cloud; check logs for import errors.
- **Owner suggestion:** Edmund / Noah  
- **Effort:** S

---

## 2. Gary Micro-Twin v1: config + generator + CLI

**Title:** Gary Micro-Twin v1 — config + generator + CLI

**Body:**
- **Goal:** Competition-safe micro-twin with 3 zones (City Hall, one high school, library) producing reproducible 1s IQ + metadata.
- **Acceptance criteria:**
  - [ ] `configs/gary_micro_twin.yaml` defines 3 zones with zone_id, name, address, lat/lon, radius_m, snr_range, occupancy_prior, signal_mix (qpsk_burst, ofdm_burst, chirp).
  - [ ] Generator produces complex QPSK/OFDM/chirp; AWGN SNR correct; invalid zone_id raises ValueError; metadata reflects actual zone.
  - [ ] CLI: `python -m edge_ran_gary.digital_twin.cli_generate --config configs/gary_micro_twin.yaml --n 50 --out data/synthetic/gary_micro_twin` outputs .npy, metadata.csv, summary.json.
  - [ ] At least one unit test for reproducibility (same seed → same output).
- **How to test:** Run CLI; run `scripts/smoke_test_digital_twin.py`.
- **Owner suggestion:** Edmund / Ananya  
- **Effort:** M

---

## 3. Micro-Twin demo notebook + docs

**Title:** Micro-Twin demo notebook + docs

**Body:**
- **Goal:** Notebook and doc so Ananya/Noah can use the micro-twin for ML tests and demos.
- **Acceptance criteria:**
  - [ ] `notebooks/03_gary_micro_twin_demo.ipynb`: generate ~30 samples; plot IQ, PSD, spectrogram; show distribution by zone/label/signal_type.
  - [ ] `docs/DIGITAL_TWIN_MICRO_TWIN.md`: what it is / what it is not (competition-safe), how it feeds ML tests and Streamlit demos, how it supports AI-RAN narrative.
- **How to test:** Run notebook; read doc for clarity.
- **Owner suggestion:** Noah / Ananya  
- **Effort:** S

---

## 4. Streamlit: micro-twin demo workflow + graceful dataset-missing handling

**Title:** Streamlit: Micro-Twin demo workflow + graceful dataset-missing handling

**Body:**
- **Goal:** In-app Micro-Twin demo and clear message when dataset folder is missing.
- **Acceptance criteria:**
  - [ ] "Generate Micro-Twin Demo Data" button generates 9–15 samples and lets user pick one; show IQ/PSD/spectrogram and prediction (or “model not loaded”).
  - [ ] When no data: show clear “No dataset found” message and option to generate micro-twin demo files.
  - [ ] “Generate Demo IQ Sample” shows visualizations immediately after click (session state set before rerun); “Demo mode active” banner when using demo data.
- **How to test:** Run Streamlit locally; click both demo buttons; verify no crash and plots appear.
- **Owner suggestion:** Noah  
- **Effort:** M

---

**Creating issues (optional):**

```bash
gh issue create --title "Fix Streamlit Cloud protobuf crash" --body-file - <<'BODY'
<paste body from issue 1>
BODY
```

Repeat for issues 2–4 with the corresponding titles and bodies.
