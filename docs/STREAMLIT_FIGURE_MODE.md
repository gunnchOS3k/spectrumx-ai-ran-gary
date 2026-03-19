## Streamlit Figure Mode / Report Mode

This repo’s Streamlit dashboard (`apps/streamlit_app.py`) supports two UI modes:

- **Standard Mode**: the original baseline-comparison dashboard (upload/demo + prediction + plots).
- **Figure Mode**: a **report-figure generator** that reorganizes content into report-oriented tabs and adds screenshot-friendly polish (titles, spacing, captions, and editable “report notes”).

### Safety (must-read)

- **Do NOT upload official competition IQ data to Streamlit Cloud.**
- Official SpectrumX data must remain **local-only** and must never be committed or uploaded.
- Use **demo/synthetic** IQ samples in the cloud app.

### How to launch locally

From repo root:

```bash
streamlit run apps/streamlit_app.py
```

(The root wrapper `streamlit_app.py` also works if you prefer.)

### How Figure Mode works

In the sidebar:

1. Set **Mode → Figure Mode**.
2. (Optional) Toggle **Screenshot Preset** for consistent figure sizing and cleaner layouts.
3. Edit **Report Notes** (section title, figure number, caption) so screenshots match your evolving report draft.
4. Load data via:
   - Upload a `.npy`, or
   - Use demo data, or
   - Use Micro-Twin demo samples (synthetic only).

Figure Mode persists across reruns using `st.session_state`.

### Tabs → Report section mapping

- **Overview** → report intro / task framing
- **Input & Preprocessing** → dataset usage + preprocessing figure panels (IQ/PSD/spectrogram)
- **Baseline Detectors** → algorithm design + baseline results (energy + spectral flatness)
- **Feature Extraction** → implementation details of handcrafted features (table + optional bar chart)
- **Prediction Path** → inference flow diagram (screenshot-friendly)
- **Results & Leaderboard** → experimental results + leaderboard progress (staged headline numbers)
- **Micro-Twin (Future Work)** → appendix / future work (clearly labeled as non-core submission)

### Recommended screenshots for the final report

- **Figure 1**: Overview tab (task framing + safety note)
- **Figure 2**: Input & Preprocessing tab (Time Domain + PSD + Spectrogram panel)
- **Figure 3**: Feature Extraction tab (feature table + top-feature bar chart)
- **Figure 4**: Prediction Path tab (decision flow)
- **Figure 5**: Results & Leaderboard tab (headline results table)
- **Figure 6**: Micro-Twin (Future Work) tab (extension panel for appendix)

### Notes

- Figure Mode is intentionally **additive**: it preserves the existing Standard Mode behavior.
- No official-data download or training is performed in the Streamlit app.

