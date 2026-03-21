## Judge Mode (Winning Project Dashboard)

This repo’s Streamlit app (`apps/streamlit_app.py`) includes a **Judge Mode** toggle intended for judges and report screenshot generation.

### Safety (must-read)

- **Do NOT upload official competition IQ data** to Streamlit Cloud.
- This app never loads official IQ data for cloud rendering.
- Judge-facing metrics are loaded only from **local** CSV files under `submissions/`.

### How Judge Mode works

In the sidebar, enable **`Judge Mode`**. The dashboard switches to a 6-tab **Judge Tour**:

1. **Problem** (Figure 1 + Figure 2)
2. **Core Submission** (Figure 4 + live inference)
3. **Results** (Figure 3 + Figure 5)
4. **Efficiency**
5. **Completed Research Extension** (Figure 6 + completed site-aware digital twin + AI-RAN controller demo)
6. **Why It Matters for Gary**

### Final Submission (Best Known) vs Submission Explorer

- **Final Submission (Best Known):** Selects the highest-priority folder under `submissions/` that contains `main.py`, using the order in `src/edge_ran_gary/submission_adapter.py` (e.g. **`leaderboard_v9`** first when that folder exists).
- **Submission Explorer:** Lets you pick any discovered package (`leaderboard_v5`, `leaderboard_v9`, …) from disk.

**Live inference:** The sidebar choice drives `evaluate()` from `submissions/<pkg>/main.py` on **synthetic Judge Mode demo IQ** only (never competition data in-cloud).

### Authoritative metrics source

Judge Mode loads the CV / leaderboard table from:

- `submissions/submission_metrics.csv` (authoritative when present)
- Optional: `submissions/leaderboard_summary.csv` (if you include it)
- Optional: `docs/final_report_figures.yaml` — per-figure caption overrides (`figure_1` … `figure_6`). Copy from `docs/final_report_figures.example.yaml`. Requires **PyYAML** (`pip install pyyaml`).

The **Canonical Final Submission** card in **Core Submission** is the **single source of truth** for judges when CSVs are present. If CSVs are missing, the app shows a labeled placeholder schema and helper text.

### Synthetic demo IQ (Judge Mode)

- The Judge demo generator **always** adds **complex Gaussian noise for the full window** and a **structured burst in the middle ~30%**.
- **Demo class = mixed** (not pure noise-only). Interpretation text is shown under **Figure 2**.

### Micro-Twin metadata (when used outside Judge Mode)

- Each sample has a metadata row: `label`, `zone_id`, `snr_db`, `cfo_hz`, `num_taps`, `sample_rate_hz`, `signal_type`, etc., plus **landmark_name** resolved from the twin config.
- **Zeros in the waveform are not a reliable “noise-only” indicator** — use the **metadata label** and fields above.

### Core vs Extension separation (non-negotiable)

Judge Mode explicitly distinguishes:

- **Core judged submission:** feature-based binary detector trained on official SpectrumX labeled data; metrics from local CSVs; optional live `evaluate()` on **synthetic** IQ.
- **Completed research extension (non-scoring):** Gary digital twin + site-aware AI-RAN controller demo (implemented in this UI) with proxy KPIs.
- **Next research scaling path (future integration):** DeepMIMO / Sionna RT realism upgrades and related beam/channel/coverage visualizations.

### Gary Micro-Twin 3D building model

The **Completed Research Extension** tab uses **pydeck** as the **main canvas**:

- **Scenario toolbar** (demand, occupancy, RF environment, **school/after-hours/weekend**, **normal vs event**), then **focus site**.
- **Central 3D map:** extruded footprints; **violet** demand hotspots; **red** interference proxies; **blue** hypothetical gNB markers; tooltips on pickable layers.
- **Radio environment cards** (always visible): gNB position, demand zones, interference story, low-7 GHz LOS / penetration / blockage **proxy model** with **implemented now** + **next scaling path** tags.
- **Users at this site** persona cards (City Hall / Library / West Side).
- **RAN controller:** five-column **pipeline** (sense → belief → site → action → KPI) plus candidate actions and a **reason** line.
- **Five proxy KPIs** (coverage, coexistence, community benefit, energy, continuity) — labeled as non-measured.
- **Plain-language** “signals × place” panel and a **6G research roadmap** (DeepMIMO / Sionna RT / beam–coverage UI) as the **next scaling path** only.

If `pydeck` is not installed, the app shows a judge-safe message (no raw tracebacks).

See **`docs/MICROTWIN_REALISM_PLAN.md`** for what is proxy vs implemented.

### Recommended screenshot sequence

- Figure 1 + Figure 2: **Problem**
- Figure 4: **Core Submission** (card + live inference panel)
- Figure 3 + Figure 5: **Results**
- Figure: **Efficiency**
- Figure 6 + extension scene + controller loop: **Completed Research Extension**

### Run locally

From repo root:

```bash
streamlit run apps/streamlit_app.py
```

See also: `docs/MICROTWIN_REALISM_PLAN.md`, `docs/STREAMLIT_FIGURE_MODE.md`.
