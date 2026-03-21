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
5. **Completed Research Extension** (Figure 6 + Gary digital twin + **Near-RT RIC–style** AI-RAN demo)
6. **Why It Matters for Gary**

**Above the tabs:** **Why judges can trust this — evidence** (three buckets) and a compact **AI-RAN / O-RAN alignment** card (extension is demo-only; no live RIC).

### Final Submission (Best Known) vs Submission Explorer

- **Final Submission (Best Known):** Highest-priority folder under `submissions/` with `main.py` (`src/edge_ran_gary/submission_adapter.py`; e.g. **`leaderboard_v9`** when present).
- **Submission Explorer:** Any discovered package, including nested paths (e.g. `leaderboard_baseline_v1/test3k`). Repo root is resolved from the app file — **not** from process CWD.

**Diagnostic expander** in the sidebar lists discovered folders, `main.py` / artifacts, and **leaderboard_v9** visibility.

**Live inference:** `evaluate()` on **synthetic Judge Mode demo IQ** only.

### Authoritative metrics source

- `submissions/submission_metrics.csv`, optional `leaderboard_summary.csv`, optional `docs/final_report_figures.yaml` (PyYAML).

### Core vs Extension separation (non-negotiable)

- **Core judged submission:** SpectrumX DAC detector; official data **offline**; CSVs + synthetic IQ in-app only.
- **Completed extension:** Gary **digital-twin wireless scene** + **O-RAN–aligned Near-RT RIC xApp abstraction** (proxies; non-scoring).
- **Next scaling:** **DeepMIMO** (channels), **Sionna RT** (ray-tracing), **NVIDIA AI Aerial / Omniverse** (twin-scale RF) — **integration-ready** paths in `data/*` and `configs/*`; **stubs** until wired; **do not** claim they run the judged detector.

### Completed Research Extension (summary)

- **Guided demo strip** → **3D radio scene** (coverage halos, gNB, demand, IF, propagation stress, links) + **triple legend**.
- **Propagation / Coverage** — proxy table + **bar chart** for screenshots.
- **Simulation backbone** — three pillars (DeepMIMO, Sionna RT, Aerial/AODT) + JSON drop-zone status.
- **Optional paths** expander: `data/deepmimo/`, `data/sionna_rt/`, `data/aerial_omniverse/`, `configs/wireless_scene/`, `configs/ric/`, etc.

If **`pydeck`** is missing, the extension shows a safe message.

### Documentation

- `docs/MICROTWIN_REALISM_PLAN.md` — extension behavior  
- `docs/SIMULATION_BACKBONE_PLAN.md` — simulation hooks + future compute  
- `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md` — capability matrix  

### Run locally

```bash
streamlit run apps/streamlit_app.py
```

See also: `docs/STREAMLIT_FIGURE_MODE.md`.
