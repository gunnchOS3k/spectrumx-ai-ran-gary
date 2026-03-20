## Judge Mode (Winning Project Dashboard)

This repo’s Streamlit app (`apps/streamlit_app.py`) includes a **Judge Mode** toggle intended for judges and report screenshot generation.

### Safety (must-read)

- **Do NOT upload official competition IQ data** to Streamlit Cloud.
- This app never loads official IQ data for cloud rendering.
- Judge-facing metrics are loaded only from **local** CSV files under `submissions/`.

### How Judge Mode works

In the sidebar, enable **`Judge Mode`**. The dashboard switches to a 6-tab **Judge Tour**:

1. **Problem** (Figure 1 + Figure 2)
2. **Core Submission** (Figure 4)
3. **Results** (Figure 3 + Figure 5)
4. **Efficiency**
5. **Future Work / Micro-Twin** (Figure 6 + simulation path story)
6. **Why It Matters for Gary**

### Authoritative metrics source

Judge Mode loads the CV / leaderboard table from:

- `submissions/submission_metrics.csv` (authoritative when present)
- Optional: `submissions/leaderboard_summary.csv` (if you include it)
- Optional: `docs/final_report_figures.yaml` — per-figure caption overrides (`figure_1` … `figure_6`). Copy from `docs/final_report_figures.example.yaml`. Requires **PyYAML** (`pip install pyyaml`).

If `submissions/submission_metrics.csv` is missing, the app shows a clearly-labeled placeholder schema and a note.

Expected CSV schema (header row):

- `submission`
- `submission_version` (optional but recommended for progress screenshots)
- `model_family`
- `artifact_present`
- `cv_accuracy`
- `cv_precision`
- `cv_recall`
- `cv_f1`
- `threshold`
- `leaderboard_rank`
- `leaderboard_accuracy`
- `notes`
- `change` (optional; “what changed” — progress view also accepts `changelog` / `note` as column aliases)
- `runtime` (optional; Efficiency tab also checks `runtime_per_sample` / `runtime_sec`)

Tip: add optional columns like `runtime` or `runtime_per_sample` if you have measured runtime and want the Efficiency tab to populate runtime per sample.

### Core vs Future Work separation (non-negotiable)

Judge Mode explicitly distinguishes:

- **Core judged submission**: metrics shown from local CSVs and summarized via a read-only “Final Submission” card.
- **Future Work / Micro-Twin**: synthetic 3D visualization and simulation concepts only (not used for official evaluation).

### Gary Micro-Twin 3D building model

The 3D building scene is implemented in `apps/streamlit_app.py` using **pydeck**:

- Approximate building footprints are **manually defined** as polygons.
- Buildings are rendered with **extruded polygons** (3D) and labeled tooltips.
- Scenario overlays (low/medium/high demand, occupancy prior, signal environment) change **color coding** to communicate impact and risk.

If `pydeck` is not installed in your runtime, the app fails gracefully with an in-app message.

### DeepMIMO / Sionna RT integration (Future Work)

Judge Mode includes a “Research-Grade 6G Simulation Path” section that explains how future integration could work:

- DeepMIMO: site-specific wireless dataset workflow
- Sionna RT: differentiable ray-tracing / radio propagation modeling
- Honest hooks: coverage map, beam/channel view, ray-tracing-backed scenario

These are presented as **future integration points** only (not claimed as part of the official submission basis unless your local code does so).

### Recommended screenshot sequence

- Figure 1 + Figure 2: **Problem**
- Figure 4: **Core Submission**
- Figure 3 + Figure 5: **Results**
- Figure: **Efficiency**
- Figure 6: **Future Work / Micro-Twin**

### Run locally

From repo root:

```bash
streamlit run apps/streamlit_app.py
```

