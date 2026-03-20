## Streamlit Figure Mode / Report Mode

This repo’s Streamlit dashboard (`apps/streamlit_app.py`) supports three UI modes:

- **Standard Mode**: the original baseline-comparison dashboard (upload/demo + prediction + plots).
- **Figure Mode**: a **report-figure generator** that reorganizes content into report-oriented tabs and adds screenshot-friendly polish (titles, spacing, captions, and editable “report notes”).
- **Judge Mode**: a polished, read-only judge tour that clearly separates the Core Judged Submission (authoritative local metrics) from Future Work / Micro-Twin (non-scoring).

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

### Judge Mode (judge-facing)

1. In the sidebar, enable **Judge Mode**.
2. The app renders a 6-tab **Judge Tour**: **Problem**, **Core Submission**, **Results**, **Efficiency**, **Future Work / Micro-Twin**, **Why It Matters for Gary**.
3. **Core metrics** are loaded from local `submissions/submission_metrics.csv` (authoritative when present). No official competition IQ data is accessed.
4. **Future Work / Micro-Twin** is shown as a clearly-labeled, synthetic, screenshot-friendly **3D building scene** (pydeck-based).

Optional: copy `docs/final_report_figures.example.yaml` → `docs/final_report_figures.yaml` to override default captions for Figures **1–6** in **Figure Mode** (and Judge Mode where wired). Requires `pip install pyyaml`.

Recommended screenshot sequence for the final report:
- Figure 1 + Figure 2: **Problem**
- Figure 4: **Core Submission**
- Figure 3 + Figure 5: **Results**
- Figure 6: **Future Work / Micro-Twin**

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
- **Submission Explorer** → read-only inventory of `submissions/*` (folder, `main.py`, `user_reqs.txt`, artifacts, light keyword hints from `main.py` text only)
- **CV Metrics** → authoritative table from **`submissions/submission_metrics.csv`** (if present); otherwise a labeled placeholder schema
- **Leaderboard Progress** → sorted view / bar chart when `leaderboard_rank` and `leaderboard_accuracy` exist in that CSV
- **Micro-Twin (Future Work)** → appendix / future work (clearly labeled as non-core submission)

### Submissions scan (read-only)

- On each run, Figure Mode calls a cached function that **lists directories under `submissions/`** and checks for:
  - `main.py`, `user_reqs.txt`, and common learned artifacts (`*.npz`, `*.pkl`, `*.joblib`)
- It optionally reads **the first ~80k characters** of `main.py` for **keyword hints** (spectral flatness, energy detector, features, sklearn, `.npz`, etc.). This is **not** a substitute for real CV metrics.
- **No** official competition `.npy` files are read or displayed.

### Authoritative CV / leaderboard table: `submission_metrics.csv`

1. Copy the template (committed example with header only):

   ```bash
   cp submissions/submission_metrics.example.csv submissions/submission_metrics.csv
   ```

2. Fill in rows from your **local** training / leaderboard runs (do not commit real competition data).

3. Expected columns (header row):

   | Column | Description |
   |--------|-------------|
   | `submission` | Folder name under `submissions/` (e.g. `leaderboard_baseline_v1`) |
   | `model_family` | Short label (e.g. `spectral_flatness`, `feature_lr`, `feature_svm`) |
   | `artifact_present` | `yes` / `no` or boolean |
   | `cv_accuracy` | Cross-validation accuracy (or holdout) |
   | `cv_precision` | Precision |
   | `cv_recall` | Recall |
   | `cv_f1` | F1 |
   | `threshold` | Decision threshold if applicable |
   | `leaderboard_rank` | Rank on organizer leaderboard (numeric) |
   | `leaderboard_accuracy` | Reported leaderboard accuracy if available |
   | `notes` | Free text |

4. Keep **`submissions/submission_metrics.csv` local** if it contains sensitive run details; the repo includes only **`submissions/submission_metrics.example.csv`** as a schema template. Add `submission_metrics.csv` to `.gitignore` if you never want it committed.

### Recommended screenshots for the final report

- **Figure 1**: Overview tab (task framing + safety note)
- **Figure 2**: Input & Preprocessing tab (Time Domain + PSD + Spectrogram panel)
- **Figure 3**: Feature Extraction tab (feature table + top-feature bar chart)
- **Figure 4**: Prediction Path tab (decision flow)
- **Figure 5**: Results & Leaderboard tab **or** CV Metrics tab (headline / authoritative table)
- **Figure 6**: Submission Explorer tab (inventory of submission packages)
- **Figure 7** (optional): Leaderboard Progress tab (bar chart from CSV)
- **Figure 8** (optional): Micro-Twin (Future Work) tab (extension panel for appendix)

### Notes

- Figure Mode is intentionally **additive**: it preserves the existing Standard Mode behavior.
- No official-data download or training is performed in the Streamlit app.
- Widget **keys** are set on sidebar controls and on duplicate-prone buttons so **StreamlitDuplicateElementId** does not occur when switching modes or rerunning.

