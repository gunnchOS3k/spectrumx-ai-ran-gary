# GitHub Issues audit — March 2026 (refresh)

**Repo:** `gunnchOS3k/spectrumx-ai-ran-gary`  
**Inventory:** Open issues from `gh issue list --repo gunnchOS3k/spectrumx-ai-ran-gary --state open` (March 23, 2026).  
**Rules:** Conservative — **verify** when uncertain. **No issues were closed** as part of this document.

---

## Executive summary

| Metric | Value |
|--------|--------|
| **Open issues audited** | **37** |
| **Open issue numbers** | `#3`–`#35` (no `#36`) + `#37`–`#40` |
| **Gap** | `#36` does not exist in the open set (likely never created or closed/archived) |

**Repo has advanced** beyond January Asana-import tasks:

- **Detector / features:** `src/edge_ran_gary/detection/` (`baselines.py`, `feature_baseline.py`, `predict.py`, `features.py`, `calibrate.py`, `ssl.py`, `anomaly.py`), `scripts/train_feature_detector.py`, `submissions/leaderboard_baseline_v1/`.
- **Streamlit:** `apps/streamlit_app.py` (Standard / Figure / **Judge Mode**, Gary micro-twin 3D tab, simulation summaries).
- **Digital twin / Gary extension:** `src/edge_ran_gary/digital_twin/`, `gary_site_geometry.py`, `gary_scenario_engine.py`, `gary_occupancy_visualization.py`, `configs/gary_micro_twin.yaml`, `configs/wireless_scene/`, `notebooks/03_gary_micro_twin_demo.ipynb`.
- **Simulation backbone (extension):** `src/edge_ran_gary/simulation_integration_hooks.py`, `examples/simulation_exports/`, `scripts/export_sionna_rt_summary.py`, `scripts/export_deepmimo_summary.py`, `scripts/check_ngc_access.py`, `scripts/bootstrap_aodt_access.py`, `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`, `docs/SIMULATION_BACKBONE_PLAN.md`.
- **Submission / reproducibility:** `submissions/`, `docs/SUBMISSION_CHECKLIST.md`, `scripts/test_leaderboard_submission.py`, `scripts/validate_feature_leaderboard_submission.py`, `runtime.txt`, pinned `requirements.txt` / Streamlit.
- **Documentation:** `docs/AUDIT_MARCH_2026.md`, `docs/EVAL_PROTOCOL.md`, `docs/JUDGE_MODE.md`, `docs/architecture/`, `docs/DIGITAL_TWIN_MICRO_TWIN.md`, `docs/MICROTWIN_REALISM_PLAN.md`.

**Board problem:** Many **open** issues still use **January filenames** that were never created or were superseded (`scripts/run_baseline.py`, `scripts/eval.py`, `docs/QA_RUNBOOK.md`, `docs/dataset_map.md`). **EPIC bodies (#37–#40)** are **checklist-stale** relative to the tree above. **CONTRIBUTING.md** / **NOAH_GUIDE.md** still reference `run_baseline.py` — doc drift.

---

## PART 1 — Issue inventory (all open)

### Historical setup / obsolete (verify then close if acceptance met)

| # | Title |
|---|--------|
| 3 | One-command baseline stub (run + report) |
| 8 | Repo skeleton + coding standards |
| 9 | Create Asana project fields + sections |

### Likely completed or superseded (verify then close)

| # | Title | Why “likely” |
|---|--------|----------------|
| 4 | Define metrics + file-level split policy | `docs/EVAL_PROTOCOL.md` + `config.py` seed |
| 7 | Automated dataset download script | `spectrumx_loader.py` has `download()`; no thin `scripts/download_*.py` wrapper |
| 30 | Baseline comparison mini-dashboard v1 | Superseded by `apps/streamlit_app.py` + Judge Mode |
| 34 | Baseline 2: Spectral flatness detector | `baselines.py` + submission path |
| 35 | Baseline 1: Energy detector | Same |

### Partially completed / needs rewrite

| # | Title |
|---|--------|
| 5 | EDA notebook sanitized for sharing — only `notebooks/03_gary_micro_twin_demo.ipynb` evidenced; competition EDA path **verify** |
| 6 | Dataset inventory + `dataset_map.md` — use `docs/OFFICIAL_DATA_TRAINING_GAPS.md` + loader docs or add `dataset_map.md` |
| 19 | Low-level design doc — points at `docs/design_doc.md`; **use** `docs/architecture/*` |
| 20 | Visualization dashboard v2 — much done; **verify** “anomaly score / explanation” vs UI |
| 29 | Sprint 1 summary — no `docs/sprint1_summary.md`; **use** `STATUS_SNAPSHOT` / `PROGRESS_REPORT` |
| 37–40 | EPICs — checklists misaligned with repo |

### Still active / final-mile critical

| # | Title |
|---|--------|
| 10 | Rotate SDS token + `.env` + gitignore |
| 11 | Robustness checklist (`docs/robustness.md` **not proven**) |
| 12 | Create submission bundle (zip-ready) |
| 13 | Submit to SpX-DAC |
| 14 | Final review meeting + freeze |
| 15 | Final QA run — no `docs/QA_RUNBOOK.md` |
| 16 | Final narrative + resume bullets |
| 17 | Final figures pack |
| 18 | Reproducibility hardening |
| 21 | Model compression / efficiency |
| 22 | Inference benchmark harness |
| 23 | Ablation table v1 |
| 28 | Decision log (`docs/decisions.md` **not proven**) |
| 31 | Error analysis v1 |
| 32 | Evaluation harness v1 — `scripts/eval.py` **not proven** |
| 33 | Baseline 3: PSD + LogReg — **verify** vs competition bar |

### Portfolio / future-work (PhD extension; defer if competition crunch)

| # | Title |
|---|--------|
| 24 | Hybrid ensemble (SSL + anomaly) |
| 25 | Calibration layer v1 |
| 26 | Anomaly detector v1 |
| 27 | Semi-supervised pipeline v1 |

---

## PART 2 — Repo evidence map

| Topic | Evidence paths |
|--------|----------------|
| Baselines | `src/edge_ran_gary/detection/baselines.py`, `submissions/leaderboard_baseline_v1/main.py` |
| Feature / train | `scripts/train_feature_detector.py`, `src/edge_ran_gary/detection/feature_baseline.py` |
| Metrics / splits | `docs/EVAL_PROTOCOL.md`, `src/edge_ran_gary/config.py` |
| Data / download | `src/edge_ran_gary/data_pipeline/spectrumx_loader.py`, `docs/OFFICIAL_DATA_TRAINING_GAPS.md` |
| Architecture | `docs/architecture/00_system_overview.md`, `docs/architecture/10_dataflow.md` |
| Streamlit + Judge | `apps/streamlit_app.py`, `docs/JUDGE_MODE.md`, `docs/STREAMLIT_FIGURE_MODE.md`, `docs/STREAMLIT_CLOUD_DEPLOY.md` |
| Gary + 3D | `configs/gary_micro_twin.yaml`, `digital_twin/gary_micro_twin.py`, `gary_site_geometry.py`, `gary_scenario_engine.py`, `gary_occupancy_visualization.py`, `notebooks/03_gary_micro_twin_demo.ipynb`, `docs/DIGITAL_TWIN_MICRO_TWIN.md` |
| Simulation extension | `simulation_integration_hooks.py`, `examples/simulation_exports/`, `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`, `scripts/export_*.py` |
| Submission | `submissions/leaderboard_baseline_v1/`, `docs/SUBMISSION_CHECKLIST.md`, `scripts/test_leaderboard_submission.py` |
| Security | `.gitignore`, `SECURITY.md` — **verify:** `.env.example` |
| **Not proven** | `scripts/run_baseline.py`, `scripts/eval.py`, `docs/dataset_map.md`, `docs/QA_RUNBOOK.md`, `docs/decisions.md`, `docs/robustness.md`, `docs/error_analysis_v1.md`, `reports/final_figures/` (populate?) |

---

## PART 3 — Per-issue recommendations

*Labels: re-check on GitHub; EPICs use `phase:*` / `type:*` / `priority:*` as in Projects.*

| # | Title | Action | Rationale | Evidence |
|---|--------|--------|-----------|----------|
| 3 | One-command baseline stub | **verify then close** | No `run_baseline.py`; `train_feature_detector` + `main.py` cover run + report path. | `scripts/train_feature_detector.py`, `submissions/leaderboard_baseline_v1/main.py` |
| 4 | Metrics + split policy | **verify then close** | Documented file-level splits + seed. | `docs/EVAL_PROTOCOL.md`, `config.py` |
| 5 | EDA notebook sanitized | **verify** | Only Gary demo notebook in `notebooks/`; competition EDA may live elsewhere or missing. | `notebooks/03_gary_micro_twin_demo.ipynb` only |
| 6 | Dataset inventory + dataset_map | **rewrite or verify close** | No `dataset_map.md`; `OFFICIAL_DATA_TRAINING_GAPS.md` inventories loader/gaps. | `docs/OFFICIAL_DATA_TRAINING_GAPS.md`, `spectrumx_loader.py` |
| 7 | Automated dataset download | **verify then close** | Download in loader; issue may want CLI wrapper (still “missing” per gaps doc). | `spectrumx_loader.py`; see gaps doc |
| 8 | Repo skeleton + standards | **verify then close** | Structured tree, tests, CONTRIBUTING (needs drift fix). | repo layout, `CONTRIBUTING.md`, `tests/` |
| 9 | Asana project fields | **verify then close** | PM/process; may be done off-repo. | **not proven** in tree |
| 10 | Rotate SDS token + .env | **verify then close** | `.env` pattern documented; rotation is operational. | `.gitignore`, `SECURITY.md`, loader token usage |
| 11 | Robustness checklist | **keep open** | No `docs/robustness.md`. | **not proven** |
| 12 | Submission bundle | **keep open** | Dry-run fresh clone must be verified. | `submissions/`, `zip_submission.py`, `SUBMISSION_CHECKLIST.md` |
| 13 | Submit to SpX-DAC | **keep open** | External milestone. | **not proven** |
| 14 | Final review + freeze | **keep open** | Process. | **not proven** |
| 15 | Final QA run | **keep open** | No `QA_RUNBOOK.md`. | **not proven** |
| 16 | Final narrative + resume | **keep open** | Content. | **not proven** |
| 17 | Final figures pack | **keep open** | `reports/final_figures/` **verify** populated. | **verify** |
| 18 | Reproducibility hardening | **keep open** | `requirements.txt`, `runtime.txt`; “one sitting” unproven. | `requirements.txt`, `runtime.txt` |
| 19 | Low-level design doc | **rewrite or verify close** | `docs/design_doc.md` missing; architecture set exists. | `docs/architecture/*` |
| 20 | Viz dashboard v2 judge-ready | **verify then close or rewrite** | Judge Mode strong; match issue bullets to UI. | `apps/streamlit_app.py`, `JUDGE_MODE.md` |
| 21 | Model compression | **keep open** or **defer** | **not proven** | — |
| 22 | Inference benchmark | **keep open** | **not proven** | — |
| 23 | Ablation table | **keep open** or **defer** | **not proven** | — |
| 24–27 | SSL / calibration / anomaly / hybrid | **defer / portfolio** | Stubs in `ssl.py`, `anomaly.py`, `calibrate.py`. | `src/edge_ran_gary/detection/*.py` |
| 28 | Decision log representation | **keep open** | `docs/decisions.md` missing. | **not proven** |
| 29 | Sprint 1 summary | **rewrite** | No `sprint1_summary.md`. | `STATUS_SNAPSHOT.md`, `PROGRESS_REPORT.md` |
| 30 | Mini-dashboard baselines | **verify then close** | Superseded by Streamlit. | `apps/streamlit_app.py` |
| 31 | Error analysis v1 | **keep open** | No `error_analysis_v1.md`. | **not proven** |
| 32 | Evaluation harness v1 | **keep open** | No `scripts/eval.py`; partial via training script. | `train_feature_detector.py` |
| 33 | Baseline 3 PSD + LogReg | **verify** | Training path exists; formal competition comparison **verify**. | `train_feature_detector.py`, `main.py` |
| 34 | Baseline 2 spectral flatness | **verify then close** | In code; artifact acceptance **verify**. | `baselines.py` |
| 35 | Baseline 1 energy | **verify then close** | Same. | `baselines.py`, `main.py` |
| 37 | EPIC 1 Competition Core Detection | **rewrite** | Checklist vs `AUDIT_MARCH_2026`, real scripts. | `docs/AUDIT_MARCH_2026.md` |
| 38 | EPIC 2 Visualization + Demo | **rewrite** | Map to Streamlit + deploy docs. | `apps/streamlit_app.py`, `docs/STREAMLIT_*` |
| 39 | EPIC 3 Reproducibility + Submission | **rewrite** | `submissions/`, not `submission/`; add QA path. | `SUBMISSION_CHECKLIST.md`, `test_leaderboard_submission.py` |
| 40 | EPIC 4 Portfolio Gary Micro-Twin | **rewrite** | Add Streamlit 3D, simulation hooks, runbooks. | `gary_*`, `simulation_integration_hooks.py`, `APPLIED_SIMULATION_BRIDGE_RUNBOOK.md`, `SIMULATION_BACKBONE_PLAN.md` |

---

## PART 4 — Safe to close *after* verification

**Do not auto-close.**

1. **#8** — Skeleton / standards (fix CONTRIBUTING drift in same PR as close comment, optional).  
2. **#3** — Baseline “one command” satisfied by train + submission entrypoints (if stakeholders agree).  
3. **#4** — Metrics + splits documented.  
4. **#7** — Download via loader (or close when thin CLI exists — **verify** against issue body).  
5. **#30, #34, #35** — Superseded or implemented.  
6. **#10** — Token rotation confirmed.  
7. **#19** — If architecture docs satisfy “design doc.”  
8. **#9** — If Asana setup is historical.

---

## PART 5 — Verify before close

- **#5** — EDA notebook path vs “sanitized for sharing.”  
- **#6** — Whether `OFFICIAL_DATA_TRAINING_GAPS.md` counts as inventory.  
- **#20** — Judge UI vs full v2 acceptance text.  
- **#33** — PSD+logreg vs leaderboard acceptance.

---

## PART 6 — Rewrite / split

- **#37–#40** — EPIC bodies and checklists.  
- **#29** — Link to existing docs or add `docs/sprint1_summary.md`.  
- **#32** — Align title with `train_feature_detector` + submission metrics, or add `scripts/eval.py`.  
- **#6** — Rename to “Dataset inventory doc” or add `dataset_map.md`.

---

## PART 7 — Keep open (final-mile minimum)

**Competition closure spine:** **#11, #12, #13, #14, #15, #16, #17, #18, #28, #31, #32** (+ **#21–#23** if in scope).

---

## PART 8 — Proposed final board (4 EPICs)

Align GitHub Project with **#37–#40** as parents; **close or relabel** January setup issues once verified.

### EPIC — Competition Core Detection (#37)

- Children: **#32, #28, #31, #11**; verify-close **#3, #4, #33–#35**.  
- Portfolio (separate column): **#24–#27**.

### EPIC — Visualization + Demo (#38)

- Verify-close: **#20, #30**.  
- Optional child: “Deploy + screenshot checklist” → `JUDGE_MODE.md` / `STREAMLIT_CLOUD_DEPLOY.md`.

### EPIC — Reproducibility + Submission (#39)

- Children: **#12, #13, #14, #15, #18, #10**.  
- Add **`docs/QA_RUNBOOK.md`** or extend `SUBMISSION_CHECKLIST.md` with fresh-machine steps.

### EPIC — Portfolio Extension: Gary Micro-Twin / 6G AI-RAN (#40)

- Rewrite checklist: micro-twin config + engine + Streamlit 3D + `simulation_integration_hooks` + exporters + honesty on DeepMIMO / Sionna / NGC (`SIMULATION_BACKBONE_PLAN.md`).

---

## PART 9 — PhD portfolio alignment (6G / AI-RAN)

**How the final board supports a strong 6G PhD application**

| Theme | Repo hook |
|--------|-----------|
| **Competition detector** | EPIC / `#37` — RF detection, eval protocol, leaderboard submission. |
| **Reproducibility** | EPIC / `#39` — frozen env, tests, zip, QA narrative. |
| **Visualization** | EPIC / `#38` — judge-safe demos, Figure Mode, failure exploration. |
| **Gary Micro-Twin** | EPIC / `#40` — place-based twin, scenario engine, civic occupancy story. |
| **6G / AI-RAN / ray-tracing direction** | `simulation_integration_hooks.py`, `export_sionna_rt_summary.py`, `export_deepmimo_summary.py`, `docs/APPLIED_SIMULATION_BRIDGE_RUNBOOK.md` — ties **physical channel realism** and **AI-RAN-style** closed-loop framing without overstating what is simulated vs stubbed. |

---

## PART 10 — Output summary (this audit)

| Recommended action | Count |
|--------------------|-------|
| **verify then close** | 11 (#3, #4, #7, #8, #9, #10, #19, #30, #34, #35, ±#6/#20) |
| **rewrite** | 7 (#6, #29, #32, #37, #38, #39, #40) |
| **keep open** (final-mile / critical) | 12 (#11, #12, #13, #14, #15, #16, #17, #18, #28, #31, #32, + optional #21–#23) |
| **verify** (uncertain) | 3 (#5, #6, #20, #33 — overlap with above; treat as **verify** first) |
| **defer / portfolio** | 4 (#24–#27) |

*Note: #32 appears in both “rewrite” and “keep open” — **keep open** until harness exists; **rewrite** the issue text.*

**Total open audited:** **37**

### Top 10 to close first (after verification)

1. **#8** — Repo skeleton  
2. **#3** — Baseline stub (superseded by train + submission)  
3. **#4** — Metrics + splits  
4. **#30** — Mini-dashboard  
5. **#35** — Energy baseline  
6. **#34** — Spectral flatness baseline  
7. **#10** — Token / `.env`  
8. **#7** — Download (if loader meets acceptance)  
9. **#19** — Design doc (if architecture suffices)  
10. **#9** — Asana (if process complete)

### Top 5 to rewrite next

1. **#40** — EPIC 4 vs Streamlit + simulation reality  
2. **#37** — EPIC 1 vs audit + scripts  
3. **#39** — EPIC 3 vs `submissions/` + QA  
4. **#38** — EPIC 2 vs deploy + judge docs  
5. **#32** — Eval harness vs `train_feature_detector`

### Top 5 that must remain open (competition closure)

1. **#13** — Submit  
2. **#12** — Bundle  
3. **#15** — QA  
4. **#14** — Freeze  
5. **#32** — Eval / metrics story (until superseded by a closed #32 or new issue)

---

*End of audit. Close/rewrite actions are **proposals** only.*
