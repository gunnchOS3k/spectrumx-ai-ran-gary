# GitHub Issues audit — March 2026

**Repo:** `gunnchOS3k/spectrumx-ai-ran-gary`  
**Scope:** Open issues only (inventory fetched 2026-03 via GitHub public API).  
**Rules applied:** Conservative — if completion is not proven in-tree, marked **verify** or **not proven**. **No issues were closed** as part of this document.

---

## Executive summary

- **37 open issues** were inventoried (including 4 EPIC umbrella issues **#37–#40** and concrete work items **#3–#35**; note **#36 is not in the open set** returned by the API — likely closed or never created).
- The codebase has **materially advanced** since January setup issues were filed: Streamlit (**`apps/streamlit_app.py`**) includes Standard / Figure / **Judge Mode**, submission packaging under **`submissions/`**, digital twin / micro-twin code (**`src/edge_ran_gary/digital_twin/`**), data loading (**`src/edge_ran_gary/data_pipeline/spectrumx_loader.py`**), evaluation protocol (**`docs/EVAL_PROTOCOL.md`**), and audit notes (**`docs/AUDIT_MARCH_2026.md`**).
- **Stale / duplicate board risk:** Many issues still describe **exact filenames** that do not exist (e.g. `scripts/run_baseline.py`, `docs/dataset_map.md`, `scripts/eval.py`) while equivalent or partial work lives elsewhere. Several **EPIC acceptance checklists** (#37–#39) are **out of sync** with repo reality.
- **Recommended cleanup:** (1) **Close or verify-close** pure PM/setup items that are superseded by GitHub Projects migration. (2) **Rewrite** EPICs and a handful of issues to point at **actual paths**. (3) **Keep open** true final-mile items: submission/QA/freeze, missing eval harness, missing dataset inventory, robustness/error-analysis docs, and any competition-critical model work you still intend to ship.

---

## Grouping (high level)

| Group | Rough contents |
|--------|----------------|
| **Historical setup / obsolete** | Asana board setup (#9); possibly token hygiene (#10) if rotation already done outside repo |
| **Likely completed (verify)** | Repo skeleton (#8); baseline detectors (#35, #34); “mini dashboard” superseded by Streamlit (#30); parts of visualization EPIC (#20 / #38) |
| **Partially completed / needs rewrite** | One-command baseline (#3), dataset download path (#7), metrics/splits filenames (#4), EDA notebook (#5), EPIC checklists (#37–#40), judge dashboard scope vs “anomaly score” (#20) |
| **Still active / final-mile critical** | Submission zip + submit + QA + freeze (#12–#15); eval harness (#32); dataset_map (#6); robustness (#11); error analysis (#31); representation decision log (#28); optional sprint summary (#29) |
| **Portfolio / future-work extension** | SSL / anomaly / calibration / ensemble (#24–#27); compression/benchmark/ablation (#21–#23); EPIC 4 / micro-twin depth (#40) |

---

## Per-issue recommendations

*Legend: **Labels** = GitHub labels at audit time. **Evidence** = paths in this repo (empty = not proven in-tree).*

| # | Title | Labels | Action | Rationale | Repo evidence |
|---|--------|--------|--------|-----------|----------------|
| 3 | One-command baseline stub (run + report) | type:eval, phase:core-detection, priority:P1 | **rewrite** | Issue asks for `scripts/run_baseline.py` — **not present**. Training/eval exists under different entrypoints. | `scripts/train_feature_detector.py` (local train + export); `submissions/leaderboard_baseline_v1/main.py` (inference contract); **not proven:** single `run_baseline.py` |
| 4 | Define metrics + file-level split policy | type:eval, phase:core-detection, priority:P1 | **rewrite → close?** | Named deliverables `docs/metrics.md` + `docs/splits.md` **not found**; substance exists in `docs/EVAL_PROTOCOL.md`. | `docs/EVAL_PROTOCOL.md` |
| 5 | EDA notebook sanitized for sharing | type:viz, phase:core-detection, priority:P0 | **verify** | Only `notebooks/03_gary_micro_twin_demo.ipynb` found; no obvious “sanitized competition EDA” notebook. | `notebooks/03_gary_micro_twin_demo.ipynb` |
| 6 | Dataset inventory + dataset_map.md | phase:core-detection, priority:P0 | **keep open** | `docs/dataset_map.md` **missing** (also noted in `docs/PROGRESS_REPORT.md`). | **not proven**; see `docs/PROGRESS_REPORT.md` |
| 7 | Automated dataset download script | phase:core-detection, priority:P0 | **rewrite** | Issue asks `scripts/download_dataset.py` — **not found**. Download implemented on `SpectrumXDataset.download()` with `__main__`. | `src/edge_ran_gary/data_pipeline/spectrumx_loader.py` |
| 8 | Repo skeleton + coding standards | type:pm, phase:core-detection, priority:P0 | **verify then close** | `src/`, `scripts/`, `configs/`, `docs/`, `tests/`, `apps/`, `submissions/` present; `README.md`, `CONTRIBUTING.md` exist. | Tree; `README.md`, `CONTRIBUTING.md` |
| 9 | Create Asana project fields + sections | type:pm, phase:core-detection, priority:P0 | **close (verify)** | Migration to GitHub Projects documented; Asana work is **external** and likely obsolete for day-to-day. | `docs/GITHUB_PROJECT_MIGRATION.md`, `PR_DESCRIPTION.md` |
| 10 | Rotate SDS token + move to .env + gitignore | type:pm, phase:core-detection, priority:P0 | **verify** | `.env` is gitignored; docs describe `SDS_SECRET_TOKEN`. **`.env.example` not found** in repo at audit. | `.gitignore` (`.env`); `SECURITY.md`; **not proven:** `.env.example` |
| 11 | Robustness checklist for unseen data | type:ssl, phase:core-detection, priority:P0 | **keep open** | `docs/robustness.md` **not found**; acceptance unchecked. | **not proven** |
| 12 | Create submission bundle (zip-ready) | type:pm, phase:core-detection, priority:P0 | **keep open** | Partial packaging exists; “dry-run fresh clone” must be **verified** end-to-end. | `submissions/leaderboard_baseline_v1/`, `zip_submission.py`, `docs/SUBMISSION_CHECKLIST.md`, `scripts/test_leaderboard_submission.py` |
| 13 | Submit to SpX-DAC | type:pm, phase:core-detection, priority:P0 | **keep open** | Operational; no in-repo proof of submission confirmation. | **not proven** (intentional) |
| 14 | Final review meeting + freeze decision | type:pm, phase:core-detection, priority:P0 | **keep open** | Process issue; `docs/decisions.md` **not found** for recorded freeze. | **not proven:** `docs/decisions.md` |
| 15 | Final QA run (fresh machine simulation) | type:qa, phase:core-detection, priority:P0 | **keep open** | No `docs/QA_RUNBOOK.md` in repo at audit (EPIC #39 also references it). | **not proven:** `docs/QA_RUNBOOK.md` |
| 16 | Final narrative + resume bullets | — | **keep open** | Docs/product; not proven. | **not proven** |
| 17 | Final figures pack | — | **keep open** | `reports/final_figures/` not audited as populated. | **not proven** |
| 18 | Reproducibility hardening | — | **keep open** | `requirements.txt` exists; “teammate one sitting” unproven. | `requirements.txt`, `runtime.txt`; **verify** full story |
| 19 | Low-level design doc draft | — | **verify** | Issue asks `docs/design_doc.md` — **not found**; architecture docs exist under `docs/architecture/`. | `docs/architecture/00_system_overview.md`, `docs/architecture/10_dataflow.md` |
| 20 | Visualization dashboard v2 (judge-ready) | — | **verify then close or rewrite** | Judge Mode + Figure Mode + docs are strong; issue also asks **predicted prob, anomaly score, explanation** — **not proven** in `apps/streamlit_app.py` at audit. | `apps/streamlit_app.py`, `docs/JUDGE_MODE.md`, `docs/STREAMLIT_FIGURE_MODE.md` |
| 21 | Model compression / efficiency pass | — | **keep open** (or **defer**) | No dedicated compression pipeline evidenced. | **not proven** |
| 22 | Inference benchmark harness | — | **keep open** | `reports/benchmarks/` / script not evidenced in quick audit. | **not proven** |
| 23 | Ablation table v1 | — | **keep open** (or **defer**) | **not proven** |
| 24 | Hybrid ensemble experiment (SSL + anomaly) | — | **defer / portfolio** | SSL/anomaly stubs per `docs/AUDIT_MARCH_2026.md`. | `src/edge_ran_gary/detection/ssl.py`, `anomaly.py` (stubs) |
| 25 | Calibration layer v1 | — | **defer / portfolio** | Calibrator largely TODO per audit doc. | `src/edge_ran_gary/detection/calibrate.py` |
| 26 | Anomaly detector v1 | — | **defer / portfolio** | Skeleton only. | `src/edge_ran_gary/detection/anomaly.py` |
| 27 | Semi-supervised pipeline v1 | — | **defer / portfolio** | SSL not implemented. | `src/edge_ran_gary/detection/ssl.py` |
| 28 | Decision log: representation locked | — | **keep open** | `docs/decisions.md` **missing**. | **not proven** |
| 29 | Sprint 1 summary writeup | — | **verify** | `docs/sprint1_summary.md` **not found**; other status docs exist. | `docs/STATUS_SNAPSHOT.md`, `docs/PROGRESS_REPORT.md` |
| 30 | Baseline comparison mini-dashboard v1 | — | **verify then close** | Superseded by richer Streamlit app; treat as duplicate if acceptance met. | `apps/streamlit_app.py` |
| 31 | Error analysis v1 | — | **keep open** | `docs/error_analysis_v1.md` **not found**. | **not proven** |
| 32 | Evaluation harness v1 (metrics + plots) | — | **keep open** | Issue specifies `scripts/eval.py` — **not found** (train script has metrics but not same deliverable). | **not proven:** `scripts/eval.py`; partial: `scripts/train_feature_detector.py` |
| 33 | Baseline 3: PSD features + Logistic Regression | — | **verify** | Local training + feature path exists; “beats baseline or document” needs **measured** evidence in repo/docs. | `scripts/train_feature_detector.py`, `submissions/leaderboard_baseline_v1/main.py` (`_extract_features`) |
| 34 | Baseline 2: Spectral flatness | — | **verify then close** | Implemented; issue also wants **held-out fold + plots** — **not proven** from file names alone. | `src/edge_ran_gary/detection/baselines.py`, Streamlit parity in `apps/streamlit_app.py` |
| 35 | Baseline 1: Energy detector | — | **verify then close** | Same as #34 re: formal eval artifacts. | `src/edge_ran_gary/detection/baselines.py`, `submissions/leaderboard_baseline_v1/main.py` |
| 37 | EPIC 1: Competition Core Detection | phase:core-detection, priority:P0 | **rewrite** | Checklist references `scripts/run_baseline.py`, full SSL/calibration/ECE targets — **misaligned** with `docs/AUDIT_MARCH_2026.md` (stubs + partial pipeline). | `docs/AUDIT_MARCH_2026.md`, `docs/EVAL_PROTOCOL.md` |
| 38 | EPIC 2: Visualization + Demo | type:viz, priority:P1 | **rewrite** | Many criteria met (Streamlit, deploy docs, demo IQ); **failure-case browser** / some bullets need explicit mapping or deferral. | `apps/streamlit_app.py`, `docs/STREAMLIT_DEPLOY.md`, `docs/STREAMLIT_CLOUD_DEPLOY.md`, `docs/JUDGE_MODE.md` |
| 39 | EPIC 3: Reproducibility + Submission | type:mlops, priority:P0 | **rewrite** | References `scripts/run_submission.py`, `submission/` dir, `docs/QA_RUNBOOK.md` — **not matching** current `submissions/` + scripts layout. | `submissions/`, `scripts/test_leaderboard_submission.py`, `docs/SUBMISSION_CHECKLIST.md` |
| 40 | EPIC 4: Portfolio Extension (Gary Micro-Twin) | phase:portfolio-digital-twin, priority:P2 | **rewrite → partial close** | Several acceptance items **done** (config, `gary_micro_twin.py`, notebook); narrative/integration bullets may remain. | `configs/gary_micro_twin.yaml`, `src/edge_ran_gary/digital_twin/gary_micro_twin.py`, `notebooks/03_gary_micro_twin_demo.ipynb`, `docs/DIGITAL_TWIN_MICRO_TWIN.md` |

---

## Issues safe to close now (after quick human verification)

> **Do not auto-close.** Confirm externally where noted.

1. **#9** — Asana setup (superseded by GitHub migration).
2. **#8** — Repo skeleton (verify team agrees standards are “good enough”).
3. **#30** — Mini-dashboard duplicate of Streamlit (if stakeholders accept `apps/streamlit_app.py` as the deliverable).

---

## Issues that need quick verification before closing

1. **#10** — Confirm token rotated org-wide; add **`/.env.example`** if you want this closable with repo evidence.
2. **#34 / #35** — Code exists; closure depends on whether you still owe **held-out evaluation artifacts** named in the issue.
3. **#20** — Judge Mode done; verify whether you still require **probabilities / anomaly scores** in-app or descope the issue.
4. **#33** — Verify documented metrics vs baselines (link a report or `submission_metrics.csv` locally).
5. **#29** — If `docs/STATUS_SNAPSHOT.md` / `PROGRESS_REPORT.md` substitute for `sprint1_summary.md`, rewrite issue then close.

---

## Issues to rewrite or split

| # | Suggested rewrite |
|---|-------------------|
| 3 | Repoint to `scripts/train_feature_detector.py` **or** add true `scripts/run_baseline.py` wrapper. |
| 4 | Close duplicate by pointing to `docs/EVAL_PROTOCOL.md` or split “metrics” vs “splits” if you want separate files. |
| 7 | Rename deliverable to `python -m src.edge_ran_gary.data_pipeline.spectrumx_loader` or add thin `scripts/download_dataset.py` shim. |
| 37–39 | Replace file paths and acceptance with **current** layout; move SSL/anomaly/ECE items to **portfolio** epic if not competition-critical. |
| 40 | Mark completed checkboxes done; spawn new issues for DeepMIMO/Sionna narrative if desired. |

---

## Issues that should remain open (final-mile critical)

- **#6, #11, #12, #13, #14, #15, #28, #31, #32** — documentation, QA, submission, eval harness, error analysis, decisions.
- **#16–#18, #21–#23** — packaging narrative, reproducibility proof, figures, efficiency — keep if competition or PhD portfolio needs them.
- **#37** (rewritten) — until competition model + eval story is frozen.

---

## Proposed final board structure (4 EPICs)

Use GitHub Project **fields** (Priority, Phase) and **only** these EPIC parents:

### EPIC A — Competition Core Detection
- Rewritten **#37** (scope: detector you will actually submit; drop unimplemented SSL unless committed).
- **#32** Eval harness (`scripts/eval.py` or rename issue to `train_feature_detector` outputs).
- **#31** Error analysis.
- **#11** Robustness checklist.
- **#28** Decision log (`docs/decisions.md`).
- **#33** Feature+linear model (if still competing on accuracy).

### EPIC B — Visualization + Demo
- Rewritten **#38** (map to Judge Mode + Figure Mode + cloud safety).
- Optional child: polish **#20** only if prob/anomaly/explainability is still a judging differentiator.

### EPIC C — Reproducibility + Submission
- Rewritten **#39** (paths: `submissions/`, `docs/SUBMISSION_CHECKLIST.md`, test scripts).
- **#12** Bundle + dry-run.
- **#15** QA runbook (`docs/QA_RUNBOOK.md` — create or rename from existing doc).
- **#18** Reproducibility hardening (pinning, seeds, README run paths).
- **#14** Freeze + **#13** Submit.

### EPIC D — Portfolio Extension (Gary Micro-Twin / 6G AI-RAN)
- Rewritten **#40** (done vs stretch).
- New issues (optional): DeepMIMO dataset workflow, Sionna RT experiment design, “AI-RAN narrative” doc — **time-boxed** after EPIC A/C green.

**Note:** Keep **#9** closed so EPIC D doesn’t inherit PM noise.

---

## How the final board supports a strong 6G PhD application

- **Competition detector** (#37 / EPIC A): Shows you can ship a **real RF ML detector** under constraints, with **documented eval** (`docs/EVAL_PROTOCOL.md`) and **reproducible** training (`scripts/train_feature_detector.py`).
- **Reproducibility** (#39, #12, #15, #18): PhD reviewers look for **artifact hygiene** — fresh clone, pinned deps, QA runbook.
- **Visualization** (#38, Judge Mode): Demonstrates **communication** of physical-layer intuition (IQ → PSD → spectrogram → features) without leaking competition data in cloud demos.
- **Gary Micro-Twin** (#40, `digital_twin/`): Positions **equity-aware scenario modeling** and controlled synthetic RF windows — a credible **digital twin** story for wireless / civic systems.
- **DeepMIMO / Sionna RT / AI-RAN** (future issues under EPIC D): Frame as **research-grade forward work** — site-specific channels, ray tracing, and RAN-aware sensing — explicitly **not** conflated with leaderboard scoring (matches in-app framing).

---

## Output tallies (this audit)

| Metric | Value | Issue numbers (exclusive buckets) |
|--------|------:|-----------------------------------|
| **Total issues audited** | 37 | #3–#35, #37–#40 |
| **Recommended: close (obsolete PM)** | 1 | #9 |
| **Recommended: verify then close** | 2 | #8, #30 |
| **Recommended: verify (may close after check)** | 8 | #5, #10, #19, #20, #29, #33, #34, #35 |
| **Recommended: rewrite** | 7 | #3, #4, #7, #37, #38, #39, #40 |
| **Recommended: keep open** | 15 | #6, #11, #12, #13, #14, #15, #16, #17, #18, #21, #22, #23, #28, #31, #32 |
| **Recommended: defer / portfolio** | 4 | #24, #25, #26, #27 |

*Buckets are mutually exclusive for counting; “verify” issues are candidates to close after evidence is attached (comment + link to commit/doc).*

### Top 10 to close first (after verification)

1. #9 (Asana)  
2. #8 (skeleton)  
3. #30 (duplicate dashboard)  
4. #35 (energy — if eval artifacts satisfied)  
5. #34 (flatness — if eval artifacts satisfied)  
6. #4 (metrics/splits — if EVAL_PROTOCOL accepted as SoT)  
7. #29 (if status docs substitute for sprint1 file)  
8. #20 (if Judge Mode satisfies “judge-ready”)  
9. #10 (if `.env.example` added + rotation confirmed)  
10. #33 (if results documented)

### Top 5 to rewrite next

1. #37 (EPIC 1)  
2. #39 (EPIC 3)  
3. #38 (EPIC 2)  
4. #3 (`run_baseline` vs actual scripts)  
5. #7 (download script path)

### Top 5 that must stay open until competition exit

1. #13 Submit  
2. #12 Submission bundle  
3. #15 Final QA  
4. #14 Freeze / decision record  
5. #32 Eval harness (or equivalent documented eval command)

---

## Evidence index (quick reference)

| Area | Paths |
|------|--------|
| Streamlit / Judge | `apps/streamlit_app.py`, `docs/JUDGE_MODE.md`, `docs/STREAMLIT_FIGURE_MODE.md` |
| Submission | `submissions/leaderboard_baseline_v1/`, `docs/SUBMISSION_CHECKLIST.md`, `scripts/test_leaderboard_submission.py` |
| Data download | `src/edge_ran_gary/data_pipeline/spectrumx_loader.py` |
| Eval / splits / metrics | `docs/EVAL_PROTOCOL.md` |
| Baselines | `src/edge_ran_gary/detection/baselines.py` |
| Feature + linear train | `scripts/train_feature_detector.py` |
| Micro-twin | `configs/gary_micro_twin.yaml`, `src/edge_ran_gary/digital_twin/gary_micro_twin.py`, `notebooks/03_gary_micro_twin_demo.ipynb` |
| Architecture docs | `docs/architecture/`, `docs/DIGITAL_TWIN*.md` |
| Prior internal audit | `docs/AUDIT_MARCH_2026.md` |

---

*End of audit — cleanup actions are proposed only; execute closes/edits in GitHub UI or CLI when ready.*
