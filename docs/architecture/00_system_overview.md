# System Overview

> **Diagrams:** GitHub-visible UML pack: [`docs/uml/README.md`](../uml/README.md) (front door) and [`docs/uml/current/index.md`](../uml/current/index.md) (browse current-state diagrams). This page is narrative; if a diagram conflicts with shipped Streamlit behavior, prefer the **current** Markdown wrappers, not legacy sources.

## Architecture Philosophy

This repository implements a two-phase approach to spectrum analysis and radio resource management:

1. **Competition Core**: Real-time spectrum occupancy detection from 1-second IQ samples (SpX-DAC competition requirement)
2. **Phase 2 Research Extension**: Gary digital twin + **detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)** in Streamlit; separate `models/` code exists for future bandit/RL study arms (not the shipped twin controller)

This separation ensures that competition judges can evaluate the core detection task independently, while demonstrating our broader research vision for equitable 6G access in under-resourced communities.

---

## Phase 1: Competition Core — Spectrum Occupancy Detection

### Problem Statement

Given a 1-second IQ sample (complex-valued time series), determine whether the spectrum is occupied (signal present) or unoccupied (noise only). This is a binary classification problem with real-time inference requirements.

### System Components

1. **Data Pipeline** (`src/edge_ran_gary/data_pipeline/`)
   - Downloads and manages the SpectrumX SDS dataset
   - Handles labeled and unlabeled IQ samples
   - Preprocesses IQ data (normalization, windowing, format conversion)

2. **Detection Module** (`src/edge_ran_gary/detection/`)
   - Feature extraction from IQ samples (time-domain, frequency-domain, statistical)
   - Baseline detectors (energy, spectral flatness, PSD-based)
   - Self-supervised learning (SSL) encoders for representation learning
   - Anomaly detection models for unsupervised scenarios
   - Calibration for confidence estimation
   - Ensemble fusion for robust predictions

3. **Evaluation** (`src/edge_ran_gary/sim/evaluation.py`)
   - Metrics: accuracy, precision, recall, F1, AUC-ROC
   - Confidence calibration metrics (ECE, Brier score)
   - Inference latency measurement
   - Submission bundle generation

4. **Visualization** (`apps/streamlit_app.py`, `src/edge_ran_gary/viz/`)
   - Interactive dashboard for IQ data exploration
   - Model prediction visualization
   - Confidence score display
   - Time-domain, frequency-domain, and constellation plots

### Data Flow

```
SpectrumX SDS Dataset
    ↓
DatasetLoader (download, organize)
    ↓
IQPreprocessor (normalize, window, format)
    ↓
FeatureExtractor (time/freq/statistical features)
    ↓
[Training Path] → EncoderSSL / AnomalyModel → ClassifierHead
    ↓
Calibrator (confidence estimation)
    ↓
EnsembleFusion (combine multiple models)
    ↓
Evaluator (metrics, submission bundle)
```

---

## Phase 2: Research Extension — Digital Twin + controller abstraction

### Problem statement

Study how **site-aware** RAN-style control (fairness, coexistence, energy proxies) can be grounded on **Gary’s three civic anchors**, with a clear path from **scenario → policy → KPIs** and from **proxy** to **simulation exports** to **external** validation.

### What ships in Streamlit today (completed extension)

1. **Gary scenario engine** (`gary_scenario_engine.py`, `gary_site_geometry.py`, digital twin generators)
   - Three anchors: **Gary City Hall**, **Gary Public Library & Cultural Center**, **West Side Leadership Academy**
   - People / device / traffic / pressure metrics from documented assumptions + UI presets
   - **Detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)** — discrete actions (hold, cautious, power, channel, prioritize, rebalance) and **heuristic KPI deltas** (`select_closed_loop_action`, `apply_action_to_kpis`)

2. **Simulation + provenance** (`simulation_integration_hooks.py`, `simulation_provenance.py`)
   - Manifest-load for DeepMIMO / Sionna / AODT / pyAerial / OTA-shaped paths; **no** full local execution of external solvers in the app

3. **pyAerial bridge** (`pyaerial_bridge/`) — import probe + **conceptual** PHY/MAC hints; real PHY remains **external**

### Additional code (future / offline study arms, not the Streamlit controller)

- **`models/`** — `bandit_policies.py`, `actor_critic.py`, `baselines.py` for **future** contextual-bandit or RL experiments (see `docs/uml/state_controller_maturity_ladder.mmd`)
- **`sim/`**, **`channels/`**, **`data_pipeline/deepmimo_scenarios.py`** — research scaffolding; heavy Sionna/DeepMIMO **execution** is an **external runtime** target (`docs/EXTERNAL_RUNTIME_GAPS.md`)

### Data flow (Streamlit extension — honest)

```
Detector belief (synthetic demo IQ in Judge Mode) + scenario state
    ↓
Rule-based policy (RIC-style abstraction)
    ↓
Discrete action + heuristic KPI update
    ↓
pydeck scene + provenance / simulation backbone panels
```

---

## Why This Architecture is Credible

This two-phase architecture separates **judged** detection (packaged `evaluate()`, offline scoring) from a **completed** Gary twin that already demonstrates **reproducible scenario → policy → KPI** semantics with explicit **provenance**. Optional modules (SSL, bandit/RL code under `models/`) support future study arms; they are **not** described as the current Streamlit controller. See `docs/uml/README.md` for diagrams.

---

## Module Organization

```
src/edge_ran_gary/
├── data_pipeline/     # Dataset loading, preprocessing
├── detection/         # Competition core: occupancy detection
├── channels/          # Phase 2: channel modeling
├── models/            # Phase 2: bandit/RL scaffolding (future study arms; not Streamlit twin controller)
├── sim/               # Phase 2: simulation environment
├── utils/             # Shared utilities (metrics, plotting)
└── viz/               # Visualization components
```

See [Data Flow Documentation](./10_dataflow.md) for detailed pipeline descriptions.
