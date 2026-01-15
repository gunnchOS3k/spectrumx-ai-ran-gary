# System Overview

## Architecture Philosophy

This repository implements a two-phase approach to spectrum analysis and radio resource management:

1. **Competition Core**: Real-time spectrum occupancy detection from 1-second IQ samples (SpX-DAC competition requirement)
2. **Phase 2 Research Extension**: Digital twin simulation with AI-RAN controller for resource allocation (research portfolio)

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

## Phase 2: Research Extension — Digital Twin + AI-RAN Controller

### Problem Statement

Design an AI-native radio access network (RAN) controller that allocates radio resources (beams, power, resource blocks) under spectral and energy constraints, with explicit fairness considerations for mid-sized cities like Gary, Indiana.

### System Components

1. **Digital Twin** (`src/edge_ran_gary/channels/`, `src/edge_ran_gary/data_pipeline/deepmimo_scenarios.py`)
   - Sionna-based ray-tracing channel models
   - DeepMIMO-style synthetic channel generation
   - GIS data integration for realistic urban environments
   - Propagation modeling for 6G-like scenarios

2. **AI-RAN Controller** (`src/edge_ran_gary/models/`)
   - Contextual bandit policies (`bandit_policies.py`)
   - Actor-critic RL agents (`actor_critic.py`)
   - Baseline heuristics (`baselines.py`)

3. **Simulation Environment** (`src/edge_ran_gary/sim/`)
   - Environment loop connecting channels and controllers
   - Constraint enforcement (spectral masks, power limits)
   - Fairness metrics across users/neighborhoods
   - Energy efficiency tracking

4. **Metrics** (`src/edge_ran_gary/utils/metrics.py`)
   - Spectral efficiency (bps/Hz/user)
   - Energy efficiency (bits/Joule)
   - Fairness index (Jain's index, Gini coefficient)
   - Constraint violation rates

### Data Flow

```
Occupancy Probability (from Phase 1)
    ↓
Digital Twin (Sionna RT + GIS)
    ↓
Channel State Information (CSI)
    ↓
AI-RAN Controller (bandit/RL)
    ↓
Resource Allocation (beams, power, RBs)
    ↓
Simulation Environment (constraints, fairness)
    ↓
Metrics & Evaluation
```

---

## Why This Architecture is Credible

This two-phase architecture demonstrates both practical engineering (competition core) and research innovation (Phase 2 extension). The competition core addresses the immediate SpX-DAC requirement with production-ready detection pipelines, while Phase 2 showcases our research vision for equitable 6G access. The separation allows judges to evaluate the detection task independently, while the Phase 2 components demonstrate our ability to extend beyond the competition scope into real-world RAN optimization problems. The use of established tools (Sionna, DeepMIMO) and modern ML techniques (SSL, RL) shows technical depth, while the focus on equity (Gary, Indiana) demonstrates social awareness—a combination valued by both competition judges and PhD admissions committees.

---

## Module Organization

```
src/edge_ran_gary/
├── data_pipeline/     # Dataset loading, preprocessing
├── detection/         # Competition core: occupancy detection
├── channels/          # Phase 2: channel modeling
├── models/            # Phase 2: AI-RAN controllers
├── sim/               # Phase 2: simulation environment
├── utils/             # Shared utilities (metrics, plotting)
└── viz/               # Visualization components
```

See [Data Flow Documentation](./10_dataflow.md) for detailed pipeline descriptions.
