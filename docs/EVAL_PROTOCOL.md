# Evaluation Protocol — 6G-Aligned AI-RAN Research Prototype

This document defines evaluation for **gunnchos 7GC Field Console** research artifacts in `spectrumx-ai-ran-gary`. Wording is intentionally conservative: this is an **IMT-2030-aligned, AI-assisted O-RAN-style control experiment**, not carrier-grade AI-RAN or deployable 6G.

---

## 1. Datasets

| Dataset | Type | Used for | Private data |
|---------|------|----------|--------------|
| Synthetic RAN snapshots | **Synthetic** | `airan_research.synthetic_data.generate()` | No |
| SpectrumX competition IQ (optional) | **External/real (competition)** | Phase-1 occupancy detection | Competition terms only |
| DeepMIMO / Sionna channels (optional) | **External/synthetic channels** | Planned twin calibration | No PII |
| Gary campus YAML profiles | **Configuration** | Scenario labels only | No coordinates/PII |

**Default reproducible path:** synthetic-only (`scripts/run_benchmark.py`, `scripts/run_ablation.py`).

---

## 2. Train / Validation / Test Split Policy

### Phase 1 — Spectrum occupancy (competition path)

- **Unit of split:** individual `.npy` IQ files (file-level; no sample leakage).
- **Ratios:** train 60% / validation 20% / test 20%.
- **Stratification:** label balance within ±5% across splits.
- **Seed:** `RANDOM_SEED = 42` (document in run config).

### Phase 2 — AI-RAN policy simulation (default today)

- **Unit of split:** deterministic scenario seeds (not IQ files).
- **Benchmark seeds:** `{42, 43, 44, 45, 46}` unless overridden.
- **Holdout:** seed `99` reserved for manual spot-check (not in default CI).
- **No fine-tuning on test seeds** for toy policies (closed-form stubs).

---

## 3. Baseline Methods

| Baseline | Role | Implementation |
|----------|------|----------------|
| Energy / flatness detectors | Phase-1 classical occupancy | `src/edge_ran_gary/models/baselines.py` |
| Uniform allocator | AI-RAN baseline | `UniformPolicy` |
| Fairness-aware policy | Equal-share stub | `FairnessAwarePolicy` |
| Energy-aware policy | Budget scaling stub | `EnergyAwarePolicy` |
| Combined fairness + energy | Multi-objective stub | `CombinedFairnessEnergyPolicy` |

Ablation entry point: `python3 scripts/run_ablation.py --seed 42`.

---

## 4. Metrics

| Metric | Definition | Evidence today |
|--------|------------|----------------|
| **Fairness** | Jain's index on resource allocations | `results/ablation/ablation_table.csv` |
| **Energy** | Toy energy score (inverse power per user) | same |
| **Spectral efficiency** | `sum(allocations)/capacity_mhz` | `results/benchmark/metrics.csv` |
| **Latency proxy** | Mean unserved demand above equal share | synthetic-only |
| **Outage / denial proxy** | Fraction of users above half equal share | synthetic-only |
| **Detection accuracy / AUC** | Phase-1 competition metrics | requires competition IQ run |

---

## 5. Random Seeds and Determinism

- Python `random.Random(seed)` in synthetic data generation.
- Benchmark CLI: `--seed` and `--seeds`.
- Plot generation uses fixed input CSV rows (no stochastic plotting).
- Document OS/Python in `docs/REPRODUCIBILITY_FRESH_MACHINE.md`.

---

## 6. What Counts as Evidence

| Level | Meaning | Example artifact |
|-------|---------|------------------|
| **Smoke** | Code executes | `make smoke` / `make test` |
| **Synthetic reproducible** | Deterministic metrics from public scripts | `results/benchmark/metrics.csv` |
| **External validation** | Third-party dataset or field run | Not claimed today |
| **Operational** | Near-RT RIC / live RAN control | **Not claimed** |

---

## 7. What This Does Not Prove

- Deployable **6G** or **carrier-grade AI-RAN**.
- Gary **citywide** spectral or equity impact.
- **Certified** hardware or unauthorized **RF transmission**.
- Superiority over commercial schedulers on live networks.
- O-RAN **E2/KPM** integration beyond export stubs.
- Fairness or energy gains that transfer to real RAN deployments without field validation.

---

## 8. Reviewer Quick Commands

```bash
pip install -r requirements.txt
make test
make benchmark
make ablation
```

Expected outputs: `results/benchmark/metrics.csv`, `figures/benchmark/benchmark_metrics.png`, `results/ablation/ablation_table.csv`, `figures/ablation/ablation_fairness_energy.png`.
