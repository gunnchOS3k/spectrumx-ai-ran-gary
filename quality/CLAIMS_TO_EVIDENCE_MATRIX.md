# Claims to Evidence Matrix — spectrumx-ai-ran-gary

Statuses: `proven` | `synthetic-only` | `planned` | `not claimed`

| Claim | Status | Evidence (file / test / figure) | Limitation |
|-------|--------|----------------------------------|------------|
| Repository installs and tests pass | proven | `make test`, `tests/test_airan_research.py` | Smoke only |
| Synthetic AI-RAN benchmark is deterministic | synthetic-only | `scripts/run_benchmark.py`, `results/benchmark/metrics.csv` | No live RAN |
| Policy ablation compares 4 toy policies | synthetic-only | `scripts/run_ablation.py`, `results/ablation/ablation_table.csv`, `figures/ablation/ablation_fairness_energy.png` | Stubs not xApps |
| Fairness metric computed (Jain index) | synthetic-only | `src/airan_research/fairness.py`, ablation CSV | Proxy fairness |
| Energy-aware objective represented | synthetic-only | `src/airan_research/energy.py`, ablation CSV | Not joules measured |
| O-RAN E2/KPM live integration | not claimed | `scripts/export_oran_kpi_schema.py` (stub export) | Export-only |
| Near-RT RIC xApp deployment | not claimed | `industry_research_stack/ORAN_XAPP_RAPP_ALIGNMENT.md` | Alignment doc |
| Gary citywide 6G impact | not claimed | `docs/LIMITATIONS_AND_NON_CLAIMS.md` | Exploratory framing |
| SpectrumX competition SOTA detection | planned | Phase-1 pipeline in `src/edge_ran_gary/` | Requires IQ run |
| Field measurements validate twin | planned | Link to `edge-io-measurement-node` | Separate repo |
| Carrier-grade AI-RAN | not claimed | — | Explicit non-claim |
| Unauthorized RF transmission | not claimed | Receive-only observation docs in umbrella | Legal boundary |
| Zenodo DOI for this repo | planned | `CITATION.cff` | Umbrella release |
| IEEE conference paper draft exists | proven | `paper/ieee_conference_draft.md` | Preprint not submitted |
