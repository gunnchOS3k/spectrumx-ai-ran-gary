# Research artifact index — RQ2 fidelity + checkpoint/recovery

| Field | Value |
|-------|-------|
| Repo | `spectrumx-ai-ran-gary` |
| Accepted base SHA | `cef3900af100c0526e8f75efd238303fd6a268bd` |
| Candidate SHA | `6591d268b8c3cb86bc0f3cf146cb270ecc047336` |
| Branch | `research/rq2-fidelity-checkpoint-001` |
| Environment | `.venv` Python (local); lean pytest path |
| Exact command (tiny fidelity repro) | `PYTHONPATH=src python -c "from airan_research.experiments.digital_programme import *; ... run_family(...)"` (see `results/experiments/rq2_fidelity_checkpoint_tiny.json`) |
| Seeds (tiny) | `[0, 1]` |
| Protocol | `paper/artifacts/experiment_protocol.yaml` (`frozen: true`) |
| New actions | `fidelity_level ∈ {target,degraded,minimum_useful}`, `checkpoint_action`, `recover_action` |
| Baselines | `fixed_target_fidelity`, `checkpoint_disabled` |
| Adaptive | `adaptive_fidelity`, `adaptive_checkpoint` |
| Outputs | `results/experiments/rq2_fidelity_checkpoint_tiny.json`, `rq2_information_equivalence_audit.json` |
| Evidence class | `SYNTHETIC_SIM` |
| README claim audit | Replaced “Production-ready detection pipeline…” with research-grade wording |
| ReadyGary | No interface change required |
| Physical / external not performed | `MMWAVE_OTA=PHYSICAL_PENDING`, `PIXEL_RF_QOS=PHYSICAL_PENDING`, `CARRIER_ACCEPTANCE=NOT_RUN` |
