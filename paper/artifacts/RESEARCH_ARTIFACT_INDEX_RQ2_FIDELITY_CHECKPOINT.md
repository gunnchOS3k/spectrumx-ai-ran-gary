# Research artifact index — RQ2 fidelity + checkpoint/recovery

| Field | Value |
|-------|-------|
| Repo | `spectrumx-ai-ran-gary` |
| Accepted base SHA | `cef3900af100c0526e8f75efd238303fd6a268bd` |
| Branch | `research/rq2-fidelity-checkpoint-001` |
| Environment | `.venv` Python 3.11 Framework; lean pytest path |
| Exact command (tiny fidelity repro) | `PYTHONPATH=src python -c "from airan_research.experiments.digital_programme import *; ... run_family(...)"` (see `results/experiments/rq2_fidelity_checkpoint_tiny.json`) |
| Frozen Paper-II seeds | train `[0..4]` n=5; held-out `[100..104]` n=5 (protocol preserved; n reported separately) |
| Protocol | `paper/artifacts/experiment_protocol.yaml` (`frozen: true`) |
| New actions | `fidelity_level ∈ {target,degraded,minimum_useful}`, `checkpoint_action`, `recover_action` |
| Baselines | `fixed_target_fidelity`, `checkpoint_disabled` |
| Adaptive | `adaptive_fidelity`, `adaptive_checkpoint` |
| Observation contract | `PolicyObservation` + `ALLOWED_OBSERVATION_FIELDS`; audit booleans computed |
| Checkpoint/recovery | Causal `CheckpointRuntimeState` (exists/age/progress/last slots); rolling `THRASH_WINDOW=2` |
| Synthetic params | Versioned `synthetic_parameters` in protocol + predeclared sensitivity JSON |
| Outputs | `rq2_fidelity_checkpoint_tiny.json`, `rq2_information_equivalence_audit.json`, `rq2_fidelity_recovery_sensitivity.json`, frozen Paper-II train/heldout/domain/ablation JSONs + tables |
| Evidence class | `SYNTHETIC_SIM` |
| CI method | Student-t 95%; SciPy `t.ppf(0.975,df)` preferred |
| README claim audit | Research-grade wording retained |
| ReadyGary | No interface change; frozen Paper-II seed protocol unchanged |
| Physical / external not performed | `MMWAVE_OTA=PHYSICAL_PENDING`, `PIXEL_RF_QOS=PHYSICAL_PENDING`, `INDEPENDENT_REPRODUCTION=EXTERNAL_PENDING`, `CERTIFICATION=NOT_RUN`, `CARRIER_ACCEPTANCE=NOT_RUN` |
