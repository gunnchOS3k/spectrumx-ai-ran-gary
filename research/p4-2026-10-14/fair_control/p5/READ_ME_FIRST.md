# P5 implemented synthetic research infrastructure

This extends the existing Paper II fair_control package in place. All three exploratory v0.1 preprints and frozen research remain unchanged. No prospective campaign, new population finding, external preregistration, physical measurement or public action is produced. Research execution remains disabled in code.

Read MODEL_AND_ARCHITECTURE.md, CONTROLLER_CONTRACT.md, REQUIREMENTS_AND_DECISIONS.md, CALIBRATION_MATRIX.md, VALIDATION_AND_LIMITS.md and REPRODUCIBILITY.md. Actual logs and engineering fixture outputs are under ../verification/p5-final/. Canonical program governance remains in the research portal p5 directory; there is no second dashboard.

Implemented: UTF-8 byte task correctness oracle with all-exogenous denominator; explicit lifecycle and deterministic task backend; fluid CPU/data/latency/queue/deadline/energy dynamics; hashed durable synthetic checkpoints and timed restoration; estimate-only finite-action controllers including matched hysteresis and beam MPC; matched opportunity/resource/cost/compute contracts; optional hard process timeout envelope; unit/provenance calibration adapters; source-pinned engineering fixture runner with permanently disabled research modes.

Scope is UNCALIBRATED synthetic engineering (code label UNCALIBRATED_SYNTHETIC). Host runtime timing records characterize these tests only. Simulated watts/joules are dimensioned assumptions, not power-meter observations. MPC is an approximate finite beam search, not a convergence certificate or globally optimal controller. Application examples inspired task contracts; production applications are not executed, imported or altered.

Before any pilot: owner/domain review must fix task semantics, real application adapters, calibrated or explicitly authored parameter tables, estimator and error rules, fidelity tradeoff, failure/arrival mixtures, practical effect, tuning/compute/sample plan and rights. A separately scoped milestone must freeze these and authorize pilot execution. Heldout and publication require distinct approval. Current approvals all false.
