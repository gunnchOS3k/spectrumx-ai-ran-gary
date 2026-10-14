# Fair-control engineering preparation — P4

This package validates interfaces on constructed fixtures. It does not implement a full research simulator, calibrated task backend, MPC or campaign runner. No historical outcome is loaded for tuning, no new seed sweep runs, and no reserved holdout is supplied.

From this directory with Python3.14 and the pinned dependencies:

```sh
python3 -m venv ../p4-env
. ../p4-env/bin/activate
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python prepare.py --check
```

Acceptance uses an existing local environment, not a fresh installation or cross-platform validation. `prepare.py --execute` exits with status2 without running a campaign. `verification/ENGINEERING_VALIDATION.json` is an engineering report, not a new scientific result. SOURCE_PROVENANCE.json hashes exact source/config/schema/test bytes and records runtime. No article/data license is inferred from repository software licensing.

Implemented contracts:

- Immutable exogenous TruthFrame with separate PathTruth dependencies and resources; no shared mutable policy truth.
- Immutable whitelisted Observation of supplied estimates, with consumed-field logging; no true-state/future object retained. This is an auditable Python cooperation contract, not isolation against malicious reflection or global side channels. A process/IPC sandbox is a future integration requirement.
- Execution-side action validation for availability, terminal/gateway/visibility, placement capacity, per-task allocations, global resources and transition energy/stall; no candidate-query access to true feasibility. Explicit defer allocates nothing and does not count as task success.
- Common transition costs, compute-evaluation and elapsed-time hooks, stable tie handling and common timeout/defer behavior. `choose` is a bounded current-stage callback hook, not a completed transition-aware controller.
- Independently specified task requirements/outcomes. Correctness, deadlines, freshness, quality and durability affect success; missing/rejected tasks remain in the exogenous denominator. The actual application oracle/backend remains to be integrated and validated.
- Paired seed-unit t analysis and prospective precision/power planning with cap handling; zero inclusion never means equivalence. Tiny arrays in tests are constructed assertions, not empirical data.
- Draft2020-12 config/report/future-result schemas, exact canonical JSON hashes and source/environment records. Future-result schema is unpopulated. Configuration is deliberately DESIGN_DRAFT_ENGINEERING_ONLY with campaign/publication authorization false and no reserved seeds/outcomes.

Before a campaign: finalize the primary hypothesis/margin and planning alternative; implement and verify task/queue/battery/switch dynamics and common predictions; implement both candidate policies and baselines with matched interfaces; calibrate or label synthetic inputs; freeze tuning/pilot/holdout procedure and resource budget; check independent validation; obtain separate execution authorization. See ../PROSPECTIVE_PROTOCOL.md. Engineering validation does not establish scientific fairness, physical realism, controller superiority or human approval.

An initial new test expected three permitted observation reads while its fixture performed two. The failed log/result is retained; the expected count was corrected to the actual two reads. No historical or scientific-result test was changed.
