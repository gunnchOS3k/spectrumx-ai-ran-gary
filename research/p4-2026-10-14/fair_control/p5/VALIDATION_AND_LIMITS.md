# Engineering evidence and limitations

Use ../verification/p5-final/TEST_RESULT.json and TESTS.txt for collected/executed counts, failures/errors and source identity; no inferred pass count. The existing32 P4 tests are unchanged. Portal p5/P4_REEXECUTED_TESTS.txt confirms actual collection/run32 against the old exact source before P5 edits. Intermediate31 count predates the final cap-boundary regression.

P5 tests use authored byte/work/time/capacity examples with known expected outputs and deliberately invalid mutations. They are not seed-level experiments, controller ranking, measured service, independent human peer review or inferential research results.

| Integrity property | Executable checks |
|---|---|
| OBSERVATION_EQUIVALENCE | common pair exogenous bytes/budget mismatch, Opportunity hardware/cost mutations |
| ACTION_FEASIBILITY | unavailable paths/common gateway; local network requirement; stale estimated choice rejected by truth |
| RESOURCE_BUDGET_COMPLIANCE | CPU/memory/power/bandwidth/energy overshoot; finite-resource catalog; endpoint changes with work capacity |
| COMPUTE_ACCOUNTING |1-call exhaustion; elapsed overrun; model failure; real process hang/failure/kill |
| SWITCHING_COST_CONSISTENCY | identical execution math across copies; exact interruption/energy; latency reset; migration progress |
| TASK_OUTCOME_INDEPENDENCE | wrong bytes despite connectivity; independent expected digest; wrong content; quality/freshness/durable failure; all-scheduled denominator |
| DETERMINISTIC_REPLAY | canonical snapshot, contract identity, restart continuation equality; controller/result non-timing parity |
| NO_FUTURE_INFORMATION_LEAKAGE | reject undeclared current/future fields; immutable estimate requests; no truth controller API; not a malicious security proof |
| NO_UNAUTHORIZED_HOLDOUT_ACCESS | modes permanently rejected before output; seeded/holdout path and authorization mutations rejected; no holdout reader |
| PROVENANCE_COMPLETENESS | missing source file/hash, missing revision, source manifest and result-hash/numerator mutations rejected |

Known mathematical answers include latency.5s +2Mbit/(2Mbps) leaving.5s CPU at2Mcycles/s→1Mcycle in2s;2W×2s→4J; switch.5s at2Mcycles/s halves one-slot work and adds.25J; timed progress restore; early completion at a sub-slot deadline; H2 edge2Mcycles/s finishes a3Mcycle task that local1Mcycle/s cannot. These isolated assertions validate equations, not broad controller optimality.

Initial90-test failure mixed network and placement costs in a placement-only fixture; fixed by explicitly zeroing network cost in that fixture. Later102-test prediction fixture supplied task-local history but previous_action=None, correctly invoking first-action zero cost; fixed by supplying the prior local action. Failed logs and all intermediate passing records are retained under verification/p5-*. No old invariant, historical test or expected scientific result was weakened. New boundaries also prompted explicit network-latency reset, attempted-output retention, byte-dependent content backend and floating-share capacity projection.

Finite synthetic dynamics omit actual application runtime, cache/download protocol, packet scheduling/retransmission/loss, OS persistence, CPU/radio-dependent power calibration, multi-interval handover, background drain/recharge, physical RF/NTN and empirically fitted arrival/correlation/forecast model. Current fidelity only changes quality; do not claim beneficial adaptation. Code equality validates implementation contracts, not scientific adequacy. No heldout populations, future seeds or old experiment inputs are loaded.

Engineering-ready means all implemented scoped components pass local/clean-copy tests and exact artifacts are recoverable. It does not mean all three P4 prospective campaigns have complete calibrations/algorithms or science approval. Paper I actual classifier/application process study and Paper III correlated architecture campaign remain future work; this implements their shared task-outcome dependency only.
