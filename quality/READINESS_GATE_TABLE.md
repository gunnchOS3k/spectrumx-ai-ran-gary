# Readiness Gate Table

| Gate | Name | Evidence required | Current status |
|------|------|-------------------|----------------|
| 0 | Concept | README, diagrams | Met |
| 1 | Smoke Test | `make e2e` / smoke only — no readiness claim | Met (smoke only) |
| 2 | Real Evidence Pipeline | `docs/REAL_EVIDENCE_PIPELINE.md` | **Open** |
| 3 | Benchmark / Field Data | Dataset or measurement | **Open** |
| 4 | Internal Validation | Baselines, ablations | **Open** |
| 5 | External Reproduction | Reproduction log | **Open** |
| 6 | Candidate Release | Paper/pilot/EVT per repo type | **Open** |
| 7 | Final | Accepted validation | **Not claimed** |

Do not tag v1.0 until Gate 7 criteria met for repo type.
