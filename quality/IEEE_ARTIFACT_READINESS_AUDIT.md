# IEEE Artifact Readiness Audit — spectrumx-ai-ran-gary

**Project framing:** 6G-aligned / IMT-2030-aligned AI-assisted O-RAN-style control experiment for Gary digital-equity context. **Not** commercial AI-RAN deployment.

| Criterion | Current evidence | Missing evidence | Priority | Cursor-actionable fix |
|-----------|------------------|------------------|----------|------------------------|
| Problem statement | `README.md`, `docs/LIMITATIONS_AND_NON_CLAIMS.md` | Field pain-point quotes with IRB | P1 | Expand `paper/ieee_conference_draft.md` §1–2 |
| Related work | `paper/related_work_seed.md`, `paper/references_stub.md` | Full BibTeX + citation graph | P1 | Complete `references_stub.md` DOIs |
| Methods | `docs/EVAL_PROTOCOL.md`, `src/airan_research/` | Live RIC integration | P2 | Keep synthetic path primary; label stubs |
| Reproducibility | `docs/REPRODUCIBILITY_FRESH_MACHINE.md`, `make benchmark` | Zenodo DOI snapshot | P1 | Umbrella repo Zenodo release |
| Datasets | Synthetic generator documented | Public field telemetry release | P2 | Link `edge-io-measurement-node` exports |
| Baselines | Uniform + 3 policy stubs | Classical RL / xApp baselines | P2 | `scripts/run_ablation.py` (done) |
| Ablations | `results/ablation/ablation_table.csv` | Multi-seed statistical tests | P2 | Extend ablation CLI `--seeds` |
| Limitations | `docs/LIMITATIONS_AND_NON_CLAIMS.md` | Independent reviewer sign-off | P1 | Mirror in paper §10 |
| Ethics | Competition + synthetic-only default | Formal IRB for Gary pilot | P1 | Cross-link umbrella ethics doc |
| Release / DOI | `CITATION.cff` present | Zenodo DOI | P1 | `gunnchos-7gc-ai-ran-field-kit` release |
| CI | `.github/workflows` smoke tests | Benchmark artifact upload | P2 | Add benchmark job to CI |
| README clarity | Status table at top | Single-page reviewer path | P1 | Conference readiness block (done) |
| External reproduction | Fresh-machine doc | Third-party rerun log | P2 | `reproducibility/EXTERNAL_RERUN_LOG.md` template |

**Overall status:** **Conference-draft ready (synthetic evidence)** — not field-validated AI-RAN deployment.
