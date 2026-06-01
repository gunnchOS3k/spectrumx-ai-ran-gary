# No More Toy Demos as Proof

## Policy

Synthetic demos are useful for **development** and **CI smoke tests**. They are **not** final evidence.

| Allowed | Not allowed |
|---------|-------------|
| Smoke test proves code runs | Smoke test proves research claim is true |
| Synthetic sanity check in CI | Toy output as conference evidence |
| Development fixture | Toy output as adoption evidence |
| Documented non-evidence run | Toy output as field validation |

## Readiness evidence must come from

- Real-world or accepted benchmark data
- Calibrated simulation with documented assumptions
- Independent external reproduction
- Field or lab pilot (privacy-reviewed)
- Hardware measurements
- Compliance artifacts (hardware only when real)
- School/community pilot records (adoption)

## Commands

- **Smoke test:** `make e2e` or `make smoke` — output labeled `results/smoke/` or `results/e2e/` with `evidence_level: smoke_test_only`
- **Real evidence:** see `docs/REAL_EVIDENCE_PIPELINE.md`
