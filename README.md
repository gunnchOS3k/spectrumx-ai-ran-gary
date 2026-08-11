# spectrumx-ai-ran-gary

AI-RAN / equitable spectrum **research** (SpectrumX competition lineage + Gary-focused policy experiments). Also known historically as EDGE-RAN Gary.

> **Current release/state:** `DIGITALLY_VALIDATED` research prototype — synthetic benchmarks; **not** carrier-deployed AI-RAN; **not** commercial 6G.

Ecosystem portal: [gunnchos-research-portal](https://github.com/gunnchOS3k/gunnchos-research-portal) · Product charter: [gunnchOS3k_PRODUCT_CHARTER.md](https://github.com/gunnchOS3k/gunnchos-7gc-ai-ran-field-kit/blob/main/program/charter/gunnchOS3k_PRODUCT_CHARTER.md)

## What is this?

Detection/ML pipelines, synthetic twin builders, AI-RAN controller experiments, and smoke/benchmark tooling.

## Why does it exist?

Study fair, efficient radio resource policies for under-resourced mid-sized cities using open/synthetic evidence first.

## Where does it fit?

Product Charter **layer 6** research. Surfaced via portal/field-kit; optional PHY sibling ReadyGary is separate.

## What is real today?

- `make test` / `make smoke` / `make benchmark` / `make ablation` paths
- Synthetic IQ / twin dataset builders
- Competition-oriented detection baselines

## What is simulated / modelled?

- Synthetic Gary twin channels and policy rollouts
- DeepMIMO/Sionna-*style* emulation where configured — not live city RF

## What is physical / external pending?

- Field IQ campaigns under ethics/privacy rules
- Carrier deployment / certification — **not claimed**
- Frontier model parity — **not claimed**

## Try / inspect in 5 minutes

```bash
pip install -r requirements.txt
make test
make smoke
# optional:
# make benchmark && make ablation
```

## Architecture

Python package (`edge_ran_gary` / related) + `configs/` + `apps/` demos + `docs/` evidence policy.

## Repo map

| Path | Role |
|---|---|
| `configs/` | Twin/policy configs |
| package modules | Detection + controller |
| `notebooks/` | Exploration |
| `quality/` | Claims/evidence |
| `docs/` | START_HERE + honesty |

## Interfaces

Research exports toward field-kit / twin consumers. No live RAN control plane.

## Tests

```bash
make lint test contract-test
```

## Evidence

`results/e2e/` + benchmark tables = synthetic unless labeled otherwise.

## Known gaps

Calibrated field data; non-synthetic equity claims; deployment evidence.

## Beginner path

A **lab for fair radio decisions** using safe synthetic data first.

## Intern path

Run smoke + one benchmark; write what is synthetic vs real.

## Expert path

Ablations + claim-to-evidence matrix without 6G commercial language.

## Contribution path

Models, tests, docs honesty. Keep research_not_carrier_deployed.

## Current release / state

**DIGITALLY_VALIDATED**. IMT-2030-aligned research language ≠ commercial 6G.

## Claim boundary

No commercial 6G · no carrier AI-RAN deployment · Cursor DRAFT-only.

---

## Retained detail (post–Cycle 3A front door)

Full prior README (incl. competition narrative): [docs/history/README_PRE_WP012.md](docs/history/README_PRE_WP012.md).

Portfolio guides: [docs/START_HERE.md](docs/START_HERE.md) · [docs/NO_MORE_TOY_DEMOS.md](docs/NO_MORE_TOY_DEMOS.md).
