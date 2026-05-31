# Supervisor README — SpectrumX AI-RAN Gary — AI-Native Spectrum Research Extension

> **Start here** if you have ~3 minutes. This is a **research prototype / simulation scaffold**, not operational 6G infrastructure.

## What is this repo?

SpectrumX AI-RAN Gary — AI-Native Spectrum Research Extension — part of the gunnchOS3k MLV **7GC AI-RAN Digital Twin Program**.

## Research problem

AI-native RAN control under spectrum and energy constraints in Gary flagship node.

## Why it matters for 6G

Aligns with trustworthy AI-native RAN, digital twins, edge measurement, NTN resilience, and digital equality for underserved communities (IMT-2030 themes).

## Program thesis

gunnchOS3k MLV is building community-scale AI-RAN, digital-twin, edge-device, and education-linked research infrastructure for trustworthy 6G connectivity in digitally underserved communities. Gary, Indiana is the flagship node; 7GC provides comparative scenario nodes for global digital equality, security, energy efficiency, NTN resilience, and inclusive edge access.

## What can be demoed today?

```bash
python scripts/demo_airan_policy.py --toy
```

See `docs/DEMO_SCRIPT_3_MIN.md` and `docs/REPRODUCIBILITY.md`.

## Real vs simulated

| Real | Simulated / scaffold |
|------|----------------------|
| Code, tests, CI, public YAML configs | Carrier-grade deployment |
| Toy/synthetic demos | Validated field measurements |
| Research direction docs | Oulu/CWC affiliation claims |

## What we do NOT claim

See `docs/LIMITATIONS_AND_NON_CLAIMS.md`. No operational 6G RAN, no certified hardware, no supervisor endorsement, no fabricated results.

## PhD supervision next steps

1. Harden metrics with calibrated datasets under supervision.
2. Connect Gary flagship measurements to twin calibration.
3. Publication-grade benchmarks (ReadyGary, SpectrumX extension).
4. Ethics review for field pilots.

## Links to sibling repos

| Repo | Role |
|------|------|
| [7gc-digital-twin](https://github.com/gunnchOS3k/7gc-digital-twin) | Spine / comparative nodes |
| [spectrumx-ai-ran-gary](https://github.com/gunnchOS3k/spectrumx-ai-ran-gary) | Gary AI-RAN / spectrum |
| [readygary-6g-beam-selection](https://github.com/gunnchOS3k/readygary-6g-beam-selection) | Beam selection benchmark |
| [edge-io-measurement-node](https://github.com/gunnchOS3k/edge-io-measurement-node) | Edge telemetry |
| [ntn-resilience-sim](https://github.com/gunnchOS3k/ntn-resilience-sim) | NTN fallback |
| [waike-research-ops](https://github.com/gunnchOS3k/waike-research-ops) | Apprenticeship OS |
