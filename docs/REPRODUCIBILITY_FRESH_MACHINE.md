# Reproducibility — Fresh Machine Guide

Reproduce the **core synthetic AI-RAN result** for `spectrumx-ai-ran-gary` without private data.

---

## Environment

| Item | Recommended |
|------|-------------|
| OS | macOS 14+ or Ubuntu 22.04+ |
| Python | 3.10 or 3.11 |
| Git | 2.40+ |

---

## Commands (clone → results)

```bash
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
make test
make benchmark
make ablation
```

Optional toy demo:

```bash
make demo-research
```

---

## Dependencies

- Core: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pyyaml` (see `requirements.txt`).
- Optional heavy stack (Sionna, Torch, Streamlit): not required for benchmark/ablation path.
- No competition IQ files required for benchmark/ablation.

---

## Expected outputs

| Path | Description |
|------|-------------|
| `results/benchmark/metrics.csv` | Multi-seed synthetic metrics |
| `figures/benchmark/benchmark_metrics.png` | Utilization + fairness plot |
| `results/ablation/ablation_table.csv` | Four-policy comparison |
| `figures/ablation/ablation_fairness_energy.png` | Ablation bar chart |
| `results/e2e/` | Smoke artifacts after `make smoke` |

---

## Runtime estimates (M1 MacBook, offline)

| Step | Time |
|------|------|
| `pip install -r requirements.txt` | 3–15 min (network) |
| `make test` | < 5 s |
| `make benchmark` | 5–15 s |
| `make ablation` | 5–15 s |
| `make smoke` | 30–90 s |

---

## Verification checklist

- [ ] `metrics.csv` contains seeds 42–46 by default
- [ ] `ablation_table.csv` lists four policies
- [ ] Figures regenerate byte-identically on same seed/OS (matplotlib backend may vary slightly)

---

## What this reproduction does **not** prove

Operational 6G, live AI-RAN control, or representative Gary field RF conditions. See `docs/EVAL_PROTOCOL.md` §7.
