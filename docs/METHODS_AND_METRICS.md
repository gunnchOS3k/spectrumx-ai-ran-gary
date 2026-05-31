# Methods and Metrics

## Data

- **Public:** synthetic site YAML, toy benchmarks, no private competition IQ data in git.
- **Future:** calibrated traces under ethics approval — not included here.

## Baselines

Proportional-fair allocation, exhaustive/hierarchical beam search, terrestrial-only vs NTN fallback.

## AI/ML methods

RL/bandit policies (SpectrumX research extension), LSTM/beam trackers (ReadyGary), policy stubs (7GC).

## Evaluation

Unit tests, toy CLI demos, Streamlit scaffold, markdown benchmark tables.

## Metrics

| Metric | Repo |
|--------|------|
| Spectral efficiency (bps/Hz) | 7gc, spectrumx, readygary |
| Energy per bit | 7gc, spectrumx |
| Latency / QoS | 7gc, edge-io |
| Fairness (Jain) | 7gc, spectrumx |
| Outage / resilience | ntn |
| Top-k / dB-loss / SE-loss | readygary |
| Inference latency | readygary, edge-io |
| Privacy risk indicators | edge-io, all ethics docs |
