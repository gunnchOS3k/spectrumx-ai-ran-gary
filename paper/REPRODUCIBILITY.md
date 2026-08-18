# Reproducibility (Paper II)

```bash
make paper-reproduce
```

Requires the frozen protocol file `paper/artifacts/experiment_protocol.yaml` (must exist before held-out JSON).

- Digital orchestration tables: `SYNTHETIC_SIM`
- Compute: `HOST_PROCESS_TIMING` (not measured RF)
- Judged SpectrumX `evaluate` path is separate and must not be overwritten
- ReadyGary companion: 28 GHz is FR2 mmWave, never Sub-6
