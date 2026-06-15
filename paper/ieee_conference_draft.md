# gunnchos 7GC Field Console: Phone-First AI-RAN Measurement Kit (IEEE Conference Draft)

## 1. Title

**Phone-First Field Console for IMT-2030-Aligned AI-RAN Policy Experiments and Exploratory Edge Measurements**

## 2. Abstract

We present a **6G-aligned research prototype** that couples (i) synthetic **AI-assisted O-RAN-style** resource-allocation experiments and (ii) a privacy-preserving **phone-first field console** for exploratory network measurements. The system targets digital-equity contexts such as Gary, Indiana, without claiming carrier-grade AI-RAN, deployable 6G, or citywide operational impact. We report **deterministic synthetic benchmarks** comparing uniform, fairness-aware, energy-aware, and combined policies, and define telemetry, ethics, and reproducibility artifacts suitable for IEEE-style review. All core results run without private IQ data or third-party traffic payload collection. Limitations, non-claims, and an evidence matrix accompany the release.

## 3. Introduction

IMT-2030 envisions intelligent, sustainable, and inclusive 6G systems [1]. Mid-sized, under-connected cities remain underrepresented in RAN experimentation yet face acute connectivity inequities [2]. We frame **gunnchos 7GC Field Console** as a **research prototype** linking:

1. **AI-RAN policy simulation** (`spectrumx-ai-ran-gary`) — toy policies inspired by near-RT RIC xApps, not deployed xApps.
2. **Edge measurement node** (`edge-io-measurement-node`) — opt-in, schema-validated telemetry for twin calibration.
3. **Pilot operations** (planned `gunnchos-digital-equity-wireless-pilot`) — community governance and consent.

We do **not** claim operational 6G, certified hardware, or unauthorized RF transmission.

## 4. Related Work

**6G vision and AI-native RAN.** IMT-2030 frameworks emphasize AI integration, sustainability, and trust [1], [3]. O-RAN specifies near-RT RIC, E2, and KPM interfaces for closed-loop RAN automation [4], [5].

**Simulation stacks.** Sionna, ns-3, 5G-LENA, srsRAN, and OpenAirInterface support repeatable wireless research [6]–[10], typically requiring careful scope control when extrapolating to field claims.

**Field measurement methodology.** Wi-Fi and active/passive probing guidelines stress consent, minimization, and aggregation [11], [12].

**Digital equity.** Community-network and municipal broadband literature documents measurement gaps in legacy industrial cities [2], [13].

**Reproducibility.** Artifact evaluation practices recommend fixed seeds, public scripts, and explicit non-claims [14], [15].

## 5. System Architecture

```text
[Phone / Mac field console] --opt-in telemetry--> [edge-io-measurement-node]
        |                                              |
        v                                              v
[spectrumx-ai-ran-gary synthetic twin] <--- calibration labels (planned)
        |
        v
[Policy stubs: uniform | fairness | energy | combined]
        |
        v
[results/benchmark, results/ablation, figures/]
```

Near-RT RIC integration is **stubbed** via KPI schema exports; live E2 is **planned**, not claimed.

## 6. Methods

**Datasets.** Default evaluation uses synthetic RAN snapshots (`seed=42…46`). Optional SpectrumX IQ and channel models are out-of-band.

**Splits.** File-level IQ splits for Phase-1 detection; seed-held policy evaluation for Phase-2 (see `docs/EVAL_PROTOCOL.md`).

**Metrics.** Jain fairness, toy energy score, spectrum utilization, latency/denial proxies.

**Baselines.** Uniform allocator plus three policy stubs (`scripts/run_ablation.py`).

## 7. AI-RAN Policy Simulation

Policies allocate MHz shares under energy budgets using closed-form stubs (`src/airan_research/policy_interface.py`). This explores **AI-assisted control objectives** without claiming near-RT RIC latency compliance.

Reproducible commands:

```bash
make benchmark   # results/benchmark/metrics.csv
make ablation    # results/ablation/ablation_table.csv
```

## 8. Edge/Field Measurement Integration

The edge node defines a minimization-first telemetry schema (hashed device ID, waypoint labels, no payload capture). Probes (`src/edge_io_node/probes/`) collect latency, loss, jitter, RSSI (platform permitting), and device status with safe fallbacks.

Field integration is **exploratory**; synthetic smoke tests are the default evidence path today.

## 9. Results

**Synthetic benchmark (deterministic).** `results/benchmark/metrics.csv` reports utilization and fairness across seeds.

**Ablation (seed 42 default).** Four policies compared in `results/ablation/ablation_table.csv`; figure at `figures/ablation/ablation_fairness_energy.png`.

These results are **synthetic-only** unless paired with field exports from the measurement node.

## 10. Limitations

- Toy policies are not learned controllers or O-RAN xApps.
- No live RAN or carrier integration.
- Gary-specific propagation not validated at city scale.
- Optional ML/detection paths may depend on competition data not bundled here.

## 11. Ethics and Privacy

Opt-in only; no third-party traffic payload inspection; aggregated reporting; retention/deletion policies documented in the edge repo. Minors require guardian/program approval. SDR discussion remains **receive-only observation** where applicable.

## 12. Reproducibility

Fresh-machine steps: `docs/REPRODUCIBILITY_FRESH_MACHINE.md`. Claims map to artifacts in `quality/CLAIMS_TO_EVIDENCE_MATRIX.md`. Zenodo DOI planned via umbrella release.

## 13. Conclusion

We contribute a **conference-ready research artifact** coupling IMT-2030-aligned AI-RAN policy stubs with a phone-first measurement schema for equitable connectivity research. The release prioritizes honest scope, reproducible synthetic evidence, and explicit non-claims over deployment rhetoric.

---

## References (numbered placeholders)

See `paper/references_stub.md` for the expanded 60-source research package mapped to `[1]`–`[60]`.
