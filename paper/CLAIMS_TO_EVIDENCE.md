# Claims (Paper II)

Evidence class: `SYNTHETIC_SIM` unless noted. Timing: `HOST_PROCESS_TIMING`. Judged IQ path not used.

| Claim | Allowed | Evidence |
|---|---|---|
| Rule-based controller exists | yes | `src/airan_research/` + Gate 2 policies |
| Judged IQ path preserved | yes — do not regress | `docs/COMPETITION_SAFETY.md`; this experiment does not import `submissions/*/main.py` |
| Frozen protocol before held-out | yes | commit with `paper/artifacts/experiment_protocol.yaml` then `scripts/run_paper_ii_heldout.py` |
| Held-out `information_equivalent` U $=0.8185$ $[0.8135,0.8235]$ | yes | `results/experiments/rq2_cross_layer_continuity_heldout.json` |
| Held-out `twin_informed` U $=0.7840$ $[0.7805,0.7874]$ | yes | same JSON; **does not beat** `no_adaptation` $0.7965$ |
| Same-information policy gap | yes | twin vs information-equivalent on identical slot features |
| Switching cost reduces twin net U | yes | ablation `no_switch_cost` $0.8142$ vs full $0.7840$ |
| 28 GHz is Sub-6 | **false** — FR2 mmWave | 3GPP TS 38.101-2 |
| Sub-ms RF inference | **no** | host-process timers only |
| Twin always helps continuity | **no** | rejected on in-distribution held-out and `energy_constrained`; twin mean exceeds `no_adaptation` on `outage_heavy` and `high_blockage` in this model |
