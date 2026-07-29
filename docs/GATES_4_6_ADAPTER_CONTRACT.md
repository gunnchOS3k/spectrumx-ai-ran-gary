# Gates 4–6 Soft Adapter Contract (spectrumx-ai-ran-gary)

Optional soft fields for the Oulu research repo
(`gunnchos-emergent-service-intent-protocols`). **No hard dependencies.**

## Non-goals

- No required import of Oulu packages from SpectrumX
- No carrier-grade or 3GPP certification claim
- No Gate 6 physical evidence from this document

## Soft adapter fields (SpectrumX → Oulu)

| Field | Type | Meaning |
|---|---|---|
| `decision_bundle_path` | string \| null | Path to policy / AI-RAN decision artifact |
| `twin_state_ref` | string \| null | Twin state used for the decision, if any |
| `spectrum_util` | number \| null | Soft utilization hint in `[0, 1]` |
| `policy_id` | string \| null | Policy or baseline identifier |
| `benchmark_csv_path` | string \| null | Optional benchmark export |
| `evidence_label` | string | Usually `SYNTHETIC_EXPERIMENT` for toy/demo runs |
| `adapter_status` | string | `AVAILABLE` \| `MISSING_SIBLING` \| `STUB` |

## Soft adapter fields (Oulu → SpectrumX, optional)

| Field | Type | Meaning |
|---|---|---|
| `service_intent_id` | string \| null | Intent id from Oulu |
| `intent_priority` | number \| null | Soft priority hint |
| `constraints` | string[] | Soft constraint tags only |
| `comm_mode` | string \| null | Communication mode label |

## Integration rule

Oulu adapters should tolerate a missing `../spectrumx-ai-ran-gary` tree.
SpectrumX remains independently runnable; this contract is soft documentation
only (no package pins, no hard imports).
