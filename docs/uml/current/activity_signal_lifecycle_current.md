# Activity — signal lifecycle (current, competition-style)

| | |
|---|---|
| **Status** | **Current** — one-second IQ path |
| **Purpose** | End-to-end flow from ingest through detector path to visualization and audit separation from CSV metrics. |
| **Source** | [`docs/uml/activity_signal_lifecycle_current.mmd`](../activity_signal_lifecycle_current.mmd) |

```mermaid
flowchart TD
  A[Ingest 1-second IQ sample\n.npy upload or synthetic demo] --> B[Normalize format\ncomplex64 vector + sample rate]
  B --> C[Optional: compact features\nextract_features for tables/charts]
  B --> D[Detector path]
  D --> D1[Submission evaluate\nor baseline threshold detectors]
  D1 --> E[Binary label 0/1\noccupied vs noise-only]
  E --> F[Optional confidence / probability]
  F --> G[Render microscope\nI/Q, Welch PSD, spectrogram]
  C --> G
  G --> H[User / judge audit trail\nCSV metrics separate from this path]

  style D1 fill:#d5e8d4
  style E fill:#d5e8d4
```

[← Current index](index.md)
