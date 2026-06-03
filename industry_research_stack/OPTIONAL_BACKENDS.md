# Optional Backends

| Backend | Purpose | Required for default e2e? |
|---------|---------|---------------------------|
| None (neutral JSON) | Site/policy export stubs | No |
| Sionna | Ray-tracing / link simulation | No |
| DeepMIMO | Realistic MIMO channels | No |
| O-RAN SC patterns | KPI / xApp alignment | No |
| ns-3 / ns-O-RAN | System simulation | No |
| NVIDIA Aerial | CUDA RAN / AI-RAN | No — access + GPU |
| srsRAN / OAI | Open RAN testbed | No — external install |
| Keysight/R&S/DEKRA | Lab validation roadmap | No — paid services |

Run optional targets: `make e2e-sionna`, `make e2e-deepmimo`, etc. (where defined).
