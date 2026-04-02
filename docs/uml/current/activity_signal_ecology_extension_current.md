# Activity — signal ecology (completed extension)

| | |
|---|---|
| **Status** | **Current extension** — scenario-driven UI layer |
| **Purpose** | Anchor the three Gary sites, scenario engine, manifest merge vs proxy, **detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)**, and ecology visualization. |
| **Source** | [`docs/uml/activity_signal_ecology_extension_current.mmd`](../activity_signal_ecology_extension_current.mmd) |

The stochastic ecology layer is **scenario-generated / illustrative**, not competition waveform ground truth.

```mermaid
flowchart TD
  subgraph anchors["Three Gary anchors"]
    CH[Gary City Hall]
    LIB[Gary Public Library and Cultural Center]
    WSLA[West Side Leadership Academy]
  end

  A[User selects focus anchor + scenario preset] --> B[Scenario engine\nScenarioInputs to SiteScenarioState]
  B --> C[People / devices / traffic demand\nfairness + coexistence pressure]
  C --> D{Simulation manifests loaded?}
  D -->|DeepMIMO / Sionna / AODT summaries| E[Merge validated fields\nprovenance finalize]
  D -->|none| F[proxy-only analytic layer]
  E --> G[Detector belief from synthetic demo IQ\noptional input to policy]
  F --> G
  G --> H[Detector-conditioned rule-based\nclosed-loop policy baseline\nRIC-style abstraction]
  H --> I[Action: hold / cautious / power /\nchannel / prioritize / rebalance]
  I --> J[Heuristic KPI deltas\napply_action_to_kpis]
  J --> K[Streamlit: pydeck scene +\necology plot + backbone cards]

  L[Illustrative stochastic signal ecology layer\nscenario-generated not competition waveform truth] -.-> K

  style H fill:#e8daef
  style L fill:#fdebd0,stroke:#e67e22
```

[← Current index](index.md)
