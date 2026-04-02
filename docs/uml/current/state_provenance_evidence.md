# State — provenance and evidence vocabulary

| | |
|---|---|
| **Status** | **Current** — documentation contract |
| **Purpose** | Relate the six canonical evidence terms to three execution surfaces (manifest-load, external runtime, lab OTA). |
| **Source** | [`docs/uml/state_provenance_evidence.mmd`](../state_provenance_evidence.mmd) |

Authoritative definitions: [`docs/PROVENANCE_LEGEND.md`](../../PROVENANCE_LEGEND.md).

```mermaid
flowchart LR
  subgraph evidence["Six evidence terms (canonical)"]
    direction TB
    proxy[proxy-only]
    demo[loaded demo]
    simex[loaded simulation export]
    inst[installer-ready]
    ext[external-runtime-required]
    ota[OTA-backed]
  end

  subgraph execution["Three execution surfaces"]
    direction TB
    mload[manifest-load-only]
    extex[external-runtime-required]
    lab[lab-OTA-workflow]
  end

  evidence ~~~ execution

  classDef dim fill:#f8f9fa,stroke:#6c757d
```

[← Current index](index.md)
