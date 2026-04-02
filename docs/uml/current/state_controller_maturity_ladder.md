# State — controller maturity ladder

| | |
|---|---|
| **Status** | **Current vs planned** — honest labeling |
| **Purpose** | Show shipped **rule-based RIC-style baseline** versus **future** study arms (contextual bandit, offline RL, RIC surrogate). |
| **Source** | [`docs/uml/state_controller_maturity_ladder.mmd`](../state_controller_maturity_ladder.mmd) |

**Current (shipped):** *Detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)*. Contextual-bandit and RL are **not** claimed as deployed controllers.

```mermaid
stateDiagram-v2
  [*] --> RuleBasedBaseline : current Streamlit extension

  state "Current (shipped)" as C {
    RuleBasedBaseline : Detector-conditioned rule-based\nclosed-loop policy baseline\nRIC-style abstraction
  }

  state "Planned study arms" as P {
    ContextualBandit : Contextual-bandit policy learner\nfuture experiment
    OfflineRL : Constrained offline RL\nfuture experiment
    RICSurrogate : Near-RT RIC surrogate target\nfuture integration
  }

  RuleBasedBaseline --> ContextualBandit : next research arm
  ContextualBandit --> OfflineRL : later arm
  OfflineRL --> RICSurrogate : productionization path

  note right of RuleBasedBaseline : select_closed_loop_action\nheuristic KPI deltas
```

[← Current index](index.md)
