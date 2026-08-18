# State — controller maturity ladder

| | |
|---|---|
| **Status** | **Current vs planned** — honest labeling |
| **Purpose** | Show shipped **rule-based RIC-style baseline** versus **future** study arms (contextual bandit, offline RL, RIC surrogate). |
| **Source** | [`docs/uml/state_controller_maturity_ladder.mmd`](../state_controller_maturity_ladder.mmd) |

**Current (shipped):** *Detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)* plus Gate 2 `twin_policies`. Contextual-bandit and RL are **not** claimed as deployed controllers.

```mermaid
stateDiagram-v2
  [*] --> RuleBasedBaseline : current Streamlit extension
  [*] --> Gate2Policies : make test / airan_research CLI

  state "Current (shipped)" as C {
    RuleBasedBaseline : Detector-conditioned rule-based\nclosed-loop policy baseline\nRIC-style abstraction
    Gate2Policies : airan_research.gate2\nstatic_uniform / twin_informed / SLSQP
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
