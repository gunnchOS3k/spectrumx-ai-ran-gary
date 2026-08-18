# Class diagram — extension / scenario / hooks (current)

| | |
|---|---|
| **Status** | **Current** — Gary extension + integration hooks |
| **Purpose** | Model scenario inputs/state, `gary_scenario_engine`, manifest hooks, provenance finalize, and pyAerial bridge **abstractions**. |
| **Source** | [`docs/uml/class_diagram_extension_current.mmd`](../class_diagram_extension_current.mmd) |

**Controller:** *Detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)* — `select_closed_loop_action` / `apply_action_to_kpis`. Gate 2 `twin_policies` is a **separate** rule-based / SLSQP path (`src/airan_research/gate2`). Full PHY execution remains an **external** target.

```mermaid
classDiagram
  direction TB

  class ScenarioInputs {
    +preset
    +rf_environment_stress
    +manual_traffic_stress
    +manual_occ_prior
    +time_context
    +event_high_load
    +site overrides
  }

  class SiteScenarioState {
    +site_id
    +people_count
    +traffic_demand_score
    +coexistence_pressure
    +coverage_pressure
    +fairness_community_priority
    +to_display_dict()
  }

  class gary_scenario_engine {
    <<module>>
    compute_all_anchor_states()
    select_closed_loop_action()
    apply_action_to_kpis()
    propagation_proxy_bundle()
  }

  class simulation_integration_hooks {
    <<module>>
    load_deepmimo_*
    load_sionna_*
    load_aerial_*
    load_pyaerial_bridge_status()
    load_ota_evidence_status()
  }

  class simulation_provenance {
    <<module>>
    finalize_simulation_status()
    civic_stack_summary()
  }

  class streamlit_extension {
    <<apps/streamlit_app.py>>
    Gary twin + pydeck + panels
  }

  class PHYBridgeStatus
  class PHYControlPlaneHints
  class CUMACSchedulerAbstraction

  class pyaerial_bridge {
    <<package>>
    describe_pyaerial_environment()
    detector_to_phy_control_plane_hints()
    cumac_scheduler_abstraction()
  }

  class TwinContext {
    <<airan_research.gate2>>
    +site_id
    +spectrum_budget
    +from_twin_state()
  }

  class twin_policies {
    <<module>>
    policy_static_uniform()
    policy_network_only()
    policy_service_priority()
    policy_optimization_based()
    policy_twin_informed()
  }

  ScenarioInputs --> gary_scenario_engine : drives
  gary_scenario_engine --> SiteScenarioState : emits per anchor
  gary_scenario_engine --> gary_scenario_engine : policy + KPI deltas

  streamlit_extension --> gary_scenario_engine
  streamlit_extension --> simulation_integration_hooks
  streamlit_extension --> simulation_provenance : finalize + display
  simulation_integration_hooks ..> simulation_provenance : hook dicts

  TwinContext --> twin_policies : rule-based / SLSQP
  twin_policies ..> gary_scenario_engine : separate Gate 2 path, not Streamlit RIC

  pyaerial_bridge ..> PHYBridgeStatus : creates
  pyaerial_bridge ..> PHYControlPlaneHints : creates
  pyaerial_bridge ..> CUMACSchedulerAbstraction : creates
  streamlit_extension --> pyaerial_bridge : optional

  note for gary_scenario_engine "Detector-conditioned rule-based closed-loop policy baseline RIC-style abstraction"
  note for twin_policies "Gate 2: still not RL. POLICY_NAMES static_uniform network_only service_priority optimization_based twin_informed"
```

[← Current index](index.md)
