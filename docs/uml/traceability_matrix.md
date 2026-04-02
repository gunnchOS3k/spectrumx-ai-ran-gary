# Traceability — lanes to diagrams

Maps **judged competition core**, **completed research extension**, and **future research-adoption** targets to the GitHub-visible UML entry points under `docs/uml/`.

| Lane | Narrative scope | Context / ecosystem | Container / component | Deployment | Behavioral (seq./act.) | Use cases | Structure / state |
|------|------------------|---------------------|------------------------|------------|-------------------------|-----------|-------------------|
| **Judged competition core** | `submissions/*/main.py`, offline eval, Judge Mode microscope | [system_context_current](current/system_context_current.md) | [container](current/container_view_current.md), [component](current/component_view_current.md) | [deployment_current](current/deployment_current.md) | [seq inference](current/sequence_competition_inference_current.md), [judge flow](current/sequence_judge_review_flow.md), [signal lifecycle](current/activity_signal_lifecycle_current.md) | [use_cases_competition_core](current/use_cases_competition_core.md) | [class detection](current/class_diagram_detection_current.md), [provenance state](current/state_provenance_evidence.md) |
| **Completed research extension** | Gary three anchors; **detector-conditioned rule-based closed-loop policy baseline (RIC-style abstraction)** | (same context diagram) | (same container/component) | (same deployment) | [scenario→KPI](current/sequence_extension_scenario_to_kpi_current.md), [manifest ingestion](current/sequence_simulation_manifest_ingestion.md), [signal ecology](current/activity_signal_ecology_extension_current.md) | [use_cases_research_extension](current/use_cases_research_extension_current.md) | [class extension](current/class_diagram_extension_current.md), [controller ladder](current/state_controller_maturity_ladder.md) |
| **Future research-adoption** | AODT / full Sionna / pyAerial execution / OTA labs — **targets**, not all in-app | [system_context_future](future/system_context_future_research_adoption.md) | [future stack](future/component_view_future_research_stack.md) | [future deployment](future/deployment_future_research_adoption.md) | [experiment program](future/activity_experiment_program_current_to_future.md) | [use_cases_future](future/use_cases_future_research_adoption.md) | [controller ladder](current/state_controller_maturity_ladder.md) (planned arms) |

**Legacy (historical only):** [legacy index](legacy/index.md)

[← UML README](README.md)
