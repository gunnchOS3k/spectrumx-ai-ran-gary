# Interface and integration catalog

Concise map of **primary contracts** and **artifact surfaces** referenced across UML and runbooks.

| Interface / artifact | Role | Where documented |
|---------------------|------|------------------|
| **`evaluate(filename)`** | Competition submission entry; binary occupancy (+ optional confidence) | [class_diagram_detection_current](current/class_diagram_detection_current.md), [sequence_competition_inference_current](current/sequence_competition_inference_current.md) |
| **`submission_adapter.run_evaluate_on_iq_array`** | In-memory IQ path into submission package | Same class diagram |
| **Submission metrics CSV** | Judge Mode headline strip (`submission_metrics.csv`) | [sequence_judge_review_flow](current/sequence_judge_review_flow.md) |
| **Manifest loaders** (`load_deepmimo_*`, `load_sionna_*`, `load_aerial_*`, OTA status) | Merge validated external-export fields into UI/provenance | [class_diagram_extension_current](current/class_diagram_extension_current.md), [sequence_simulation_manifest_ingestion](current/sequence_simulation_manifest_ingestion.md) |
| **Provenance finalize** (`finalize_simulation_status`, civic summaries) | Evidence-tier labeling without implying full external runtime in Streamlit | [class_diagram_extension_current](current/class_diagram_extension_current.md), [`docs/PROVENANCE_LEGEND.md`](../PROVENANCE_LEGEND.md) |
| **pyAerial bridge abstractions** | `describe_pyaerial_environment`, control-plane hints, CU MAC scheduler abstraction — **interfaces**, not guaranteed in-process PHY | [class_diagram_extension_current](current/class_diagram_extension_current.md), [`docs/PYAERIAL_BRIDGE.md`](../PYAERIAL_BRIDGE.md) |
| **OTA evidence manifest read** | Data-lake / lab workflow hooks as **manifest-load** or external-runtime tiers | [`docs/PROVENANCE_LEGEND.md`](../PROVENANCE_LEGEND.md), [`docs/EXTERNAL_RUNTIME_GAPS.md`](../EXTERNAL_RUNTIME_GAPS.md) |
| **Scenario engine** (`ScenarioInputs`, `SiteScenarioState`, `select_closed_loop_action`, `apply_action_to_kpis`) | Extension semantics; **rule-based** baseline | [class_diagram_extension_current](current/class_diagram_extension_current.md), [activity_signal_ecology_extension_current](current/activity_signal_ecology_extension_current.md) |

[← UML README](README.md)
