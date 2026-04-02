# Class diagram — detection / judged competition core (current)

| | |
|---|---|
| **Status** | **Current** — modules wired for Streamlit + submissions |
| **Purpose** | Bound `submission_adapter`, `submissions/*/main.py` (`evaluate`), baselines, and Streamlit responsibilities. |
| **Source** | [`docs/uml/class_diagram_detection_current.mmd`](../class_diagram_detection_current.mmd) |

For historical aspirational stacks, see [legacy class diagram (detection)](../legacy/class_diagram_detection_legacy.md).

```mermaid
classDiagram
  direction TB

  class submission_adapter {
    <<module>>
    discover_submission_folders()
    default_best_submission_folder()
    run_evaluate_on_iq_array()
    resolve_repo_root()
  }

  class submission_package {
    <<submissions/pkg/main.py>>
    evaluate(filename) int
    optional artifact files
  }

  class feature_baseline {
    <<module>>
    extract_features(iq, sr) list
  }

  class detection_baselines {
    <<module>>
    energy_detector()
    spectral_flatness_detector()
  }

  class judged_headline_metrics {
    <<module>>
    build_judged_headlines()
  }

  class streamlit_app {
    <<apps/streamlit_app.py>>
    load_iq_data()
    compute_psd()
    compute_spectrogram()
    Judge Mode + Standard Mode
  }

  submission_adapter --> submission_package : loads
  streamlit_app --> submission_adapter : inference
  streamlit_app --> feature_baseline : optional features
  streamlit_app --> detection_baselines : baseline models
  streamlit_app --> judged_headline_metrics : CSV headlines

  note for submission_package "Judged competition core boundary.\nOrganizer scoring is offline."
```

[← Current index](index.md)
