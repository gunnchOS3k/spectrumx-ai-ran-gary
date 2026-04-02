# Sequence — competition inference (current)

| | |
|---|---|
| **Status** | **Current** — Streamlit + `evaluate()` path |
| **Purpose** | Trace upload/demo IQ through loaders, submission adapter, optional baselines, and UI visualization. |
| **Source** | [`docs/uml/sequence_competition_inference_current.mmd`](../sequence_competition_inference_current.mmd) |

Official leaderboard scoring remains **organizer-offline**; this sequence is the **in-app** audit and demo path.

```mermaid
sequenceDiagram
  autonumber
  actor User as User / Judge
  participant ST as Streamlit app\napps/streamlit_app.py
  participant LO as IQ loader\nload_iq_data / synthetic demo
  participant AD as submission_adapter\nrun_evaluate_on_iq_array
  participant PKG as submissions/*/main.py\nevaluate
  participant FB as feature_baseline\noptional extract_features
  participant VZ as Plotly / UI metrics

  User->>ST: Upload .npy OR demo / Judge Mode synthetic IQ
  ST->>LO: Normalize to complex64 window
  LO-->>ST: iq array + sample_rate

  alt Final submission package
    ST->>AD: Load module + run on iq array
    AD->>PKG: evaluate-like path on in-memory IQ
    PKG-->>AD: pred 0/1 + optional confidence
    AD-->>ST: prediction + info dict
  else Baseline detectors in Standard Mode
    ST->>ST: energy_detector / spectral_flatness / PSD+LogReg
    ST-->>ST: prediction + confidence
  end

  opt Interpretability panel
    ST->>FB: extract_features(iq, sr)
    FB-->>ST: feature rows
  end

  ST->>VZ: Time / PSD / spectrogram / microscope
  VZ-->>User: Rendered views

  Note over PKG,LE: Official leaderboard scoring uses organizer pipeline\nnot this Streamlit session
```

[← Current index](index.md)
