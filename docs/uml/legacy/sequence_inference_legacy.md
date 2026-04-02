# LEGACY — sequence inference (SSL / ensemble path)

| | |
|---|---|
| **Status** | **Legacy** |
| **Why archived** | Does not match default Streamlit `evaluate()` / submission adapter path. |
| **Source** | [`docs/uml/sequence_inference.mmd`](../sequence_inference.mmd) |
| **Prefer** | [Competition inference (current)](../current/sequence_competition_inference_current.md) |

```mermaid
sequenceDiagram
    actor User
    participant Streamlit as Streamlit Dashboard\napps/streamlit_app.py
    participant Loader as IQ Data Loader\nload_iq_data()
    participant Preproc as IQPreprocessor
    participant Features as FeatureExtractor
    participant Model as Detection Model\nClassifierHead/AnomalyModel
    participant Calib as Calibrator
    participant Ensemble as EnsembleFusion
    participant Viz as Visualization\nPlotly Charts

    User->>Streamlit: Upload .npy file
    Streamlit->>Loader: load_iq_data(uploaded_file)
    Loader-->>Streamlit: iq_data (complex64 array)

    Streamlit->>Preproc: normalize(iq_data)
    Preproc-->>Streamlit: preprocessed_iq

    Streamlit->>Features: extract_all(preprocessed_iq)
    Features->>Features: extract_time()
    Features->>Features: extract_freq()
    Features->>Features: extract_statistical()
    Features-->>Streamlit: feature_dict

    Streamlit->>Model: predict(feature_dict)
    Model->>Model: forward pass
    Model-->>Streamlit: raw_logits/scores

    Streamlit->>Calib: calibrate(raw_logits)
    Calib-->>Streamlit: calibrated_proba

    alt Multiple Models
        Streamlit->>Ensemble: fuse([proba1, proba2, ...])
        Ensemble-->>Streamlit: final_proba, confidence
    else Single Model
        Streamlit->>Calib: get_confidence(calibrated_proba)
        Calib-->>Streamlit: confidence
    end

    Streamlit->>Streamlit: prediction = argmax(proba)

    Streamlit->>Viz: create_time_plots(iq_data)
    Viz-->>Streamlit: time_domain_fig

    Streamlit->>Viz: create_constellation_plot(iq_data)
    Viz-->>Streamlit: constellation_fig

    Streamlit->>Viz: create_psd_plot(iq_data)
    Viz-->>Streamlit: psd_fig

    Streamlit->>Viz: create_spectrogram(iq_data)
    Viz-->>Streamlit: spectrogram_fig

    Streamlit->>User: Display: prediction, confidence, plots
```

[← Legacy index](index.md)
