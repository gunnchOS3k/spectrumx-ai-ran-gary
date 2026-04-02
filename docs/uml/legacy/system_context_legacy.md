# LEGACY — system context (superseded)

| | |
|---|---|
| **Status** | **Legacy** — do not use for post-project accuracy |
| **Why archived** | Wrong repo topology; RL/notebooks and old package names over-emphasized vs. current `submissions/*/main.py` + Streamlit truth. |
| **Source** | [`docs/uml/system_context.mmd`](../system_context.mmd) |
| **Prefer** | [System context (current)](../current/system_context_current.md) |

```mermaid
flowchart TB
    subgraph External["External Systems"]
        SDS[SpectrumX SDS\nDataset Provider]
        Judges[Competition Judges\nEvaluation Panel]
    end

    subgraph Team["Development Team"]
        Edmund[Edmund Gunn, Jr.\nTeam Lead\n6G/AI-RAN Design]
        Ananya[Ananya Jha\nML Modeling\nOptimization/MLOps]
        Noah[Noah Newman\nData Pipeline\nVisualization]
    end

    subgraph Local["Local Development Environment"]
        Compute[Local Compute\nPython 3.10+\nPyTorch/CUDA]
        Notebooks[Jupyter Notebooks\nEDA, Baselines, RL]
    end

    subgraph Repo["Repository: spectrumx-ai-ran-gary"]
        DataPipeline[data_pipeline/\nDataset Loading\nPreprocessing]
        Detection[detection/\nOccupancy Detection\nModels & Inference]
        Channels[channels/\nSionna RT\nChannel Modeling]
        Models[models/\nAI-RAN Controllers\nBandit/RL]
        Sim[sim/\nEnvironment\nEvaluation]
        Viz[viz/\nVisualization\nStreamlit Integration]
    end

    subgraph Cloud["Streamlit Community Cloud"]
        Dashboard[Streamlit Dashboard\napps/streamlit_app.py\nUser Upload & Visualization]
    end

    SDS -->|Download Dataset| DataPipeline
    Team -->|Develop & Commit| Repo
    Judges -->|Evaluate| Repo

    Edmund -.->|Designs| Channels
    Edmund -.->|Designs| Models
    Ananya -.->|Implements| Detection
    Ananya -.->|Implements| Models
    Noah -.->|Implements| DataPipeline
    Noah -.->|Maintains| Dashboard

    Team -->|Develop| Local
    Local -->|Run Experiments| Notebooks
    Notebooks -->|Use| Repo

    DataPipeline -->|Preprocessed IQ| Detection
    Detection -->|Occupancy Probabilities| Sim
    Channels -->|CSI| Sim
    Models -->|Resource Allocations| Sim
    Sim -->|Metrics| Viz
    Detection -->|Predictions| Viz

    Repo -->|Deployed| Cloud
    Cloud -->|Serves| Dashboard
    Dashboard -->|Visualizes| Detection
    Dashboard -->|Displays| Viz

    Judges -.->|Reviews| Dashboard
    Team -.->|Maintains via PRs| Cloud
```

[← Legacy index](index.md)
