# Sequence — judge review flow

| | |
|---|---|
| **Status** | **Current** — Judge Mode UX narrative |
| **Purpose** | How a SpectrumX judge navigates headline metrics, scope lanes, core submission, results, and optional extension. |
| **Source** | [`docs/uml/sequence_judge_review_flow.mmd`](../sequence_judge_review_flow.mmd) |

Competition IQ is **not** uploaded to Cloud; synthetic demo supports live inference contract review.

```mermaid
sequenceDiagram
  autonumber
  actor Judge as SpectrumX judge
  participant ST as Streamlit Judge Mode
  participant Hero as Hero metrics strip\nsubmission_metrics.csv
  participant Scope as Scope strip\njudged / extension / future tier
  participant Core as Core Submission tab
  participant Res as Results tab
  participant Exp as Extension tab\noptional

  Judge->>ST: Open app + enable Judge Mode
  ST->>Hero: Load headline row if CSV present
  Hero-->>Judge: Package / rank / accuracy / runtime placeholders if empty

  ST->>Scope: Render three-lane scope
  Scope-->>Judge: Judged core vs extension vs validation path

  Judge->>Core: Open Core Submission
  Core->>Core: Judged-sample microscope IQ/PSD/spec
  Core->>Core: Live evaluate on synthetic demo IQ
  Core-->>Judge: Prediction + confidence + inference contract note

  Judge->>Res: Open Results
  Res-->>Judge: CV / leaderboard table from CSV

  opt Deeper review
    Judge->>Exp: Completed Research Extension
    Exp-->>Judge: Gary twin + rule-based controller semantics\nprovenance panels
  end

  Note over Judge,ST: Competition IQ not uploaded to Cloud\nsynthetic demo only for live inference
```

[← Current index](index.md)
