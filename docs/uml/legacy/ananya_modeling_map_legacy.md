# LEGACY — modeling decision map

| | |
|---|---|
| **Status** | **Legacy** — historical design exploration |
| **Purpose (historical)** | Representation / learning / calibration / threshold decision blocks. |
| **Source** | [`docs/uml/ananya_modeling_map.mmd`](../ananya_modeling_map.mmd) |
| **Prefer** | [Class diagram detection (current)](../current/class_diagram_detection_current.md) |

```mermaid
flowchart TD
    Start[1-second IQ Sample\ncomplex64, shape=N] --> RepBlock{REPRESENTATION\nDecision Block A}

    RepBlock -->|Option A1| Raw1D[Raw IQ -> 1D\nDirect complex array\nor I/Q concatenated]
    RepBlock -->|Option A2| STFT2D[STFT -> 2D Spectrogram\nTime-frequency matrix\nshape=time_bins, freq_bins]
    RepBlock -->|Option A3| PSD[PSD Features\nWelch PSD + statistics\nshape=feature_dim]

    Raw1D -.->|When: Simple baseline\nFast inference\nLow memory| RepNote1[Note: Works well for\nenergy-based detectors\nMinimal preprocessing]
    STFT2D -.->|When: Need time-freq structure\nCNN/Transformer friendly\nRich representation| RepNote2[Note: Captures temporal\nand spectral patterns\nGood for SSL/CNN]
    PSD -.->|When: Classical features\nInterpretable\nSmall model size| RepNote3[Note: Hand-crafted features\nWorks with MLP/SVM\nFast training]

    Raw1D --> LearnBlock{LEARNING STRATEGY\nDecision Block B}
    STFT2D --> LearnBlock
    PSD --> LearnBlock

    LearnBlock -->|Option B1| Supervised[Supervised Baseline\nTrain on labeled set only\nCross-entropy loss]
    LearnBlock -->|Option B2| SSL[SSL Pretrain + Finetune\nPretrain on unlabeled\nFinetune on labeled]
    LearnBlock -->|Option B3| SemiSup[Mean Teacher / FixMatch\nConsistency regularization\nPseudo-labeling]

    Supervised -.->|When: Sufficient labeled data\n>1000 samples\nBaseline comparison| LearnNote1[Note: Simple and fast\nGood if labels are reliable\nNo unlabeled data needed]
    SSL -.->|When: Large unlabeled set\nWant transfer learning\nLimited labels <500| LearnNote2[Note: Leverages unlabeled data\nBetter generalization\nRequires pretext task design]
    SemiSup -.->|When: Medium labeled set\n500-2000 samples\nWant consistency| LearnNote3[Note: Uses both labeled\nand unlabeled effectively\nMore complex training]

    Supervised --> CalBlock{CALIBRATION\nDecision Block C}
    SSL --> CalBlock
    SemiSup --> CalBlock

    CalBlock -->|Option C1| Platt[Platt Scaling\nLogistic regression\non logits]
    CalBlock -->|Option C2| Isotonic[Isotonic Regression\nNon-parametric\nPiecewise constant]
    CalBlock -->|Option C3| TempScale[Temperature Scaling\nSingle parameter\nNeural net friendly]

    Platt -.->|When: Small validation set\nSimple calibration\nFast inference| CalNote1[Note: 2 parameters\nWorks well with\nlimited data]
    Isotonic -.->|When: Need strong calibration\nLarger validation set\nNon-linear needed| CalNote2[Note: More flexible\nRequires more data\nBetter ECE typically]
    TempScale -.->|When: Neural network models\nSingle parameter\nFast| CalNote3[Note: Simplest method\nGood for deep models\nMinimal overhead]

    Platt --> ThreshBlock{THRESHOLD POLICY\nDecision Block D}
    Isotonic --> ThreshBlock
    TempScale --> ThreshBlock

    ThreshBlock -->|Option D1| MaxF1[Maximize F1 Score\nTune on validation\nBalanced precision/recall]
    ThreshBlock -->|Option D2| FixedFPR[Fixed FPR\ne.g., FPR=0.01\nControl false alarms]
    ThreshBlock -->|Option D3| ROC[ROC Optimal\nYouden's J statistic\nmaximize TPR-FPR]

    MaxF1 -.->|When: Balanced dataset\nCare about both\nprecision and recall| ThreshNote1[Note: Good default\nWorks well when\nclasses balanced]
    FixedFPR -.->|When: False alarms costly\nNeed low FPR\nCompetition constraint| ThreshNote2[Note: Safety-critical\nSpectrum sensing\nRegulatory compliance]
    ROC -.->|When: Want optimal\nROC curve point\nGeneral purpose| ThreshNote3[Note: Maximizes\ndiscrimination ability\nGood baseline]

    MaxF1 --> Output[Final Prediction\nlabel: 0/1\nprob: calibrated\nconfidence: threshold-based]
    FixedFPR --> Output
    ROC --> Output

    LearnBlock -.->|Optional| Anomaly[Anomaly Detection\nIsolation Forest\nAutoencoder\nOne-Class SVM]
    Anomaly -.->|When: Unlabeled only\nor extreme imbalance\nNovelty detection| AnomalyNote[Note: Unsupervised\nNo labels needed\nGood for fusion]
    Anomaly --> Ensemble[Ensemble Fusion\nCombine multiple models\nWeighted voting]
    Output --> Ensemble
    Ensemble --> Final[Final Decision\n+ Metadata]

    style RepBlock fill:#e1f5ff
    style LearnBlock fill:#fff4e1
    style CalBlock fill:#e8f5e9
    style ThreshBlock fill:#f3e5f5
    style Output fill:#ffebee
    style Final fill:#ffebee
```

[← Legacy index](index.md)
