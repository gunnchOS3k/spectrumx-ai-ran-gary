# Evaluation Protocol

This document defines the evaluation methodology for spectrum occupancy detection, ensuring reproducibility and preventing data leakage.

---

## Split Policy

### File-Level Splits (No Leakage)

**Principle:** Each `.npy` file is assigned to exactly one split. No file appears in multiple splits.

**Splits:**
- **Train:** 60% of labeled files
- **Validation:** 20% of labeled files  
- **Test:** 20% of labeled files

**Implementation:**
```python
# Use file-level stratification to maintain label balance
from sklearn.model_selection import train_test_split

# Split by filename, not by samples
train_files, temp_files = train_test_split(
    labeled_filenames, 
    test_size=0.4, 
    stratify=labels,
    random_state=42
)
val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    stratify=temp_labels,
    random_state=42
)
```

**Seed:** `RANDOM_SEED = 42` (documented in `src/edge_ran_gary/config.py`)

**Verification:**
- [ ] No file appears in multiple splits
- [ ] Label distribution similar across splits (within 5%)
- [ ] Split assignment reproducible (same seed = same splits)

---

## Metrics

### Primary Metrics

1. **Accuracy**
   - Definition: (TP + TN) / (TP + TN + FP + FN)
   - Use case: Overall correctness
   - Target: > 85% on validation set

2. **ROC-AUC**
   - Definition: Area under ROC curve
   - Use case: Ranking quality, threshold-independent
   - Target: > 0.90

3. **PR-AUC**
   - Definition: Area under Precision-Recall curve
   - Use case: Imbalanced dataset performance
   - Target: > 0.85

### Secondary Metrics

4. **Precision**
   - Definition: TP / (TP + FP)
   - Use case: False alarm control
   - Target: > 0.80

5. **Recall (Sensitivity)**
   - Definition: TP / (TP + FN)
   - Use case: Signal detection rate
   - Target: > 0.85

6. **F1 Score**
   - Definition: 2 * (Precision * Recall) / (Precision + Recall)
   - Use case: Balanced performance
   - Target: > 0.82

7. **False Alarm Rate (FAR)**
   - Definition: FP / (FP + TN)
   - Use case: Regulatory compliance
   - Target: < 0.05

### Calibration Metrics

8. **Expected Calibration Error (ECE)**
   - Definition: Weighted average of |confidence - accuracy| across bins
   - Bins: 10 bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
   - Use case: Reliability of confidence scores
   - Target: < 0.10 (good), < 0.05 (excellent)

9. **Brier Score**
   - Definition: Mean squared error of probabilities
   - Use case: Overall probability quality
   - Target: < 0.15

---

## Evaluation Workflow

### 1. Data Loading
```python
from src.edge_ran_gary.data_pipeline.spectrumx_loader import SpectrumXDataset

dataset = SpectrumXDataset(cfg)
X_labeled, y_labeled, filenames = dataset.load_labeled()

# Apply file-level split
train_files, val_files, test_files = split_files(filenames, y_labeled, seed=42)
```

### 2. Model Training
```python
# Train on train split only
model.fit(X_train, y_train)

# Validate on validation split
val_predictions = model.predict(X_val)
val_proba = model.predict_proba(X_val)
```

### 3. Calibration
```python
# Fit calibrator on validation set
calibrator.fit(val_proba, y_val)

# Apply calibration
calibrated_proba = calibrator.predict_proba(val_proba)
```

### 4. Threshold Selection
```python
# Select threshold on validation set
threshold = select_threshold(y_val, calibrated_proba, policy="max_f1")

# Apply threshold to get final predictions
final_predictions = (calibrated_proba >= threshold).astype(int)
```

### 5. Evaluation
```python
# Compute all metrics on validation set (for model selection)
metrics_val = compute_all_metrics(y_val, final_predictions, calibrated_proba)

# Final evaluation on test set (only once, after model selection)
metrics_test = compute_all_metrics(y_test, final_predictions_test, calibrated_proba_test)
```

---

## Anti-Leakage Checklist

Before running any evaluation, verify:

- [ ] **No test set usage during training**
  - Test set is only loaded for final evaluation
  - No hyperparameter tuning on test set
  - No model selection based on test performance

- [ ] **File-level splits**
  - Same file never appears in multiple splits
  - Splits are based on filenames, not samples

- [ ] **Deterministic splits**
  - Same seed always produces same splits
  - Split assignment saved to file for verification

- [ ] **No data snooping**
  - EDA performed on train set only (or separate exploration set)
  - Feature engineering based on train set statistics only
  - No manual inspection of test set

- [ ] **Calibration on validation**
  - Calibrator fitted on validation set only
  - Threshold selected on validation set only
  - Test set used only for final reporting

- [ ] **Reproducible seeds**
  - All random operations use fixed seed
  - Seeds documented in config files
  - Results reproducible across runs

---

## Reporting Format

### Model Comparison Table

| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 | FAR | ECE |
|-------|----------|---------|--------|-----------|--------|----|----|-----|
| Energy Detector | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Spectral Flatness | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| SSL Model | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| Final Ensemble | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |

### Calibration Curve
- Plot: Predicted probability vs. Actual frequency
- Bins: 10 bins (0.0-0.1, ..., 0.9-1.0)
- Ideal: Diagonal line (perfect calibration)
- Report: ECE value

### ROC Curve
- Plot: TPR vs. FPR
- Report: AUC value
- Mark: Operating point (selected threshold)

### Precision-Recall Curve
- Plot: Precision vs. Recall
- Report: AUC value
- Mark: Operating point (selected threshold)

---

## Submission Requirements

### Artifacts
1. **Model checkpoints** - Trained models (`.pth` or `.pkl`)
2. **Config files** - All hyperparameters and settings (`.yaml` or `.json`)
3. **Split files** - Train/val/test file assignments (`.csv` or `.json`)
4. **Evaluation results** - Metrics on test set (`.json`)
5. **Plots** - ROC, PR, calibration curves (`.png` or `.pdf`)

### Documentation
1. **Model card** - Architecture, training procedure, limitations
2. **Evaluation report** - Metrics, calibration quality, failure analysis
3. **Reproducibility guide** - How to reproduce results from scratch

---

## Reproducibility Contract

### Seeds
- **Split seed:** 42
- **Training seed:** 42 (PyTorch, NumPy)
- **Data augmentation seed:** None (random per epoch, OK)

### Environment
- Python 3.10+
- Dependencies: `requirements.txt`
- CUDA version: Document if using GPU

### Artifacts
- **Configs:** `configs/experiment_name.yaml`
- **Checkpoints:** `results/experiment_name/checkpoints/`
- **Metrics:** `results/experiment_name/metrics.json`
- **Plots:** `results/experiment_name/plots/`

### Verification
- [ ] Fresh clone → install deps → download data → run eval → same results
- [ ] All random operations use documented seeds
- [ ] No manual intervention required
- [ ] Results match previous runs (within floating-point tolerance)

---

## Notes

- **Test set:** Only evaluate once, after all model selection is complete
- **Validation set:** Use for hyperparameter tuning, calibration, threshold selection
- **Train set:** Use for model training only
- **Unlabeled set:** Use for SSL pretraining, not for evaluation
