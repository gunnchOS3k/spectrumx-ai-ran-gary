# Next Steps for ML Implementation (Ananya)

## Week 1: Foundation & Decision Making

### Day 1-2: Representation Decision
**Goal**: Choose and implement representation method

1. **Evaluate representation options**:
   - Test Raw IQ (1D) on a small subset
   - Test STFT Spectrogram (2D) on a small subset
   - Test PSD Features on a small subset
   - Compare: memory usage, training speed, baseline accuracy

2. **Decision criteria**:
   - Which gives best baseline accuracy with simple model?
   - Which is fastest to train?
   - Which works best with available labeled data size?

3. **Implement chosen representation**:
   - Fill in `FeatureExtractor.extract_*()` methods
   - Add unit tests for representation consistency
   - Document output shapes and formats

**Deliverable**: Working representation pipeline with one method implemented

---

### Day 3-4: Learning Strategy Decision
**Goal**: Choose and implement learning strategy

1. **Evaluate learning strategies**:
   - **Supervised Baseline**: Quick test with small labeled set
     - Implement simple MLP/CNN classifier
     - Train on labeled data only
     - Baseline accuracy target: >70%
   
   - **SSL Pretrain + Finetune**: If unlabeled data is large
     - Choose SSL method (SimCLR vs BYOL vs wav2vec)
     - Implement augmentations (start with time shift + noise)
     - Pretrain on unlabeled data (100-200 epochs)
     - Finetune on labeled data (20-50 epochs)
   
   - **Semi-supervised**: If labeled set is medium (500-2000)
     - Implement Mean Teacher or FixMatch
     - Requires consistency regularization

2. **Decision criteria**:
   - Labeled data size: <500 → SSL, 500-2000 → Semi-sup, >2000 → Supervised
   - Unlabeled data size: Large → SSL beneficial, Small → Supervised better
   - Training time: SSL takes longer, but may improve accuracy

3. **Implement chosen strategy**:
   - Fill in `EncoderSSL.pretrain()` or supervised training loop
   - Implement augmentations
   - Set up training schedule (learning rate, epochs, early stopping)

**Deliverable**: Working training pipeline with one learning strategy

---

### Day 5: Calibration & Threshold
**Goal**: Implement calibration and threshold selection

1. **Implement calibration**:
   - Start with Platt scaling (simplest)
   - Use validation set (20% of labeled data)
   - Compute ECE before and after calibration
   - Target ECE: <0.10 (good), <0.05 (excellent)

2. **Implement threshold selection**:
   - Start with "max_f1" policy
   - Compute F1, precision, recall for different thresholds
   - Visualize precision-recall curve
   - If competition has FPR constraint, implement "fixed_fpr" policy

3. **Evaluate calibration quality**:
   - Plot calibration curve (reliability diagram)
   - Check ECE improvement
   - Verify confidence scores are meaningful

**Deliverable**: Calibrated model with threshold selection

---

## Week 2: Integration & Evaluation

### Day 1-2: End-to-End Pipeline
**Goal**: Complete inference pipeline

1. **Implement `DetectionPipeline.predict()`**:
   - Fill in all TODO hooks
   - Test with single IQ sample
   - Verify output format: `{prob, label, confidence, metadata}`

2. **Implement batch prediction**:
   - Optimize for speed (vectorization)
   - Add progress bar for large batches
   - Memory-efficient processing

3. **Integration testing**:
   - Test with Streamlit dashboard
   - Verify predictions match expectations
   - Check metadata is useful for debugging

**Deliverable**: Working end-to-end inference pipeline

---

### Day 3-4: Evaluation & Metrics
**Goal**: Comprehensive evaluation

1. **Implement evaluation metrics**:
   - Accuracy, Precision, Recall, F1
   - AUC-ROC, AUC-PR
   - ECE (Expected Calibration Error)
   - Confusion matrix

2. **Cross-validation**:
   - 5-fold CV on labeled data
   - Report mean ± std for all metrics
   - Identify best hyperparameters

3. **Error analysis**:
   - Analyze false positives and false negatives
   - Visualize difficult samples
   - Identify failure modes

**Deliverable**: Evaluation report with metrics

---

### Day 5: Optional Enhancements
**Goal**: Improve model if time permits

1. **Anomaly detection** (if needed):
   - Implement Isolation Forest or Autoencoder
   - Use for fusion with supervised model
   - Test ensemble performance

2. **Ensemble fusion**:
   - Combine multiple models (if you trained multiple)
   - Test weighted voting vs stacking
   - Measure improvement

3. **Hyperparameter tuning**:
   - Learning rate, batch size, architecture
   - Use validation set for tuning
   - Document best hyperparameters

**Deliverable**: Final model with best performance

---

## Decision Log Template

For each decision point, document:

```markdown
### Decision: [Representation/Learning Strategy/Calibration]

**Options considered**: 
- Option 1: [name] - [pros/cons]
- Option 2: [name] - [pros/cons]
- Option 3: [name] - [pros/cons]

**Chosen**: Option X

**Reasoning**: 
- Data characteristics: [labeled size, unlabeled size, distribution]
- Performance: [baseline accuracy, training time]
- Constraints: [memory, compute, time]

**Results**:
- Accuracy: X%
- Training time: Y minutes
- ECE: Z

**Next steps**: [what to try next if needed]
```

---

## Quick Reference: Decision Points

### Representation
- **Raw IQ (1D)**: Fast, simple, good for energy-based
- **Spectrogram (2D)**: Rich features, good for CNN/SSL
- **PSD Features**: Interpretable, fast training

### Learning Strategy
- **Supervised**: >2000 labeled samples
- **SSL**: <500 labeled, >10000 unlabeled
- **Semi-supervised**: 500-2000 labeled, medium unlabeled

### Calibration
- **Platt**: Small validation set, simple
- **Isotonic**: Large validation set, better ECE
- **Temperature**: Neural networks, minimal overhead

### Threshold
- **Max F1**: Balanced dataset, default
- **Fixed FPR**: Competition constraint, safety-critical
- **ROC Optimal**: General purpose, maximize discrimination

---

## Questions to Answer This Week

1. What representation works best for our data?
2. Do we have enough labeled data for supervised learning?
3. How much does SSL help vs supervised baseline?
4. What calibration method gives best ECE?
5. What threshold policy optimizes for competition metrics?

---

## Resources

- **SSL Methods**: 
  - SimCLR paper: "A Simple Framework for Contrastive Learning"
  - BYOL paper: "Bootstrap Your Own Latent"
  - wav2vec paper: "wav2vec: Unsupervised Pre-training for Speech Recognition"

- **Calibration**:
  - "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017)
  - sklearn.calibration documentation

- **Evaluation**:
  - scikit-learn metrics documentation
  - ROC curve interpretation guide
