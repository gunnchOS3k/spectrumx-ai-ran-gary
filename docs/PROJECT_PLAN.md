# Project Plan: SpX-DAC 2026

## Scope

This project implements a spectrum occupancy detection system for the SpectrumX Student Data & Algorithm Competition (SpX-DAC). The system detects "structured transmission present" from 1-second IQ samples using semi-supervised/self-supervised learning approaches.

### Phase 1: Competition Core (Required)
- Binary classification: occupied (signal present) vs. unoccupied (noise only)
- Uses small labeled set + larger unlabeled set
- Production-ready detection pipeline with calibration
- Evaluation harness with reproducible splits

### Phase 2: Research Extension (Portfolio)
- Digital Twin simulation (Sionna RT + GIS)
- AI-RAN controller (bandit/RL) for resource allocation
- Fairness considerations for under-resourced communities

---

## Milestones

### Sprint 0: Setup (Week 1)
**Target Date:** 2026-01-15
- [x] Repo skeleton + coding standards
- [x] Dataset download script
- [x] Dataset inventory + documentation
- [x] EDA notebook sanitized
- [x] Metrics + split policy defined
- [x] One-command baseline stub

**Definition of Done:**
- All team members can clone repo and run baseline
- Dataset accessible via one command
- Evaluation framework ready

---

### Sprint 1: Baselines (Week 2-3)
**Target Date:** 2026-01-29
- [x] Energy detector implemented
- [x] Spectral flatness detector implemented
- [ ] PSD + Logistic Regression baseline
- [ ] Baseline evaluation on labeled set
- [ ] Baseline comparison report

**Definition of Done:**
- All baseline methods produce predictions
- Evaluation metrics computed (ROC, PR, F1)
- Baseline performance documented

---

### Sprint 2: SSL (Week 4-5)
**Target Date:** 2026-02-12
- [ ] SSL method selection (SimCLR/BYOL/wav2vec)
- [ ] SSL pretraining on unlabeled data
- [ ] Supervised finetuning on labeled data
- [ ] SSL model evaluation

**Definition of Done:**
- SSL model outperforms baselines
- Training pipeline reproducible
- Checkpoints saved and loadable

---

### Sprint 3: Anomaly + Fusion (Week 6)
**Target Date:** 2026-02-19
- [ ] Anomaly detection model (optional)
- [ ] Ensemble fusion implementation
- [ ] Calibration (Platt/Isotonic)
- [ ] Threshold selection policy

**Definition of Done:**
- Final model selected and calibrated
- ECE < 0.10 (good calibration)
- Threshold optimized for competition metrics

---

### Sprint 4: Polish + Submission (Week 7)
**Target Date:** 2026-02-26
- [ ] Robustness testing
- [ ] Final QA run (fresh machine)
- [ ] Submission bundle creation
- [ ] Final review meeting + freeze
- [ ] Submit to SpX-DAC

**Definition of Done:**
- Submission bundle works on fresh clone
- All tests pass
- Documentation complete
- Submission confirmed

---

## Definition of Done (General)

For any task to be considered "Done":

1. **Code:**
   - [ ] Code implemented and tested
   - [ ] Code reviewed by at least one team member
   - [ ] Tests pass (if applicable)
   - [ ] No hardcoded paths or secrets

2. **Documentation:**
   - [ ] Docstrings/comments added
   - [ ] README/docs updated if needed
   - [ ] Usage examples provided

3. **Reproducibility:**
   - [ ] Can be run with one command (or documented steps)
   - [ ] Uses deterministic seeds
   - [ ] Outputs are reproducible

4. **Integration:**
   - [ ] Works with existing codebase
   - [ ] No breaking changes (or migration guide provided)
   - [ ] Follows project coding standards

---

## Timeline Mapping

**Asana CSV Dates → GitHub Project Dates:**

- Sprint 0 tasks (2026-01-05 to 2026-01-11): Already completed (marked Done)
- Sprint 1 tasks (2026-01-12 to 2026-01-29): In Progress
- Sprint 2 tasks (2026-01-30 to 2026-02-12): Backlog
- Sprint 3 tasks (2026-02-13 to 2026-02-19): Backlog
- Sprint 4 tasks (2026-02-20 to 2026-02-27): Backlog

**Adjusted Dates (if CSV dates are past):**
- If current date > CSV due date and task not done: Move to next available sprint
- If task is clearly done (code exists): Mark as Done regardless of date
- If task is partially done: Mark as In Progress

---

## Risk Management

### High-Risk Items
1. **SSL pretraining convergence** - May take longer than expected
   - Mitigation: Start with simpler SSL method (SimCLR), have supervised fallback
2. **Calibration quality** - May not achieve target ECE
   - Mitigation: Try multiple calibration methods, document tradeoffs
3. **Submission bundle** - May have dependency issues
   - Mitigation: Test on fresh machine early, document all dependencies

### Blockers
- Dataset access (requires SDS token)
- GPU availability for SSL training
- Competition deadline (hard constraint)

---

## Success Metrics

### Competition Metrics
- **Primary:** Accuracy on test set (competition-provided)
- **Secondary:** ROC-AUC, PR-AUC, False Alarm Rate
- **Calibration:** ECE < 0.10

### Process Metrics
- All tasks tracked in GitHub Project
- Zero secrets in repository
- 100% reproducibility (fresh clone works)
- All team members contributing regularly

---

## Team Responsibilities

### Edmund Gunn, Jr. (Lead/PM)
- Systems design and architecture
- Evaluation policy and fairness framing
- Paper/story narrative
- Integration and final review

### Ananya Jha (ML Lead)
- Baseline → SSL → Calibration pipeline
- Model optimization and ablation studies
- Reproducibility and MLOps
- Model selection and tuning

### Noah Newman (Data/Eval/Viz Lead)
- Dataset plumbing and preprocessing
- Evaluation harness construction
- Failure analysis and debugging
- Streamlit dashboard maintenance
- Packaging and QA checks
