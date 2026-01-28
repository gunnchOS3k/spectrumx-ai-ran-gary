# Noah's Guide: Streamlit, Eval Harness, Sprint Summary

**Role:** Data/Eval/Viz Lead  
**Focus:** Dataset plumbing, evaluation harness, failure analysis, Streamlit dashboard, packaging/QA checks

---

## Quick Start

### Setup
```bash
# Clone repo
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary

# Create venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up .env (ask Edmund for SDS token)
cp .env.example .env
# Edit .env with your SDS token
```

---

## Streamlit Dashboard

### Local Development
```bash
# Run locally
streamlit run apps/streamlit_app.py

# Or use root wrapper
streamlit run streamlit_app.py
```

### Features
- **File Upload**: Supports `.npy` files (complex, float [I,Q], int16 interleaved)
- **Demo Data**: Click "Generate Demo IQ Sample" for quick testing
- **Visualizations**: Time/IQ, Constellation, PSD, Spectrogram
- **Baselines**: Energy Detector, Spectral Flatness (with tunable thresholds)

### Common Tasks

#### Add a New Visualization
1. Add toggle in sidebar:
   ```python
   show_new_viz = st.toggle("New Viz", value=False)
   ```
2. Add plot in main content:
   ```python
   if show_new_viz:
       # Your plotting code here
   ```

#### Add a New Baseline Model
1. Implement detector function in `apps/streamlit_app.py`:
   ```python
   def new_detector(iq_data, sample_rate, threshold):
       # Your detection logic
       return prediction, confidence, metadata
   ```
2. Add to model dropdown and call it in prediction panel

#### Debug Demo Data Not Showing
- Check `st.session_state['use_demo']` is set to `True` before `st.rerun()`
- Ensure `demo_iq` and `demo_sample_rate` are set in session state
- Verify data-loading conditional checks demo state correctly

### Streamlit Cloud Deployment

**Note:** Edmund (repo owner) handles initial deployment. You maintain via PRs.

1. **Test locally first**:
   ```bash
   streamlit run apps/streamlit_app.py
   ```

2. **Create PR branch**:
   ```bash
   git checkout -b noah/streamlit-feature-name
   ```

3. **Make changes and test**:
   - Edit `apps/streamlit_app.py`
   - Test locally
   - Commit: `git commit -m "Add feature X to Streamlit dashboard"`

4. **Push and open PR**:
   ```bash
   git push origin noah/streamlit-feature-name
   # Open PR on GitHub
   ```

5. **After merge**: Streamlit Cloud auto-deploys from `main` branch

---

## Evaluation Harness

### Structure
- **Location**: `src/edge_ran_gary/sim/evaluation.py`
- **Metrics**: ROC-AUC, PR-AUC, F1, Precision, Recall, ECE (Expected Calibration Error)

### Running Evaluation

#### Single Model
```python
from edge_ran_gary.sim.evaluation import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate(
    y_true=labels,
    y_pred=predictions,
    y_proba=probabilities
)

print(f"ROC-AUC: {results['roc_auc']:.3f}")
print(f"PR-AUC: {results['pr_auc']:.3f}")
print(f"F1: {results['f1']:.3f}")
print(f"ECE: {results['ece']:.3f}")
```

#### Baseline Comparison
```bash
# Run all baselines (when script exists)
python scripts/run_baseline.py --data-dir data/competition_dataset --output results/baselines/
```

### Adding New Metrics

1. Edit `src/edge_ran_gary/utils/metrics.py`:
   ```python
   def compute_new_metric(y_true, y_pred):
       # Your metric logic
       return metric_value
   ```

2. Add to `Evaluator.evaluate()`:
   ```python
   results['new_metric'] = compute_new_metric(y_true, y_pred)
   ```

### Failure Analysis

#### Generate Failure Cases
```python
# Find false positives (predicted signal, actual noise)
fp_mask = (predictions == 1) & (labels == 0)
fp_samples = data[fp_mask]

# Find false negatives (predicted noise, actual signal)
fn_mask = (predictions == 0) & (labels == 1)
fn_samples = data[fn_mask]

# Save for analysis
np.save("results/false_positives.npy", fp_samples)
np.save("results/false_negatives.npy", fn_samples)
```

#### Visualize in Streamlit
- Add "Failure Case Browser" section
- Load failure cases from `results/`
- Show IQ plots, PSD, spectrogram for each case
- Add notes: "Why did model fail here?"

---

## Sprint Summary Template

### Format
```markdown
# Sprint X Summary - [Your Name]

## Completed ✅
- [Task 1] - [Brief description]
- [Task 2] - [Brief description]

## In Progress 🚧
- [Task 3] - [Status update]

## Blockers 🚨
- [Blocker 1] - [What's blocking you?]

## Next Sprint 📋
- [Task 4] - [What you'll do next]
```

### Example
```markdown
# Sprint 1 Summary - Noah Newman

## Completed ✅
- Streamlit dashboard deployed to Streamlit Cloud
- Added demo data generation button
- Fixed IQ loading for int16 interleaved format

## In Progress 🚧
- Evaluation harness - implementing ECE computation
- Failure case browser - collecting false positive samples

## Blockers 🚨
- None

## Next Sprint 📋
- Complete evaluation harness with all metrics
- Add failure case browser to Streamlit
- Generate baseline comparison report
```

---

## How to PR

### Workflow
1. **Create branch**:
   ```bash
   git checkout -b noah/feature-name
   ```

2. **Make changes**:
   - Edit files
   - Test locally
   - Commit: `git commit -m "Descriptive message"`

3. **Push**:
   ```bash
   git push origin noah/feature-name
   ```

4. **Open PR**:
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out PR template
   - Request review from Edmund/Ananya

### PR Template
```markdown
## Description
[What does this PR do?]

## Changes
- [Change 1]
- [Change 2]

## Testing
- [ ] Tested locally
- [ ] Streamlit dashboard works (if applicable)
- [ ] No breaking changes

## Screenshots (if UI changes)
[Add screenshots]
```

### Code Standards
- **Functions**: Small, testable, with docstrings
- **Variables**: Descriptive names
- **Comments**: Explain "why", not "what"
- **Formatting**: Use `black` or follow existing style

---

## QA Checks

### Before PR
- [ ] Code runs locally
- [ ] No hardcoded paths (use config/env)
- [ ] No secrets committed (check `.gitignore`)
- [ ] Docstrings added for new functions
- [ ] Streamlit app works (if changed)

### Before Submission
- [ ] All tests pass
- [ ] Fresh machine QA run completed
- [ ] Documentation updated
- [ ] Submission bundle works

---

## Common Issues

### Streamlit App Not Loading
- Check `streamlit_app.py` exists at root or `apps/`
- Verify `requirements.txt` has all dependencies
- Check Streamlit Cloud logs

### Evaluation Metrics Wrong
- Verify label encoding (0=noise, 1=signal)
- Check probability calibration
- Ensure no data leakage (file-level splits)

### Git Issues
- **Merge conflicts**: `git pull origin main` then resolve
- **Forgot to commit**: `git add .` then `git commit`
- **Wrong branch**: `git checkout correct-branch`

---

## Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Docs**: https://plotly.com/python/
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Repo Issues**: https://github.com/gunnchOS3k/spectrumx-ai-ran-gary/issues

---

## Questions?

- **Edmund**: PM/integration questions
- **Ananya**: ML model questions
- **GitHub Issues**: Technical questions
