# Edmund's Guide: Micro-Twin Build, Integration, Issue Hygiene, Submission

**Role:** Lead/PM + 6G/AI-RAN + Digital Twin + Narrative  
**Focus:** Systems design, evaluation policy, fairness framing, paper/story, integration

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

# Set up .env (get SDS token from SpectrumX)
cp .env.example .env
# Edit .env with SDS_TOKEN=your_token_here
```

---

## Gary Micro-Twin Build

### Overview
The Gary Micro-Twin is a focused 3-zone synthetic data generator:
1. **Gary City Hall** - Civic center, moderate traffic
2. **West Side Leadership Academy** - High school, variable occupancy (equity focus)
3. **Gary Public Library & Cultural Center** - Community hub, steady baseline

### Generate Dataset
```bash
# Generate micro-twin dataset
python -c "
from src.edge_ran_gary.digital_twin.gary_micro_twin import generate_micro_twin_dataset

micro_twin, samples, metadata_df = generate_micro_twin_dataset(
    output_dir='data/gary_micro_twin',
    n_per_zone=1000,
    label_balance=0.5,
    seed=42
)
"
```

### Modify Zones
Edit `configs/gary_micro_twin.yaml`:
```yaml
zones:
  gary_city_hall:
    name: "Gary City Hall"
    lat: 41.6026
    lon: -87.3372
    radius_m: 500
    weight: 1.0
    occupancy_prior: 0.5
    noise_floor_prior: -85.0
    snr_range: [5, 20]
    # ... etc
```

### Integration Points for Ananya
The micro-twin provides:
- **Zone-aware synthetic data** for controlled ML testing
- **Metadata** (zone_id, SNR, CFO, multipath) for ablation studies
- **Reproducible seeds** for deterministic experiments

**Usage in ML pipeline**:
```python
# Ananya can use this for:
# 1. Zone-aware training/testing
# 2. SNR ablation studies
# 3. Fairness evaluation across zones
```

---

## Integration Workflow

### Merge PRs
1. **Review PR**:
   - Check code quality
   - Verify tests pass
   - Ensure no secrets committed

2. **Merge**:
   ```bash
   # On GitHub: Click "Merge pull request"
   # Or locally:
   git checkout main
   git pull origin main
   git merge --no-ff feature-branch
   git push origin main
   ```

3. **Verify Integration**:
   - Run tests: `pytest tests/` (if exists)
   - Check Streamlit: `streamlit run apps/streamlit_app.py`
   - Verify docs updated

### Handle Conflicts
```bash
# Pull latest main
git checkout main
git pull origin main

# Rebase feature branch
git checkout feature-branch
git rebase main

# Resolve conflicts, then:
git add .
git rebase --continue
git push origin feature-branch --force-with-lease
```

---

## Issue Hygiene

### Creating Issues

#### Template
```markdown
## Context
[Why does this matter?]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Dependencies
- [ ] Depends on issue #X
- [ ] Blocks issue #Y

## Owner Growth Outcome
- [Skill 1]
- [Skill 2]

## Deliverable
[Code/Experiment/Figure/Doc/Demo/Submission]

## Notes
[Any risks, including data leakage]
```

#### Example
```markdown
## Context
We need to implement PSD+LogReg baseline to compare against energy detector.

## Acceptance Criteria
- [ ] PSD feature extraction implemented
- [ ] Logistic regression model trained on labeled data
- [ ] Evaluation shows ROC-AUC > 0.7
- [ ] Code documented with docstrings

## Dependencies
- None

## Owner Growth Outcome
- Feature engineering for RF signals
- Scikit-learn model training
- Evaluation metrics interpretation

## Deliverable
Code

## Notes
No data leakage risk - using labeled set only.
```

### Labeling Issues

**Labels**:
- `type:baseline`, `type:ssl`, `type:anomaly`, `type:viz`, `type:eval`, `type:mlops`, `type:paper`, `type:pm`, `type:qa`
- `phase:core-detection`, `phase:portfolio-digital-twin`, `phase:portfolio-ai-ran`
- `priority:P0`, `priority:P1`, `priority:P2`, `priority:P3`
- `blocked`

**Project Fields**:
- **Status**: Backlog, Ready, In Progress, Blocked, In Review, Done
- **Sprint**: Sprint 0-4, Portfolio (Digital Twin), Portfolio (AI-RAN)
- **Owner**: Edmund, Ananya, Noah
- **Effort**: XS, S, M, L, XL
- **Deliverable**: Code, Experiment, Figure, Doc, Demo, Submission
- **Due Date**: Date

### Closing Issues
- **Done**: Close with PR link: "Fixed in PR #X"
- **Duplicate**: Close with link to original issue
- **Won't Fix**: Close with explanation

---

## Submission Checklist

### Pre-Submission (Week Before)
- [ ] All tests pass
- [ ] Fresh machine QA run completed
- [ ] Documentation complete
- [ ] Submission bundle created

### Submission Bundle Structure
```
submission/
├── README.md              # How to run submission
├── requirements.txt      # Dependencies
├── config.yaml           # Model config
├── model.pth             # Trained model checkpoint
├── predict.py            # Prediction script
└── notebooks/            # EDA/analysis notebooks (sanitized)
```

### Fresh Machine QA Run
```bash
# On a fresh machine (or Docker):
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary

# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run submission
python submission/predict.py --data-dir /path/to/test/data --output predictions.csv

# Verify predictions.csv format matches competition requirements
```

### Final Review Meeting
**Agenda**:
1. Review submission bundle
2. Verify reproducibility
3. Check documentation
4. Final narrative review
5. Submit!

---

## Narrative & Story

### Competition Core Story
**Problem**: Detect "structured transmission present" from 1-second IQ samples  
**Approach**: Semi-supervised learning (SSL) + calibration  
**Results**: [ROC-AUC, F1, ECE metrics]  
**Impact**: [Fairness considerations, real-world deployment]

### Portfolio Extension Story
**Digital Twin**: Gary Micro-Twin for controlled ML testing  
**AI-RAN Controller**: Bandit/RL for resource allocation  
**Fairness**: Zone-aware evaluation for under-resourced communities

### Resume Bullets
- "Built semi-supervised spectrum occupancy detector achieving [X]% ROC-AUC"
- "Designed zone-aware digital twin for controlled ML testing"
- "Implemented calibration pipeline reducing ECE to [X]"
- "Led team of 3 in SpectrumX competition"

---

## GitHub Project Management

### Create Project Board
```bash
# Create project (one-time)
gh project create --owner gunnchOS3k --title "SpX-DAC 2026 — spectrumx-ai-ran-gary"

# Add custom fields (use GraphQL or GitHub UI)
# Status, Sprint, Owner, Effort, Deliverable, Due Date
```

### Add Issues to Project
```bash
# Add issue to project
gh project item-add <project-number> --owner gunnchOS3k --url <issue-url>

# Set field values (use GraphQL)
gh api graphql -f query='
mutation {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: "<project-id>"
      itemId: "<item-id>"
      fieldId: "<field-id>"
      value: {text: "In Progress"}
    }
  ) {
    projectV2Item {
      id
    }
  }
}
'
```

### Weekly Standup
**Format**:
- What did you complete last week?
- What are you working on this week?
- Any blockers?

**Update Project Board**:
- Move issues to "Done"
- Update "In Progress" issues
- Add new issues to "Backlog"

---

## Common Tasks

### Update Documentation
```bash
# After major changes, update:
# - README.md
# - docs/architecture/*.md
# - docs/PROGRESS_REPORT.md
```

### Release Notes
```markdown
# Release v0.1.0

## Added
- Gary Micro-Twin v1
- Streamlit dashboard improvements
- SSL scaffolding

## Fixed
- QPSK symbol generation (complex constellation)
- AWGN SNR calculation
- Streamlit demo data display

## Changed
- Updated evaluation protocol
```

### Security Audit
```bash
# Check for secrets
grep -r "password\|token\|key\|secret" --exclude-dir=.git --exclude-dir=.venv .

# Verify .gitignore
cat .gitignore

# Check .env not committed
git ls-files | grep -E "\.env$|secrets"
```

---

## Resources

- **GitHub Projects**: https://docs.github.com/en/issues/planning-and-tracking-with-projects
- **GitHub CLI**: https://cli.github.com/manual/
- **Submission Guidelines**: [SpectrumX competition docs]
- **Repo**: https://github.com/gunnchOS3k/spectrumx-ai-ran-gary

---

## Questions?

- **Ananya**: ML model questions
- **Noah**: Evaluation/viz questions
- **GitHub Issues**: Project management questions
