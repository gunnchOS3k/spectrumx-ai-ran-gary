# Contributing Guide

This guide explains how to contribute to the SpX-DAC 2026 project.

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your SDS_SECRET_TOKEN
```

### 2. Download Dataset

```bash
# Download dataset (requires SDS_SECRET_TOKEN in .env)
python scripts/download_dataset.py
# Or use the dataset loader directly:
python -c "from src.edge_ran_gary.data_pipeline.spectrumx_loader import SpectrumXDataset; from src.edge_ran_gary.config import SpectrumXDatasetConfig; ds = SpectrumXDataset(SpectrumXDatasetConfig()); ds.download()"
```

### 3. Run Baselines

```bash
# Run baseline detection (when implemented)
python scripts/run_baseline.py

# Or use Streamlit dashboard
streamlit run apps/streamlit_app.py
```

---

## Running Baselines

### Energy Detector

```python
from src.edge_ran_gary.detection.baselines import EnergyDetector
import numpy as np

# Load IQ sample
iq = np.load("path/to/sample.npy")

# Create detector
detector = EnergyDetector(threshold=1.0)

# Detect
prediction, confidence, power = detector.detect(iq)
print(f"Prediction: {prediction}, Confidence: {confidence:.3f}, Power: {power:.3f}")
```

### Spectral Flatness Detector

```python
from src.edge_ran_gary.detection.baselines import SpectralFlatnessDetector

detector = SpectralFlatnessDetector(threshold=0.5, sample_rate=1e6)
prediction, confidence, flatness = detector.detect(iq)
```

### Using Streamlit Dashboard

1. Start dashboard: `streamlit run apps/streamlit_app.py`
2. Upload `.npy` file via sidebar
3. Select baseline model
4. Adjust threshold slider
5. View predictions and visualizations

---

## Data Management

### Where to Put Data

**Never commit data files to git!**

- **Raw dataset:** `competition_dataset/` (gitignored)
- **Processed data:** `data/processed/` (gitignored)
- **Results:** `results/` (gitignored)
- **Plots:** `results/{experiment_name}/plots/` (gitignored)

### Data Formats

- **IQ samples:** `.npy` files, complex64 dtype, shape `(N,)`
- **Labels:** CSV with columns `filename, label`
- **Configs:** YAML or JSON files in `configs/`

### Environment Variables

Store all secrets in `.env` (gitignored):
```bash
SDS_SECRET_TOKEN=your_token_here
```

Never commit:
- `.env` files
- API keys or tokens
- Dataset files (`.npy`)
- Large result files

---

## Reproducing Results

### Deterministic Runs

All random operations use fixed seeds:

```python
import numpy as np
import torch

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
```

### One-Command Reproducibility

```bash
# Reproduce baseline results
python scripts/run_baseline.py --config configs/baseline_config.yaml

# Reproduce SSL training
python scripts/train_ssl.py --config configs/ssl_config.yaml

# Reproduce evaluation
python scripts/evaluate.py --checkpoint results/model.pth --split test
```

### Verification Checklist

- [ ] Fresh clone works without manual edits
- [ ] Same seed produces same results
- [ ] All dependencies in `requirements.txt`
- [ ] Config files document all hyperparameters
- [ ] Results match previous runs (within tolerance)

---

## Adding Experiments

### 1. Create Config File

```yaml
# configs/my_experiment.yaml
experiment_name: my_experiment
seed: 42

model:
  type: ssl
  encoder_arch: transformer
  learning_rate: 1e-3
  batch_size: 256

data:
  train_split: 0.6
  val_split: 0.2
  test_split: 0.2
```

### 2. Implement Experiment Script

```python
# scripts/run_experiment.py
import yaml
from pathlib import Path

config = yaml.safe_load(open("configs/my_experiment.yaml"))

# Load data
# Train model
# Evaluate
# Save results
```

### 3. Save Results

```python
# Save to results/{experiment_name}/
results_dir = Path(f"results/{config['experiment_name']}")
results_dir.mkdir(parents=True, exist_ok=True)

# Save metrics
with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics, f)

# Save plots
fig.savefig(results_dir / "plots" / "roc_curve.png")
```

### 4. Document Results

Add to `docs/experiments_log.md`:
```markdown
## Experiment: my_experiment

**Date:** 2026-01-14
**Config:** `configs/my_experiment.yaml`
**Results:** `results/my_experiment/metrics.json`

**Key Findings:**
- Accuracy: X.XX%
- ROC-AUC: X.XX
- ECE: X.XX
```

---

## Adding Figures

### Where to Put Figures

- **EDA plots:** `reports/eda/`
- **Experiment plots:** `results/{experiment_name}/plots/`
- **Final submission:** `submission/figures/`

### Figure Requirements

- **Format:** PNG (for reports) or PDF (for papers)
- **Resolution:** 300 DPI minimum
- **Labels:** Clear axis labels, legends, titles
- **Captions:** Descriptive captions explaining what the figure shows

### Example

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="Model A")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True)

plt.savefig("results/my_experiment/plots/roc_curve.png", dpi=300, bbox_inches="tight")
```

---

## Code Standards

### Style

- **Python:** Follow PEP 8
- **Docstrings:** Use Google style
- **Type hints:** Use type hints for function signatures
- **Imports:** Group imports (stdlib, third-party, local)

### Structure

```python
"""
Module docstring explaining what this module does.
"""

from typing import Optional
import numpy as np

from src.edge_ran_gary.config import SomeConfig


class MyClass:
    """
    Class docstring.
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg: SomeConfig):
        """Initialize."""
        self.cfg = cfg
    
    def my_method(self, x: np.ndarray) -> np.ndarray:
        """
        Method docstring.
        
        Args:
            x: Input array
            
        Returns:
            Output array
        """
        return x
```

### Testing

- Write unit tests for new functions
- Test edge cases (empty arrays, None values, etc.)
- Run tests before committing: `pytest tests/`

---

## Git Workflow

### Branching

- **Main branch:** Protected, only via PRs
- **Feature branches:** `feature/description`
- **Bug fixes:** `fix/description`
- **Docs:** `docs/description`

### Commits

- Use descriptive commit messages
- Reference issue numbers: `Fix #123`
- Keep commits atomic (one logical change per commit)

### Pull Requests

1. Create feature branch
2. Make changes
3. Test locally
4. Push branch
5. Open PR with description
6. Request review from team member
7. Address review comments
8. Merge after approval

---

## Questions?

- **Technical questions:** Open a GitHub Issue
- **Process questions:** Check `docs/PROJECT_PLAN.md`
- **Evaluation questions:** Check `docs/EVAL_PROTOCOL.md`

---

## Security Reminders

- ✅ Never commit secrets (tokens, API keys)
- ✅ Use `.env` for all secrets
- ✅ Check `.gitignore` before committing
- ✅ Review diffs before pushing
- ✅ Rotate tokens if accidentally committed

See `SECURITY.md` for more details.
