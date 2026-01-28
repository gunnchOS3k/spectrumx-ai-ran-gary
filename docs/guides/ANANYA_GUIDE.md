# Ananya's Guide: Baseline → SSL → Calibration → Ablation Workflow

**Role:** ML Lead  
**Focus:** Baselines → SSL → calibration → optimization/ablation → reproducibility/MLOps

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

# Install PyTorch (if needed for SSL)
pip install torch torchvision  # Or use conda
```

---

## Workflow: Baseline → SSL → Calibration

### Phase 1: Baselines (Sprint 1)

#### 1. Run Existing Baselines
```bash
# Energy Detector
python -c "
from src.edge_ran_gary.detection.baselines import EnergyDetector
import numpy as np

detector = EnergyDetector(threshold=1.0)
iq_data = np.random.normal(0, 1, 1000000) + 1j * np.random.normal(0, 1, 1000000)
pred, conf, power = detector.predict(iq_data)
print(f'Prediction: {pred}, Confidence: {conf:.3f}, Power: {power:.6f}')
"

# Spectral Flatness
python -c "
from src.edge_ran_gary.detection.baselines import SpectralFlatnessDetector
import numpy as np

detector = SpectralFlatnessDetector(threshold=0.5, sample_rate=1e6)
iq_data = np.random.normal(0, 1, 1000000) + 1j * np.random.normal(0, 1, 1000000)
pred, conf, flatness = detector.predict(iq_data, sample_rate=1e6)
print(f'Prediction: {pred}, Confidence: {conf:.3f}, Flatness: {flatness:.6f}')
"
```

#### 2. Implement PSD + Logistic Regression
**File**: `src/edge_ran_gary/detection/baselines.py`

```python
from sklearn.linear_model import LogisticRegression
from scipy import signal

class PSDLogRegDetector:
    def __init__(self):
        self.model = LogisticRegression()
        self.fitted = False
    
    def extract_psd_features(self, iq_data, sample_rate):
        """Extract PSD features for LogReg."""
        freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=1024)
        # Use magnitude of PSD
        psd_mag = np.abs(psd)
        # Extract features: mean, std, max, min, etc.
        features = np.array([
            np.mean(psd_mag),
            np.std(psd_mag),
            np.max(psd_mag),
            np.min(psd_mag),
            np.percentile(psd_mag, 25),
            np.percentile(psd_mag, 75)
        ])
        return features
    
    def fit(self, X_iq, y, sample_rate):
        """Fit on labeled data."""
        X_features = np.array([
            self.extract_psd_features(iq, sample_rate) 
            for iq in X_iq
        ])
        self.model.fit(X_features, y)
        self.fitted = True
    
    def predict(self, iq_data, sample_rate):
        """Predict on single sample."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        features = self.extract_psd_features(iq_data, sample_rate)
        prob = self.model.predict_proba([features])[0, 1]
        pred = 1 if prob > 0.5 else 0
        return pred, prob
```

#### 3. Evaluate Baselines
```python
from edge_ran_gary.sim.evaluation import Evaluator
import numpy as np

# Load labeled data (replace with actual loader)
# X_train, y_train, X_val, y_val = load_labeled_data()

# Evaluate Energy Detector
evaluator = Evaluator()
energy_preds = [energy_detector.predict(x)[0] for x in X_val]
energy_probs = [energy_detector.predict(x)[1] for x in X_val]
energy_results = evaluator.evaluate(y_val, energy_preds, energy_probs)

print("Energy Detector:")
print(f"  ROC-AUC: {energy_results['roc_auc']:.3f}")
print(f"  F1: {energy_results['f1']:.3f}")

# Repeat for Spectral Flatness and PSD+LogReg
```

---

### Phase 2: SSL (Sprint 2)

#### Decision Point: Choose SSL Method

**Options**:
1. **SimCLR** - Contrastive learning, good for 1D signals
2. **BYOL** - Bootstrap Your Own Latent, no negative pairs
3. **wav2vec-style** - Masked prediction, good for temporal signals

**When to choose**:
- **SimCLR**: If you have many unlabeled samples, want simple implementation
- **BYOL**: If you want to avoid negative sampling complexity
- **wav2vec**: If you want to leverage temporal structure

#### Implementation Template

**File**: `src/edge_ran_gary/detection/ssl.py`

```python
import torch
import torch.nn as nn

class EncoderSSL(nn.Module):
    """SSL encoder with interchangeable methods."""
    
    def __init__(self, method='simclr', input_dim=1000000):
        super().__init__()
        self.method = method
        # TODO: Choose SSL method
        # - SimCLR: contrastive loss with augmentations
        # - BYOL: predictor + target network
        # - wav2vec: masked prediction
        
        # Example encoder (1D CNN)
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),
            nn.Flatten(),
            nn.Linear(256 * 128, 512)
        )
    
    def forward(self, x):
        # x: (batch, 2, length) - [I, Q] channels
        return self.encoder(x)
    
    def pretrain(self, unlabeled_data, epochs=100):
        """Pretrain on unlabeled data."""
        # TODO: Implement SSL pretraining
        # - Define augmentations (time shift, freq shift, noise)
        # - Define loss (contrastive/masked prediction)
        # - Train with optimizer
        pass
    
    def finetune(self, labeled_data, labels, epochs=50):
        """Finetune on labeled data."""
        # TODO: Add classifier head
        # - Freeze encoder (optional)
        # - Train classifier on labeled data
        pass
```

#### Augmentations for IQ Data

```python
def augment_iq(iq_data, rng):
    """Apply augmentations for SSL."""
    # Time shift
    shift = rng.integers(-1000, 1000)
    iq_shifted = np.roll(iq_data, shift)
    
    # Frequency shift (CFO)
    cfo_hz = rng.uniform(-500, 500)
    t = np.arange(len(iq_data)) / sample_rate
    iq_cfo = iq_shifted * np.exp(1j * 2 * np.pi * cfo_hz * t)
    
    # Add noise
    noise_power = rng.uniform(0.01, 0.1)
    noise = np.sqrt(noise_power / 2) * (
        rng.normal(0, 1, len(iq_data)) + 1j * rng.normal(0, 1, len(iq_data))
    )
    iq_noisy = iq_cfo + noise
    
    return iq_noisy
```

---

### Phase 3: Calibration (Sprint 3)

#### Decision Point: Choose Calibration Method

**Options**:
1. **Platt Scaling** - Logistic regression on logits
2. **Isotonic Regression** - Non-parametric, more flexible
3. **Temperature Scaling** - Single parameter, simple

**When to choose**:
- **Platt**: If you have limited calibration data (<1000 samples)
- **Isotonic**: If you have enough data and want best calibration
- **Temperature**: If you want simplicity and model is already well-calibrated

#### Implementation

**File**: `src/edge_ran_gary/detection/calibrate.py`

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class Calibrator:
    def __init__(self, method='platt'):
        self.method = method
        self.calibrator = None
    
    def fit(self, y_proba, y_true):
        """Fit calibration on validation set."""
        if self.method == 'platt':
            # Platt scaling: logit(y_proba) -> calibrated proba
            logits = np.log(y_proba / (1 - y_proba + 1e-10))
            self.calibrator = LogisticRegression()
            self.calibrator.fit(logits.reshape(-1, 1), y_true)
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba, y_true)
        # TODO: Add temperature scaling
    
    def calibrate(self, y_proba):
        """Calibrate probabilities."""
        if self.method == 'platt':
            logits = np.log(y_proba / (1 - y_proba + 1e-10))
            return self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_proba)
    
    def compute_ece(self, y_proba_cal, y_true, n_bins=10):
        """Compute Expected Calibration Error."""
        # TODO: Implement ECE
        # - Bin probabilities
        # - Compute accuracy per bin
        # - Weighted average of |acc - conf|
        pass
```

---

### Phase 4: Ablation Studies

#### Test Different Components

```python
# 1. Test different SSL methods
for method in ['simclr', 'byol', 'wav2vec']:
    model = EncoderSSL(method=method)
    # Train and evaluate
    results = evaluate_model(model)
    print(f"{method}: ROC-AUC={results['roc_auc']:.3f}")

# 2. Test different augmentations
augmentations = ['time_shift', 'freq_shift', 'noise', 'all']
for aug in augmentations:
    # Train with specific augmentation
    results = train_with_augmentation(aug)
    print(f"{aug}: ROC-AUC={results['roc_auc']:.3f}")

# 3. Test calibration methods
for cal_method in ['platt', 'isotonic', 'temperature']:
    calibrator = Calibrator(method=cal_method)
    # Fit and evaluate
    ece = evaluate_calibration(calibrator)
    print(f"{cal_method}: ECE={ece:.3f}")
```

---

## Using Gary Micro-Twin for Controlled Testing

### Generate Synthetic Data
```python
from edge_ran_gary.digital_twin.gary_micro_twin import generate_micro_twin_dataset

# Generate dataset
micro_twin, samples, metadata_df = generate_micro_twin_dataset(
    output_dir="data/gary_micro_twin_test",
    n_per_zone=100,
    label_balance=0.5,
    seed=42
)

# Test detector on specific zone
city_hall_data = metadata_df[metadata_df['zone_id'] == 'gary_city_hall']
# Load IQ samples and test detector
```

### Zone-Aware Evaluation
```python
# Compare performance across zones
for zone_id in ['gary_city_hall', 'west_side_leadership_academy', 'gary_public_library']:
    zone_data = metadata_df[metadata_df['zone_id'] == zone_id]
    # Evaluate detector
    results = evaluate_on_zone(zone_data)
    print(f"{zone_id}: ROC-AUC={results['roc_auc']:.3f}")
```

---

## Reproducibility Checklist

### Before Committing
- [ ] Set random seeds (`np.random.seed()`, `torch.manual_seed()`)
- [ ] Save config files (YAML/JSON)
- [ ] Log hyperparameters
- [ ] Save model checkpoints

### Config Management
```python
# Use config files
import yaml

with open('configs/ssl_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set seeds from config
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

# Use hyperparameters from config
model = EncoderSSL(
    method=config['ssl_method'],
    hidden_dim=config['hidden_dim']
)
```

---

## Common Commands

### Train SSL Model
```bash
python scripts/train_ssl.py \
    --config configs/ssl_config.yaml \
    --unlabeled-dir data/unlabeled \
    --labeled-dir data/labeled \
    --output models/ssl_checkpoint.pth
```

### Evaluate Model
```bash
python scripts/evaluate.py \
    --model models/ssl_checkpoint.pth \
    --data-dir data/val \
    --output results/eval_report.json
```

### Calibrate Probabilities
```bash
python scripts/calibrate.py \
    --model models/ssl_checkpoint.pth \
    --cal-method platt \
    --val-dir data/val \
    --output models/calibrated_model.pth
```

---

## Resources

- **SSL Papers**: SimCLR, BYOL, wav2vec 2.0
- **Calibration**: "On Calibration of Modern Neural Networks" (Guo et al.)
- **PyTorch Docs**: https://pytorch.org/docs/
- **Scikit-learn**: https://scikit-learn.org/

---

## Questions?

- **Edmund**: Integration/architecture questions
- **Noah**: Evaluation/metrics questions
- **GitHub Issues**: Technical questions
