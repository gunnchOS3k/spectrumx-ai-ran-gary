# Gary Spectrum Digital Twin

## Overview

The Gary Spectrum Digital Twin is a lightweight prototype for generating synthetic 1-second IQ windows with metadata. It enables:

- **ML Pipeline Testing:** Generate reproducible synthetic data for training SSL/anomaly models
- **Robustness Evaluation:** Test detector performance across different zones (equity-focused)
- **Domain Shift Analysis:** Compare real vs. synthetic data performance
- **AI-RAN Integration:** Connect detector outputs to digital twin for controller training

## Architecture

```
Detector (Competition Core)
    ↓
Occupancy Probability
    ↓
Digital Twin (Zone Model + Signal Generator)
    ↓
Synthetic IQ Windows + Metadata
    ↓
AI-RAN Controller (Bandit/RL)
```

## Quick Start

### Generate Synthetic Dataset

```bash
# Generate 2000 samples (default)
python -m edge_ran_gary.digital_twin.dataset_builder \
    --out data/synth_gary_twin \
    --n 2000 \
    --seed 123 \
    --config configs/digital_twin_gary.yaml

# Output:
# - data/synth_gary_twin/sample_000000.npy
# - data/synth_gary_twin/sample_000001.npy
# - ...
# - data/synth_gary_twin/metadata.csv
```

### Use in Python

```python
from edge_ran_gary.digital_twin import generate_iq_window, build_synth_dataset

# Generate single window
iq_data, metadata = generate_iq_window(
    seed=123,
    label=1,  # 0=noise, 1=signal
    config_path="configs/digital_twin_gary.yaml"
)

# Build full dataset
build_synth_dataset(
    output_dir="data/synth_gary_twin",
    n_samples=2000,
    seed=123
)
```

### Visualize in Streamlit

1. Generate dataset (see above)
2. Open Streamlit app: `streamlit run streamlit_app.py`
3. Upload any `.npy` file from `data/synth_gary_twin/`
4. View visualizations and run baseline detectors

## Zone Model

Zones represent abstract neighborhoods (not real PII) with different characteristics:

- **Equity Weights:** Higher weight = more emphasis in sampling (for fairness evaluation)
- **Occupancy Priors:** Probability of signal presence
- **Noise Floor:** Background noise level (dBm)
- **SNR Range:** Signal-to-noise ratio range (dB)
- **CFO Range:** Carrier frequency offset range (Hz)
- **Multipath Taps:** Number of multipath channel taps

### Configuration

Edit `configs/digital_twin_gary.yaml` to customize zones:

```yaml
zones:
  zone_01:
    weight: 1.5  # Higher = under-resourced zone
    occupancy_prior: 0.3
    noise_floor_prior: -95.0
    snr_range: [0, 15]
    cfo_range: [-2000, 2000]
    multipath_taps_range: [3, 8]
```

## Signal Generation

### Noise-Only (Label=0)

- AWGN with zone-specific noise floor
- Reproducible via seed

### Structured Signal (Label=1)

- **QPSK-like:** RRC pulse-shaped QPSK with impairments
- **OFDM-like:** Multi-carrier OFDM with cyclic prefix
- **Impairments:**
  - Carrier frequency offset (CFO)
  - Multipath channel (FIR filter)
  - AWGN (target SNR)

## Integration Points for Ananya

### 1. SSL Pretraining

```python
# TODO: Plug SSL pretraining here
# - Use synthetic data for self-supervised learning
# - Compare with real data for domain shift evaluation
# Location: src/edge_ran_gary/detection/ssl.py
```

**Guardrails:**
- Use separate seeds for train/val/test
- Document domain shift metrics (real vs. synth)
- Calibrate on real data, not synthetic

### 2. Anomaly Detection

```python
# TODO: Compare real vs. synthetic for anomaly detection
# - Train autoencoder on synthetic data
# - Evaluate on real data
# Location: src/edge_ran_gary/detection/anomaly.py
```

**Guardrails:**
- Validate anomaly scores on real data
- Check for distribution shift
- Use synthetic data for augmentation only

### 3. Calibration

```python
# TODO: Calibrate thresholds using synthetic data
# - Generate calibration set with known labels
# - Fit Platt/Isotonic on synthetic
# - Validate on real data
# Location: src/edge_ran_gary/detection/calibrate.py
```

**Guardrails:**
- **Never calibrate only on synthetic data**
- Always validate on real holdout set
- Report ECE on both synthetic and real

### 4. Domain Shift Evaluation

```python
# Compare detector performance:
# - Train on synthetic, test on real
# - Train on real, test on synthetic
# - Measure distribution shift (KL divergence, etc.)
```

## Metadata Format

`metadata.csv` contains:

- `file`: Filename (e.g., `sample_000000.npy`)
- `label`: 0 (noise) or 1 (signal)
- `zone_id`: Zone identifier
- `snr_db`: Signal-to-noise ratio (NaN for noise-only)
- `cfo_hz`: Carrier frequency offset (NaN for noise-only)
- `num_taps`: Multipath taps (NaN for noise-only)
- `seed`: Random seed for reproducibility

## Reproducibility

All generation is deterministic:

- **Seeds:** Every sample uses a unique seed (base_seed + sample_idx)
- **Configs:** Zone parameters in `configs/digital_twin_gary.yaml`
- **Outputs:** `.npy` files in `data/` (gitignored)

## Use Cases

### 1. ML Pipeline Testing

Generate synthetic data to:
- Test SSL methods without real data
- Validate preprocessing pipelines
- Debug model architectures

### 2. Robustness Evaluation

Test detector across zones:
- Under-resourced zones (higher noise, lower SNR)
- Well-resourced zones (lower noise, higher SNR)
- Measure fairness metrics

### 3. Domain Shift Analysis

Compare real vs. synthetic:
- Train on synthetic, test on real
- Measure distribution shift
- Identify failure modes

### 4. AI-RAN Controller

Connect to controller:
- Detector outputs → occupancy probability
- Digital twin generates synthetic scenarios
- Controller learns policies (bandit/RL)

## Next Steps

1. **Ananya:** Use synthetic data for SSL pretraining
2. **Noah:** Integrate with evaluation harness
3. **Edmund:** Connect to AI-RAN controller for Phase 2

## References

- Competition core: `src/edge_ran_gary/detection/`
- Digital twin: `src/edge_ran_gary/digital_twin/`
- Config: `configs/digital_twin_gary.yaml`
- Streamlit app: `streamlit_app.py`
