# Spectrum Occupancy Detection -- Technical Report
## SpectrumX AI-RAN Competition (UMass / New York University Team)

**Team:** Ananya Gunn  
**Affiliation:** New York University  
**Language / Framework:** Python 3.13 · NumPy · SciPy · scikit-learn  
**Best Submission:** v9 (GBM + prior correction) -- Leaderboard accuracy **0.85**

---

## 2.4 Dataset Usage and Preprocessing

### 2.4.1 Dataset Structure

The competition dataset is organised as follows:

```
competition_dataset/
  files/
    kdoke@hamilton.edu/
      trainingData/          200 unlabeled .npy files  (IQ captures)
      VLA_brutal/
        groundtruth.csv      Labels for 240 files
        <labeled .npy files>  40 labeled files in this folder
```

The groundtruth CSV contains two columns: `filename` and `label` (0 = unoccupied, 1 = occupied).  
Across the 240 labeled files the class distribution is **imbalanced**: 160 occupied (67%) and 80 unoccupied (33%), a 2:1 positive-to-negative ratio. The hidden test set is understood to be **balanced (50/50)**, which has important consequences for threshold calibration (see §2.5.4).

The 200 files in `trainingData/` carry no labels and were not used for supervised training. Semi-supervised approaches (pseudo-labeling) were evaluated but rejected because the unlabeled files showed no consistent structure that could be reliably annotated without ground truth.

### 2.4.2 IQ Sample Format

Each `.npy` file contains a single broadband IQ capture. Three distinct binary formats were encountered in the dataset:

| Format | dtype | Shape | Interpretation |
|--------|-------|-------|----------------|
| A | complex64 | (N,) | Direct complex samples |
| B | float32 / float64 | (N, 2) | Columns = [I, Q] |
| C | int16 | (2N,) interleaved | Even indices = I, odd = Q |

A unified loader (`_load_iq_auto`) detects the format at runtime and converts each file to a 1-D complex64 array.

### 2.4.3 Sample Size and Memory Management

Full files contain up to ~10 million IQ samples. To balance feature quality against runtime, up to **5 million samples** are loaded per file (via NumPy memory-mapping, `mmap_mode="r"`). This avoids loading entire files into RAM while still capturing enough spectral content for reliable PSD estimation.

For the time-windowed consistency feature introduced in v14, the first 5 M samples are divided into **8 non-overlapping windows** of up to 524 288 samples each.

### 2.4.4 Preprocessing Steps

1. **Format normalization** -- convert to complex64 as described above.  
2. **Length capping** -- truncate to `MAX_IQ = 5 000 000` samples.  
3. **Feature extraction** -- 33 engineered features derived from the PSD and time-domain statistics (detailed in §2.5.2).  
4. **Z-score standardization** -- a `StandardScaler` (mean and standard deviation computed on the 240 labeled training files) normalizes each feature to zero mean and unit variance. The scaler parameters are hardcoded into the submitted `main.py` so that inference on the server requires no retraining.  
5. **NaN/Inf sanitization** -- `np.nan_to_num` replaces any degenerate feature values (arising from edge cases such as all-zero IQ segments) with 0.0.

---

## 2.5 Algorithm Design

### 2.5.1 Overview and Motivation

The task is **binary spectrum occupancy detection**: given a raw broadband IQ capture, determine whether the channel is occupied (1) or idle (0). The challenge is that IQ samples from different signal types (e.g., narrowband tones, wideband communications waveforms, partial-band transmissions) can vary enormously in amplitude, bandwidth, and waveform structure, while thermal noise can occasionally produce high-power spectral artefacts that mimic weaker signals.

Our approach evolved through six distinct design generations, each motivated by analysis of failure modes on the competition leaderboard. Figure 6 (see `docs/report_figures/fig6_algorithm_evolution.png`) illustrates the full progression.

---

### 2.5.2 Baseline Methods (v3)

Three classical signal-processing baselines were implemented and evaluated first:

#### Energy Detector
The simplest occupancy test: compute mean received power and compare to a threshold.

```
P_mean = mean(|IQ|^2)
decision = 1  if  P_mean > theta_E  else  0
```

**Threshold selection:** A grid sweep over the labeled data maximising F1-score yielded `theta_E = 0.00127`.  
**Intuition:** Occupied channels carry a transmitted signal whose amplitude adds on top of thermal noise; idle channels carry only thermal noise. The mean power therefore shifts upward when a signal is present.  
**Limitation:** Highly sensitive to path loss and channel gain variations. A weak distant signal may have lower mean power than a noisy nearby antenna. Leaderboard accuracy: **0.50** (random-chance performance, indicating the threshold generalised poorly to the test set).

#### Spectral Flatness Detector
The power spectral density (PSD) of white noise is approximately flat, while structured signals concentrate energy in specific frequency bands. Spectral flatness (Wiener entropy) quantifies this:

```
SF = geometric_mean(PSD) / arithmetic_mean(PSD)   in [0, 1]
decision = 1  if  SF < theta_SF  else  0
```

Welch's method with `nperseg=1024` was used to estimate the PSD. The threshold `theta_SF = 0.9953` was selected by F1 optimisation on the labeled data.  
**Intuition:** Flat noise → SF close to 1; structured signal → SF close to 0.  
**Limitation:** Narrowband signals embedded in broadband noise may not significantly lower the global flatness if the signal occupies only a small fraction of the band. Leaderboard accuracy: **0.50** (same as energy detector -- shared failure mode).

#### PSD + Logistic Regression (6 features)
A logistic regression classifier trained on 6 summary statistics of the Welch PSD:

```
features = [mean(PSD), std(PSD), max(PSD), min(PSD), p25(PSD), p75(PSD)]
```

Weights were learned on the labeled data and hardcoded into the submission. Threshold was swept on the balanced subset of labels. Leaderboard accuracy: **0.50** with 6 features -- the linear model underfitted the signal diversity.

---

### 2.5.3 Feature Engineering Progression (v4 → v7)

The core insight from v3 failures was that simple summary statistics of the PSD do not capture signal structure well enough. We systematically expanded the feature set while keeping logistic regression as the classifier.

#### v4 -- 10 Features (+IQ-domain statistics)

Added four features capturing time-domain IQ signal structure:

| New feature | Formula | Intuition |
|---|---|---|
| `spectral_entropy` | -Σ p_k log2(p_k) where p_k = PSD_k / Σ PSD | Low for structured, high for flat noise |
| `peak_to_mean` | max(PSD) / mean(PSD) | Spiky signals score high |
| `band_energy_std` | std of energy in 4 equal PSD quarters | Signals unevenly distributed across bands |
| `autocorr` | |mean(IQ[1:] · IQ*[:-1])| | Coherent signals have high lag-1 autocorrelation |

**Result:** Leaderboard accuracy improved from 0.50 → **0.68**. CV accuracy: 0.717.

#### v5 -- 21 Features (+amplitude statistics)

Added 11 more features capturing amplitude distribution and higher-order moments:

| Group | Features added |
|---|---|
| Spectral kurtosis | `spectral_kurt_mean`, `spectral_kurt_max` -- kurtosis of PSD across segments |
| Band energy ratios | `band_e0`–`band_e3` -- fraction of energy in each spectral quarter |
| Magnitude statistics | `mag_kurtosis`, `mag_skewness` -- higher-order moments of |IQ| |
| Power | `mean_power`, `power_var_ratio` -- temporal power variation |
| IQ balance | `i_std`, `q_std`, `iq_corr` -- I/Q imbalance indicators |

**Result:** Leaderboard accuracy: **0.79**. Balanced CV: 0.794.  
**Key insight:** Spectral kurtosis (`spectral_kurt_max`) proved particularly effective at detecting signals with intermittent high-energy spikes -- a signature of burst-mode transmissions.

#### v6 -- 30 Features (+robust PSD percentiles, 5 M IQ)

Completed the feature set with:
- Full PSD percentile suite: `psd_p10`, `psd_p90`, `psd_iqr`
- `crest_factor` (peak-to-RMS amplitude ratio)
- `mag_std` (amplitude envelope variability)

Also increased the IQ sample budget to **5 million samples**, providing more spectral resolution.

**Result:** Leaderboard accuracy: **0.82**. Balanced CV: 0.919. The gap between CV (0.92) and leaderboard (0.82) first became apparent here -- indicating overfitting to the training label distribution.

#### v7 -- 30 Features, 10 M IQ

Doubled the IQ budget to 10 million samples to improve PSD estimation reliability.

**Result:** Leaderboard accuracy: **0.82** (unchanged). Balanced CV: 0.938.  
**Conclusion:** More data did not help generalization; the model was already extracting as much information as logistic regression could use. The bottleneck was model capacity, not feature estimation quality.

---

### 2.5.4 Gradient Boosting Machine (v8)

**Motivation:** Logistic regression assumes linear separability in the feature space. The 0.92 CV accuracy but 0.82 leaderboard accuracy strongly suggested that the true decision boundary is non-linear and that logistic regression was both underfitting the training data and failing to generalize to the test distribution.

**Model:** `sklearn.ensemble.GradientBoostingClassifier` with:
- `n_estimators = 200` (200 additive trees)  
- `max_depth = 3` (shallow trees for regularization)  
- `learning_rate = 0.05` (slow learning to prevent overfitting)  
- `subsample = 0.8` (stochastic gradient boosting)  
- `min_samples_leaf = 5` (minimum leaf population)

**Feature set:** Same 30 features as v6/v7, using 5 M IQ samples.

**Training:** All 240 labeled files were used for training. 5-fold stratified cross-validation was used to estimate generalization performance.

**Balanced CV accuracy: 0.931.** Leaderboard accuracy: **0.85** (v8 → submitted as v9 after prior correction, see below).

**Feature Importance:** After training, the GBM's impurity-based feature importances revealed that `psd_max` (raw maximum spectral power) carried **77%** of the total decision weight. This heavy reliance on a single feature became the subject of later investigation (see §2.5.6).

See Figure 5 (`fig5_feature_importance.png`) for the full importance ranking.

---

### 2.5.5 Prior Correction (v9)

**Problem:** The training set has a 2:1 positive-to-negative class ratio (160 occupied, 80 idle). The GBM's intercept (init log-odds) was set to log(2/3 ÷ 1/3) = log(2) ≈ 0.693, biasing all predictions toward "occupied." The hidden test set is believed to be balanced (50/50), so this bias causes systematic over-prediction of occupancy → excess false positives.

**Fix:** Set `INIT_LOG_ODDS = 0.0` in the exported model. Mathematically, this removes the 2:1 prior and replaces it with a 1:1 prior:

```
# At inference time:
score = INIT_LOG_ODDS + sum(learning_rate * leaf_value for each tree)
prob  = sigmoid(score)
```

Setting `INIT_LOG_ODDS = 0.0` instead of `log(2)` subtracts `log(2)` from the log-odds, shifting all probabilities toward the neutral 0.5 boundary. The decision threshold remains 0.5.

**Result:** This single change with no retraining moved the operating point from FP=0.08 to FP=0.04, improving accuracy from the raw v8 to **0.85** on the leaderboard:

| | TP | FP | FN | TN | Acc |
|---|---|---|---|---|---|
| v8 (no correction) | 0.41 | 0.08 | 0.09 | 0.42 | 0.83 |
| v9 (corrected) | 0.39 | 0.04 | 0.11 | 0.46 | **0.85** |

---

### 2.5.6 Threshold and Ensemble Experiments (v10–v12)

After establishing v9 as the best baseline, three additional experiments were conducted to push accuracy higher by reducing the false positive rate.

#### v10 -- Conservative Threshold (0.6)
Raising the decision threshold from 0.5 to 0.6 reduces FP at the cost of more FN:

| Threshold | TP | FP | FN | TN | Acc |
|---|---|---|---|---|---|
| 0.50 (v9) | 0.39 | 0.04 | 0.11 | 0.46 | 0.85 |
| 0.60 (v10) | 0.38 | 0.03 | 0.12 | 0.47 | 0.85 |

**Observation:** 2 of the 8 FP cases were resolved, but 6 FP noise files had probabilities > 0.60 -- they genuinely resemble occupied channels in the feature space.

#### v11 -- Sample Weighting (noise weight = 3)
Trained a new GBM with `class_weight = {0: 3, 1: 1}` to penalize false positives more during training.

**Result:** The model became less sensitive overall; FP increased to 0.05 while TP held at 0.39. Accuracy dropped to **0.84**. Weighting the noise class too heavily caused the model to mis-classify ambiguous signal files.

#### v12 -- Ensemble (30% LogReg + 70% GBM)
Blended the logistic regression model (v6) with the GBM (v9) by averaging their probabilities:

```
prob_ensemble = 0.30 * prob_logreg + 0.70 * prob_gbm
```

With a threshold of 0.754, this ensemble achieved FP = 0.00 on the balanced CV, but at the cost of missing many borderline signals. Leaderboard result: TP = 0.33, FP = 0.00, accuracy = **0.83** -- worse than v9 because the ensemble over-rejected true signals.

**Key insight from v10–v12:** The 6 persistent FP noise files have **high probability in the v9 GBM's feature space**. These files are not borderline -- they look like real signals according to `psd_max` and related features. Simply adjusting thresholds or ensembles cannot fix a model that fundamentally misrepresents these samples.

See Figure 7 (`fig7_fp_fn_tradeoff.png`) for the FP/FN trade-off curve illustrating why no threshold perfectly separates the distributions.

---

### 2.5.7 Leaderboard Context

| Rank | Team | Time (s) | Acc | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| #1 | sub_test3 (hwy/uf) | 2060 | **0.87** | 0.37 | 0.00 | 0.13 | 0.50 |
| #2 | sub_test2 (hwy/uf) | 2213 | 0.86 | 0.36 | 0.00 | 0.14 | 0.50 |
| #3 | test10 (Zhao/UCLA) | 3694 | 0.86 | 0.39 | 0.03 | 0.11 | 0.47 |
| **Ours** | **v9 (Gunn/NYU)** | **798** | **0.85** | **0.39** | **0.04** | **0.11** | **0.46** |

**Analysis:** The top two submissions achieve FP = 0.00 by trading away 2 true positives (0.39 → 0.37 TP). They spend 2.5× more execution time -- most likely running a more expensive second-stage confirmation on any file that passes initial screening. Our v9 detects the **same signals** as test10 (identical TP = 0.39 and FN = 0.11) but has one extra false positive. The gap to #1 reduces to: -2 TP gained, +8 FP removed, net +6 correct predictions.

---

## 2.6 Algorithm Pseudocode

### Main Detection Pipeline

```
FUNCTION evaluate(filename):
    # 1. Load IQ
    iq = load_npy(filename, max_samples=5_000_000)
    iq = convert_to_complex64(iq)               // handle A/B/C formats

    # 2. Extract 33 features (30 legacy + 3 new)
    features = extract_features(iq, sample_rate=1e6)
    features = sanitize(features)               // replace NaN/Inf with 0

    # 3. Z-score normalize (hardcoded scaler)
    x_scaled = (features - SCALER_MEAN) / (SCALER_STD + 1e-30)

    # 4. GBM inference
    score = INIT_LOG_ODDS                        // 0.0 (prior-corrected)
    FOR i = 0 TO N_TREES - 1:
        node = 0
        WHILE TREE_LEFT[TREE_OFFSETS[i] + node] != -1:
            feat_idx = TREE_FEATURE[TREE_OFFSETS[i] + node]
            IF x_scaled[feat_idx] <= TREE_THRESHOLD[TREE_OFFSETS[i] + node]:
                node = TREE_LEFT[TREE_OFFSETS[i] + node]
            ELSE:
                node = TREE_RIGHT[TREE_OFFSETS[i] + node]
        score += LEARNING_RATE * TREE_VALUE[TREE_OFFSETS[i] + node]

    prob = sigmoid(score)

    # 5. Threshold decision
    RETURN 1 IF prob >= PROB_THRESHOLD ELSE 0
```

### Feature Extraction Pseudocode

```
FUNCTION extract_features(iq, sample_rate):
    N = length(iq)
    mag   = abs(iq)
    power = mag^2

    # --- PSD via Welch (nperseg = min(8192, N)) ---
    psd = welch(iq, fs=sample_rate, return_onesided=False)
    psd = abs(psd)

    # Basic statistics
    psd_mean = mean(psd)
    psd_std  = std(psd)
    psd_max  = max(psd)
    psd_min  = min(psd)
    p10, p25, p75, p90 = percentile(psd, [10, 25, 75, 90])
    psd_iqr  = p75 - p25

    # Spectral shape
    flatness         = geometric_mean(psd) / arithmetic_mean(psd)
    spectral_entropy = -sum(p_k * log2(p_k))  where p_k = psd_k / sum(psd)
    peak_to_mean     = psd_max / psd_mean

    # Spectral kurtosis (across 1024-sample segments)
    seg_psds  = [welch(iq[i:i+1024]) for i in 0..N//1024]
    sk_per_bin = kurtosis(seg_psds, axis=0)
    spectral_kurt_mean = mean(sk_per_bin)
    spectral_kurt_max  = max(sk_per_bin)

    # Band energy ratios (4 equal frequency quarters)
    band_energies = [sum(psd[quarter*k : quarter*(k+1)]) / sum(psd) for k in 0..3]
    band_energy_std = std(band_energies)

    # Time-domain
    autocorr       = |mean(iq[1:] * conj(iq[:-1]))|
    mean_power     = mean(power)
    mag_kurtosis   = kurtosis(mag)
    mag_skewness   = skew(mag)
    crest_factor   = max(mag) / sqrt(mean_power)
    i_std          = std(real(iq))
    q_std          = std(imag(iq))
    iq_corr        = |corrcoef(real(iq), imag(iq))[0,1]|
    power_var_ratio= var(power) / mean_power^2

    # --- NEW anti-FP features ---
    # 1. Narrowband concentration
    psd_top10_frac = sum(top-10 PSD bins) / sum(psd)

    # 2. Peak vs noise floor
    psd_max_to_p10 = psd_max / p10

    # 3. Temporal consistency of spectral peak
    FOR w in 8 windows of length 524288:
        peak_bins[w] = argmax(|FFT(iq[window_w])|^2)
    peak_bin_cv = std(peak_bins) / (mean(peak_bins) + 1)

    RETURN [psd_mean, psd_std, psd_max, psd_min, p10, p25, p75, p90, psd_iqr,
            flatness, spectral_entropy, peak_to_mean,
            spectral_kurt_mean, spectral_kurt_max,
            band_e0, band_e1, band_e2, band_e3, band_energy_std,
            autocorr, mean_power, std_power, mag_std,
            mag_kurtosis, mag_skewness, crest_factor,
            i_std, q_std, iq_corr, power_var_ratio,
            psd_top10_frac, psd_max_to_p10, peak_bin_cv]
```

### Threshold Tuning (Training Phase)

```
FUNCTION tune_threshold(probs_corrected, labels_balanced):
    best_acc = 0.0
    best_thr = 0.5
    FOR thr in linspace(0.01, 0.99, 2000):
        predictions = (probs_corrected >= thr)
        acc = accuracy(predictions, labels_balanced)
        IF acc > best_acc:
            best_acc = acc
            best_thr = thr
    RETURN best_thr, best_acc
```

---

## 2.7 Implementation Details

### 2.7.1 Software Structure

```
spectrumx-ai-ran-gary/
  src/edge_ran_gary/detection/
    baselines.py          EnergyDetector, SpectralFlatnessDetector, PSDLogRegDetector
  scripts/
    run_phase1_baselines.py   threshold sweep + metric logging
    train_v8_model.py         GBM training, tree export
    train_v14_model.py        v14 training (33 features)
    generate_v14_submission.py  auto-generates main.py from JSON model file
    generate_report_plots.py    all figures in this report
  results/
    baseline_thresholds.json  energy + flatness thresholds
    v4_model.json .. v14_model.json  per-version model weights + CV metrics
  submissions/
    leaderboard_v9/main.py    self-contained inference (30 features, GBM)
    leaderboard_v14/main.py   self-contained inference (33 features, GBM)
    leaderboard_v9.zip        submitted zip
    leaderboard_v14.zip       submitted zip
  docs/report_figures/        PNG figures for this report
```

### 2.7.2 Libraries Used

| Library | Version | Purpose |
|---|---|---|
| NumPy | ≥1.26 | Array operations, FFT, memory-mapped file I/O |
| SciPy | ≥1.12 | `signal.welch` (PSD), `stats.kurtosis`, `stats.skew` |
| scikit-learn | ≥1.4 | `GradientBoostingClassifier`, `StandardScaler`, `StratifiedKFold` |
| pandas | ≥2.0 | CSV loading, DataFrame operations |
| matplotlib | ≥3.8 | Report figures |

### 2.7.3 Self-Contained Submission Design

The competition requires a single `main.py` file with no internet access and minimal dependencies (`numpy`, `scipy`). All learned parameters -- scaler mean/std, GBM tree structure (split features, thresholds, leaf values), and tree offsets -- are **hardcoded as Python lists** in the submission file. The GBM inference is a pure-NumPy tree traversal loop (no scikit-learn at inference time).

This was automated via `generate_v14_submission.py`, which reads `results/v14_model.json` (produced by training) and writes a complete `submissions/leaderboard_v14/main.py`.

### 2.7.4 Key Design Decisions

- **Memory mapping (`mmap_mode="r"`):** Avoids loading 10 M-sample files fully into RAM. Only the portion accessed (first 5 M samples) is paged in.
- **`return_onesided=False` in Welch:** Complex IQ signals have a two-sided spectrum; the one-sided PSD would discard the negative-frequency content which may contain signal energy.
- **Hardcoded scaler:** Normalization statistics are computed on training data only, preventing data leakage and enabling deployment without a fit step.

---

## 2.8 Computational Complexity and Efficiency

### 2.8.1 Time Complexity per File

| Step | Operation | Complexity | Typical Runtime |
|---|---|---|---|
| IQ Load | Memory-map + copy 5 M samples | O(N) | ~0.05 s |
| Welch PSD | FFT of N/nperseg segments, nperseg=8192 | O(N log nperseg) | ~0.30 s |
| Spectral kurtosis | Vectorized FFT on N/1024 segments × 1024 | O(N log 1024) | ~0.20 s |
| Band/amplitude stats | Element-wise operations on N samples | O(N) | ~0.10 s |
| Peak-bin CV (8 FFTs × 524 288) | 8 × O(M log M), M=524 288 | O(8M log M) | ~1.10 s |
| GBM inference (200 trees × depth 3) | O(200 × 3) leaf lookups | O(1) | <0.01 s |
| **Total per file** | | | **~1.87 s** |

### 2.8.2 Leaderboard Runtime

On the competition evaluation server (approximately 2–3× slower than our development machine):

| Version | Per-file (local) | Est. per-file (server) | 240 files (est.) | Actual LB time |
|---|---|---|---|---|
| v3 (baseline) | ~0.4 s | ~1.0 s | ~240 s | 566 s |
| v6 (LogReg 30f, 5M) | ~0.8 s | ~2.0 s | ~480 s | 882 s |
| v7 (LogReg 30f, 10M) | ~2.6 s | ~4.7 s | ~1130 s | 1135 s |
| v9 (GBM 30f, 5M) | ~1.4 s | ~3.3 s | ~795 s | 798 s |
| v14 (GBM 33f, 5M) | ~1.9 s | ~4.0 s | ~960 s (est.) | -- |

The peak-bin CV feature adds ~1.1 s per file (8 large FFTs) but substantially improves FP rejection.

### 2.8.3 Memory Requirements

| Resource | Requirement |
|---|---|
| IQ buffer (5 M complex64) | 40 MB per file |
| PSD array (8192 bins) | < 0.1 MB |
| GBM tree arrays (200 trees × ~12 nodes) | ~0.5 MB |
| Scaler arrays (33 values × 2) | negligible |
| **Peak RAM** | **~60 MB** |

### 2.8.4 Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i5 (development laptop) |
| RAM | 16 GB DDR4 |
| Storage | NVMe SSD (important for mmap performance) |
| OS | Windows 10 / 11 |
| Python | 3.13 (Anaconda distribution) |
| GPU | Not used -- all computation is CPU-bound |

---

## 2.9 Experimental Results

### 2.9.1 Cross-Validation Table

All CV results use **5-fold stratified cross-validation** on the 240 labeled files. "Balanced CV" refers to evaluation on an equal-size subsample of positive and negative examples (min-count class size), with prior correction applied to account for the 2:1 training imbalance. The threshold is swept to maximise balanced accuracy.

| Version | Model | Features | IQ samples | Balanced CV Acc | Opt. Threshold | LB Acc | LB TP | LB FP | LB FN | LB TN | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | Energy + Flatness | 1 each | 5 M | -- | tuned | 0.50 | 0.50 | 0.50 | 0.00 | 0.00 | 566 |
| v3 | PSD+LogReg | 6 | all | -- | 0.5 | 0.50 | 0.50 | 0.50 | 0.00 | 0.00 | 566 |
| v4 | LogReg | 10 | 5 M | 0.717 | 0.291 | 0.68 | 0.43 | 0.25 | 0.07 | 0.25 | 542 |
| v5 | LogReg | 21 | 5 M | 0.794 | 0.487 | 0.79 | 0.31 | 0.01 | 0.19 | 0.49 | 555 |
| v6 | LogReg | 30 | 5 M | 0.919 | 0.299 | 0.82 | 0.35 | 0.03 | 0.15 | 0.47 | 882 |
| v7 | LogReg | 30 | 10 M | 0.938 | 0.311 | 0.82 | 0.35 | 0.02 | 0.15 | 0.48 | 1135 |
| v8/v9 | GBM | 30 | 5 M | 0.931 | 0.500 | 0.85 | 0.39 | 0.04 | 0.11 | 0.46 | 798 |
| v10 | GBM (thr=0.6) | 30 | 5 M | 0.931 | 0.600 | 0.85 | 0.38 | 0.03 | 0.12 | 0.47 | 1514 |
| v11 | GBM (wt=3) | 30 | 5 M | 0.931 | 0.873 | 0.84 | 0.39 | 0.05 | 0.11 | 0.45 | 895 |
| v12 | Ensemble 30%LR+70%GBM | 30 | 5 M | 0.938 | 0.754 | 0.83 | 0.33 | 0.00 | 0.17 | 0.50 | 1529 |
| **v14** | **GBM** | **33** | **5 M** | **0.931** | **0.986** | **(pending)** | -- | -- | -- | -- | -- |

Note: v9 CV accuracy (0.931) equals v8 because only the INIT_LOG_ODDS constant was changed (no retraining). The threshold 0.986 for v14 reflects the high-certainty operating point after adding the new anti-FP features; the CV estimate of FP=0.01 is likely optimistic.

### 2.9.2 Confusion Matrix Detail (Leaderboard)

Assuming 200 hidden test files (100 occupied, 100 idle):

#### v9 -- Best Performing Submission

```
                Predicted: Idle   Predicted: Occupied
Actual: Idle        92 (TN)            8 (FP)
Actual: Occupied    22 (FN)           78 (TP)
```
Accuracy = (92 + 78) / 200 = **0.85**

#### v12 -- Zero FP at Cost of TP

```
                Predicted: Idle   Predicted: Occupied
Actual: Idle       100 (TN)            0 (FP)
Actual: Occupied    34 (FN)           66 (TP)
```
Accuracy = (100 + 66) / 200 = **0.83** ← over-conservative

#### Leaderboard #1 (sub_test3) -- Target

```
                Predicted: Idle   Predicted: Occupied
Actual: Idle       100 (TN)            0 (FP)
Actual: Occupied    26 (FN)           74 (TP)
```
Accuracy = (100 + 74) / 200 = **0.87**

### 2.9.3 Performance Metric Summary

| Metric | v3 (baseline) | v9 (best) | v12 (no-FP) | #1 target |
|---|---|---|---|---|
| Accuracy | 0.50 | **0.85** | 0.83 | 0.87 |
| True Positive Rate (Recall) | 1.00 | 0.78 | 0.66 | 0.74 |
| True Negative Rate (Specificity) | 0.00 | 0.92 | **1.00** | **1.00** |
| False Positive Rate | **1.00** | 0.08 | **0.00** | **0.00** |
| False Negative Rate | 0.00 | 0.22 | 0.34 | 0.26 |
| F1 Score | 0.667 | 0.838 | 0.795 | 0.850 |
| Miss Detection Rate (MDR = FN rate) | 0.00 | 0.22 | 0.34 | 0.26 |
| False Alarm Rate (FAR = FP rate) | 1.00 | 0.08 | 0.00 | 0.00 |

### 2.9.4 Figure References

| Figure | File | Description |
|---|---|---|
| Fig. 1 | `fig1_accuracy_progression.png` | Leaderboard accuracy across all versions |
| Fig. 2 | `fig2_tp_fp_breakdown.png` | TP/FP/FN/TN breakdown by version |
| Fig. 3 | `fig3_runtime_vs_accuracy.png` | Runtime vs accuracy scatter (all teams) |
| Fig. 4 | `fig4_cv_vs_lb.png` | CV accuracy vs leaderboard accuracy (optimism gap) |
| Fig. 5 | `fig5_feature_importance.png` | v9 GBM feature importances |
| Fig. 6 | `fig6_algorithm_evolution.png` | Version-by-version algorithm evolution |
| Fig. 7 | `fig7_fp_fn_tradeoff.png` | FP/FN trade-off as threshold varies |
| Fig. 8 | `fig8_new_feature_separation.png` | New feature class separation (v14) |

All figures are saved in `docs/report_figures/`. Run `python scripts/generate_report_plots.py` to regenerate.

---

## 2.10 Discussion

### 2.10.1 Strengths of the Approach

**1. Generalizable feature engineering:** The 30 features in v6–v12 cover a broad set of spectral and temporal signal characteristics. By combining global PSD statistics (mean, std, percentiles, flatness), higher-order spectral measures (kurtosis, entropy), and time-domain amplitude statistics (crest factor, mag kurtosis, I/Q correlation), the model is not tuned to any specific signal type. This is why v9 achieves near-identical recall (TP rate) to the UCLA team (test10) despite using less compute time.

**2. Efficient inference:** Pure-NumPy tree traversal requires no scikit-learn at test time. With 200 trees of depth 3, inference takes microseconds. Feature extraction dominates runtime (~1.9 s/file), leaving ample budget for more expensive second-stage analysis if needed.

**3. Prior correction:** Explicitly accounting for the mismatch between training class distribution (2:1) and expected test distribution (1:1) was a crucial step that improved leaderboard accuracy by an estimated 2 percentage points. This is a principled Bayesian correction rather than a heuristic.

**4. Conservative threshold tuning on a balanced subset:** Evaluating threshold selection on a class-balanced validation subset prevents the optimistic accuracy inflation that would occur if the imbalanced training distribution were used directly.

### 2.10.2 Limitations

**1. Over-reliance on `psd_max`:** The GBM's 77% dependence on the maximum PSD value makes the model brittle to noise files containing a single anomalously high spectral spike. These 8 persistent false positives drove our leaderboard ceiling to 0.85 instead of 0.87+.

**2. CV optimism gap:** Balanced 5-fold CV consistently overestimates leaderboard accuracy by 6–11 percentage points (Fig. 4). This suggests covariate shift between training and test distributions -- the hidden test files likely contain signal types and noise conditions not well-represented in the 240 labeled files.

**3. No two-stage confirmation:** The top two leaderboard teams (sub_test3, sub_test2) achieve FP = 0 by running a more expensive second-pass confirmation, spending ~2000 s for 240 files (versus our 800 s). Our single-pass approach cannot eliminate the borderline FP cases without also rejecting borderline TP cases.

**4. Unlabeled data not utilized:** The 200 unlabeled files in `trainingData/` were not incorporated. Self-supervised or contrastive approaches could potentially improve representation learning, but the competition time frame and submission logistics limited this exploration.

### 2.10.3 Design Trade-offs

| Decision | Trade-off | Chosen |
|---|---|---|
| 5 M vs 10 M IQ samples | Speed vs PSD accuracy | 5 M (leaderboard same accuracy, faster) |
| LogReg vs GBM | Interpretability vs accuracy | GBM (+3% leaderboard) |
| Low threshold vs high threshold | Recall vs precision | Low (v9): maximise TP |
| Ensemble blending | Robustness vs conservatism | Single model (ensemble over-rejected) |
| Sample weighting | FP reduction vs TP stability | Not used (destabilized model) |

### 2.10.4 Potential Improvements

1. **Temporal consistency features (implemented in v14):** `psd_top10_frac` (fraction of energy in top-10 spectral bins), `psd_max_to_p10` (peak above the 10th-percentile noise floor), and `peak_bin_cv` (coefficient of variation of the dominant frequency bin across 8 time windows) address the `psd_max` brittleness directly. CV results show FP=0.01, TP=0.44, estimated accuracy 0.93.

2. **Probability calibration:** Platt scaling or isotonic regression applied to the GBM's CV predictions would sharpen the probability distribution, making threshold selection more reliable across different test distributions.

3. **Two-stage pipeline:** Run a fast GBM (Stage 1, high recall) to flag candidates, then apply an expensive temporal persistence check (Stage 2) only to positives. This matches the inferred strategy of the top-ranked teams and is estimated to achieve FP=0, TP≥0.37 within the ~2000 s server budget.

4. **Richer PSD resolution:** Using `nperseg=32768` for a narrowband signal with a 1 MHz sample rate gives frequency resolution of ~30 Hz, potentially resolving very narrow tones that blur into background noise at 8192-point resolution.

5. **Ensemble with temporal model:** Combining the spectral GBM with a recurrence-based model that exploits the temporal ordering of signal frames could improve discrimination of bursty or intermittent signals.

---

*End of Report -- Sections 2.4 through 2.10*
