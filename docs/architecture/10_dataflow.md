# Data Flow and Reproducibility

## Competition Core: Detection Pipeline

### Training Pipeline

```
1. Dataset Download
   Input: SDS_SECRET_TOKEN (from .env)
   Output: competition_dataset/files/{user}/training data/*.npy
   Component: SpectrumXDataset.download()
   Config: SpectrumXDatasetConfig (dataset_root, sds_host)

2. Data Loading
   Input: competition_dataset/files/{user}/training data/*.npy
   Input: competition_dataset/files/{user}/VLABrutal/groundtruth.csv
   Output: List[np.ndarray] (IQ samples), np.ndarray (labels), List[str] (filenames)
   Component: SpectrumXDataset.load_labeled()
   Format: Each .npy file contains 1-second IQ sample (complex64, shape=(N,))

3. Preprocessing
   Input: Raw IQ samples (variable length, various formats)
   Operations:
     - Normalization (zero-mean, unit-variance or power normalization)
     - Windowing (if needed for STFT)
     - Format standardization (complex64)
   Output: Preprocessed IQ samples (shape=(N,), dtype=complex64)
   Component: IQPreprocessor.normalize(), IQPreprocessor.window()
   Config: PreprocessingConfig (normalization_method, window_type, window_length)

4. Feature Extraction
   Input: Preprocessed IQ samples
   Operations:
     - Time-domain: I(t), Q(t), |x(t)|, phase(t), instantaneous power
     - Frequency-domain: PSD (Welch), spectrogram (STFT), spectral features
     - Statistical: moments, kurtosis, skewness, spectral flatness
   Output: Feature vectors (dict or np.ndarray)
   Component: FeatureExtractor.extract_time(), FeatureExtractor.extract_freq(), FeatureExtractor.extract_statistical()
   Config: FeatureExtractorConfig (feature_types, n_fft, overlap)

5. Model Training
   A) Supervised Learning Path:
      Input: Features + labels
      Models: EncoderSSL (self-supervised pretraining) → ClassifierHead (supervised fine-tuning)
      Output: Trained model checkpoints
      Component: EncoderSSL.train(), ClassifierHead.train()
      Config: ModelConfig (encoder_arch, classifier_arch, learning_rate, batch_size, epochs)
   
   B) Unsupervised Learning Path:
      Input: Features (unlabeled)
      Models: AnomalyModel (isolation forest, autoencoder, etc.)
      Output: Trained anomaly detector
      Component: AnomalyModel.fit()
      Config: AnomalyModelConfig (model_type, contamination_rate)

6. Calibration
   Input: Model predictions (logits or probabilities)
   Operations: Temperature scaling, Platt scaling, or isotonic regression
   Output: Calibrated confidence scores
   Component: Calibrator.calibrate()
   Config: CalibrationConfig (method, validation_split)

7. Ensemble Fusion
   Input: Multiple model predictions + confidence scores
   Operations: Weighted voting, stacking, or Bayesian model averaging
   Output: Final prediction + confidence
   Component: EnsembleFusion.fuse()
   Config: EnsembleConfig (fusion_method, model_weights)

8. Evaluation
   Input: Predictions, ground truth labels
   Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, ECE, Brier score
   Output: Metrics dictionary, confusion matrix, ROC curve
   Component: Evaluator.evaluate()
   Config: EvaluationConfig (metrics_list, output_dir)

9. Submission Bundle
   Input: Trained models, preprocessing configs, inference code
   Output: submission/ directory with:
     - model_checkpoints/
     - configs/
     - inference.py (entry point)
     - requirements.txt
     - README.md
   Component: ModelRegistry.export_submission()
```

### Inference Pipeline

```
1. User Upload (Streamlit)
   Input: .npy file (user-uploaded)
   Component: apps/streamlit_app.py (file uploader)

2. IQ Loading
   Input: .npy file
   Formats supported:
     - complex64/complex128 array (N,)
     - float array (N, 2) as [I, Q]
     - int16 interleaved (N*2,) as [I0, Q0, I1, Q1, ...]
   Output: complex64 array (N,)
   Component: load_iq_data() in streamlit_app.py

3. Preprocessing
   Same as training pipeline step 3
   Component: IQPreprocessor (reuse from training)

4. Feature Extraction
   Same as training pipeline step 4
   Component: FeatureExtractor (reuse from training)

5. Model Inference
   Input: Features
   Operations:
     - Load model checkpoint
     - Forward pass
     - Get logits/probabilities
   Output: Raw prediction + uncalibrated confidence
   Component: EncoderSSL.predict(), ClassifierHead.predict(), AnomalyModel.predict()

6. Calibration
   Same as training pipeline step 6
   Component: Calibrator (reuse from training)

7. Ensemble Fusion
   Same as training pipeline step 7
   Component: EnsembleFusion (reuse from training)

8. Visualization
   Input: IQ data, predictions, confidence scores
   Output: Interactive plots (Plotly)
   Components:
     - Time domain: I(t), Q(t), |x(t)|
     - Constellation: I vs Q scatter
     - PSD: Power spectral density (Welch)
     - Spectrogram: Time-frequency heatmap
   Component: apps/streamlit_app.py (plotting functions)
```

---

## Phase 2: Digital Twin + AI-RAN Pipeline

### Simulation Pipeline

```
1. Occupancy Probability Input
   Input: Occupancy predictions from Phase 1 (per time slot, per frequency band)
   Format: np.ndarray (time_slots, frequency_bands) with values in [0, 1]
   Component: OccupancyMapper (maps detection results to spectrum grid)

2. Digital Twin Initialization
   Input: GIS data (Gary, Indiana), Sionna scene configuration
   Operations:
     - Load 3D building models
     - Set up ray-tracing scene
     - Configure antenna arrays
   Output: Sionna scene object
   Component: SionnaSceneBuilder.build_scene()
   Config: SceneConfig (city_name, frequency, antenna_config)

3. Channel Generation
   Input: Scene, user positions, base station positions
   Operations:
     - Ray-tracing computation
     - Channel impulse response (CIR) generation
     - CSI extraction
   Output: Channel state information (CSI) tensors
   Component: SionnaSceneBuilder.generate_channels()
   Config: ChannelConfig (num_users, num_bs, num_paths)

4. AI-RAN Controller Decision
   Input: CSI, occupancy probability, fairness constraints
   Operations:
     - Contextual bandit: select action (beam, power, RB) based on context
     - RL agent: policy network forward pass
   Output: Resource allocation decisions
   Component: BanditPolicy.select_action(), ActorCritic.select_action()
   Config: ControllerConfig (policy_type, exploration_rate, fairness_weight)

5. Environment Step
   Input: Resource allocation decisions
   Operations:
     - Apply allocations to channel
     - Compute link performance (SINR, capacity)
     - Check constraint violations (spectral mask, power limits)
     - Compute fairness metrics
   Output: Rewards, next state, done flag
   Component: SimulationEnvironment.step()
   Config: EnvironmentConfig (constraint_thresholds, reward_weights)

6. Metrics Computation
   Input: Episode history (allocations, rewards, violations)
   Metrics:
     - Spectral efficiency (bps/Hz/user)
     - Energy efficiency (bits/Joule)
     - Fairness index (Jain's index, Gini coefficient)
     - Constraint violation rate
   Output: Metrics dictionary
   Component: Metrics.compute_spectral_efficiency(), Metrics.compute_fairness()
```

---

## Reproducibility Contract

### Seeds and Randomness

All random operations use fixed seeds for reproducibility:

```python
# In config.py or environment setup
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
```

### Configuration Management

All configurations are stored in:
- `src/edge_ran_gary/config.py` (Python dataclasses)
- `configs/` directory (YAML files for experiments)
- Environment variables (`.env` file, not committed)

### Artifact Paths

Standard directory structure:

```
competition_dataset/          # Dataset (gitignored)
  files/
    {user}/
      training data/          # Unlabeled samples
      VLABrutal/              # Labeled samples + groundtruth.csv

results/                      # Experiment outputs (gitignored)
  {experiment_name}/
    checkpoints/              # Model checkpoints
    logs/                      # Training logs
    metrics/                  # Evaluation metrics
    plots/                    # Visualization outputs
    config.yaml               # Experiment config snapshot

submission/                   # Submission bundle
  model_checkpoints/
  configs/
  inference.py
  requirements.txt
  README.md
```

### Environment Specification

- Python version: 3.10+
- Dependencies: `requirements.txt`
- CUDA version (if using GPU): Documented in `docs/setup.md`
- OS: Linux/macOS (tested on both)

### Version Control

- Git tags for major milestones: `v1.0-detection-core`, `v2.0-phase2-extension`
- Commit messages follow conventional commits format
- Experiment configs are committed with results (in `results/{experiment_name}/config.yaml`)

### Data Versioning

- Dataset UUID: `458c3f72-8d7e-49cc-9be3-ed0b0cd7e03d` (SpectrumX SDS)
- Dataset version: Tracked via SDS API or manual version file
- Preprocessed data cache: `data/processed/` (gitignored, but hash-based naming for reproducibility)

---

## Dependencies Between Phases

Phase 2 (Digital Twin + AI-RAN) depends on Phase 1 (Detection) only through:
- **Occupancy probability maps**: Output of Phase 1 detection models
- **No direct code dependencies**: Phase 2 can run independently with synthetic occupancy maps

This separation allows:
- Competition judges to evaluate Phase 1 independently
- Phase 2 to be demonstrated as a research extension
- Independent development and testing of each phase
