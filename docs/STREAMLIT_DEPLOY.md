# Streamlit Cloud Deployment Guide

## Quick Start

The Streamlit app is configured for easy deployment on Streamlit Cloud.

### Entry Point

**Main file path:** `streamlit_app.py` (root level)

The root-level `streamlit_app.py` is a wrapper that imports and runs the main app from `apps/streamlit_app.py`. This ensures Streamlit Cloud can find the entry point at the repo root.

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

1. **Connect Repository:**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select repository: `gunnchOS3k/spectrumx-ai-ran-gary`
   - Branch: `main` (or your deployment branch)

2. **Configure App:**
   - **Main file path:** `streamlit_app.py`
   - **Python version:** 3.9+ (auto-detected)

3. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete
   - App will be available at `https://[app-name].streamlit.app`

## Viewing Logs

If the app fails to load:

1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Manage app" → "Logs"
4. Check for:
   - Import errors
   - Missing dependencies
   - File path issues

## Common Failure Causes

### 1. Missing Dependencies

**Symptom:** `ModuleNotFoundError` in logs

**Fix:** Ensure `requirements.txt` includes all dependencies:
- streamlit
- plotly
- numpy
- scipy
- pandas
- matplotlib

### 2. Wrong File Path

**Symptom:** "App not found" or "No module named apps"

**Fix:** 
- Main file path must be `streamlit_app.py` (root level)
- The wrapper imports `apps.streamlit_app` automatically

### 3. Loading Local Dataset Paths

**Symptom:** FileNotFoundError for competition_dataset/

**Fix:** 
- The app does NOT require the competition dataset
- It works with:
  - User-uploaded .npy files
  - Demo data generation (click "Generate Demo IQ Sample")
  - Synthetic data from digital twin (see `docs/DIGITAL_TWIN.md`)

### 4. Secrets/Environment Variables

**Symptom:** Missing API keys or tokens

**Fix:**
- Use Streamlit Secrets: "Manage app" → "Secrets"
- Add secrets in TOML format
- Access via `st.secrets["key_name"]`
- **Never commit secrets.toml to git** (already in .gitignore)

## App Features

The dashboard supports:

- **File Upload:** Upload .npy files with various formats:
  - Complex array: `(N,)` dtype `complex64`/`complex128`
  - Float array: `(N, 2)` as `[I, Q]` pairs
  - int16 interleaved: `(N*2,)` with checkbox enabled

- **Demo Data:** Generate synthetic IQ samples for testing (no upload needed)

- **Visualizations:**
  - Time domain: I(t), Q(t), |x(t)|
  - IQ constellation scatter
  - Power Spectral Density (Welch)
  - Spectrogram (STFT)

- **Baseline Models:**
  - Energy Detector
  - Spectral Flatness Detector
  - PSD+LogReg (when available)

- **Error Handling:** Graceful error messages instead of crashes

## Troubleshooting

### App Shows "Oh no" Page

1. Check logs (see "Viewing Logs" above)
2. Verify `streamlit_app.py` exists at repo root
3. Ensure all imports work (test locally first)
4. Check that `apps/streamlit_app.py` exists

### App Loads But No Visualizations

1. Upload a .npy file or click "Generate Demo IQ Sample"
2. Check file format matches supported formats
3. Adjust sample rate if needed
4. Enable visualization toggles in sidebar

### Import Errors

1. Verify `requirements.txt` is up to date
2. Check that all Python packages are listed
3. Ensure Python version compatibility (3.9+)

## Security Notes

- **No secrets committed:** `.streamlit/secrets.toml` is in `.gitignore`
- **No dataset required:** App works without competition dataset
- **Environment variables:** Use Streamlit Secrets for sensitive data

## Support

For issues:
1. Check logs first
2. Test locally: `streamlit run streamlit_app.py`
3. Verify file paths and imports
4. Check requirements.txt completeness
