"""
Leaderboard v5 submission entrypoint.

Uses PSD + Logistic Regression with:
- 21 features (PSD stats, flatness, spectral entropy, kurtosis, crest factor, etc.)
- Feature standardization (z-score with embedded scaler)
- Accuracy-tuned probability threshold for 50/50 test distribution
- Trained on full labeled dataset (240 files, 160 pos / 80 neg)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

DEFAULT_SAMPLE_RATE = 1e6
MAX_IQ_SAMPLES = 2_000_000

LOGREG_WEIGHTS = [
    0.1547977702725788,
    0.4374238198832431,
    0.49601972086432805,
    -0.264956888397388,
    -0.12395261397723113,
    -0.12847253807718367,
    0.2706681713153779,
    0.3436838061297657,
    0.47696135842626547,
    -0.5641174104762854,
    -0.22766088700233306,
    0.20897217793039108,
    0.15462090861809658,
    0.15433214802568798,
    0.03908897306585177,
    -0.016712433576598195,
    -0.0491721853298175,
    -0.05920527480913937,
    0.04282719732392281,
    0.042364241196604605,
    -0.06419747568169076,
]
LOGREG_BIAS = 0.5831208468550225

SCALER_MEAN = [
    1.0464570066037738e-08,
    3.2500561585162755e-09,
    1.1001059175101228e-07,
    3.0790135228479325e-09,
    8.780629396724453e-09,
    9.079419415377229e-09,
    1.2166684196235345e-08,
    1.3188444474010463e-08,
    3.087264780858116e-09,
    0.9746588793533333,
    11.945569976170857,
    11.025447388511632,
    0.010466832263045945,
    0.010418447711951254,
    0.04297162552829832,
    0.31511023044586184,
    0.6346956593294938,
    3.8673551310978893,
    0.0657123838706563,
    0.06571539143721263,
    0.0005717897494562589,
]
SCALER_STD = [
    7.638170051625778e-09,
    7.271481875268134e-09,
    3.041672337923749e-07,
    1.961731191442624e-09,
    5.610082744851265e-09,
    5.80363948172561e-09,
    1.1034626210437182e-08,
    1.3015517083582593e-08,
    7.825225934748339e-09,
    0.05915450680640072,
    0.2585858549783264,
    55.55421968385656,
    0.007639178240786598,
    0.007419162286836768,
    0.019662365759745545,
    1.3266823724353987,
    0.15449568111138925,
    0.2329695997235527,
    0.03024831048519976,
    0.030250753571608915,
    0.0004035197137990605,
]

PROB_THRESHOLD = 0.4872344689378758


def _load_iq_from_npy(path: Path, is_int16_interleaved: bool = False) -> np.ndarray:
    data = np.load(str(path), allow_pickle=False)

    if is_int16_interleaved:
        if data.dtype != np.int16:
            raise ValueError(f"Expected int16, got {data.dtype}")
        if len(data.shape) != 1 or data.shape[0] % 2 != 0:
            raise ValueError("Interleaved IQ must be 1D with even length")
        i = data[::2].astype(np.float32)
        q = data[1::2].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)

    if np.iscomplexobj(data):
        if len(data.shape) != 1:
            raise ValueError(f"Complex data must be 1D, got shape {data.shape}")
        return data.astype(np.complex64)

    if len(data.shape) == 2 and data.shape[1] == 2:
        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)
        return (i + 1j * q).astype(np.complex64)

    if len(data.shape) == 1 and np.issubdtype(data.dtype, np.floating):
        return (data.astype(np.float32) + 0j).astype(np.complex64)

    raise ValueError(f"Unsupported data shape {data.shape}, dtype {data.dtype}")


def _load_iq_auto(path: Path) -> np.ndarray:
    try:
        return _load_iq_from_npy(path, is_int16_interleaved=False)
    except Exception:
        return _load_iq_from_npy(path, is_int16_interleaved=True)


def _extract_features(iq: np.ndarray, sample_rate: float) -> np.ndarray:
    mag = np.abs(iq)
    power = mag ** 2
    mean_power = float(np.mean(power))

    _, psd = signal.welch(
        iq, fs=sample_rate, nperseg=min(4096, len(iq)),
        return_onesided=False, scaling="density",
    )
    psd_mag = np.abs(psd)
    psd_mean = float(np.mean(psd_mag))

    psd_pos = psd_mag[psd_mag > 0]
    geo_mean = float(np.exp(np.mean(np.log(psd_pos)))) if psd_pos.size > 0 else 0.0
    flatness = geo_mean / psd_mean if psd_mean > 0 else 0.0

    psd_norm = psd_mag / (np.sum(psd_mag) + 1e-30)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))

    return np.array([
        psd_mean,
        float(np.std(psd_mag)),
        float(np.max(psd_mag)),
        float(np.min(psd_mag)),
        float(np.percentile(psd_mag, 10)),
        float(np.percentile(psd_mag, 25)),
        float(np.percentile(psd_mag, 75)),
        float(np.percentile(psd_mag, 90)),
        float(np.percentile(psd_mag, 75) - np.percentile(psd_mag, 25)),
        flatness,
        spectral_entropy,
        float(np.max(psd_mag) / (psd_mean + 1e-30)),
        mean_power,
        float(np.std(power)),
        float(np.std(mag)),
        float(scipy_kurtosis(mag, fisher=True)),
        float(scipy_skew(mag)),
        float(np.max(mag) / (np.sqrt(mean_power) + 1e-30)),
        float(np.std(iq.real)),
        float(np.std(iq.imag)),
        float(np.abs(np.corrcoef(iq.real, iq.imag)[0, 1])) if len(iq) > 1 else 0.0,
    ], dtype=float)


def evaluate(filename: str) -> int:
    try:
        path = Path(filename)
        if not path.is_file():
            return 0

        iq = _load_iq_auto(path)
        if iq is None or iq.size == 0 or iq.ndim != 1:
            return 0
        if MAX_IQ_SAMPLES and iq.size > MAX_IQ_SAMPLES:
            iq = iq[:MAX_IQ_SAMPLES]

        feats = _extract_features(iq, DEFAULT_SAMPLE_RATE)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        scaler_mean = np.array(SCALER_MEAN, dtype=float)
        scaler_std = np.array(SCALER_STD, dtype=float)
        feats_scaled = (feats - scaler_mean) / (scaler_std + 1e-30)

        w = np.array(LOGREG_WEIGHTS, dtype=float)
        b = float(LOGREG_BIAS)
        logit = float(np.dot(w, feats_scaled) + b)
        prob = 1.0 / (1.0 + np.exp(-logit))

        return int(1 if prob >= PROB_THRESHOLD else 0)
    except Exception:
        return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/sample.npy")
        raise SystemExit(1)

    result = evaluate(sys.argv[1])
    print(int(result))
