# Gary Micro-Twin v1 — Output spec for Ananya

Competition-safe synthetic IQ only. **Never upload or commit the official competition dataset.**

## File layout

After running:

```bash
python -m edge_ran_gary.digital_twin.generate --config configs/gary_micro_twin.yaml --n 200 --out data/synthetic/gary_micro_twin
```

you get:

```
data/synthetic/gary_micro_twin/
  iq/
    sample_000000.npy
    sample_000001.npy
    ...
  metadata.csv
  manifest.json
```

- **iq/*.npy** — One file per sample; each is complex64 shape `(N,)` (1-second IQ at configured sample_rate).
- **metadata.csv** — One row per sample with columns aligned to `DigitalTwinSample.metadata` (sample_id, zone_id, landmark_name, center_lat, center_lon, snr_db, signal_type, sample_rate_hz, n_samples, seed, config_hash, generator_version, file, label).
- **manifest.json** — Resolved config path, config_hash, generator_version, n_samples, seed, zone_ids, timestamp_utc.

Output directory is gitignored (`data/synthetic/`).

## DigitalTwinSample schema

| Field | Meaning |
|-------|--------|
| `iq` | complex64 array shape (N,) |
| `label` | 0 (noise only) or 1 (structured signal present) |
| `metadata.sample_id` | Unique ID (e.g. zone_id_seed_label) |
| `metadata.zone_id` | One of city_hall, high_school, library |
| `metadata.landmark_name` | Human-readable name |
| `metadata.center_lat`, `center_lon` | Approximate zone center (from config) |
| `metadata.snr_db` | SNR in dB (None for label=0) |
| `metadata.signal_type` | "noise" or "qpsk" / "ofdm" |
| `metadata.sample_rate_hz` | Sample rate |
| `metadata.n_samples` | Length of IQ |
| `metadata.seed` | Reproducibility seed |
| `metadata.config_hash` | Hash of config used |
| `metadata.generator_version` | e.g. "1.0.0" |

## How Ananya consumes it

1. **Load IQ and metadata**
   - Load `iq/*.npy` and `metadata.csv` (or use manifest to locate files).
   - Map each row to `(iq, sample_rate_hz, metadata)`.

2. **Run detector**
   - `model.predict_iq(iq, sample_rate_hz, optional metadata) -> prob_occupied` (or logits).
   - Use metadata for zone-aware evaluation and stress tests.

3. **Recommended stress tests**
   - Sweep **snr_db** (low / mid / high).
   - Sweep **zone_id** (city_hall, high_school, library).
   - Vary **waveform_mix** in config (qpsk vs ofdm weights).
   - Vary **occupancy_prior** per zone.

## Competition-safe warning

- **Never upload or commit the official SpectrumX competition dataset.**
- The synthetic micro-twin is for demos, tests, and ML pipeline development only.
- Evaluation and submission must use the official competition data according to competition rules.
