# Reproducibility

## Quickstart

```bash
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/demo_airan_policy.py --toy
```

## Tests

```bash
pytest -q
```

## Sample data policy

Synthetic/toy only. **No private competition data.** No student PII.

## Regenerate artifacts

Demo commands write to `results/` or `docs/generated/` where applicable.

## Citation

See `CITATION.cff` in repo root.
