# Reproducibility — spectrumx-ai-ran-gary

RQ2 AI-RAN / SpectrumX artifact. **Judged competition core** is `submissions/*/main.py` (`evaluate`). Do not rewrite those packages for this digital pass.

**Controller (current):** detector-conditioned **rule-based** closed-loop policy baseline (RIC-style abstraction). **Not** trained RL unless a future commit says otherwise.

Not an institutional affiliation claim. No private competition IQ in the public tree.

## Quickstart (laptop)

```bash
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary
git checkout <frozen-sha>
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install pytest numpy scipy jsonschema
make test
```

Equivalent: `make reproduce` (alias of `make test`).

## Canonical commands

| Target | What it proves |
|--------|----------------|
| `make test` | `tests/test_airan_research.py` always; `tests/gate2` when sibling `gunnchos-7gc-ai-ran-field-kit` fixtures exist (otherwise skipped) |
| `make smoke` / `make e2e` | Broader local smoke (`demo_airan_policy.py --toy` + optional tool exports). Still not field validation. Needs extra deps from `requirements.txt` |
| `pytest -q` | Full pytest tree if you install Streamlit/sklearn; **not** required for CI |

Do **not** `pip install -r requirements.txt` in the default CI job: that file pulls optional stacks (Sionna, DeepMIMO, torch) and is not the competition `evaluate()` contract.

## Competition contract

- Leave `submissions/*/main.py` `evaluate(filename) -> int` unchanged unless a judged resubmission is explicitly in scope.
- Streamlit adapter `PREFERRED_SUBMISSION_ORDER` currently lists `leaderboard_v9` first; `leaderboard_v14` exists on disk. Preferred order is **not** a leaderboard rank.

## UML

Authoritative lanes: [docs/uml/README.md](docs/uml/README.md). Current diagrams were audited against this checkout (Gate 2 `src/airan_research` added as completed extension; controller remains rule-based).

## Independent reproduction

[docs/packets/EXTERNAL_REPRODUCTION_PACKET.md](docs/packets/EXTERNAL_REPRODUCTION_PACKET.md)

## Citation

[`CITATION.cff`](CITATION.cff) · [`LICENSE`](LICENSE) (MIT)
