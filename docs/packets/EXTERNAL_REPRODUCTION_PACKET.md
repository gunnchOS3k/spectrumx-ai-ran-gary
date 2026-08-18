# External / independent reproduction packet

**Repo:** `spectrumx-ai-ran-gary`  
**RQ:** RQ2 (AI-RAN / SpectrumX occupancy + Gary extension)  
**Status:** `INDEPENDENT_REPRODUCTION_PENDING` (digital `SYNTHETIC_SIM` tables exist; no outside-lab PASS on file)

Cursor cannot sign this on another person’s behalf. This packet is for an **outside researcher** on a clean machine.

## Why still pending

No returned evidence record from an independent researcher exists in this tree.

## What this repo can reproduce today

- `make test` — uniform baseline; Paper II digital unit tests; Gate 2 twin-policy tests **if** the sibling field-kit repo is present
- `make paper-reproduce` — frozen protocol + held-out digital programme + generated `paper/tables/` (`SYNTHETIC_SIM`)
- Streamlit demo IQ and Gary **rule-based** controller (not RL)
- Official SpectrumX IQ scoring remains **organizer-offline** and is **not** in this public checkout

## Frozen checkout

```bash
git clone https://github.com/gunnchOS3k/spectrumx-ai-ran-gary.git
cd spectrumx-ai-ran-gary
git checkout <frozen-sha-from-the-draft-PR>
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install pytest numpy scipy jsonschema matplotlib
make test
make paper-reproduce
```

Expect: pytest PASS. Gate 2 tests skip with a reason if `../gunnchos-7gc-ai-ran-field-kit` is absent.

Optional smoke (does not prove OTA or leaderboard score):

```bash
# heavier; may fail if optional extras are missing
make smoke
```

## Competition contract

Do not modify `submissions/*/main.py` to “make CI greener.” Judged interface is `evaluate(filename) -> int`.

## Expected evidence form

Store as `artifacts/independent_reproduction/<lab-or-person-id>.md`.

```text
system:
commit:
command: make test
start:
end:
result:
output_hashes:
deviations:
PASS_FAIL:
notes: rule-based controller; Gate 2 skipped or ran; no private IQ
```

## Physical / external blockers (remain after this packet)

- No independent PASS on file
- No public competition IQ
- Organizer leaderboard scoring is offline
- AODT / full Sionna RT / pyAerial / OTA remain external (see `docs/EXTERNAL_RUNTIME_GAPS.md`)
- Gate 2 contract tests need the field-kit sibling repo
