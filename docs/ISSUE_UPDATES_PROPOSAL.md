# Issue updates proposal

Use this to align GitHub issues with current reality. **Do not auto-edit issues** — apply manually or via scripts you control.

## Proposed issue rewrites (titles + acceptance criteria)

### Competition Core (must not be blocked by portfolio)

1. **Eval protocol locked**
   - Acceptance: `docs/EVAL_PROTOCOL.md` exists; split policy and metrics defined; no data leakage.
2. **Baselines run via one command**
   - Acceptance: Single script or command runs energy detector, spectral flatness (and PSD+LogReg when available); outputs metrics.
3. **Calibration + thresholding**
   - Acceptance: Calibration (e.g. Platt/isotonic) applied; threshold policy documented; ECE reported.
4. **SSL using unlabeled data**
   - Acceptance: SSL method chosen and trained on unlabeled data; finetuning on labeled; metrics logged.
5. **Submission bundle**
   - Acceptance: One-command run on fresh machine; submission format matches competition; no secrets in repo.

### Portfolio / AI-RAN (time-boxed; do not block core)

6. **Gary Micro-Twin v1**
   - Acceptance: Config (city_hall, high_school, library); `generate_sample` → `DigitalTwinSample`; CLI writes iq/*.npy, metadata.csv, manifest.json; invalid zone_id raises; tests for contract + SNR + QPSK.
7. **Digital Twin → AI-RAN narrative**
   - Acceptance: Short doc or section linking detector → occupancy prob → micro-twin → AI-RAN story.

## Duplicates to close (from Asana import)

- Any issue that duplicates the above (e.g. “Implement baselines” when baselines are already implemented): close with a comment like “Done in main; see commit X” or “Superseded by issue #Y.”
- If multiple “Micro-Twin” or “Streamlit” issues exist, keep one and close the rest as duplicates, linking to the canonical issue.

## Separation of concerns

- **Competition Core:** Eval protocol, baselines, calibration, SSL, submission. These are unblocked by repo state; no dependency on portfolio tasks.
- **Portfolio / AI-RAN:** Micro-twin, narrative, optional AI-RAN controller. Clearly label as portfolio so core is never blocked.

## Suggested labels

- `phase:core-detection` — competition deliverable.
- `phase:portfolio-digital-twin` — micro-twin / digital twin.
- `phase:portfolio-ai-ran` — AI-RAN / narrative.
- `priority:P0` for core; `priority:P1`/`P2` for portfolio.
