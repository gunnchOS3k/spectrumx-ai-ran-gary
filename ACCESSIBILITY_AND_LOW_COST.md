# Accessibility and Low Cost — SpectrumX AI-RAN Gary

## Design principles

1. **Phone-first** — prioritize hardware learners already own.
2. **Offline-first where possible** — document what works without continuous connectivity.
3. **Low-bandwidth paths** — synthetic/smoke modes for constrained networks.
4. **Accessible docs** — plain language README sections; structured headings for screen readers.
5. **No paywall on learning** — MIT/open tooling; external services optional and documented.

## Accessibility checklist (repo)

- [ ] README states real vs synthetic vs planned
- [ ] CLI / UI errors are human-readable
- [ ] Color/contrast notes for any UI artifacts
- [ ] Keyboard / switch navigation noted for console UIs
- [ ] Multilingual support documented (if applicable)

## Cost assumptions

| Item | Assumption |
|------|------------|
| Developer machine | Existing laptop (Linux/macOS/WSL) |
| Phone / edge device | Android or iOS already owned — **not** required to buy custom hardware |
| Cloud | Optional; local `make test` / `pytest` / `npm test` paths documented |
| Data | Synthetic by default; field data **opt-in only** |

## Console alignment

See [gunnchos-device-os](https://github.com/gunnchOS3k/gunnchos-device-os) for mode contracts
(School, Developer, Play, Research Measurement).
