# Streamlit UI copy: style and terminology

Use this guide when editing `apps/streamlit_app.py` or related UI strings.

## Tone

- **Professional and plain.** Prefer clear explanations over hype.
- **Honest scope.** State what is judged, what is a demo extension, and what is a future scaling path.
- **Active voice** and **short sentences** in instructional text.

## Three product buckets (do not blur)

1. **Core judged submission** – SpectrumX DAC detector (offline evaluation, packaged `evaluate()`).
2. **Completed research extension** – Gary digital twin, scenario engine, AI-RAN-style controller demo (not scored).
3. **Next scaling path** – DeepMIMO, Sionna RT, NVIDIA AI Aerial / Omniverse integration hooks and exports.

## Approved phrasing (examples)

- Competition-ready / judge-ready dashboard
- Completed research extension
- Next scaling path
- Integration-ready (for directories, stubs, exporters)
- Non-scoring prototype (for the extension)
- Synthetic demo IQ (in-app only)
- Local CSV metrics / submission package

## Discouraged or banned phrasing

- “Winning project,” “no-brainer,” “blow away judges,” “industry-defining”
- “Conference-grade storytelling” (use **transparent demo narrative** or **declared scope**)
- Implying official competition IQ runs in Streamlit Cloud
- Claiming a full production RIC, live E2, or ray-traced truth where only proxies exist

## Punctuation and typography

- **Do not use the em dash (U+2014) in titles, tab labels, section headers, card titles, or figure headings.** Use a **colon**, **parentheses**, a **short subtitle**, or a **period** instead.
- **Missing values** in tables and metrics: use the shared constant `UI_EMPTY` (`"N/A"`) in code, not em dash.
- **Middle dot (·)** is acceptable **inline** to separate short labels (e.g. file path hints), not as a substitute for sentence structure in long prose.

## Capitalization

- **Section headers:** Title Case (e.g. “Judge Tour Overview,” “Simulation Summaries”).
- **Widget labels and help text:** Sentence case (e.g. “Judge mode,” “Upload .npy file”).
- **Technical names:** Preserve official casing (SpectrumX, DeepMIMO, Sionna RT, Streamlit).

## Status and simulation labels

Prefer short, scannable labels:

- **Loaded (simulation export)** / **Loaded (demo summary)** when a parser succeeded
- **Not loaded** when optional files are absent
- **Access confirmed** only when referring to NGC/installer probe artifacts, not a full twin

## Contrast and HTML callouts

- Custom HTML cards should keep **explicit dark text** (`#111827`–`#212529`) on **light** card backgrounds (`#f1f3f5`–`#f8f9fa`) so they stay readable if the app theme is dark.
- Avoid low-contrast gray-on-gray for primary explanations.

## Local run command

From the repository root (with dependencies installed):

```bash
streamlit run apps/streamlit_app.py
```
