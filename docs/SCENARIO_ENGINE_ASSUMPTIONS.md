# Gary scenario engine — public defaults vs assumptions

This document supports the **Completed Research Extension** (`gary_scenario_engine.py` + Gary micro-twin Streamlit tab). It does **not** describe the **core judged** SpectrumX detector.

## Sourced vs scenario (in-repo policy)

| Item | Classification | Notes |
|------|----------------|-------|
| West Side Leadership Academy — **1,232 students** | **Public-report-style default** | Common figure from IDOE/school reporting channels; **verify current year** before citing as authoritative. |
| WSLA staff / teacher headcount (default **95 FTE**) | **Assumption** | Editable in UI; replace with district HR or state staffing roll-ups if available. |
| Student **attendance ratio** (fraction of enrollment on campus) | **Scenario parameter** | Driven by preset + sliders; **not** a live census. |
| Gary City Hall — employee count (default **120**) | **Assumption** | No reliable public roll-up documented in-repo; editable. |
| City Hall — visitor session baseline (default **45**) | **Assumption** | Separate from employees; scale for events in UI. |
| Library — **~240** programmed public-space capacity baseline | **Scenario aggregate (assumption)** | Aggregated from programmed meeting/study-style capacities for **storytelling**; **not** a substitute for fire-code occupancy or OFO. |
| Library occupancy as fraction of baseline | **Scenario parameter** | Default tied to preset; editable slider. |
| Library IP devices / occupant (**1.3–1.5** default band, **1.4** default) | **Assumption** | Optional **1.0** “simple mode” in UI. |
| Device concurrency (active share of provisioned devices) | **Scenario parameter** | From preset + manual traffic stress. |
| Propagation / coverage / KPI numbers in the extension UI | **Proxy** | Deterministic abstractions until DeepMIMO / Sionna RT / Aerial summaries replace them. |

## Site-specific device models (defaults)

- **School (WSLA):** **2** IP devices / student; **3** IP + **1** control (e.g. LMR stand-in) / staff present.
- **City Hall:** **4** IP / employee present; **~1.2** IP / visitor session (configurable constants in engine).
- **Library:** IP / occupant as above; minimal control devices scaled weakly with occupancy (assumption).

## Presets (`normal_day`, `peak_day`, `after_hours`, `emergency_special`)

Presets modulate:

- Student/staff **attendance** multipliers  
- **Visitor** scaling  
- **Concurrency** (active device share)  
- **Emergency / special ops** control-traffic boost (school staff control devices)

Exact coefficients live in `src/edge_ran_gary/gary_scenario_engine.py` (`_preset_core`).

## Replacing assumptions with local data

1. **Enrollment / staffing:** drop verified counts into UI overrides or future YAML under `configs/wireless_scene/`.  
2. **Library capacity:** replace `LIBRARY_PROGRAMMED_CAPACITY` with a documented facility program figure.  
3. **City Hall workforce:** use HR or budgeted FTE if publishable.  
4. **Attendance / events:** calibrate presets against bell schedules, board meeting calendars, or special-event manifests.

## Controller loop (extension only)

- **State:** `SiteScenarioState` + RF sliders + detector belief (`select_closed_loop_action`).  
- **Action:** discrete set — hold, cautious transmit, reduce power, change channel, prioritize site, rebalance service.  
- **KPIs:** `apply_action_to_kpis` nudges proxy metrics from scenario-derived bases — **not** measured network KPIs.

See also `docs/MICROTWIN_REALISM_PLAN.md` and `docs/INDUSTRY_GRADE_EXTENSION_PLAN.md`.
