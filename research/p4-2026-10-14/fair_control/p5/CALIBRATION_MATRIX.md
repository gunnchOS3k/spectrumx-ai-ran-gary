# Calibration inventory and parameter boundaries

No legitimate physical calibration evidence was established for this model. This is a bounded search of existing identified repositories/interfaces and P4 sources, not proof that no measurement exists anywhere. No external dataset was downloaded, physical/device test performed, learner data accessed or production code altered. CALIBRATION_EVIDENCE_AVAILABLE:false.

| Candidate source | Provenance/method/rights | Actual status and applicability |
|---|---|---|
| Device OS device_lab/calibration.py + wp010 readiness JSON | Read-only source at4f321c2565669a0e476fc73c524ab84dffed90c0,MIT notice inspected | PHYSICAL_PENDING, CALIBRATED_EVT0=false, LAB_FUTURE007/008/009=false. Ingestion schemas/test toy values are not real power/latency observations |
| Device OS offline_sync.py and continuity sync.py | Same SHA,MIT; clock/vector/LWW/idempotent planner interfaces inspected | Workload semantics only. Planner apply lacks transport/deadline measurement; mock:false labels do not prove physical experiments |
| WAIKE syncCoordinator.ts |32e9c8e9a7eef71c49b8c68292b28e57b2ca88c1; minimal transport,durable receipt/lease/idempotency inspected | Suitable task contract inspiration; no runtime/learner measurements imported. No standalone LICENSE found by filename search; reuse/distribution rights pending |
| waike-research-ops learner-journey test |dd1c5ae0a917ce3fc45e7def2439cc6981bf0ab7,MIT | Anonymous constructed fixture semantics,not actual student activity/capacity calibration |
| EdgeDroid/energy-offloading/predictive-placement sources already reviewed in P4 | Portal p4/NOVELTY_SOURCES.json records identities and actual depth | Published workload/method candidates only. No numeric parameter imported,trace/license/applicability unverified for this model |
| Public network data/standards/NTN simulator literature | No specific licensed raw calibration input selected in P5 | Unavailable for calibration here. Standards targets would be STANDARDS_REQUIREMENT,not MEASURED. Literature-derived numbers need verified methods/units/context |

| Model parameter | Units | Current source/status | Method/uncertainty/limits |
|---|---|---|---|
| Task work/memory/input and deadlines/freshness | Mcycles/MB/Mbit/s | Authored fixture; SYNTHETIC_ASSUMPTION | Exact configured values for equation checks; no device/workload uncertainty estimate |
| CPU capacities/bandwidth/memory | Mcycles/s,Mbps,MB | Hardware and constructed truth; UNCALIBRATED | Scalar fluid capacities,not OS/RF/carrier traces |
| Network latency and gate availability | s,Boolean | Authored frames; SYNTHETIC_ASSUMPTION | One-time latency and imposed outages; no fitted delay/loss/Markov process |
| Power/battery | W,J | Model allocations; UNCALIBRATED | E=P×Δ plus cost; dimensional consistency only,no meter/idle/nonlinear model |
| Switch/checkpoint/restore duration/energy | s,J | Configured synthetic Model/API; UNCALIBRATED | Positive-time operations; no migration-byte or measured terminal cost calibration |
| Fidelity quality | normalized0..1 | Target1/degraded.5; SYNTHETIC_ASSUMPTION | Not learning outcome/prediction accuracy; no work reduction or evidence of benefit |
| Forecast/state uncertainty | estimated frames | Constructed prediction,UNCALIBRATED | Not physically synchronized twin; no accuracy fit or physical observation |
| Statistical SD/effect/cap | fractions/unit counts | P4 prospective hypotheses,not calibration | SD.15 hypothetical; approval pending,no outcome generated |

Calibration class vocabulary: MEASURED needs actual verified source bytes/hash,value,units,method,uncertainty,rights/applicability; LITERATURE_DERIVED needs a verified extract/context and permissions; STANDARDS_REQUIREMENT is a target/constraint,not observation; SYNTHETIC_ASSUMPTION is authored engineering input; UNCALIBRATED carries no established physical transfer. The adapter validates parameter/unit/status/source identity and reference samples, calculates descriptive mean/sample SD, and never promotes status automatically. A constructed1W/3W reference gives mean2W/sample SDsqrt2W but is not a power measurement. Invalid J-as-W or source hashes and constructed-as-measured mutations fail.

Next acquisition needs separately approved legitimate raw records/instruments,source licenses,exact hardware/network/workload context,measurement/cycle/clock definitions, uncertainty and causal forecast train/validation split. Before physical work obtain scope approval. Old paper proxies cannot be relabeled measured values. Existing MIT software permissions are separate from article/dataset license and author/IP decisions.
