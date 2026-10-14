# P7-A scientific-design and bounded runtime engineering

This extends the existing P6 without editing its source,verification,config or archives.
No pilot,confirmatory,heldout or external action is enabled. All scientific approvals
are pending. The standalone p7a_run.py refuses every mode before config/output access.

Use Python3.14.7 and the existing P6 requirements.txt versions. From this directory:

```sh
python p7a_check.py --output /tmp/p7a-new-engineering-check
python p7a_design.py --review path/to/p7a/review --base-config path/to/p6/P6_EXPERIMENT_FREEZE_CANDIDATE.json --verify
```

Choose a new output directory. Source-root paths are relative to this directory,
not absolute machine paths. A clean package retains sibling p4-2026-10-14/fair_control
and its unchanged PROSPECTIVE_PROTOCOL.md. No Git is needed for tests or design verify.
The source revision JSON pins exact committed content;verification outputs are separately
hashed. Timing diagnostics already exist in verification/runtime-v2;do not repeat them
merely to review the design. p7a_benchmark.py accepts only a new output path and executes
72fixed current-input controller requests,no Environment.step/task outcomes or seeds.
The initial optional CPU-label metadata failure is retained in runtime-v1.

P7A proposal uses an explicit synthetic byte-surrogate task catalog,not real application
service. Reference adapters remain P6 engineering controls. The candidate's8pair initial
resource/harness feasibility proposal is unsigned. P6's40pair proposal is preserved.
Reuse prototype is not selected primary;fresh P6 wrapper remains recommended.
No new resource-saving fidelity relationship or host-to-virtual delay is invented.

Timing/memory are Unix/macOS host diagnostics. Memory current RSS uses a Darwin API;
other systems return null for it. No hostile process sandbox,hard deadline,full-harness
runtime,physical calibration or scientific superiority is established. Retained project
MIT notice in LICENSE concerns software,not article/data licensing or patent clearance.
