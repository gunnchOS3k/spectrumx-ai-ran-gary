# Policy information/action matrix

Original receives fullSlot except static reuses its frozen action. Revised source masks declare the fields listed below, while all policies receive action-space availability/capacities,N,B and priorities. Future state access is absent; only oracle has a privileged current-model evaluation hook. Declarations and consumption are distinct.

| Artifact policy | Original consumed inputs / revised use | Revised declared observations | Actions | Allocation/compute |
|---|---|---|---|---|
| no_adaptation | availability | energy_budget | cloud fallback | uniform/unit power |
| static | first-slot rule fields only thereafter frozen | rule_based at firstslot | frozen placement/network | frozen priority shares |
| local_only | availability | energy_budget | local/preference fallback | uniform/.7power |
| cloud_only | availability | energy_budget | cloud/TNthenNTN | uniform/unitpower |
| edge_only | availability/priorities | edge_capacity, energy_budget | edge/WiFithendegraded | priority/.9power |
| rule_based | outage,latency,edgecapacity | edge_capacity, latency_ms, terrestrial_outage | threshold edge/cloud and path | priority/unitpower |
| optimization_based | availability,N,B | energy_budget | cloud/preferred path | SLSQP/unitpower |
| twin_informed | outage,latency,energy,edgecapacity,strictness | continuity_strict, edge_capacity, energy_budget, latency_ms, terrestrial_outage | threshold placement/path | SLSQP/powerthreshold |
| information_equivalent | latency,loss,outage,energy,strictness plus space | continuity_strict, energy_budget, latency_ms, packet_loss_pct, terrestrial_outage | linear candidate argmax | uniform/unitpower |
| oracle | current full metric inputs,previous action | blockage, checkpoint_age_slots, checkpoint_available, continuity_strict, edge_capacity, energy_budget, latency_ms, mobility, model_oracle_metric_eval, packet_loss_pct, recovery_budget, task_progress, terrestrial_outage | greedy path/placement, Rfidelity | uniform/unitpower |
| fixed_target_fidelity | Rextension only; no_adaptation base | energy_budget | Rfixed target | uniform/unitpower |
| adaptive_fidelity | Rextension only; twin base +energy,outage,latency | continuity_strict, edge_capacity, energy_budget, latency_ms, terrestrial_outage | Rthreshold service fidelity | SLSQP/basepower |
| checkpoint_disabled | Rextension only; no_adaptation base | energy_budget | Rdegraded/nocheckpoint | uniform/unitpower |
| adaptive_checkpoint | Rextension only; twinbase +progress,age,availability,budget,blockage | blockage, checkpoint_age_slots, checkpoint_available, continuity_strict, edge_capacity, energy_budget, latency_ms, recovery_budget, task_progress, terrestrial_outage | Rcausal synthetic checkpoint/recover | SLSQP/basepower |

All compare configured dimensionless utility, not measured tasks. See BASELINE_FAIRNESS_AUDIT.md for unavailable-action scoring, unequal compute/optimization, common charged costs versus ignored selection costs, and fidelity/checkpoint confounding. Observations are current-slot; checkpoint state is endogenous policy history, not future access. Source code may expose Python internals; the mask is a documented invocation contract, not a security boundary.
