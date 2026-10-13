# Paper II calculations and reviewer orientation

## Worked switching-cost example

Consider an illustrative two-slot trace, not a new experiment. Policy A obtains gross continuity .90 in both slots and never changes action. Its average net score is .90. Policy B obtains .94 in both slots, but in slot2 changes both network and placement. With first-slot charge zero and second charge .05+.03=.08, its scores are .94 and .86, averaging .90. A gross gain can disappear through transition charges even though no task or physical handover duration was measured. Changing only the charge is an accounting probe, not removal of switches.

## Archived decomposition

Original held-out twin gross mean is approximately .82315; cloud fallback .81621. Episode charges are .48180 and .15330 respectively. Convert sums to mean-slot deductions by division by16: .03011 and .00958. Twin gains about .00694 gross but pays .02053 more, yielding net difference -.01359863. The seed-paired post-hoc Student-t interval is [-.01477592,-.01242134], n=5, df=4. All five seed differences are negative. This is evidence against the archived superiority hypothesis, within its score and priors.

For fixed traces rescale scored cost by lambda: U(lambda)=U(1)+(1-lambda)*C/16 while clipping is inactive. Solve the two mean lines for crossing; lambda=.33766185. No action recomputation, training or preregistered sweep occurs. This threshold is not a physical switch-cost estimate or an adaptive-control robustness guarantee.

## Fairness and ablations

All original non-static controllers can access the full Slot, but read different fields and consume different computational work. The revised PolicyObservation runtime mask still gives different declared features and action-space capacity access. Information-access parity, equal feature consumption and equal compute opportunity are separate requirements. None is established merely by identical object arguments.

`no_edge_observations` replaces latency/loss in the state later used by scoring, producing an artificial score gain. `no_energy_constraint` changes an energy covariate while energy has no enforced depletion dynamics. `no_twin_context` and `no_continuity_objective` change branches/flags without removing the mathematical continuity score. These interventions are source-level sensitivity probes, not isolated causal estimates of a physical observation channel. Static actions surviving unavailable states also invalidate a direct deployed-feasibility ranking.

## Revised negative evidence

The separately reproduced revised twin-minus-cloud-fallback difference is -.01254544, 95% paired interval [-.01479626,-.01029462] over the five held-out seeds. Adaptive service-fidelity and checkpoint policies are separately unfavorable in the preserved revised table. Do not combine revised and original seeds as ten interchangeable replications or infer real twin-prediction fidelity from the service-fidelity multiplier.

A prospective comparison must use a new frozen identifier, separate truth/observation state, enforce feasible actions, equalize observations and compute budgets, specify primary contrast and precision rationale, and charge measured or justified transitions. A finite-horizon solver can be a privileged comparator for a small model. Field traces and task/energy measurements are absent and remain future work.
