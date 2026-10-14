# Linear-selector semantic audit

Sources O86a5a6f policy_information_equivalent and Rb44357f counterpart were independently inspected and executed for focused assertions. The function name is retained as an artifact label, not a guarantee. Score weights are latency .004, loss .02, outage mismatch .35, low-energy cloud penalty .002, strict-edge bonus .25 and high-latency-edge bonus .15. Alpha paths are TN1,localWiFi1.1,degradedLocal1.6,NTN2.2 (revised may also offer offline10); beta placements cloud1.15,edge1,local1.25.

For edge versus cloud at a fixed path n, the score difference is

```text
.004*L*alpha_n*.15 + .25*I(strict) + .15*I(L>40)
 + .002*max(50-e,0).
```

For edge versus local it is `.004*L*alpha_n*.25 + .25*I(strict)+.15*I(L>40)`. Strict edge dominance requires edge to be a candidate and both differences>0 against candidate competitors. A sufficient domain is L>0,alpha_n>0,edge capacity>0 and the frozen nonnegative bonuses/penalties with beta_edge smaller than beta_cloud/local. This includes all five original/full evaluated families (L20..140,edge capacity .2..1) and, separately checked, revised full-family bounds. It does not cover arbitrary weights, missing edge, negative/zero latency, new failure models or all transformed inputs.

Loss adds a constant to all candidates. Under the generator outage removes TN, so mismatch penalty does not distinguish available candidates. Edge selection leaves network argmax equal to minimum alpha among available paths. Uniform shares and unit power are returned; neither SLSQP nor transition planning occurs. Extra features can change absolute scores and margins without changing this argmax. Different delay/loss values still change evaluated utility after selection; they are not irrelevant to the scoring model. Resource allocation and power are fixed by this selector. Revised default Action adds target fidelity and no checkpoint/recover; there is no fidelity dimension in its linear score.

No retained per-slot action trace is present in raw summaries. audit_science.py reconstructs precisely train seeds0–4 and heldout/domain seeds100–104,24×16 slots per seed, five families: in_distribution,held_out,high_blockage,energy_constrained,outage_heavy. It calls only the frozen selector and independent simplified-rule/inequality assertions:9600states per version, separately, all pass. Revised extra RNG draws are preserved; its states are not pooled with original states. This is a focused verification of historical input-domain behavior, not a new heldout performance evaluation. Continuous-domain validity comes from the inequality argument, not finite enumeration alone.

Zero-latency exception: no_edge_observations zeros L,jitter,loss. At L=0,e>=50,non-strict class, every available placement/path score is0. Candidate order is cloud,edge,local, then listed networks; updates use strict `>` so the first cloud/path wins. With strictness or low energy, remaining bonuses/penalties can remove or change ties; edge is not automatically unique. The illustrative boundary assertion confirms cloud on the first TN candidate. With no_local_edge, edge is absent and the theorem cannot apply.

Historical ablation JSON evaluates the procedural twin policy. Applying the transformation to the linear selector here is a NEW boundary assertion, not an archived linear-policy ablation result. The historical no_edge_observations score gain instead comes from changing truth used for scoring; it cannot identify the benefit of observing less. No frozen source, ablation, policy weight or outcome was modified. Evidence: verification/SCIENTIFIC_AUDIT.json, selector section; mathematical derivation also in MATHEMATICAL_CHARACTERIZATION.md.
