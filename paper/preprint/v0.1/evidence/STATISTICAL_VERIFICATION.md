# Independent statistical verification

1477 retained non-timing summary means, sampleSD(ddof1) and intervals checked. Every family uses5 seed means, each24 episodes×16 slots. Train0–4; heldout and shifted families100–104. Slots/episodes/users are not extra independent inferential units. Original marginal summaries use roundedt4=2.776; revised and new paired arithmetic use SciPy quantile2.7764451052. No original result is replaced by a differently rounded calculation. PooledCohen d in original artifacts is separately verified and is not paired d_z.

| Vintage / paired whole-policy contrast | n | Mean difference | Sample SD | 95% paired CI | paired d_z |
|---|---:|---:|---:|---|---:|
| original: twin_informed minus no_adaptation | 5 | -0.0135986301 | 0.0009481532 | [-0.0147759178, -0.0124213424] | -14.34223 |
| original: information_equivalent minus twin_informed | 5 | 0.0343691705 | 0.0008907635 | [0.0332631415, 0.0354751995] | 38.58394 |
| original: information_equivalent minus oracle | 5 | 0.0003051532 | 0.0015207922 | [-0.0015831597, 0.0021934661] | 0.20065 |
| original: no_switch_cost minus full | 5 | 0.0301145833 | 0.0011829053 | [0.0286458125, 0.0315833542] | 25.45815 |
| revised: twin_informed minus no_adaptation | 5 | -0.0125454397 | 0.0018127423 | [-0.0147962565, -0.0102946228] | -6.92070 |
| revised: information_equivalent minus twin_informed | 5 | 0.0345116576 | 0.0014768347 | [0.0326779251, 0.0363453900] | 23.36867 |
| revised: information_equivalent minus oracle | 5 | -0.0000795350 | 0.0015215460 | [-0.0019687839, 0.0018097140] | -0.05227 |
| revised: adaptive_fidelity minus fixed_target_fidelity | 5 | -0.0545219232 | 0.0033314869 | [-0.0586585105, -0.0503853359] | -16.36564 |
| revised: adaptive_checkpoint minus checkpoint_disabled | 5 | -0.0585490885 | 0.0012874984 | [-0.0601477290, -0.0569504481] | -45.47508 |
| revised: no_switch_cost minus full | 5 | 0.0302083333 | 0.0008104357 | [0.0292020444, 0.0312146222] | 37.27419 |

Pairing uses matching within-vintage/family exogenous seed streams; revised checkpoint state evolves separately by policy. Original and revised generators consume different random draws and must not be pooled or treated as a feature-isolated paired ablation. All five twin-minus-fallback differences are negative separately. Linear-minus-greedy intervals includezero: inconclusive, not equivalence or proof of optimality. Adaptive fidelity/checkpoint differences are negative whole-policy comparisons, not isolated component effects.

Large standardized paired effects reflect very small dispersion under narrow fixedpriors andn=5; they do not imply large practical population effects. Intervals assume suitable seed-mean sampling behavior; normality/tails cannot be assessed well with5units. No multiplicity-adjusted confirmatory claim is made. More seeds alone would not resolve construct/physical/fairness validity. Protocol history establishes repository freezing only, not an externally verified preregistration chronology.

verification/SCIENTIFIC_AUDIT.json records every comparison and the summary/parity audit. Its runtime reconstructs selector states without generating new primary performance samples. Recovered historical reproduction/test records remain labeled historical. A future clean archive rerun is an instruction acceptance check, not independent human replication.
