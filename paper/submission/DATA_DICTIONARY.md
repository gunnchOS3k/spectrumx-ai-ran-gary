# Paper II metric and artifact dictionary

| Field | Meaning/unit | Boundary |
|---|---|---|
| vintage | Original or revised source/protocol identity | Different action/score semantics; keep separate |
| seed_means | 24 episode means from one seed, each with 16 slots | n=5 for held-out intervals |
| service_continuity_utility U | Mean max(-1,g-c); dimensionless | Analytical score, not completed-task fraction |
| predicted_latency_ms | Base latency times network/placement/utilization multipliers | Uncalibrated prediction, no end-to-end packet delay |
| switch_cost C | Sum of configured network .05 and placement .03 transition charges per episode | Divide by16 to compare with slot-mean utility |
| n_network_switches / n_placement_switches | Counts per episode, first slot uncharged | No measured attachment/migration time |
| energy_use_j | Formula-based energy proxy labeled joules | No battery state or calibrated device power |
| fairness | Jain index of allocated shares | Does not establish task fairness or delivered throughput |
| compute_time_ms | Host process time around policy execution | Excluded from exact parity; not radio slot time |
| `no_adaptation` | Cloud placement plus fallback network | Changes network when terrestrial unavailable |
| `static` | First action retained | Later unavailable action can still be scored |
| `information_equivalent` | Fixed linear discrete-action selector | Name does not prove identical consumed features or compute budget |
| `oracle` | Greedy evaluator over restricted current action set | Not a global episode optimum/upper bound |
| fidelity (revised) | Synthetic service-quality multiplier/action | Different from predictive agreement of a digital twin |

Domain-shift cells share the same held-out seed integers under different authored priors. Ablations may change scoring truth, not solely controller observations. Missing feasible-action rejection is preserved and disclosed. No SpectrumX judged IQ, OTA data, learner records or private instructor keys are part of the orchestration evidence.
