# FMEA (excerpt)

| Function | Failure | Effect | S | O | D | RPN | Mitigation |
|----------|---------|--------|---|---|---|-----|------------|
| Demo run | Missing deps | Cannot reproduce | 7 | 3 | 4 | 84 | runbook + CI |
| Claims | Overclaim 6G | Supervisor distrust | 8 | 2 | 3 | 48 | CLAIMS_AUDIT |
| Privacy | PII in repo | Harm | 9 | 1 | 2 | 18 | data policy |
