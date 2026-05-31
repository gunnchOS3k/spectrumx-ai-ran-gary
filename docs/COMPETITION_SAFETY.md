# Competition Safety

## Research vs competition paths

| Path | Location | Purpose |
|------|----------|---------|
| Competition | `submissions/*/main.py` via `edge_ran_gary.submission_adapter` | Preserves `evaluate(filename)` contract |
| Research | `src/airan_research/` + `scripts/demo_airan_policy.py --toy` | Synthetic metrics only |

## Rules

- Do not commit private IQ or leaderboard datasets.
- Research demos must use `--toy` and write only to `results/`.
- Never modify competition `evaluate()` signatures without explicit review.
- PRs must state whether competition path was exercised.

## Verification

```bash
python3 scripts/demo_airan_policy.py --toy
# Competition smoke (optional, local data only):
# python3 -c "from edge_ran_gary.submission_adapter import list_submissions; print(list_submissions())"
```
