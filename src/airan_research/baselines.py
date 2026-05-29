from .policy_interface import proportional_fair_policy

def round_robin_baseline(n_users: int, total_rb: int = 100) -> list[int]:
    base = total_rb // max(n_users, 1)
    return [base] * n_users
