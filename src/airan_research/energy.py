def energy_score(power_w: float, n_users: int) -> float:
    """Higher is better for toy demo (lower power per user)."""
    return 1.0 / (1.0 + power_w / max(n_users, 1))
