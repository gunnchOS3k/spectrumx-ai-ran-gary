"""CI-neutral free-space / log-distance propagation. Optional backends fail-closed."""
from __future__ import annotations

import math

OPTIONAL_BACKENDS = ("sionna", "deepmimo", "ns3", "ns_o_ran", "nvidia_aerial")


def log_distance_path_loss(
    distance_m: float,
    *,
    exponent: float = 2.2,
    pl0_db: float = 40.0,
    d0_m: float = 1.0,
    material_db: float = 6.0,
    floor_db: float = 0.0,
    blockage_db: float = 0.0,
) -> float:
    d = max(distance_m, d0_m)
    return pl0_db + 10.0 * exponent * math.log10(d / d0_m) + material_db + floor_db + blockage_db


def coverage_from_path_loss(path_loss_db: float, threshold_db: float = 95.0) -> float:
    return max(0.0, min(1.0, 1.0 - (path_loss_db / (threshold_db * 1.4))))


def optional_backend_status() -> dict[str, str]:
    return {name: "fail_closed_unused" for name in OPTIONAL_BACKENDS}


def require_optional_backend(name: str) -> None:
    raise RuntimeError(f"{name} backend is optional and fail-closed; default CI must not require it")
