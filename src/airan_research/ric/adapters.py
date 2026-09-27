"""Safe RIC adapters. No real E2 claim."""
from __future__ import annotations

import os
from typing import Any


def real_actuation_enabled() -> bool:
    return os.environ.get("REAL_ACTUATION_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class SimulatedRICAdapter:
    name = "simulated"

    def read_telemetry(self, site_id: str) -> dict[str, Any]:
        return {"adapter": self.name, "site_id": site_id, "e2_claimed": False}

    def recommend(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"adapter": self.name, "mode": "recommendation_only", "e2_claimed": False}

    def actuate(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "applied": False,
            "e2_claimed": False,
            "decision": "dry_run_recorded" if request.get("dry_run") else "denied",
            "real_actuation_enabled": real_actuation_enabled(),
        }


class ReadOnlyTelemetryAdapter:
    name = "read_only_telemetry"

    def read_telemetry(self, site_id: str) -> dict[str, Any]:
        return {"adapter": self.name, "site_id": site_id, "mode": "read_only", "e2_claimed": False}

    def recommend(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"adapter": self.name, "e2_claimed": False}

    def actuate(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"applied": False, "decision": "denied", "e2_claimed": False}


class MockTestbedRICAdapter:
    name = "mock_testbed"

    def read_telemetry(self, site_id: str) -> dict[str, Any]:
        return {"adapter": self.name, "site_id": site_id, "e2_claimed": False}

    def recommend(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"adapter": self.name, "mode": "shadow", "e2_claimed": False}

    def actuate(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"applied": False, "decision": "shadow_recorded", "e2_claimed": False}


class _FailClosed:
    def __init__(self, name: str) -> None:
        self.name = name

    def read_telemetry(self, site_id: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} backend is optional and fail-closed")

    def recommend(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} backend is optional and fail-closed")

    def actuate(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"{self.name} backend is optional and fail-closed; no E2")


class SrsRANAdapter(_FailClosed):
    def __init__(self) -> None:
        super().__init__("srsRAN")


class OAIAdapter(_FailClosed):
    def __init__(self) -> None:
        super().__init__("OAI")


class ORANSCAdapter(_FailClosed):
    def __init__(self) -> None:
        super().__init__("O-RAN-SC")
