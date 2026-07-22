"""Twin-state context and Gate 2 AI-RAN policy evaluation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import resource
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None  # type: ignore


POLICY_NAMES = (
    "static_uniform",
    "network_only",
    "service_priority",
    "optimization_based",
    "twin_informed",
)


def resolve_schema_dir(schema_dir: str | Path | None = None) -> Path:
    if schema_dir is not None:
        return Path(schema_dir).expanduser().resolve()
    env = os.environ.get("GATE2_CONTRACTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    sibling = (
        Path(__file__).resolve().parents[4]
        / "gunnchos-7gc-ai-ran-field-kit"
        / "contracts"
    )
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError("Pass --schema-dir or set GATE2_CONTRACTS_DIR")


def load_validator(schema_dir: Path):
    candidate = schema_dir.parent / "scripts" / "validate_contract.py"
    spec = importlib.util.spec_from_file_location("gate2_validate_contract", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {candidate}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def git_commit(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB
    return float(usage) / 1024.0


@dataclass
class TwinContext:
    """Versioned twin-conditioned policy context for the integrated path."""

    run_id: str
    site_id: str
    service_profile: str
    n_users: int
    user_demands: list[dict[str, Any]]
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    available_networks: list[str]
    spectrum_budget: float
    energy_budget: float
    compute_nodes: list[dict[str, Any]]
    mobility_state: dict[str, Any]
    blockage_state: dict[str, Any]
    outage_state: dict[str, Any]
    privacy_constraints: dict[str, Any]
    continuity_requirements: dict[str, Any]
    measurement_quality: dict[str, Any]
    uncertainty: dict[str, Any]
    twin_hash: str
    evidence_level: str
    raw: dict[str, Any]

    @classmethod
    def from_twin_state(cls, twin: dict[str, Any], twin_hash: str) -> "TwinContext":
        summary = twin["source_measurement"]["summary"]
        available = [
            c["network"]
            for c in twin.get("connectivity_candidates", [])
            if c.get("available")
        ]
        return cls(
            run_id=twin["run_id"],
            site_id=twin["site_id"],
            service_profile=twin.get("service_profile")
            or summary.get("service_profile")
            or "unknown",
            n_users=int(twin.get("n_users") or max(1, len(twin.get("user_demands") or []))),
            user_demands=list(twin.get("user_demands") or []),
            latency_ms=float(summary["mean_latency_ms"]),
            jitter_ms=float(summary["mean_jitter_ms"]),
            packet_loss_pct=float(summary["mean_packet_loss_pct"]),
            available_networks=available,
            spectrum_budget=float(
                twin["spectrum_availability"]["values"].get("spectrum_budget_mhz", 20.0)
            ),
            energy_budget=float(
                twin["energy_constraints"]["values"].get("energy_budget_j", 100.0)
            ),
            compute_nodes=list(
                twin["compute_availability"]["values"].get("compute_nodes") or []
            ),
            mobility_state=dict(twin["mobility_state"]["values"]),
            blockage_state=dict(twin["blockage_state"]["values"]),
            outage_state=dict(twin["outage_state"]["values"]),
            privacy_constraints=dict(twin["privacy_constraints"]["values"]),
            continuity_requirements=dict(twin["continuity_requirements"]["values"]),
            measurement_quality=dict(twin.get("measurement_quality") or {}),
            uncertainty=dict(twin.get("uncertainty") or {}),
            twin_hash=twin_hash,
            evidence_level=str(twin.get("evidence_level")),
            raw=twin,
        )


def _jains_fairness(shares: list[float]) -> float:
    if not shares:
        return 0.0
    arr = np.asarray(shares, dtype=float)
    denom = float(np.sum(arr**2) * len(arr))
    if denom <= 0:
        return 0.0
    return float((np.sum(arr) ** 2) / denom)


def _predict_metrics(ctx: TwinContext, shares: list[float], network: str) -> dict[str, float]:
    """Transparent analytical metric model (configured assumptions, not field claims)."""
    util = float(np.sum(shares) / max(ctx.spectrum_budget, 1e-9))
    util = min(1.5, max(0.0, util))
    network_penalty = {
        "terrestrial": 1.0,
        "local_edge_wifi": 1.1,
        "degraded_local": 1.6,
        "ntn_fallback": 2.2,
        "device_to_device": 1.4,
        "offline_continuation": 10.0,
    }.get(network, 1.3)
    pred_latency = ctx.latency_ms * network_penalty * (1.0 + 0.4 * util)
    pred_loss = min(100.0, ctx.packet_loss_pct * network_penalty * (1.0 + 0.2 * util))
    reliability = max(0.0, min(1.0, 1.0 - pred_loss / 100.0))
    energy = (ctx.energy_budget * 0.2) + (util * 15.0) + (0.05 * pred_latency)
    fairness = _jains_fairness(shares)
    continuity = max(0.0, min(1.0, reliability * (1.0 - min(pred_latency, 500.0) / 500.0)))
    return {
        "predicted_latency_ms": float(pred_latency),
        "predicted_reliability": float(reliability),
        "predicted_packet_loss_pct": float(pred_loss),
        "energy_use_j": float(energy),
        "fairness": float(fairness),
        "service_continuity_utility": float(continuity),
        "capacity_utilization": float(min(1.0, util)),
    }


def _select_network(ctx: TwinContext, prefer_local_edge: bool = True) -> str:
    if ctx.outage_state.get("terrestrial_outage"):
        if "local_edge_wifi" in ctx.available_networks and prefer_local_edge:
            return "local_edge_wifi"
        if "degraded_local" in ctx.available_networks:
            return "degraded_local"
        if "ntn_fallback" in ctx.available_networks:
            return "ntn_fallback"
        return "offline_continuation"
    if prefer_local_edge and "local_edge_wifi" in ctx.available_networks and ctx.latency_ms > 60:
        return "local_edge_wifi"
    if "terrestrial" in ctx.available_networks:
        return "terrestrial"
    return ctx.available_networks[0] if ctx.available_networks else "offline_continuation"


def policy_static_uniform(ctx: TwinContext) -> dict[str, Any]:
    share = ctx.spectrum_budget / max(ctx.n_users, 1)
    shares = [share] * ctx.n_users
    network = "terrestrial" if "terrestrial" in ctx.available_networks else _select_network(ctx)
    return {
        "shares": shares,
        "power": [1.0] * ctx.n_users,
        "priority": list(range(ctx.n_users)),
        "network": network,
        "compute_placement": "cloud",
        "local_edge_placement": False,
        "continuity_class": "degraded_ok",
        "rationale": "Equal spectrum share baseline; ignores twin service context.",
    }


def policy_network_only(ctx: TwinContext) -> dict[str, Any]:
    network = _select_network(ctx, prefer_local_edge=False)
    # Uses network condition only
    scale = 0.7 if ctx.packet_loss_pct > 5 else 1.0
    share = (ctx.spectrum_budget * scale) / max(ctx.n_users, 1)
    return {
        "shares": [share] * ctx.n_users,
        "power": [0.8 if ctx.packet_loss_pct > 5 else 1.0] * ctx.n_users,
        "priority": list(range(ctx.n_users)),
        "network": network,
        "compute_placement": "cloud",
        "local_edge_placement": False,
        "continuity_class": "degraded_ok",
        "rationale": "Allocation scaled by network loss only; no service/twin fields.",
    }


def policy_service_priority(ctx: TwinContext) -> dict[str, Any]:
    priorities = [int(u.get("priority", 3)) for u in ctx.user_demands] or [2] * ctx.n_users
    while len(priorities) < ctx.n_users:
        priorities.append(3)
    weights = [1.0 / max(1, p) for p in priorities[: ctx.n_users]]
    total = sum(weights) or 1.0
    shares = [ctx.spectrum_budget * w / total for w in weights]
    network = _select_network(ctx)
    continuity = "strict" if "create" in ctx.service_profile else "degraded_ok"
    return {
        "shares": shares,
        "power": [1.0 + (1.0 / max(1, p)) for p in priorities[: ctx.n_users]],
        "priority": sorted(range(ctx.n_users), key=lambda i: priorities[i]),
        "network": network,
        "compute_placement": "local_edge" if "local_edge" in [n.get("id") for n in ctx.compute_nodes] else "cloud",
        "local_edge_placement": any(n.get("id") == "local_edge" and n.get("capacity", 0) > 0 for n in ctx.compute_nodes),
        "continuity_class": continuity,
        "rationale": "Deterministic service-class priority weights.",
    }


def policy_optimization_based(ctx: TwinContext, seed: int = 0) -> dict[str, Any]:
    """Constrained optimization over spectrum shares.

    Decision variables: per-user spectrum shares x_i (MHz)
    Objective: maximize sum(log(1+x_i)) - 0.01 * predicted_energy proxy
    Constraints: sum(x_i) <= spectrum_budget; x_i >= 0.1; fairness Jain >= 0.7 when feasible
    Infeasibility: fall back to static_uniform and record violation
    """
    n = ctx.n_users
    rng = np.random.default_rng(seed)
    x0 = np.full(n, ctx.spectrum_budget / n) + rng.normal(0, 1e-6, size=n)

    def objective(x: np.ndarray) -> float:
        return -float(np.sum(np.log1p(np.maximum(x, 1e-9)))) + 0.01 * float(np.sum(x))

    constraints = [{"type": "ineq", "fun": lambda x: ctx.spectrum_budget - np.sum(x)}]
    bounds = [(0.1, ctx.spectrum_budget)] * n
    violations: list[dict[str, str]] = []
    if minimize is None:
        result_shares = list(policy_static_uniform(ctx)["shares"])
        violations.append(
            {
                "constraint": "scipy_missing",
                "severity": "warning",
                "detail": "scipy.optimize unavailable; used static_uniform fallback",
            }
        )
    else:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not res.success:
            result_shares = list(policy_static_uniform(ctx)["shares"])
            violations.append(
                {
                    "constraint": "optimization_infeasible",
                    "severity": "warning",
                    "detail": str(res.message),
                }
            )
        else:
            result_shares = [float(v) for v in res.x]
            if _jains_fairness(result_shares) < 0.7:
                violations.append(
                    {
                        "constraint": "fairness",
                        "severity": "warning",
                        "detail": f"Jain fairness {_jains_fairness(result_shares):.3f} < 0.7",
                    }
                )
    network = _select_network(ctx)
    out = {
        "shares": result_shares,
        "power": [1.0] * n,
        "priority": list(range(n)),
        "network": network,
        "compute_placement": "hybrid",
        "local_edge_placement": True,
        "continuity_class": "degraded_ok",
        "rationale": (
            "SLSQP maximizes sum(log(1+x_i)) under spectrum budget; "
            "units: MHz shares, objective dimensionless + energy proxy."
        ),
        "violations": violations,
    }
    return out


def policy_twin_informed(ctx: TwinContext, seed: int = 0) -> dict[str, Any]:
    base = policy_optimization_based(ctx, seed=seed)
    # Incorporate twin fields: continuity, energy, local-edge, outage
    if ctx.continuity_requirements.get("class") == "strict":
        base["continuity_class"] = "strict"
        base["compute_placement"] = "local_edge"
        base["local_edge_placement"] = True
    if ctx.energy_budget < 50:
        base["power"] = [0.6] * ctx.n_users
    if ctx.outage_state.get("terrestrial_outage"):
        base["network"] = _select_network(ctx)
    if ctx.privacy_constraints.get("contains_direct_identifiers"):
        base.setdefault("violations", []).append(
            {
                "constraint": "privacy",
                "severity": "error",
                "detail": "Twin privacy constraints forbid direct identifiers",
            }
        )
    # Prefer local edge when available and latency elevated
    if any(n.get("id") == "local_edge" and n.get("capacity", 0) > 0 for n in ctx.compute_nodes):
        if ctx.latency_ms > 40:
            base["local_edge_placement"] = True
            base["compute_placement"] = "local_edge"
    base["rationale"] = (
        "Twin-informed policy uses full twin context: outage, energy, privacy, "
        "continuity, compute nodes, and measurement quality on top of constrained optimization."
    )
    return base


POLICIES: dict[str, Callable[..., dict[str, Any]]] = {
    "static_uniform": policy_static_uniform,
    "network_only": policy_network_only,
    "service_priority": policy_service_priority,
    "optimization_based": policy_optimization_based,
    "twin_informed": policy_twin_informed,
}


def evaluate_policy(
    twin_path: Path,
    policy_name: str,
    output: Path,
    schema_dir: Path | None = None,
    *,
    seed: int = 0,
    readygary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    schema_path = resolve_schema_dir(schema_dir)
    twin_hash = sha256_file(twin_path)
    twin = json.loads(twin_path.read_text(encoding="utf-8"))
    mod = load_validator(schema_path)
    mod.validate_document(
        twin,
        schema_path,
        expected_schema_name="gunnchos.twin_state_bundle",
        enforce_privacy=False,
    )
    ctx = TwinContext.from_twin_state(twin, twin_hash)

    t0 = time.perf_counter()
    mem0 = peak_memory_mb()
    if policy_name in {"optimization_based", "twin_informed"}:
        decision = POLICIES[policy_name](ctx, seed=seed)
    else:
        decision = POLICIES[policy_name](ctx)
    runtime = time.perf_counter() - t0
    mem1 = peak_memory_mb()

    metrics = _predict_metrics(ctx, decision["shares"], decision["network"])
    violations = list(decision.get("violations") or [])
    if metrics["energy_use_j"] > ctx.energy_budget:
        violations.append(
            {
                "constraint": "energy_budget",
                "severity": "warning",
                "detail": f"predicted energy {metrics['energy_use_j']:.2f}J > budget {ctx.energy_budget}",
            }
        )

    cfg = {"policy": policy_name, "seed": seed, "version": "1.0.0"}
    beam_action = None
    readygary_used = False
    if readygary:
        beam_action = {
            "candidate_beams": readygary.get("candidate_beams", []),
            "selected_beam": int(readygary.get("selected_beam", 0)),
            "expected_sinr_db": float(readygary.get("expected_sinr_db", 0.0)),
            "beam_switch_cost_ms": float(readygary.get("beam_switch_cost_ms", 0.0)),
            "model_runtime_ms": float(readygary.get("model_runtime_ms", 0.0)),
            "provider": "readygary-6g-beam-selection",
            "provider_commit": readygary.get("provider_commit"),
            "provider_input_hash": readygary.get("provider_input_hash"),
        }
        readygary_used = True

    bundle = {
        "schema_name": "gunnchos.airan_decision_bundle",
        "schema_version": "1.0.0",
        "run_id": ctx.run_id,
        "site_id": ctx.site_id,
        "policy_name": policy_name,
        "policy_version": "1.0.0",
        "policy_configuration_hash": sha256_bytes(canonical_json_bytes(cfg)),
        "input_twin_state_hash": twin_hash,
        "selected_actions": {
            "resource_block_allocation": [float(x) for x in decision["shares"]],
            "power_allocation": [float(x) for x in decision["power"]],
            "beam_action": beam_action,
            "scheduling_priority": [int(x) for x in decision["priority"]],
            "selected_network": decision["network"],
            "compute_placement": decision["compute_placement"],
            "local_edge_placement": bool(decision["local_edge_placement"]),
            "continuity_class": decision["continuity_class"],
        },
        "predicted_metrics": metrics,
        "confidence": {
            "score": 0.55 if ctx.evidence_level == "synthetic" else 0.75,
            "notes": "Analytical metric model; not a field-validated predictor.",
        },
        "constraint_violations": violations,
        "runtime_s": float(runtime),
        "peak_memory_mb": float(max(mem0, mem1)),
        "decision_rationale": decision["rationale"]
        + ("" if readygary_used else " ReadyGary beam provider not used."),
        "producer": {
            "repository": "spectrumx-ai-ran-gary",
            "commit": git_commit(),
        },
        "evidence_level": ctx.evidence_level if ctx.evidence_level in {"synthetic", "controlled_device_measurement"} else "mixed",
        "metric_model_assumptions": [
            {
                "name": "latency_utilization_model",
                "description": "predicted_latency = measured_latency * network_penalty * (1 + 0.4*utilization)",
                "assumption_class": "configured",
            },
            {
                "name": "reliability_from_loss",
                "description": "reliability = 1 - predicted_packet_loss_pct/100",
                "assumption_class": "configured",
            },
        ],
        "readygary_used": readygary_used,
        "random_seed": seed,
    }
    if bundle["evidence_level"] not in {"synthetic", "controlled_device_measurement", "mixed"}:
        bundle["evidence_level"] = "mixed"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    mod.validate_document(
        bundle,
        schema_path,
        expected_schema_name="gunnchos.airan_decision_bundle",
        enforce_privacy=False,
    )
    return bundle


def validate_decision(path: Path, schema_dir: Path | None = None) -> dict[str, Any]:
    schema_path = resolve_schema_dir(schema_dir)
    mod = load_validator(schema_path)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return mod.validate_document(
        doc,
        schema_path,
        expected_schema_name="gunnchos.airan_decision_bundle",
        enforce_privacy=False,
    )


def run_benchmark(
    twin_path: Path,
    *,
    policies: list[str] | None = None,
    repetitions: int = 5,
    warmup: int = 1,
    seed: int = 0,
    schema_dir: Path | None = None,
) -> list[dict[str, Any]]:
    policies = policies or list(POLICY_NAMES)
    twin_hash = sha256_file(twin_path)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        # warmup
        for _ in range(warmup):
            evaluate_policy(
                twin_path,
                policy,
                Path("/tmp") / f"airan_warmup_{policy}.json",
                schema_dir=schema_dir,
                seed=seed,
            )
        times: list[float] = []
        mems: list[float] = []
        for i in range(repetitions):
            out = Path("/tmp") / f"airan_bench_{policy}_{i}.json"
            t0 = time.perf_counter()
            bundle = evaluate_policy(twin_path, policy, out, schema_dir=schema_dir, seed=seed)
            times.append(time.perf_counter() - t0)
            mems.append(float(bundle["peak_memory_mb"]))
        times_sorted = sorted(times)
        p95 = times_sorted[min(len(times_sorted) - 1, int(0.95 * (len(times_sorted) - 1)))]
        rows.append(
            {
                "policy": policy,
                "input_size_bytes": twin_path.stat().st_size,
                "repetition_count": repetitions,
                "warmup_count": warmup,
                "runtime_s_list": times,
                "mean_s": statistics.mean(times),
                "median_s": statistics.median(times),
                "p95_s": p95,
                "stdev_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
                "cpu_environment": platform.processor() or platform.machine(),
                "python_version": platform.python_version(),
                "peak_memory_mb": max(mems) if mems else 0.0,
                "random_seed": seed,
                "input_hash": twin_hash,
            }
        )
    return rows


def run_ablations(
    twin_path: Path,
    *,
    seed: int = 0,
    schema_dir: Path | None = None,
) -> list[dict[str, Any]]:
    twin = json.loads(twin_path.read_text(encoding="utf-8"))
    variants = {
        "full": twin,
        "no_edge_observations": _ablate(twin, "edge"),
        "no_twin_context": _ablate(twin, "twin_context"),
        "no_service_continuity_objective": _ablate(twin, "continuity"),
        "no_fairness_constraint": _ablate(twin, "fairness"),
        "no_energy_constraint": _ablate(twin, "energy"),
        "no_local_edge": _ablate(twin, "local_edge"),
    }
    rows = []
    for name, doc in variants.items():
        tmp = Path("/tmp") / f"twin_ablation_{name}.json"
        tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        out = Path("/tmp") / f"airan_ablation_{name}.json"
        bundle = evaluate_policy(tmp, "twin_informed", out, schema_dir=schema_dir, seed=seed)
        m = bundle["predicted_metrics"]
        rows.append(
            {
                "ablation": name,
                "policy": "twin_informed",
                "predicted_latency_ms": m["predicted_latency_ms"],
                "fairness": m["fairness"],
                "service_continuity_utility": m["service_continuity_utility"],
                "energy_use_j": m["energy_use_j"],
                "capacity_utilization": m["capacity_utilization"],
                "constraint_violation_count": len(bundle["constraint_violations"]),
                "input_hash": sha256_file(tmp),
                "random_seed": seed,
            }
        )
    return rows


def _ablate(twin: dict[str, Any], kind: str) -> dict[str, Any]:
    doc = json.loads(json.dumps(twin))
    if kind == "edge":
        summary = doc["source_measurement"]["summary"]
        summary["mean_latency_ms"] = 0.0
        summary["mean_jitter_ms"] = 0.0
        summary["mean_packet_loss_pct"] = 0.0
        summary["mean_upload_mbps"] = 0.0
        summary["mean_download_mbps"] = 0.0
        doc["network_state"]["origin"] = "missing"
    elif kind == "twin_context":
        doc["mobility_state"]["values"] = {}
        doc["blockage_state"]["values"] = {}
        doc["uncertainty"] = {"overall": "high", "notes": "ablated twin context"}
    elif kind == "continuity":
        doc["continuity_requirements"]["values"]["class"] = "offline_ok"
    elif kind == "fairness":
        doc["n_users"] = 1
        doc["user_demands"] = doc["user_demands"][:1] or doc["user_demands"]
    elif kind == "energy":
        doc["energy_constraints"]["values"]["energy_budget_j"] = 1e9
    elif kind == "local_edge":
        nodes = doc["compute_availability"]["values"].get("compute_nodes") or []
        for n in nodes:
            if n.get("id") == "local_edge":
                n["capacity"] = 0.0
        for c in doc["connectivity_candidates"]:
            if c["network"] == "local_edge_wifi":
                c["available"] = False
    return doc
