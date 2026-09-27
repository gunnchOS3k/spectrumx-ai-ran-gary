"""Adapter for 7GC digital-twin site summaries (synthetic).

Backward compatible: adapt_site_summary remains the original tiny mapping.
The rich campus-design path is additive via adapt_campus_twin.
"""


def adapt_site_summary(site_summary: dict) -> dict:
    return {
        "site_id": site_summary.get("site_id", "gary"),
        "n_users": site_summary.get("n_users", 100),
        "fairness_stub": site_summary.get("jains_fairness", 0.5),
    }


def adapt_campus_twin(twin: dict, design: dict | None = None) -> dict:
    """Rich twin path used by the planning optimizer."""
    base = adapt_site_summary(twin)
    base["geometry_fidelity"] = (design or {}).get(
        "geometry_fidelity",
        (twin.get("service_demand") or {}).get("values", {}).get("geometry_fidelity"),
    )
    base["source_manifest_sha256"] = (design or {}).get(
        "source_manifest_sha256",
        (twin.get("service_demand") or {}).get("values", {}).get("source_manifest_sha256"),
    )
    base["n_users"] = twin.get("n_users", base["n_users"])
    return base
