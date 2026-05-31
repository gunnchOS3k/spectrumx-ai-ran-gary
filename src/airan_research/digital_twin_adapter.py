"""Adapter stub for 7GC digital-twin site summaries (synthetic)."""


def adapt_site_summary(site_summary: dict) -> dict:
    return {
        "site_id": site_summary.get("site_id", "gary"),
        "n_users": site_summary.get("n_users", 100),
        "fairness_stub": site_summary.get("jains_fairness", 0.5),
    }
