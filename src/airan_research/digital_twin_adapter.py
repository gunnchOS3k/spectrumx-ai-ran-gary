from .policy_interface import load_gary_site_schema_example

def export_site_summary() -> dict:
    site = load_gary_site_schema_example()
    return {"digital_twin_export": site, "format": "7gc-site-schema-v1"}
