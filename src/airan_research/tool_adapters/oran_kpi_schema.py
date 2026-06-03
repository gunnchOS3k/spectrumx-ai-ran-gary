"""O-RAN-aligned KPI schema (neutral YAML)."""
from pathlib import Path
import yaml

def export(out: Path | None = None) -> Path:
    out = out or Path("results/tool_exports/oran_kpi_schema.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = {"kpi_version": "stub-0.1", "metrics": ["latency_ms", "throughput_mbps", "packet_loss"], "evidence_status": "schema_alignment_only"}
    out.write_text(yaml.dump(schema), encoding="utf-8")
    return out
