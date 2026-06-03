import json
from pathlib import Path
def export(out: Path | None = None) -> Path:
    out = out or Path("results/tool_exports/aerial_kpi_stub.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": "nvidia_aerial", "evidence_status": "stub"}, indent=2) + "\n", encoding="utf-8")
    return out
