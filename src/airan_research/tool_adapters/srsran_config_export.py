from pathlib import Path
import yaml
def export(out: Path | None = None) -> Path:
    out = out or Path("results/tool_exports/srsran_testbed_stub.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump({"testbed": "srsran_optional", "evidence_status": "stub"}), encoding="utf-8")
    return out
