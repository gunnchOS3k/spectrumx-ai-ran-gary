from pathlib import Path

def export(out: Path | None = None) -> Path:
    out = out or Path("results/tool_exports/xapp_policy_stub.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# xApp policy stub\n\nSmoke alignment with O-RAN research patterns. Not deployed RIC.\n", encoding="utf-8")
    return out
