#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airan_research.tool_adapters.oran_kpi_schema import export
print(export())
