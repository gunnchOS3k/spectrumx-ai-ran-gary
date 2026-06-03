#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airan_research.tool_adapters import oran_kpi_schema, xapp_policy_stub, aerial_kpi_export, srsran_config_export, oai_config_export, ns_oran_scenario_export
for m in [oran_kpi_schema, xapp_policy_stub, aerial_kpi_export, srsran_config_export, oai_config_export, ns_oran_scenario_export]:
    print(m.export())
