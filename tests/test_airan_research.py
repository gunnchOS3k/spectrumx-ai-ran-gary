import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airan_research.metrics import jain_fairness, energy_per_bit
from airan_research.digital_twin_adapter import export_site_summary


def test_fairness():
    assert jain_fairness([10, 10, 10]) == 1.0


def test_twin_export():
    assert export_site_summary()["digital_twin_export"]["site_id"] == "gary"
