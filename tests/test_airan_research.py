import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airan_research.baselines import list_ablation_policies, run_baseline
from airan_research.policy_interface import PolicyContext


def test_uniform_baseline():
    ctx = PolicyContext(n_users=10, spectrum_mhz=100.0, energy_budget_w=5.0)
    alloc = run_baseline("uniform", ctx)
    assert len(alloc) == 10


def test_ablation_policies_allocate():
    ctx = PolicyContext(n_users=5, spectrum_mhz=50.0, energy_budget_w=6.0)
    for name in list_ablation_policies():
        alloc = run_baseline(name, ctx)
        assert len(alloc) == 5
        assert all(v >= 0 for v in alloc)
