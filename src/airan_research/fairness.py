from .metrics import jain_fairness

def fairness_report(allocations: list[float]) -> dict:
    return {"jain_fairness": jain_fairness(allocations), "note": "toy research metric"}
