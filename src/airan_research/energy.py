from .metrics import energy_per_bit

def energy_report(power_w: float, throughput_bps: float) -> dict:
    return {"energy_per_bit_j": energy_per_bit(power_w, throughput_bps)}
