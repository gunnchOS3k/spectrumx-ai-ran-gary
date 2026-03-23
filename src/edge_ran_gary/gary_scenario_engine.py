"""
Gary anchor-site scenario engine for the Completed Research Extension.

Computes people, device, traffic, and pressure metrics from **scenario assumptions**
and a small set of **public-grounded defaults** (documented in UI + docs).

This is **not** a propagation solver and **not** the judged SpectrumX detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from math import tanh
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Public-grounded defaults (verify / replace with local data)
# ---------------------------------------------------------------------------

# West Side Leadership Academy — enrollment commonly reported from IDOE-style
# school reports (verify current year in `docs/SCENARIO_ENGINE_ASSUMPTIONS.md`).
WSLA_ENROLLMENT_STUDENTS: int = 1232

# Staff + teachers: **scenario assumption** (not a verified public headcount in-app).
WSLA_STAFF_FTE_ASSUMED: int = 95

# Gary Public Library & Cultural Center — **aggregated scenario baseline** from
# programmed public meeting / study capacities (~240); treat as **assumption**.
LIBRARY_PROGRAMMED_CAPACITY: int = 240

# City Hall — **no reliable public FTE count** in-repo; editable assumption.
CITY_HALL_EMPLOYEES_ASSUMED: int = 120
CITY_HALL_VISITORS_BASE_ASSUMED: int = 45

# Device models (scenario parameters)
WSLA_IP_PER_STUDENT: float = 2.0
WSLA_IP_PER_STAFF: float = 3.0
WSLA_CONTROL_PER_STAFF: float = 1.0  # e.g. LMR / walkie stand-in

CITY_HALL_IP_PER_EMPLOYEE: float = 4.0
CITY_HALL_IP_PER_VISITOR: float = 1.2

LIBRARY_IP_PER_OCCUPANT_DEFAULT: float = 1.4
LIBRARY_IP_PER_OCCUPANT_SIMPLE: float = 1.0

PresetName = Literal["normal_day", "peak_day", "after_hours", "emergency_special"]


@dataclass
class ScenarioInputs:
    """User + UI inputs driving the engine."""

    preset: PresetName
    rf_environment_stress: float  # 0..1 from UI (quieter → noisy)
    manual_traffic_stress: float  # 0..1 from demand select
    manual_occ_prior: float  # 0..1 from occupancy select
    time_context: str  # "School hours" | "After hours" | "Weekend"
    event_high_load: bool
    # Overrides (optional)
    wsla_staff_count: Optional[int] = None
    wsla_attendance_ratio: Optional[float] = None  # fraction of enrollment on campus
    city_hall_employees: Optional[int] = None
    city_hall_visitors: Optional[int] = None
    library_occupancy_ratio: Optional[float] = None  # of programmed capacity
    library_ip_per_occupant: Optional[float] = None
    library_simple_device_mode: bool = False


@dataclass
class SiteScenarioState:
    site_id: str
    site_name: str
    people_count: float
    students_present: float
    staff_present: float
    visitors_present: float
    ip_device_count: float
    control_device_count: float
    active_ip_devices: float
    active_control_devices: float
    traffic_demand_score: float
    coexistence_pressure: float
    coverage_pressure: float
    fairness_community_priority: float
    sourcing_notes: List[str] = field(default_factory=list)
    assumption_notes: List[str] = field(default_factory=list)

    def to_display_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _preset_core(preset: PresetName, time_context: str, site_id: str) -> Dict[str, float]:
    """Returns multipliers: attendance_students, attendance_staff, visitor_factor, concurrency, emergency_control_boost."""
    base = {
        "normal_day": (0.90, 0.92, 1.0, 0.34, 1.0),
        "peak_day": (0.96, 0.98, 1.55, 0.52, 1.0),
        "after_hours": (0.10, 0.22, 0.35, 0.20, 1.0),
        "emergency_special": (0.78, 0.95, 1.25, 0.48, 1.45),
    }[preset]

    att_s, att_st, vis_f, conc, ctrl_b = base
    if site_id == "west_side_leadership":
        if time_context == "School hours" and preset == "normal_day":
            att_s = max(att_s, 0.88)
        elif time_context == "After hours":
            att_s = min(att_s, 0.12)
            att_st = max(att_st, 0.18)
        elif time_context == "Weekend":
            att_s = 0.05
            att_st = 0.08
    if site_id == "public_library":
        if time_context == "Weekend" and preset == "normal_day":
            vis_f *= 1.15
    return {
        "attendance_students": att_s,
        "attendance_staff": att_st,
        "visitor_factor": vis_f,
        "concurrency": conc,
        "control_boost": ctrl_b,
    }


def compute_site_state(site_id: str, site_name: str, inp: ScenarioInputs) -> SiteScenarioState:
    notes: List[str] = []
    assumptions: List[str] = []
    pf = _preset_core(inp.preset, inp.time_context, site_id)

    if site_id == "west_side_leadership":
        enroll = float(WSLA_ENROLLMENT_STUDENTS)
        staff = float(inp.wsla_staff_count if inp.wsla_staff_count is not None else WSLA_STAFF_FTE_ASSUMED)
        att = inp.wsla_attendance_ratio if inp.wsla_attendance_ratio is not None else pf["attendance_students"]
        att_staff = pf["attendance_staff"]
        students = enroll * _clamp01(att)
        staff_p = staff * att_staff
        visitors = 0.0
        people = students + staff_p
        ip_dev = students * WSLA_IP_PER_STUDENT + staff_p * WSLA_IP_PER_STAFF
        ctrl_dev = staff_p * WSLA_CONTROL_PER_STAFF * pf["control_boost"]
        notes.append(f"Enrollment **{int(enroll)} students** — public-report-style figure; **verify** current IDOE/school report.")
        assumptions.append(f"Staff/teacher headcount **{int(staff)} FTE** — **scenario assumption** (editable).")
        assumptions.append(f"Attendance ratio **{att:.2f}** of enrollment — **scenario** (not a live census).")
    elif site_id == "public_library":
        cap = float(LIBRARY_PROGRAMMED_CAPACITY)
        occ_r = inp.library_occupancy_ratio if inp.library_occupancy_ratio is not None else (0.38 * pf["visitor_factor"])
        occ_r = _clamp01(occ_r)
        ip_each = (
            LIBRARY_IP_PER_OCCUPANT_SIMPLE
            if inp.library_simple_device_mode
            else (inp.library_ip_per_occupant or LIBRARY_IP_PER_OCCUPANT_DEFAULT)
        )
        people = cap * occ_r
        students = staff_p = 0.0
        visitors = people
        ip_dev = people * ip_each
        ctrl_dev = max(8.0, people * 0.02)  # minimal ops / security control devices (assumption)
        notes.append(
            f"Baseline capacity **~{int(cap)}** (programmed public-space **scenario aggregate**; **not** building fire code OFO)."
        )
        assumptions.append(f"Occupancy **{occ_r:.2f} × baseline** — **scenario**.")
        assumptions.append(f"IP devices / occupant **{ip_each:.2f}** — **assumption** (simple mode = {inp.library_simple_device_mode}).")
    elif site_id == "city_hall":
        emp = float(inp.city_hall_employees if inp.city_hall_employees is not None else CITY_HALL_EMPLOYEES_ASSUMED)
        vis = float(inp.city_hall_visitors if inp.city_hall_visitors is not None else CITY_HALL_VISITORS_BASE_ASSUMED)
        presence_emp = 0.55 + 0.40 * pf["attendance_staff"]  # more staff on-site peak
        presence_vis = vis * pf["visitor_factor"] * (0.6 if inp.preset == "after_hours" else 1.0)
        workers = emp * presence_emp
        visitors = presence_vis
        people = workers + visitors
        students = 0.0
        staff_p = workers
        ip_dev = workers * CITY_HALL_IP_PER_EMPLOYEE + visitors * CITY_HALL_IP_PER_VISITOR
        ctrl_dev = workers * 0.15 + 12.0  # radios / security (assumption)
        assumptions.append(f"Municipal employees **{int(emp)}** — **scenario assumption** (no verified public roll-up in-app).")
        assumptions.append(f"Visitor session count **~{int(vis)}** baseline × scenario — **assumption**.")
    else:
        people = students = staff_p = visitors = ip_dev = ctrl_dev = 0.0
        assumptions.append("Unknown site — zeroed.")

    if inp.event_high_load:
        people *= 1.08
        ip_dev *= 1.10
        notes.append("**Special event** modifier applied (+load).")

    conc = pf["concurrency"] * (0.85 + 0.25 * inp.manual_traffic_stress)
    conc = _clamp01(conc)
    active_ip = ip_dev * conc
    active_ctrl = ctrl_dev * min(1.0, conc * 1.1)

    # Normalize traffic demand (0..1) — site-relative reference
    ref_ip = {"west_side_leadership": 3200.0, "public_library": 450.0, "city_hall": 650.0}.get(site_id, 800.0)
    raw_load = active_ip / ref_ip
    traffic_demand = _clamp01(0.15 + 0.55 * tanh(raw_load) + 0.12 * inp.manual_traffic_stress)

    coexist = _clamp01(
        0.22 * traffic_demand
        + 0.38 * inp.rf_environment_stress
        + 0.18 * inp.manual_occ_prior
        + 0.12 * (active_ctrl / max(50.0, active_ip + 1.0))
    )

    cap_proxy = max(people, 50.0)
    coverage_pressure = _clamp01(active_ip / (cap_proxy * 2.8))

    # Fairness: prioritize education + public access sites under stress
    if site_id == "west_side_leadership":
        fair = 0.52 + 0.28 * traffic_demand + 0.12 * (1.0 - inp.rf_environment_stress)
    elif site_id == "public_library":
        fair = 0.58 + 0.22 * traffic_demand
    else:
        fair = 0.45 + 0.20 * traffic_demand
    fair = _clamp01(fair)

    return SiteScenarioState(
        site_id=site_id,
        site_name=site_name,
        people_count=round(people, 1),
        students_present=round(students, 1),
        staff_present=round(staff_p, 1),
        visitors_present=round(visitors, 1),
        ip_device_count=round(ip_dev, 1),
        control_device_count=round(ctrl_dev, 1),
        active_ip_devices=round(active_ip, 1),
        active_control_devices=round(active_ctrl, 1),
        traffic_demand_score=round(traffic_demand, 4),
        coexistence_pressure=round(coexist, 4),
        coverage_pressure=round(coverage_pressure, 4),
        fairness_community_priority=round(fair, 4),
        sourcing_notes=notes,
        assumption_notes=assumptions,
    )


def compute_all_anchor_states(
    site_defs: List[Dict[str, Any]], inp: ScenarioInputs
) -> Dict[str, SiteScenarioState]:
    out: Dict[str, SiteScenarioState] = {}
    for b in site_defs:
        sid = str(b["id"])
        out[sid] = compute_site_state(sid, str(b["name"]), inp)
    return out


def propagation_proxy_bundle(
    site_id: str,
    height_m: float,
    rf_env: float,
    coverage_pressure: float,
    coexistence_pressure: float,
) -> Dict[str, Any]:
    """Deterministic propagation *abstraction* tied to scenario pressure (still proxy)."""
    env = _clamp01(rf_env)
    pen_db = 12 + 8 * env + (4 if height_m > 40 else 0)
    block_score = _clamp01(0.25 + 0.45 * env + 0.15 * (height_m / 70.0) + 0.12 * coverage_pressure)
    if height_m >= 55:
        los_label = "NLOS-heavy proxy (tall civic mass)"
    elif site_id == "west_side_leadership":
        los_label = "Mixed LOS / NLOS proxy (campus + parking)"
    else:
        los_label = "Partial LOS proxy (suburban civic)"
    cov = _clamp01(
        0.52 + 0.20 * (1.0 - env) + 0.08 * (1.0 - height_m / 70.0) - 0.14 * coverage_pressure - 0.06 * coexistence_pressure
    )
    p_challenge = _clamp01(
        0.28 * env + 0.18 * (height_m / 70.0) + 0.26 * block_score + 0.18 * coverage_pressure
    )
    return {
        "coverage_proxy": cov,
        "challenge_proxy": p_challenge,
        "los_nl_os": los_label,
        "penetration_db_proxy": pen_db,
        "blockage_proxy": block_score,
    }


ActionKey = Literal[
    "hold",
    "cautious",
    "power",
    "channel",
    "prioritize",
    "rebalance",
]


def select_closed_loop_action(
    state: SiteScenarioState,
    detector_pred: Optional[int],
    detector_conf: Optional[float],
    rf_env: float,
) -> Tuple[ActionKey, str]:
    """
    Discrete policy from **scenario state** + sensing (not narrative-only).
    """
    td = state.traffic_demand_score
    cx = state.coexistence_pressure
    fair = state.fairness_community_priority
    covp = state.coverage_pressure
    env = _clamp01(rf_env)

    if detector_pred is None:
        return "hold", "No detector belief on demo IQ — **hold** until sensing available (open Core Submission tab)."

    if detector_pred == 1:
        if cx > 0.72 and td > 0.62:
            return "cautious", "Occupied belief + **high coexistence** + **traffic** → **cautious transmit**."
        if env > 0.68:
            return "channel", "Occupied belief + **elevated RF stress** → **change channel** to reduce overlap (proxy policy)."
        if covp > 0.72:
            return "power", "Occupied belief + **high coverage pressure** → **reduce power** to protect margin (proxy)."
        return "cautious", "Occupied belief with moderate stress → **cautious transmit**."

    # pred == 0 (vacant / noise-only belief)
    if env > 0.72:
        return "hold", "Vacant belief but **hostile RF** — **hold** and sense (proxy)."
    if fair > 0.74 and td > 0.55:
        return "prioritize", "Vacant belief + **equity priority** + demand → **prioritize site** capacity (proxy)."
    if td > 0.78 and covp > 0.55:
        return "rebalance", "Vacant belief + **imbalanced load** → **rebalance service** across resources (proxy)."
    if td > 0.65:
        return "prioritize", "Vacant belief + capacity opportunity → **prioritize** focus site."
    return "hold", "Vacant belief — **hold** / scan (default conservative policy)."


def apply_action_to_kpis(
    action: ActionKey,
    base_coverage: float,
    base_coex: float,
    base_fair: float,
    base_energy: float,
    base_cont: float,
) -> Dict[str, float]:
    """Shift KPI proxies by chosen action (transparent heuristic)."""
    c, x, f, e, t = base_coverage, base_coex, base_fair, base_energy, base_cont
    if action == "hold":
        e = min(1.0, e + 0.08)
        t = max(0.0, t - 0.04)
    elif action == "cautious":
        c = max(0.0, c - 0.05)
        x = min(1.0, x + 0.06)
        e = min(1.0, e + 0.04)
    elif action == "power":
        x = min(1.0, x + 0.10)
        c = max(0.0, c - 0.08)
        e = min(1.0, e + 0.12)
    elif action == "channel":
        x = min(1.0, x + 0.08)
        c = min(1.0, c + 0.06)
        t = min(1.0, t + 0.04)
    elif action == "prioritize":
        f = min(1.0, f + 0.12)
        c = min(1.0, c + 0.08)
        x = max(0.0, x - 0.06)
        e = max(0.0, e - 0.05)
    elif action == "rebalance":
        f = min(1.0, f + 0.08)
        t = min(1.0, t + 0.08)
        c = min(1.0, c + 0.04)
        x = max(0.0, x - 0.04)
    return {
        "coverage": _clamp01(c),
        "coexistence": _clamp01(x),
        "fairness": _clamp01(f),
        "energy": _clamp01(e),
        "continuity": _clamp01(t),
    }


def public_defaults_summary() -> List[Tuple[str, str, str]]:
    """(site, item, classification) for UI."""
    return [
        ("West Side Leadership Academy", f"{WSLA_ENROLLMENT_STUDENTS} students (public-report-style; verify IDOE)", "sourced_default"),
        ("West Side Leadership Academy", f"{WSLA_STAFF_FTE_ASSUMED} staff FTE", "assumption"),
        ("Gary Public Library & Cultural Center", f"~{LIBRARY_PROGRAMMED_CAPACITY} programmed capacity baseline", "assumption_aggregate"),
        ("Gary City Hall", f"{CITY_HALL_EMPLOYEES_ASSUMED} employees, {CITY_HALL_VISITORS_BASE_ASSUMED} visitor baseline", "assumption"),
    ]
