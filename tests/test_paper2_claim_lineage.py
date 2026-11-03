"""Paper II lineage: committed tables/CSVs equal a fresh declared-seed run, and Results prose matches that data.

Host-process timing is excluded. It is not a scientific output of the seed list.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest

from airan_research.experiments.digital_programme import PROTOCOL_RELPATH, load_protocol, run_family

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
EXP = ROOT / "results" / "experiments"
ORIGINAL = PAPER / "submission" / "evidence" / "original"
AUDIT = PAPER / "preprint" / "v0.1" / "verification" / "SCIENTIFIC_AUDIT.json"

POLICY_TABLES = {"rq2_train_policies.tex", "rq2_heldout_policies.tex"}
HELD_QUOTED = (
    "information_equivalent",
    "oracle",
    "static",
    "edge_only",
    "no_adaptation",
    "optimization_based",
    "rule_based",
    "twin_informed",
    "cloud_only",
    "local_only",
)


def _fmt(x: float, nd: int = 4) -> str:
    return f"{float(x):.{nd}f}"


def _u(block: dict, policy: str) -> dict:
    return block["policies"][policy]["service_continuity_utility"]


@pytest.fixture(scope="module")
def regen(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper2")
    os.environ["MPLCONFIGDIR"] = str(out / "mpl")
    (out / "mpl").mkdir()
    proto = load_protocol(ROOT / PROTOCOL_RELPATH)
    split = proto["split"]
    held = run_family(proto, family="held_out", seeds=list(split["held_out_seeds"]))
    train = run_family(proto, family="in_distribution", seeds=list(split["train_seeds"]))
    shifts = {
        str(fam): run_family(proto, family=str(fam), seeds=list(split["held_out_seeds"]))
        for fam in split["domain_shift_families"]
    }
    ablations = {
        str(name): run_family(
            proto,
            family="held_out",
            seeds=list(split["held_out_seeds"]),
            policies=["twin_informed"],
            ablation=str(name),
        )
        for name in proto["ablations"]
    }
    exp = out / "experiments"
    exp.mkdir()
    for name, obj in {
        "rq2_cross_layer_continuity_heldout.json": held,
        "rq2_cross_layer_continuity_train.json": train,
        "rq2_cross_layer_continuity_domain_shift.json": shifts,
        "rq2_cross_layer_continuity_ablation.json": ablations,
    }.items():
        (exp / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("p2_generate_tables", PAPER / "scripts" / "generate_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.HELD = exp / "rq2_cross_layer_continuity_heldout.json"
    gen.TRAIN = exp / "rq2_cross_layer_continuity_train.json"
    gen.SHIFT = exp / "rq2_cross_layer_continuity_domain_shift.json"
    gen.ABL = exp / "rq2_cross_layer_continuity_ablation.json"
    gen.TABLES = out / "tables"
    gen.FIGS = out / "figures"
    gen.TABLES.mkdir()
    gen.FIGS.mkdir()
    assert gen.main() == 0
    return {"held": held, "train": train, "shift": shifts, "ablation": ablations, "out": out, "proto": proto}


def _norm(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _blank_compute_tex(name: str, text: str) -> str:
    if name not in POLICY_TABLES:
        return text
    lines = []
    for line in text.split("\n"):
        parts = line.split("&")
        if len(parts) == 8:
            parts[6] = " COMPUTE "
            line = "&".join(parts)
        lines.append(line)
    return "\n".join(lines)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            {k: v for k, v in row.items() if k != "compute_time_ms_mean"}
            for row in csv.DictReader(fh)
        ]


def _section(name: str) -> str:
    return _norm(PAPER / "sections" / f"{name}.tex")


@pytest.mark.parametrize("name", sorted(p.name for p in (PAPER / "tables").glob("*.tex")))
def test_committed_table_matches_regeneration(regen, name):
    fresh = regen["out"] / "tables" / name
    assert _blank_compute_tex(name, _norm(PAPER / "tables" / name)) == _blank_compute_tex(
        name, _norm(fresh)
    ), f"stale table {name}: run make paper-reproduce"


@pytest.mark.parametrize("name", sorted(p.name for p in (PAPER / "tables").glob("*.csv")))
def test_committed_table_csv_matches_regeneration(regen, name):
    fresh = regen["out"] / "tables" / name
    assert _rows(PAPER / "tables" / name) == _rows(fresh), f"stale table csv {name}"


@pytest.mark.parametrize("name", sorted(p.name for p in (PAPER / "figures").glob("*.csv")))
def test_committed_figure_csv_matches_regeneration(regen, name):
    fresh = regen["out"] / "figures" / name
    assert _rows(PAPER / "figures" / name) == _rows(fresh), f"stale figure data {name}"


def test_prose_seeds_match_declared_protocol(regen):
    proto = regen["proto"]
    assert list(proto["split"]["train_seeds"]) == [0, 1, 2, 3, 4]
    assert list(proto["split"]["held_out_seeds"]) == [100, 101, 102, 103, 104]
    assert regen["held"]["seeds"] == [100, 101, 102, 103, 104]
    assert regen["train"]["seeds"] == [0, 1, 2, 3, 4]
    method = _section("method")
    assert r"Train seeds $\{0,1,2,3,4\}$" in method
    assert r"held-out seeds $\{100,\ldots,104\}$" in method


def test_prose_heldout_utilities_match_regeneration(regen):
    held = regen["held"]
    text = _section("results")
    for name in HELD_QUOTED:
        stats = _u(held, name)
        assert _fmt(stats["mean"]) in text, name
        assert _fmt(stats["ci_low"]) in text, name
        assert _fmt(stats["ci_high"]) in text, name
    twin_d = held["effect_sizes"]["twin_informed"]["cohens_d_vs_no_adaptation_continuity"]
    opt_d = held["effect_sizes"]["optimization_based"]["cohens_d_vs_no_adaptation_continuity"]
    assert _fmt(twin_d) in text
    assert _fmt(opt_d) in text
    for stale in ("0.8274", "0.7930", "0.8066", "0.7046", "0.6310", "0.9797", "-5.92", "0.3714", "1.0244", "<0.1"):
        assert stale not in text


def test_prose_switch_cost_names_the_actual_maximum(regen):
    held = regen["held"]
    switches = {name: block["switch_cost"]["mean"] for name, block in held["policies"].items()}
    top = max(switches, key=switches.get)
    assert top == "adaptive_checkpoint"
    text = _section("results")
    assert "highest mean switch cost" in text
    assert "adaptive\\_checkpoint" in text
    assert _fmt(switches[top]) in text
    assert _fmt(switches["twin_informed"]) in text
    assert _fmt(switches["no_adaptation"]) in text


def test_prose_h1_rejected_and_h2_is_policy_structure(regen):
    held = regen["held"]
    twin = _u(held, "twin_informed")["mean"]
    base = _u(held, "no_adaptation")["mean"]
    linear = _u(held, "information_equivalent")["mean"]
    assert twin < base
    assert linear > twin
    text = _section("results")
    assert "is \\textbf{not} supported" in text
    assert "policy-structure effect" in text
    info = _u(held, "information_equivalent")
    oracle = _u(held, "oracle")
    assert info["ci_low"] <= oracle["ci_high"] and oracle["ci_low"] <= info["ci_high"]
    assert "CIs overlap" in text


def test_prose_slsqp_compute_exceeds_linear_without_quoting_wall_clock(regen):
    held = regen["held"]
    linear = held["policies"]["information_equivalent"]["compute_time_ms"]["mean"]
    assert held["policies"]["optimization_based"]["compute_time_ms"]["mean"] > linear
    assert held["policies"]["twin_informed"]["compute_time_ms"]["mean"] > linear
    text = _section("results")
    assert "HOST\\_PROCESS\\_TIMING" in text
    assert "4.3" not in text


def test_prose_ablation_matches_regeneration(regen):
    abl = regen["ablation"]
    full = _u(abl["full"], "twin_informed")
    nosw = _u(abl["no_switch_cost"], "twin_informed")
    edge = _u(abl["no_edge_observations"], "twin_informed")
    assert nosw["mean"] > full["mean"]
    text = _section("results")
    for stats in (full, nosw):
        assert _fmt(stats["mean"]) in text
        assert _fmt(stats["ci_low"]) in text
        assert _fmt(stats["ci_high"]) in text
    assert _fmt(edge["mean"]) in text
    for name, block in abl.items():
        if name in ("full", "no_edge_observations", "no_switch_cost"):
            continue
        other = _u(block, "twin_informed")["mean"]
        assert abs(other - full["mean"]) < 0.01
    assert "near the full twin policy" in text


def test_prose_domain_shift_matches_regeneration(regen):
    held = regen["held"]
    shift = regen["shift"]
    text = _section("results")
    outage_twin = _u(shift["outage_heavy"], "twin_informed")["mean"]
    outage_base = _u(shift["outage_heavy"], "no_adaptation")["mean"]
    assert outage_twin > outage_base
    assert _fmt(outage_twin) in text and _fmt(outage_base) in text
    listed = ("no_adaptation", "rule_based", "twin_informed", "information_equivalent")
    block = shift["high_blockage"]["policies"]
    leader = max(listed, key=lambda name: block[name]["service_continuity_utility"]["mean"])
    assert leader == "information_equivalent"
    assert _fmt(block[leader]["service_continuity_utility"]["mean"]) in text

    def rank(src, names):
        return tuple(sorted(names, key=lambda name: -src["policies"][name]["service_continuity_utility"]["mean"]))

    names = ("no_adaptation", "rule_based", "twin_informed", "information_equivalent", "oracle")
    assert rank(held, names) == rank(shift["energy_constrained"], names)
    assert "energy\\_constrained" in text and "ranked like the in-distribution held-out split" in text


def test_claims_match_current_json_not_the_older_generator(regen):
    held = regen["held"]
    shift = regen["shift"]
    abl = regen["ablation"]
    text = _norm(PAPER / "CLAIMS_TO_EVIDENCE.md")
    info = _u(held, "information_equivalent")
    twin = _u(held, "twin_informed")
    base = _u(held, "no_adaptation")
    assert _fmt(info["mean"]) in text and _fmt(info["ci_low"]) in text and _fmt(info["ci_high"]) in text
    assert _fmt(twin["mean"]) in text and _fmt(base["mean"]) in text
    assert _fmt(_u(abl["no_switch_cost"], "twin_informed")["mean"]) in text
    assert _fmt(_u(abl["full"], "twin_informed")["mean"]) in text
    assert _u(held, "twin_informed")["mean"] < base["mean"]
    assert _u(shift["energy_constrained"], "twin_informed")["mean"] < _u(shift["energy_constrained"], "no_adaptation")["mean"]
    assert _u(shift["outage_heavy"], "twin_informed")["mean"] > _u(shift["outage_heavy"], "no_adaptation")["mean"]
    assert _u(shift["high_blockage"], "twin_informed")["mean"] > _u(shift["high_blockage"], "no_adaptation")["mean"]
    assert "outage_heavy" in text and "high_blockage" in text
    assert "only on `outage_heavy`" not in text
    for stale in ("0.8274", "0.7930", "0.8066"):
        assert stale not in text


def _original_mean(family_json: dict, policy: str) -> str:
    return f"{family_json['policies'][policy]['service_continuity_utility']['mean']:.5f}"


def test_canonical_results_display_matches_preserved_json_and_revised_run(regen):
    """Original vintage stays the preserved JSON. Revised means follow the fresh declared-seed run."""
    original = json.loads((ORIGINAL / "rq2_cross_layer_continuity_heldout.json").read_text(encoding="utf-8"))
    shifted = json.loads((ORIGINAL / "rq2_cross_layer_continuity_domain_shift.json").read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    expected = [
        _original_mean(original, "twin_informed"),
        _original_mean(original, "no_adaptation"),
        _original_mean(original, "information_equivalent"),
        _original_mean(original, "oracle"),
        f"{original['policies']['twin_informed']['switch_cost']['mean']:.5f}",
        f"{original['policies']['no_adaptation']['switch_cost']['mean']:.5f}",
        _original_mean(shifted["high_blockage"], "twin_informed"),
        _original_mean(shifted["outage_heavy"], "twin_informed"),
        _original_mean(shifted["outage_heavy"], "no_adaptation"),
        _original_mean(shifted["energy_constrained"], "twin_informed"),
        f"{audit['switching']['gross_twin']:.5f}",
        f"{audit['switching']['gross_fallback']:.5f}",
        f"{audit['switching']['gross_gain']:.5f}",
        _fmt(_u(regen["held"], "adaptive_fidelity")["mean"], 5),
        _fmt(_u(regen["held"], "adaptive_checkpoint")["mean"], 5),
        _fmt(_u(regen["held"], "checkpoint_disabled")["mean"], 5),
    ]
    stale = (
        "0.79303",
        "0.80663",
        "0.82740",
        "0.82710",
        "0.82315",
        "0.81621",
        "0.48180",
        "0.15330",
        "0.00694",
        "0.61589",
        "0.70461",
        "0.67092",
        "0.79098",
        "0.74197",
        "0.62374",
        "0.68232",
    )
    for path in (PAPER / "owner_review_results.tex", PAPER / "submission" / "manuscript.tex"):
        text = _norm(path)
        for token in expected:
            assert token in text, f"{path.name} missing {token}"
        for token in stale:
            assert token not in text, f"{path.name} still has stale {token}"
