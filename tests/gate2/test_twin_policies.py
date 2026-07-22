"""SpectrumX Gate 2 twin-conditioned policy tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from airan_research.gate2.twin_policies import evaluate_policy, run_ablations, run_benchmark, sha256_file

FK = Path(__file__).resolve().parents[3] / "gunnchos-7gc-ai-ran-field-kit"
TWIN = FK / "fixtures/valid/twin_state_bundle.valid.json"
SCHEMA = FK / "contracts"


def test_uses_supplied_twin_state(tmp_path):
    out = tmp_path / "d.json"
    bundle = evaluate_policy(TWIN, "twin_informed", out, schema_dir=SCHEMA, seed=0)
    assert bundle["input_twin_state_hash"] == sha256_file(TWIN)


def test_policies_respond_to_input_changes(tmp_path):
    twin = json.loads(TWIN.read_text())
    twin["source_measurement"]["summary"]["mean_latency_ms"] = 5.0
    twin["outage_state"]["values"]["terrestrial_outage"] = False
    t1 = tmp_path / "t1.json"
    t1.write_text(json.dumps(twin))
    twin["source_measurement"]["summary"]["mean_latency_ms"] = 250.0
    twin["outage_state"]["values"]["terrestrial_outage"] = True
    for c in twin["connectivity_candidates"]:
        if c["network"] == "terrestrial":
            c["available"] = False
    t2 = tmp_path / "t2.json"
    t2.write_text(json.dumps(twin))
    b1 = evaluate_policy(t1, "twin_informed", tmp_path / "a.json", schema_dir=SCHEMA)
    b2 = evaluate_policy(t2, "twin_informed", tmp_path / "b.json", schema_dir=SCHEMA)
    assert b1["predicted_metrics"]["predicted_latency_ms"] != b2["predicted_metrics"]["predicted_latency_ms"]


def test_invalid_twin_fails(tmp_path):
    twin = json.loads(TWIN.read_text())
    twin["schema_version"] = "2.0.0"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(twin))
    with pytest.raises(Exception):
        evaluate_policy(bad, "static_uniform", tmp_path / "o.json", schema_dir=SCHEMA)


def test_benchmark_and_ablation_execute(tmp_path):
    rows = run_benchmark(TWIN, repetitions=2, warmup=0, seed=0, schema_dir=SCHEMA)
    assert rows and "mean_s" in rows[0]
    ab = run_ablations(TWIN, seed=0, schema_dir=SCHEMA)
    assert len(ab) >= 5
