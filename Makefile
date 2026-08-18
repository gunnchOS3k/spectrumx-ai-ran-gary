.PHONY: setup lint test contract-test benchmark ablation demo-research e2e clean smoke reproduce paper paper-reproduce

PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

setup:
	python3 -m pip install -r requirements.txt
	python3 -m pip install 'scipy>=1.11' 'numpy>=1.24' jsonschema pytest

lint:
	PYTHONPATH=src $(PY) -m compileall -q src/airan_research/gate2

test:
	PYTHONPATH=src $(PY) -m pytest -q tests/test_airan_research.py tests/test_paper_ii_digital.py tests/gate2

contract-test:
	PYTHONPATH=src pytest -q tests/gate2

reproduce: test

paper-reproduce:
	@test -f paper/artifacts/experiment_protocol.yaml
	$(PY) scripts/demo_airan_policy.py --toy || true
	PYTHONPATH=src $(PY) scripts/run_paper_ii_heldout.py
	$(PY) paper/scripts/generate_tables.py

paper: paper-reproduce
	@test -f paper/manuscript.tex
	@test -f paper/MANUSCRIPT_STATUS.md
	@test -f paper/CITATION_AUDIT.md
	@test -f results/experiments/rq2_cross_layer_continuity_heldout.json
	@echo "Paper II: judged core preserved; digital tables SYNTHETIC_SIM; timing HOST_PROCESS_TIMING"

benchmark:
	@test -n "$(TWIN_STATE)" || (echo "Set TWIN_STATE=path/to/02_twin_state.json" && exit 1)
	PYTHONPATH=src python3 -m airan_research benchmark --twin-state $(TWIN_STATE) --output results/benchmark_results.csv --schema-dir $(SCHEMA_DIR)

ablation:
	@test -n "$(TWIN_STATE)" || (echo "Set TWIN_STATE=path/to/02_twin_state.json" && exit 1)
	PYTHONPATH=src python3 -m airan_research ablation --twin-state $(TWIN_STATE) --output results/ablation_results.csv --schema-dir $(SCHEMA_DIR)

demo-research:
	python3 scripts/demo_airan_policy.py --toy

clean:
	rm -rf results/benchmark_results.csv results/ablation_results.csv

e2e:
	@mkdir -p results/e2e
	PYTHONPATH=src pytest -q tests/test_airan_research.py 2>&1 | tee results/e2e/e2e_terminal_output.txt
	python3 scripts/demo_airan_policy.py --toy >> results/e2e/e2e_terminal_output.txt
	python3 scripts/run_all_tool_exports.py 2>> results/e2e/e2e_terminal_output.txt || true
	$(MAKE) e2e-tooling 2>> results/e2e/e2e_terminal_output.txt || true
	python3 scripts/e2e_check_required_artifacts.py

# Smoke test only — not evidence of readiness
smoke: e2e

e2e-tooling:
	@mkdir -p results/tool_exports
	python3 scripts/run_all_tool_exports.py 2>/dev/null || python3 scripts/check_optional_backends.py || true

e2e-sionna e2e-deepmimo e2e-aerial e2e-oran:
	@echo "Optional target $@ — requires external install; not run in default CI"
