.PHONY: setup lint test contract-test benchmark ablation demo-research e2e clean smoke reproduce paper paper-reproduce

# Prefer an in-tree venv. System /usr/bin/python3 on macOS is often 3.9 and cannot
# import PEP 604 annotations used throughout src/ (requires-python >=3.10).
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHON_BOOTSTRAP := $(shell \
	for c in python3.14 python3.13 python3.12 python3.11 python3.10; do \
		command -v $$c >/dev/null 2>&1 && $$c -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null && { command -v $$c; break; }; \
	done)

setup:
	@if [ -z "$(PYTHON_BOOTSTRAP)" ]; then \
		echo "ERROR: need Python >=3.10 on PATH to create .venv (found none)."; \
		echo "Install python3.11+ then re-run: make setup"; \
		exit 1; \
	fi
	@if [ ! -x .venv/bin/python ]; then \
		echo "Creating .venv with $(PYTHON_BOOTSTRAP)"; \
		$(PYTHON_BOOTSTRAP) -m venv .venv; \
	fi
	.venv/bin/python -c 'import sys; assert sys.version_info >= (3,10), sys.version'
	.venv/bin/python -m pip install -U pip wheel
	.venv/bin/python -m pip install -r requirements.txt
	.venv/bin/python -m pip install 'scipy>=1.11' 'numpy>=1.24' jsonschema pytest
	@echo "setup OK: $$(.venv/bin/python --version)"

lint:
	@test -x .venv/bin/python || (echo "Run make setup first (Python >=3.10 venv required)"; exit 1)
	PYTHONPATH=src $(PY) -m compileall -q src/airan_research/gate2

test:
	@test -x .venv/bin/python || (echo "Run make setup first (Python >=3.10 venv required)"; exit 1)
	.venv/bin/python -c 'import sys; assert sys.version_info >= (3,10), sys.version'
	PYTHONPATH=src $(PY) -m pytest -q tests/test_airan_research.py tests/test_paper_ii_digital.py tests/gate2 tests/test_planning_closed_loop.py

contract-test:
	@test -x .venv/bin/python || (echo "Run make setup first (Python >=3.10 venv required)"; exit 1)
	PYTHONPATH=src $(PY) -m pytest -q tests/gate2

reproduce: test

paper-reproduce:
	@test -x .venv/bin/python || (echo "Run make setup first (Python >=3.10 venv required)"; exit 1)
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
