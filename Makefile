.PHONY: test demo-research e2e

test:
	PYTHONPATH=src pytest -q tests/test_airan_research.py

demo-research:
	python3 scripts/demo_airan_policy.py --toy

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
