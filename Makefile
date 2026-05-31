.PHONY: test demo-research e2e

test:
	PYTHONPATH=src pytest -q tests/test_airan_research.py

demo-research:
	python3 scripts/demo_airan_policy.py --toy

e2e:
	@mkdir -p results/e2e
	PYTHONPATH=src pytest -q tests/test_airan_research.py 2>&1 | tee results/e2e/e2e_terminal_output.txt
	python3 scripts/demo_airan_policy.py --toy >> results/e2e/e2e_terminal_output.txt
	python3 scripts/e2e_check_required_artifacts.py
