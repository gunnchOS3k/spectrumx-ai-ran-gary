.PHONY: test demo demo-research benchmark-toy map

test:
	pytest -q

demo-research:
	python3 scripts/demo_airan_policy.py --toy
