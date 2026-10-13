# Paper II reproduction commands and version boundary

Original source: `86a5a6f3000fce7593ee1bdfe02bae6bc37d2498`; protocol predecessor: `72044d4008dc7624b1246b880387a693c37150e7`; revised extension source: `b44357f764d440f983132bd44a9c23c180591f70`. Recovered manuscript HEAD is `28917e3f58902ba4327825ad117d5597e2b903cc`. These are three distinct identities. Original/revised results must not be pooled.

Use Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and PyYAML 6.0.3. The exact current environment is in reproducibility/requirements-lock.txt. SciPy SLSQP must be installed: uniform fallback alone does not reproduce the intended optimization path. From the repository root:

```sh
PYTHONPATH=src python paper/submission/scripts/verify_heldout.py
PYTHONPATH=src python -m pytest -q tests/test_paper_ii_digital.py tests/test_rq2_fidelity_checkpoint.py
python paper/submission/scripts/build_evidence.py
python paper/submission/scripts/package_submission.py
```

The verifier imports each retained, self-contained digital_programme.py separately, executes its frozen held-out protocol and compares every field after recursively excluding compute_time_ms. Python/NumPy/SciPy versions and fresh output hashes are recorded in audit/independent_current_reproduction.json. Timing is platform-sensitive; a timing discrepancy is not silently counted as numerical parity. The archived original/revised full train/domain/ablation comparisons remain in audit/reproduction_report.json. Current selected tests are recorded separately rather than substituted for the earlier 5/31 historical source-test counts.

Each vintage has train seeds 0..4 and held-out seeds 100..104, 24 episodes of 16 slots per seed. The unit for interval estimation is a seed mean: five held-out units, not 1,920 independent users or packets. Student-t df=4; differences pair the same seed. Original report intervals used a rounded t=2.776; independent post-hoc intervals use SciPy's exact quantile. Protocol-before-outcome history is preserved, but no independently audited prospective registration is claimed.

To reproduce full historical packages, check out each exact source in an isolated directory and invoke its run_programme with include_heldout=True, include_domain_shift=True and include_ablations=True. The source snapshot and protocol retained in evidence/ are sufficient for the self-contained digital engine. The current verifier provides a narrower exact held-out rerun without modifying repository results or judged IQ code.

Compile the synchronized standalone manuscript from paper/submission with pdfLaTeX, BibTeX and two further pdfLaTeX passes, all with no shell escape. All eight pages were visually inspected. Essential figures have text descriptions and table/JSON equivalents; no tagged-PDF accessibility certification is claimed.
