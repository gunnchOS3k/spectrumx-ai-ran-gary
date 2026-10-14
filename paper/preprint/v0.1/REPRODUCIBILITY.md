# Bounded reproducibility instructions
Run from the extracted evidence directory. No Git checkout, network, absolute host path or journal template is required. Dependencies must already be installed or installed by the reviewer in their own environment.

```sh
python3 -m venv ../review-env
. ../review-env/bin/activate
python -m pip install -r environment/acceptance-requirements.txt
python scripts/verify_integrity.py
python scripts/audit_science.py
python scripts/optimizer_bounds.py
MPLCONFIGDIR=generated/mpl python scripts/generate_figures.py
python scripts/verify_claims.py
python scripts/reproduce_heldout.py
python scripts/verify_integrity.py
```

The venv above is OUTSIDE the evidence root so exact manifest membership remains checkable. Generated outputs are written only under generated/ and are explicitly excluded from immutable-payload membership; Python __pycache__ is also excluded. No input is rewritten. SHA256SUMS enumerates every shipped file other than itself; source/MANIFEST.json separately binds19 payload files to exact Git objects. PROVENANCE.json distinguishes original/revised experimental SHAs and publication source SHA.

The final clean-extraction acceptance uses an existing recorded Python 3.14.7 environment (NumPy 2.5.4,SciPy 1.18.1,Matplotlib 3.11.2), not a new installation or external laboratory. Historical rerun environment Python 3.12.14/NumPy 2.3.5 is preserved separately. Frozen requirements describe broad historical packages; acceptance-requirements pins the actual audit environment. No cross-platform binary equality guarantee is made. The scientific script has1477 independent summary checks,9600 reconstructed input/selector assertions per version and10 small optimizer checks; optimizer_bounds adds100 existing seed/N component cases. These reconstruct existing inputs, not new primary experimental outcomes.

Only one final packaging acceptance reruns held-out seeds 100–104,24 episodes×16 slots/version, calling the frozen run_family and required SciPy SLSQP path. This is a bounded test of extracted instructions; no train/domain/ablation sweep, new seed or retrospective retuning. All retained old-family data remains supplied and independently checked arithmetically. Exact non-timing whole-payload comparison excludes only exact compute_time_ms branches; timing schema/finite values/seed mean/SD are separately validated and both old/new timings retained in generated/heldout_acceptance/REPORT.json. Scientific-value and seed-order mutations must fail; valid timing change must pass; invalid timing must fail. No producer/timestamp/hash exclusions are used in this Paper II payload comparison. File/provenance bytes are independently hashed.

Expected outputs: generated/SCIENTIFIC_AUDIT.json,OPTIMIZER_BOUNDS.json, figures/,tables/,CLAIM_VERIFICATION.json and heldout_acceptance/{original,revised,REPORT}.json. Archived verification/ contains completed audit records; generated results are new checks, not silently substituted evidence. Retained prior_audit test logs are historical, not a new test run. No per-slot action trace exists. Raw supplied JSON contains original summaries and seed means.

To build the separate source archive: extract it, put a TeX distribution with article,geometry,lmodern,amsmath,amssymb,booktabs,array,graphicx,url,natbib,hyperref and plainnat on PATH, then `sh BUILD.sh`. Figure exports/tables are included; figure_source/ contains relative plotting code and bounded inputs for regeneration using the same Python dependencies. No IEEE assets are needed. Clean build validates relative paths and references, not platform upload acceptance.
