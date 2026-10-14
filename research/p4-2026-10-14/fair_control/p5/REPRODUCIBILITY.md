# Reproduce P5 engineering acceptance

From this fair_control directory, with Python3.14.7 and packages pinned in requirements.txt (jsonschema4.26.0,numpy2.5.4,scipy1.18.1):

```sh
python p5_check.py --output verification/p5-recheck
python p5_run.py --mode ENGINEERING_TEST --output verification/p5-fixture-recheck
python p5_run.py --mode PILOT_EXPERIMENT --output verification/must-not-exist
```

The last command must exit2 without creating output. CONFIRMATORY_EXPERIMENT and HELDOUT_EVALUATION also exit2. Never change execution guards in this milestone. Do not run old frozen simulations or prepare.py to rewrite old P4 records.

The package has no absolute machine-path dependency. Scripts locate their own src/fixtures/schemas. In a clean copy SOURCE_REVISION.json retains the exact implementation-content commit; Git working copies report current HEAD as provenance. Source-tree hashes separately identify exact code/config/schema/test/doc inputs. Result records preserve configuration, model/controller/environment revisions, Python/packages, empty constructed seed manifest, observable/compute/resource declarations, primary all-task schema, secondary errors/queue/energy, atomic run state and raw checkpoint/outcomes. No new random seed list or inferential result is created.

Timing/OS/Python producer metadata is retained separately from the canonical deterministic payload. Repeat on the same source/runtime: payload hash must match; elapsed time is allowed to change and is checked finite/nonnegative. Across environments source/dynamic outcome checks are required; no bitwise numerical/cross-platform guarantee is implied. All current acceptance uses the existing local runtime, not a fresh network installation. To install later create a virtual environment and `python -m pip install -r requirements.txt`; fresh-install/cross-platform acceptance remains untested. Clean-copy acceptance is actual extraction/test/fixture/guard execution in a new temporary directory.

The portal P5 bundle includes this complete package, unchanged original P4 license notices plus repository LICENSE, review docs and SHA256SUMS. Each payload is checked before extraction; traversal/extra/missing payloads and byte mutation are rejected. Run the extracted `fair_control/p5_check.py` and p5_run.py. A checksum proves identity, not rights, calibration or science. No archive extraction needs unrelated production applications or their private data.
