#!/usr/bin/env python3
from pathlib import Path
import sys,json,unittest,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.p6_provenance import manifest
p=argparse.ArgumentParser(description='Constructed engineering validation only; no research execution');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));n=suite.countTestCases()
with (a.output/'TESTS.txt').open('w') as out:r=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
m=manifest()
v={'evidence_class':'CONSTRUCTED_ENGINEERING_VALIDATION','collected':n,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'expected_failures':len(r.expectedFailures),'unexpected_successes':len(r.unexpectedSuccesses),'source_tree_sha256':m['source_tree_sha256'],'campaign_executed':False,'publication_authorized':False}
(a.output/'TEST_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');(a.output/'SOURCE_PROVENANCE.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(v));sys.exit(not r.wasSuccessful())
