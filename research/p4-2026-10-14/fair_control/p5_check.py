#!/usr/bin/env python3
from pathlib import Path
import sys,unittest,json,argparse
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.runner import tree_manifest,run
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));count=suite.countTestCases()
with (a.output/'TESTS.txt').open('w') as out:r=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
v={'evidence_class':'CONSTRUCTED_ENGINEERING_VALIDATION','collected':count,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'campaign_executed':False,'publication_authorized':False,'source_tree_sha256':tree_manifest()['source_tree_sha256']}
(a.output/'TEST_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v));sys.exit(not r.wasSuccessful())
