#!/usr/bin/env python3
from pathlib import Path
import sys,unittest,argparse,json,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import p7a
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
s=unittest.defaultTestLoader.discover(str(ROOT/'tests'));n=s.countTestCases()
with (a.output/'TESTS.txt').open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(s)
x={'evidence_class':'P7A_CONSTRUCTED_ENGINEERING_TESTS','collected':n,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'campaign_executed':False,'pilot_execution_authorized':False}
(a.output/'TEST_RESULT.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x));sys.exit(not r.wasSuccessful())
