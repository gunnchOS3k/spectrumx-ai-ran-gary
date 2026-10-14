#!/usr/bin/env python3
from pathlib import Path
import sys,json,unittest,argparse,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.runner import tree_manifest
p=argparse.ArgumentParser(description='Constructed engineering validation only; no research execution');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));n=suite.countTestCases()
with (a.output/'TESTS.txt').open('w') as out:r=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
m=tree_manifest()
for p in [ROOT/'p6_check.py',ROOT/'p6_worker.py']:
 m['files'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
from fair_control.core import digest
m['source_tree_sha256']=digest(m['files'])
v={'evidence_class':'CONSTRUCTED_ENGINEERING_VALIDATION','collected':n,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'source_tree_sha256':m['source_tree_sha256'],'campaign_executed':False,'publication_authorized':False}
(a.output/'TEST_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');(a.output/'SOURCE_PROVENANCE.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(v));sys.exit(not r.wasSuccessful())
