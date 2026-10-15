#!/usr/bin/env python3
from pathlib import Path
import sys,unittest,argparse,json,hashlib,platform
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
p=argparse.ArgumentParser(description='Constructed measurement QA only;loopback traffic only');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
s=unittest.defaultTestLoader.discover(str(ROOT/'tests'));n=s.countTestCases()
with (a.output/'TESTS.txt').open('w') as out:r=unittest.TextTestRunner(stream=out,verbosity=2).run(s)
files={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for folder in ['src','tests','schemas','android/src','vendor'] for q in sorted((ROOT/folder).rglob('*')) if q.is_file() and '__pycache__' not in q.parts}
for name in ['p7r_check.py','p7r_demo.py','p7r_run.py','android/build.py','android/AndroidManifest.xml','README.md','LICENSE']:
 q=ROOT/name
 if q.exists():files[name]=hashlib.sha256(q.read_bytes()).hexdigest()
x={'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','collected':n,'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'python':platform.python_version(),'source_file_sha256':files,'source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'external_network_requests':0,'device_measurements':0,'prospective_campaign_executed':False,'publication_authorized':False}
(a.output/'TEST_RESULT.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='source_file_sha256'},indent=2));sys.exit(not r.wasSuccessful())
