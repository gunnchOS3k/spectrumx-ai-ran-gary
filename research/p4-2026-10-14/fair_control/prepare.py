#!/usr/bin/env python3
"""Validate constructed contracts. This executable cannot launch a research campaign."""
from pathlib import Path
import sys,json,unittest,argparse,platform,importlib.metadata,hashlib
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.validation import validate_config,validate_result,provenance
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.add_argument('--execute',action='store_true');a=parser.parse_args()
if a.execute:parser.exit(2,'Campaign execution is outside P4; no runner, reserved seeds or authorization are supplied.\n')
if not a.check:parser.error('Use --check for engineering validation only')
validate_config(json.loads((ROOT/'configs/design.json').read_text()))
V=ROOT/'verification';V.mkdir(exist_ok=True)
with (V/'UNIT_TESTS.txt').open('w') as out:
 result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
p=provenance(ROOT)
record={'schema_version':'0.1','evidence_class':'ENGINEERING_VALIDATION','campaign_executed':False,'publication_authorized':False,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_tree_sha256':p['source_tree_sha256']};validate_result(record)
(V/'ENGINEERING_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n');(V/'SOURCE_PROVENANCE.json').write_text(json.dumps(p,indent=2)+'\n')
(V/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{x:importlib.metadata.version(x) for x in ['jsonschema','numpy','scipy']},'fresh_installation_tested':False,'source':'Existing local environment; future dependency installation/cross-platform acceptance still pending.'},indent=2)+'\n')
print(json.dumps(record,indent=2));sys.exit(not result.wasSuccessful())
