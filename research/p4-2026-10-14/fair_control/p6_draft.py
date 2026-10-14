#!/usr/bin/env python3
"""Export/recheck review-only configuration and hypothetical planning; never run it."""
from pathlib import Path
import sys,json,argparse,hashlib,subprocess
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from fair_control.p6_configuration import candidate,validate_candidate,verify_seal,AUTH
from fair_control.p6_statistics import planning
from fair_control.core import digest
p=argparse.ArgumentParser();p.add_argument('--export',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
if a.export:
 pin=ROOT/'P6_SOURCE_REVISION.json'
 source=json.loads(pin.read_text())['implementation_content_sha'] if pin.exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 c=candidate(source,hashlib.sha256((ROOT.parent/'PROSPECTIVE_PROTOCOL.md').read_bytes()).hexdigest());validate_candidate(c);c['configuration_status']['configuration_technically_validated']=True;status=validate_candidate(c)
 a.export.mkdir(parents=True,exist_ok=True)
 seal={'candidate_sha256':digest(c),'source':c['source'],'authorization':dict(AUTH)};verify_seal(c,seal)
 for n,v in [('P6_EXPERIMENT_FREEZE_CANDIDATE.json',c),('P6_FREEZE_SEAL.json',seal),('P6_CONFIGURATION_VERIFICATION.json',status),('P6_SAMPLE_PLANNING.json',[planning(s) for s in [.05,.1,.15,.2,.3]])]:
  (a.export/n).write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps(status))
elif a.verify:
 c=json.loads((a.verify/'P6_EXPERIMENT_FREEZE_CANDIDATE.json').read_text());seal=json.loads((a.verify/'P6_FREEZE_SEAL.json').read_text());verify_seal(c,seal);print(json.dumps(validate_candidate(c)))
else:p.error('Choose --export or --verify; neither executes a research configuration')
